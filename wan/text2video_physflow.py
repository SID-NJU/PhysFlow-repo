# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import cv2
from contextlib import contextmanager
from functools import partial
import torch.nn.functional as TF  
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import numpy as np
from typing import Optional, Literal
import itertools

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import\
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 
                 use_physics=False,       
                 nsg_alpha=0.1,           
                 straight_beta=0.05,      
                 physics_decay=0.1,       
                 physics_start_ratio=0.0, 
                 physics_end_ratio=1.0,   
                 safe_threshold=0.5,      
                 
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            use_physics (`bool`, *optional*, defaults to False):
                Enable Stage 1 physics-guided velocity correction (NSG + straightness).
            nsg_alpha (`float`, *optional*, defaults to 0.1):
                Strength of NSG probability flow conservation correction.
                Larger values apply stronger correction toward physical consistency.
            straight_beta (`float`, *optional*, defaults to 0.05):
                Strength of latent trajectory straightness correction.
                Reduces second-order curvature along the frame axis.
            physics_decay (`float`, *optional*, defaults to 0.1):
                Decay factor for physics mask edge softening.
            physics_start_ratio (`float`, *optional*, defaults to 0.0):
                Fraction of total steps at which physics correction begins (0.0 = from start).
            physics_end_ratio (`float`, *optional*, defaults to 1.0):
                Fraction of total steps at which physics correction ends (1.0 = until end).
            safe_threshold (`float`, *optional*, defaults to 0.5):
                Maximum allowed magnitude for combined physics correction.
                If correction exceeds this, it is proportionally scaled down.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """

        num_frames = frame_num
        target_shape = (self.vae.model.z_dim, (num_frames - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            physics_step_start = int(physics_start_ratio * sampling_steps)
            physics_step_end   = int(physics_end_ratio   * sampling_steps)

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond, _ = self.model(
                    latent_model_input, t=timestep, **arg_c)
                noise_pred_uncond, _ = self.model(
                    latent_model_input, t=timestep, **arg_null)
                
                noise_pred_cond, noise_pred_uncond = noise_pred_cond[0], noise_pred_uncond[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                if use_physics and physics_step_start <= step_idx < physics_step_end:
                    noise_pred = self.apply_stage1_physics(
                        noise_pred      = noise_pred,
                        z_t             = latents[0],
                        nsg_alpha       = nsg_alpha,
                        straight_beta   = straight_beta,
                        decay_factor    = physics_decay,
                        safe_threshold  = safe_threshold,  
                    )

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def compute_nsg_correction(
        self,
        v_theta: torch.Tensor,
        z_t:     torch.Tensor,
        alpha:   float = 0.1,
        eps:     float = 1e-4,
    ):
        """Compute float32 NSG corrections with per-pixel normalized directions."""
        
        v_f32 = v_theta.float()
        z_f32 = z_t.float()

        C, nf, H, W = v_f32.shape
        if nf < 2:
            return torch.zeros_like(v_f32), torch.zeros(1, 1, H, W, device=v_f32.device)

        score = v_f32[:, :-1]   

        dp_dt = z_f32[:, 1:] - z_f32[:, :-1]   

        denom = -dp_dt
        denom_safe = torch.where(
            denom.abs() < eps,
            torch.sign(denom + eps) * eps,
            denom
        )
        g_raw = score / denom_safe   

        g_norm = torch.norm(g_raw, p=2, dim=0, keepdim=True).clamp(min=eps)  
        g_nsg = g_raw / g_norm   

        vg_dot = (v_f32[:, :-1] * g_nsg).sum(dim=0, keepdim=True)  
        residual = vg_dot - 1.0

        delta_inner = -alpha * residual * g_nsg   

        delta_v = TF.pad(delta_inner, (0, 0, 0, 0, 0, 1))  

        return delta_v, residual

    def apply_stage1_physics(
        self,
        noise_pred:     torch.Tensor,
        z_t:            torch.Tensor,
        nsg_alpha:      float = 0.1,
        straight_beta:  float = 0.05,
        decay_factor:   float = 0.1,
        nsg_eps:        float = 1e-4,
        safe_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Combine masked NSG and straightness corrections, clip, and restore dtype."""
        original_dtype = noise_pred.dtype

        noise_pred_f32 = noise_pred.float()
        z_t_f32 = z_t.float()

        C, nf, H, W = noise_pred_f32.shape

        delta_nsg, residual = self.compute_nsg_correction(
            noise_pred_f32, z_t_f32, alpha=nsg_alpha, eps=nsg_eps
        )

        delta_str = self.compute_straightness_correction(
            z_t_f32, beta=straight_beta
        )

        residual_mag = residual.abs()   
        r_max = residual_mag.max()

        if r_max > nsg_eps:
            
            mask_inner = residual_mag / r_max
            last_frame = mask_inner[:, -1:, :, :]
            physics_mask = torch.cat([mask_inner, last_frame], dim=1)
            physics_mask = physics_mask.expand(C, nf, H, W)
        else:
            physics_mask = torch.ones(C, nf, H, W,
                                      device=noise_pred_f32.device,
                                      dtype=torch.float32)

        delta_total = (delta_nsg + delta_str) * physics_mask

        delta_total = torch.nan_to_num(delta_total, nan=0.0, posinf=0.0, neginf=0.0)

        max_correction = delta_total.abs().max()
        if max_correction > safe_threshold:
            delta_total = delta_total * (safe_threshold / max_correction)

        result = noise_pred_f32 + delta_total
        return result.to(original_dtype)

    def compute_straightness_correction(
        self,
        z_t:  torch.Tensor,
        beta: float = 0.05,
    ) -> torch.Tensor:
        """
        Latent trajectory straightness correction.

        Physics principle: natural video trajectories are approximately
        straight (low second-order curvature) in representation space.
        Reference: perceptual straightening hypothesis (ReStraV, CVPR2024).

        Curvature κ[f] = z[f+1] - 2·z[f] + z[f-1]  (2nd-order finite diff)
        Correction Δv = -beta * κ  (reduce curvature).

        This operates directly on z_t rather than v_theta because curvature
        is a property of the latent trajectory, not the velocity magnitude.

        Args:
            z_t:  current noisy latent [C, F, H, W]
            beta: correction strength

        Returns:
            delta_v: correction tensor [C, F, H, W]  (zero at frame boundaries)
        """
        C, num_frames, H, W = z_t.shape
        if num_frames < 3:
            return torch.zeros_like(z_t)

        curvature = z_t[:, 2:] - 2.0 * z_t[:, 1:-1] + z_t[:, :-2]  

        delta_inner = -beta * curvature      

        delta_v = TF.pad(delta_inner, (0, 0, 0, 0, 1, 1))  

        return delta_v

