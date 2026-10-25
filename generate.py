

import argparse
from datetime import datetime
import hashlib
import logging
import os
import re
import sys
import warnings
import json
import csv

warnings.filterwarnings('ignore')

import torch
import random

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, str2bool

def _safe_prompt_slug(prompt: str, prompt_idx: int, max_bytes: int = 180) -> str:
    """Build a filesystem-safe slug with a strict byte budget."""
    text = (prompt or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9._ -]", "", text)
    text = text.strip(" .")

    if not text:
        text = f"prompt_{prompt_idx:03d}"

    budget = max(32, int(max_bytes))
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) > budget:
        digest = hashlib.md5(encoded).hexdigest()[:8]
        keep = budget - len(digest) - 1
        truncated = encoded[:max(1, keep)].decode("utf-8", errors="ignore").strip(" .")
        text = f"{truncated}_{digest}" if truncated else f"prompt_{prompt_idx:03d}_{digest}"

    return text

def _init_logging(rank: int, log_all_workers: bool = False):
    level = logging.INFO if (rank == 0 or log_all_workers) else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Batch physics-guided T2V generation for VBench evaluation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)

    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt. Mutually exclusive with --prompt_file.")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to txt/json/jsonl/csv file with prompts.")
    parser.add_argument("--prompt_key", type=str, default="prompt",
                        help="Key name for prompt in json/jsonl/csv files.")
    parser.add_argument("--n_prompt", type=str, default="")
    parser.add_argument("--size", type=str, default="832*480",
                        choices=list(SIZE_CONFIGS.keys()),
                        help="Video resolution. 832*480 for VBench-2.0 (480P).")
    parser.add_argument("--frame_num", type=int, default=81,
                        help="Number of frames. 81 = ~5s @ 16fps for VBench.")
    parser.add_argument("--base_seed", type=int, default=None,
                        help="Base seed for reproducibility. If omitted, a random seed is sampled per output video.")
    parser.add_argument("--num_videos_per_prompt", type=int, default=3,
                        help="Number of videos to generate per prompt.")
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)

    parser.add_argument("--use_physics", type=str2bool, default=True)
    parser.add_argument("--compare_physics", type=str2bool, default=True,
                        help="Generate both physics + no_physics per prompt.")
    parser.add_argument("--nsg_alpha", type=float, default=0.1,
                        help="NSG correction strength (fraction of ||v||).")
    parser.add_argument("--straight_beta", type=float, default=0.05,
                        help="Straightness correction strength.")
    parser.add_argument("--physics_decay", type=float, default=0.1,
                        help="(Kept for API compat; unused in new pipeline)")
    parser.add_argument("--physics_start_ratio", type=float, default=0.2,
                        help="Start physics at this fraction of steps.")
    parser.add_argument("--physics_end_ratio", type=float, default=0.8,
                        help="Stop physics at this fraction of steps.")
    parser.add_argument("--nsg_lambda", type=float, default=1e-2,
                        help="(Kept for API compat; lambda is now adaptive)")
    parser.add_argument("--safe_threshold", type=float, default=1.0,
                        help="Max correction magnitude (× mean ||v||).")

    parser.add_argument("--output_dir", type=str, default="videos")
    parser.add_argument("--shard_rank", type=int, default=0,
                        help="Shard worker rank for independent multi-process launch.")
    parser.add_argument("--shard_world_size", type=int, default=1,
                        help="Total shard workers for independent multi-process launch.")
    parser.add_argument("--log_all_workers", type=str2bool, default=False,
                        help="If true, all shard workers log at INFO level.")

    return parser.parse_args()

def load_prompts(args):
    """Load prompts from --prompt or --prompt_file."""
    prompts = []
    if args.prompt_file and os.path.exists(args.prompt_file):
        ext = os.path.splitext(args.prompt_file)[-1].lower()
        try:
            if ext == '.json':
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        prompts = [str(item.get(args.prompt_key, ""))
                                   for item in data if isinstance(item, dict)]
            elif ext == '.jsonl':
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            prompts.append(str(item.get(args.prompt_key, "")))
            elif ext == '.csv':
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    prompts = [str(row.get(args.prompt_key, "")) for row in reader]
            else:  
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to parse prompt file {args.prompt_file}: {e}")
            sys.exit(1)
        prompts = [p for p in prompts if p.strip()]
    elif args.prompt:
        prompts = [args.prompt]

    if not prompts:
        raise ValueError("No valid prompts found. Use --prompt or --prompt_file.")
    return prompts

def optimize(args):
    rank       = args.shard_rank
    world_size = args.shard_world_size
    device     = 0

    if world_size <= 0:
        raise ValueError("--shard_world_size must be >= 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid shard rank: rank={rank}, world_size={world_size}.")

    _init_logging(rank, args.log_all_workers)

    assert args.ckpt_dir is not None, "Please specify --ckpt_dir."
    assert args.prompt is not None or args.prompt_file is not None,\
        "Please specify either --prompt or --prompt_file."
    if args.num_videos_per_prompt <= 0:
        raise ValueError("--num_videos_per_prompt must be >= 1.")
    if args.base_seed is not None and args.base_seed < 0:
        logging.warning("--base_seed < 0 detected. Falling back to random seed mode.")
        args.base_seed = None
    if args.offload_model is None:
        args.offload_model = True

    logging.info(f"Shard mode enabled: rank={rank}, world_size={world_size}")

    prompts = load_prompts(args)
    logging.info(f"Loaded {len(prompts)} prompts.")
    seed_mode = "random per sample" if args.base_seed is None else f"base_seed={args.base_seed}"
    logging.info(
        f"Generation config: {args.num_videos_per_prompt} video(s) per prompt, seed mode: {seed_mode}."
    )

    cfg = WAN_CONFIGS[args.task]
    logging.info("Building WanT2V pipeline (this only happens once)...")
    wan_t2v = wan.WanT2VPhysFlow(
    
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    physics_modes = [True, False] if args.compare_physics else [args.use_physics]
    seed_tag_suffix = f"_seed{args.base_seed}" if args.base_seed is not None else ""

    for p_idx, current_prompt in enumerate(prompts):
        for mode_idx, use_phys in enumerate(physics_modes):
            physics_tag = (
                f"nsg{args.nsg_alpha}_str{args.straight_beta}"
                f"_s{args.physics_start_ratio:.1f}-{args.physics_end_ratio:.1f}"
                f'_th{args.safe_threshold:.1f}-decay{args.physics_decay:.2f}_frame{args.frame_num}-fixed'
                if use_phys else f"no_physics_frame{args.frame_num}-fixed"
            )
            physics_tag = f"{physics_tag}{seed_tag_suffix}"

            logging.info("-" * 60)
            logging.info(f"Task [{p_idx+1}/{len(prompts)}] | mode={physics_tag}")
            logging.info(f"Prompt: {current_prompt[:80]}...")
            if args.prompt_file:
                dimension_name = os.path.splitext(
                    os.path.basename(args.prompt_file))[0]
            else:
                dimension_name = "Single_Prompts"

            config_dir = os.path.join(
                args.output_dir, physics_tag, dimension_name)
            os.makedirs(config_dir, exist_ok=True)

            prompt_slug = _safe_prompt_slug(current_prompt, p_idx + 1, max_bytes=180)
            legacy_prompt_slug = current_prompt[:180]

            for sample_idx in range(args.num_videos_per_prompt):
                sample_no = sample_idx + 1
                global_sample_idx = (
                    (p_idx * len(physics_modes) + mode_idx) * args.num_videos_per_prompt
                    + sample_idx
                )

                if world_size > 1 and (global_sample_idx % world_size) != rank:
                    continue

                if args.base_seed is None:
                    current_seed = random.randint(0, sys.maxsize)
                else:
                    current_seed = args.base_seed + sample_idx

                save_file = os.path.join(
                    config_dir, f"{prompt_slug}-{sample_no}.mp4")
                legacy_save_file = os.path.join(
                    config_dir, f"{legacy_prompt_slug}-{sample_no}.mp4")

                if os.path.exists(save_file):
                    logging.info(f"Output already exists at {save_file}, skipping generation.")
                    continue

                if legacy_save_file != save_file and os.path.exists(legacy_save_file):
                    logging.info(f"Legacy output already exists at {legacy_save_file}, skipping generation.")
                    continue

                logging.info(
                    f"Sample [{sample_no}/{args.num_videos_per_prompt}] | seed={current_seed}"
                )

                generate_kwargs = dict(
                    input_prompt        = current_prompt,
                    size                = SIZE_CONFIGS[args.size],
                    frame_num           = args.frame_num,
                    shift               = args.sample_shift,
                    sample_solver       = args.sample_solver,
                    sampling_steps      = args.sample_steps,
                    guide_scale         = args.sample_guide_scale,
                    n_prompt            = args.n_prompt,
                    seed                = current_seed,
                    offload_model       = args.offload_model,
                    
                    use_physics         = use_phys,
                    nsg_alpha           = args.nsg_alpha,
                    straight_beta       = args.straight_beta,
                    physics_decay       = args.physics_decay,
                    physics_start_ratio = args.physics_start_ratio,
                    physics_end_ratio   = args.physics_end_ratio,
                    safe_threshold      = args.safe_threshold,
                )
                import time
                start_time = time.time()
                video = wan_t2v.generate(**generate_kwargs)
                end_time = time.time()
                print(f"Generation time for sample {sample_no}: {end_time - start_time:.2f} seconds")
                if video is None:
                    logging.warning("Generation returned None, retrying once with a different seed.")
                    retry_seed = current_seed + 1 if isinstance(current_seed, int) else random.randint(0, sys.maxsize)
                    generate_kwargs["seed"] = retry_seed
                    video = wan_t2v.generate(**generate_kwargs)

                if video is None:
                    raise RuntimeError(
                        f"Generation returned None twice for sample={sample_no}, prompt_idx={p_idx}, shard={rank}."
                    )

                logging.info(f"Saving → {save_file}")
                cache_video(
                    tensor      = video[None],
                    save_file   = save_file,
                    fps         = cfg.sample_fps,
                    nrow        = 1,
                    normalize   = True,
                    value_range = (-1, 1),
                )

    if rank == 0:
        logging.info("All tasks completed.")

if __name__ == "__main__":
    args = _parse_args()
    optimize(args)