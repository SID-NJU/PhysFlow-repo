# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from time import time
import torch
import torch.cuda.amp as amp
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.model import sinusoidal_embedding_1d
from typing import List, Union, Optional, Tuple
import torch.nn.functional as F
import torch

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        output.append(x_i)
    return torch.stack(output).float()

@torch.no_grad() 
def generate_attention_mask(
    attention_map: torch.Tensor,
    grid_sizes: torch.Tensor,
    target_x_shape: Tuple[int, int, int, int], 
    batch_index: int = 0,
    target_word_indices: Union[List[int], slice] = None,
    head_index: Optional[int] = None, 
    word_aggregation_method: str = 'mean', 
    upsample_mode_spatial: str = 'nearest', 
    upsample_mode_temporal: str = 'nearest', 
    output_dtype: torch.dtype = torch.float32 
) -> torch.Tensor:
    """
    Generates a binary mask from an attention map based on attention towards target words.

    The mask identifies regions in the video (x) that attend strongly to the specified
    context words, exceeding a given threshold. The mask has the same dimensions as x.

    Args:
        attention_map (torch.Tensor): Attention weights [B, Head_num, Lx, Lctx].
                                      Lx = flattened video tokens (patches),
                                      Lctx = context tokens (words).
        target_word_indices (Union[List[int], slice]): Indices or slice for the target
                                                     word(s) in the Lctx dimension.
        grid_sizes (torch.Tensor): Patch grid dimensions [B, 3] -> (F, H_patch, W_patch)
                                   for each batch item, corresponding to Lx.
                                   F, H_patch, W_patch should be integers.
        target_x_shape (Tuple[int, int, int, int]): The desired output shape [C, T, H, W],
                                                    matching the original video tensor x.
        threshold (float): Value between 0 and 1. Attention scores >= threshold become 1 (True),
                           otherwise 0 (False).
        batch_index (int, optional): Batch item to process. Defaults to 0.
        head_index (Optional[int], optional): Specific head to use. If None, average
                                              attention across all heads. Defaults to None.
        word_aggregation_method (str, optional): How to aggregate scores if multiple
                                                 target_word_indices are given ('mean',
                                                 'sum', 'max'). Defaults to 'mean'.
        upsample_mode_spatial (str, optional): PyTorch interpolate mode for H, W dimensions.
                                           Defaults to 'nearest'.
        upsample_mode_temporal (str, optional): PyTorch interpolate mode for T dimension.
                                            Defaults to 'nearest'.
        output_dtype (torch.dtype, optional): Data type of the output mask.
                                              Defaults to torch.bool.

    Returns:
        torch.Tensor: A binary mask tensor of shape target_x_shape [C, T, H, W].

    Raises:
        TypeError: If inputs are not torch.Tensors.
        ValueError: If tensor dimensions or indices are invalid, or if
                    aggregation/upsample modes are unknown.
        IndexError: If batch_index or head_index are out of bounds.
    """
    
    if not isinstance(attention_map, torch.Tensor):
        raise TypeError("attention_map must be a torch.Tensor")
    if not isinstance(grid_sizes, torch.Tensor):
        raise TypeError("grid_sizes must be a torch.Tensor")
    if attention_map.dim() != 4:
        raise ValueError(f"attention_map must be [B, H, Lx, Lctx], got {attention_map.dim()} dims")
    if grid_sizes.dim() != 2 or grid_sizes.shape[1] != 3:
        raise ValueError(f"grid_sizes must be [B, 3], got {grid_sizes.shape}")
    if len(target_x_shape) != 4:
         raise ValueError(f"target_x_shape must be [C, T, H, W], got length {len(target_x_shape)}")

    B, H, Lx, Lctx = attention_map.shape
    C_out, T_out, H_out, W_out = target_x_shape

    if not 0 <= batch_index < B:
        raise IndexError(f"batch_index {batch_index} out of range for batch size {B}")
    if head_index is not None and not 0 <= head_index < H:
         raise IndexError(f"head_index {head_index} out of range for head count {H}")
    if word_aggregation_method not in ['mean', 'sum', 'max']:
        raise ValueError(f"Unknown word_aggregation_method: {word_aggregation_method}")
    if upsample_mode_spatial not in ['nearest', 'bilinear']:
        raise ValueError(f"Unknown upsample_mode_spatial: {upsample_mode_spatial}")
    if upsample_mode_temporal not in ['nearest', 'linear']:
         raise ValueError(f"Unknown upsample_mode_temporal: {upsample_mode_temporal}")

    if head_index is None:
        
        attn_map_processed = attention_map[batch_index].mean(dim=0)
    else:
        
        attn_map_processed = attention_map[batch_index, head_index]

    if isinstance(target_word_indices, slice):
        _slice_indices = range(*target_word_indices.indices(Lctx))
        if not _slice_indices: 
             num_words = 0
        elif _slice_indices.start >= Lctx or _slice_indices.stop < -Lctx : 
             num_words = len(_slice_indices) 
        else:
             num_words = len(_slice_indices)
        word_indices_str = f"slice({_slice_indices.start}:{_slice_indices.stop}:{_slice_indices.step})"
        word_attn_scores = attn_map_processed[:, target_word_indices] 
    elif isinstance(target_word_indices, list):
         
         valid_indices = [idx for idx in target_word_indices if -Lctx <= idx < Lctx]
         if not valid_indices:
             num_words = 0
             word_attn_scores = torch.empty((Lx, 0), device=attention_map.device, dtype=attention_map.dtype) 
         else:
            word_attn_scores = attn_map_processed[:, valid_indices] 
            num_words = len(valid_indices)
         word_indices_str = str(valid_indices) 
    else:
        raise TypeError(f"target_word_indices must be list or slice, got {type(target_word_indices)}")

    if num_words > 1:
        if word_aggregation_method == 'mean':
            aggregated_scores = word_attn_scores.mean(dim=-1)
        elif word_aggregation_method == 'sum':
            aggregated_scores = word_attn_scores.sum(dim=-1)
        elif word_aggregation_method == 'max':
            aggregated_scores = word_attn_scores.max(dim=-1).values
        
    elif num_words == 1:
         aggregated_scores = word_attn_scores.squeeze(-1) 
    else: 
         return torch.zeros(target_x_shape, dtype=output_dtype, device=attention_map.device)

    f_patch, h_patch, w_patch = map(int, grid_sizes[batch_index].tolist())
    actual_num_tokens = f_patch * h_patch * w_patch

    if actual_num_tokens == 0:
         return torch.zeros(target_x_shape, dtype=output_dtype, device=attention_map.device)

    if actual_num_tokens > Lx:
         
         padding_size = actual_num_tokens - aggregated_scores.numel()
         scores_padded = F.pad(aggregated_scores, (0, padding_size), "constant", 0)
         scores_unpadded = scores_padded 
         
    elif actual_num_tokens < Lx:
        scores_unpadded = aggregated_scores[:actual_num_tokens]
    else:
        scores_unpadded = aggregated_scores 

    try:
        
        attention_patch_grid = scores_unpadded.reshape(f_patch, h_patch, w_patch)
    except RuntimeError as e:
        raise e

    grid_for_upsample = attention_patch_grid.unsqueeze(0).unsqueeze(0).float() 

    target_size_3d = (T_out, H_out, W_out)

    if upsample_mode_temporal == 'linear' and f_patch != T_out:
        upsample_mode_3d = 'trilinear'
        align_corners_3d = False 
    else:

        upsample_mode_3d = 'nearest'
        align_corners_3d = None 

    upsampled_scores_grid = F.interpolate(grid_for_upsample,
                                       size=target_size_3d,
                                       mode=upsample_mode_3d,
                                       align_corners=align_corners_3d)

    upsampled_scores = upsampled_scores_grid.squeeze(0).squeeze(0)

    binary_mask_thw = (upsampled_scores / torch.max(upsampled_scores)) 

    final_mask = binary_mask_thw.unsqueeze(0).expand(C_out, T_out, H_out, W_out)

    return final_mask.to(dtype=output_dtype)

def usp_dit_forward( 
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    words_indices=None,
    block_id=-1,
    type=None,
    timestep=None
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        collect_attn_map=False)

    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]
    
    save_block_id = block_id 
    attn_map = None
    binary_mask = None
    for i, block in enumerate(self.blocks):
        kwargs["collect_attn_map"] = False
        if i == save_block_id:
            kwargs["collect_attn_map"] = True
            x, attn_map = block(x, **kwargs)
        else:
            x = block(x, **kwargs)

    x = self.head(x, e)
    
    x = get_sp_group().all_gather(x, dim=1)

    x = self.unpatchify(x, grid_sizes)

    if save_block_id != -1 and words_indices is not None:
        attention_map = get_sp_group().all_gather(attn_map, dim=2)
        binary_mask = generate_attention_mask(
                        attention_map=attention_map, 
                        target_word_indices=words_indices,
                        grid_sizes=grid_sizes, 
                        target_x_shape=x[0].shape, 
                        batch_index=0, 
                        head_index=None, 
                        word_aggregation_method='mean'
                        )

    return [u.float() for u in x], binary_mask

def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)

    x = x.flatten(2)
    x = self.o(x)
    return x
