#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText as AutoVisionLanguageModel
except Exception:
    try:
        from transformers import AutoModelForVision2Seq as AutoVisionLanguageModel
    except Exception:
        AutoVisionLanguageModel = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Glm4vForConditionalGeneration
except Exception:
    Glm4vForConditionalGeneration = None

try:
    from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
except Exception:
    VideoLlavaForConditionalGeneration = None
    VideoLlavaProcessor = None

try:
    from deepseek_vl2.models import DeepseekVLV2Processor
except Exception:
    DeepseekVLV2Processor = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.webm', '.avi'}

def build_safe_prompt_slug(prompt: str, prompt_idx: int, max_bytes: int = 180) -> str:
    """Mirror generator-side filename slug policy for robust matching."""
    text = (prompt or '').replace('\n', ' ').replace('\r', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9._ -]', '', text)
    text = text.strip(' .')

    if not text:
        text = f'prompt_{prompt_idx:03d}'

    budget = max(32, int(max_bytes))
    encoded = text.encode('utf-8', errors='ignore')
    if len(encoded) > budget:
        digest = hashlib.md5(encoded).hexdigest()[:8]
        keep = budget - len(digest) - 1
        truncated = encoded[:max(1, keep)].decode('utf-8', errors='ignore').strip(' .')
        text = f'{truncated}_{digest}' if truncated else f'prompt_{prompt_idx:03d}_{digest}'

    return text

def read_prompts(path: Path):
    return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def load_cfg(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def cat_for_idx(idx: int, cfg: dict):
    for c in cfg['categories']:
        s, e = c['range']
        if s <= idx <= e:
            return c
    raise ValueError(f'No category for prompt {idx}')

def build_prompt_slug_index(prompts: List[str]) -> Dict[str, List[int]]:
    slug_to_idx = {}
    for idx, prompt in enumerate(prompts, start=1):
        full_slug = prompt.strip().lower()
        if full_slug:
            slug_to_idx.setdefault(full_slug, []).append(idx)

        safe_slug = build_safe_prompt_slug(prompt, idx, max_bytes=180).lower()
        if safe_slug and safe_slug != full_slug:
            slug_to_idx.setdefault(safe_slug, []).append(idx)

        short_slug_raw = full_slug[:180]
        short_slug = short_slug_raw.strip()
        if short_slug and short_slug != full_slug:
            slug_to_idx.setdefault(short_slug, []).append(idx)
        if short_slug_raw and short_slug_raw != short_slug and short_slug_raw != full_slug:
            slug_to_idx.setdefault(short_slug_raw, []).append(idx)
    return slug_to_idx

def parse_idx_and_repeat(p: Path, prompt_slug_to_idx: Dict[str, List[int]]):
    f = p.name.lower()

    m = re.match(r'^(?:prompt[_-]?)?0*(\d{1,3})-(\d+)(?:[_-].*)?\.[a-z0-9]+$', f)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.match(r'^(?:prompt[_-]?)?0*(\d{1,3})(?:[_-].*)?\.[a-z0-9]+$', f)
    if m:
        return int(m.group(1)), 10**9

    parent = p.parent.name.lower()
    m = re.match(r'^(?:prompt[_-]?)?0*(\d{1,3})$', parent)
    if m:
        return int(m.group(1)), 10**9

    stem = p.stem.strip().lower()
    m = re.match(r'^(.*)-(\d+)$', stem)
    if m:
        slug = m.group(1).strip()
        repeat_id = int(m.group(2))
        ids = prompt_slug_to_idx.get(slug, [])
        if len(ids) == 1:
            return ids[0], repeat_id

    ids = prompt_slug_to_idx.get(stem, [])
    if len(ids) == 1:
        return ids[0], 10**9

    return None, None

def find_videos(video_dir: Path, n_prompts: int, prompt_slug_to_idx: Dict[str, List[int]]):
    mp = {i: [] for i in range(1, n_prompts + 1)}
    order = {i: [] for i in range(1, n_prompts + 1)}
    for p in sorted(video_dir.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        idx, repeat_id = parse_idx_and_repeat(p, prompt_slug_to_idx)
        if idx is not None and 1 <= idx <= n_prompts:
            order[idx].append((repeat_id, str(p), p))

    for idx in range(1, n_prompts + 1):
        order[idx].sort(key=lambda x: (x[0], x[1]))
        mp[idx] = [x[2] for x in order[idx]]

    return mp

def sample_frames(video: Path, n: int, tmp: Path):
    outs = []
    if n <= 0:
        return outs

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return outs

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return outs

        frame_indices = []
        for i in range(n):
            idx = int((i + 0.5) * total_frames / n)
            idx = max(0, min(total_frames - 1, idx))
            frame_indices.append(idx)

        for i, frame_idx in enumerate(frame_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = tmp / f'frame_{i:02d}.jpg'
            Image.fromarray(frame_rgb).save(img, format='JPEG', quality=90)
            if img.exists() and img.stat().st_size > 0:
                outs.append(img)
    finally:
        cap.release()

    return outs

def norm(v: str):
    t = (v or '').strip().lower()
    if t.startswith('y'):
        return 'yes'
    if t.startswith('n'):
        return 'no'
    return 'unknown'

def expected_answer_for_question(cat: dict, q_idx: int):
    
    exp = cat.get('better_answer', [])
    if isinstance(exp, list) and 0 <= q_idx < len(exp):
        x = norm(str(exp[q_idx]))
        if x in {'yes', 'no'}:
            return x
    return 'yes'

def parse_json(text: str):
    stripped = (text or '').strip()
    stripped = re.sub(r'<think>[\s\S]*?</think>', '', stripped, flags=re.IGNORECASE)
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped, flags=re.IGNORECASE)
        stripped = re.sub(r'\s*```$', '', stripped)
    try:
        return json.loads(stripped)
    except Exception:
        pass

    candidates = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(stripped):
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(stripped[start:i + 1])
                    start = -1

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {}

def get_dtype(name: str):
    n = name.lower()
    if n == 'bfloat16':
        return torch.bfloat16
    if n == 'float16':
        return torch.float16
    if n == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')

def resolve_model_type(model_path: str, requested_type: str) -> str:
    model_type = requested_type.lower()
    valid_types = {
        'auto', 'qwen', 'qwen3vl', 'internvl', 'minicpmv',
        'glm', 'deepseekvl', 'videollava',
    }
    if model_type not in valid_types:
        raise ValueError(f'Unsupported model type: {requested_type}')
    if model_type != 'auto':
        return model_type

    markers = [model_path.lower()]
    config_path = Path(model_path).expanduser() / 'config.json'
    if config_path.is_file():
        try:
            markers.append(json.dumps(json.loads(config_path.read_text(encoding='utf-8'))).lower())
        except (OSError, ValueError):
            pass
    if any('minicpm' in marker for marker in markers):
        return 'minicpmv'
    if any('internvl' in marker for marker in markers):
        return 'internvl'
    if any(token in marker for marker in markers for token in ('deepseek-vl', 'deepseek_vl')):
        return 'deepseekvl'
    if any(token in marker for marker in markers for token in ('video-llava', 'video_llava', 'videollava')):
        return 'videollava'
    if any(token in marker for marker in markers for token in ('glm4v', 'glm-4.6v', 'glm-4.5v', 'glm-4.1v')):
        return 'glm'
    qwen3vl_markers = ('qwen3_vl', 'qwen3-vl', 'qwen3vl')
    if any(token in marker for marker in markers for token in qwen3vl_markers):
        return 'qwen3vl'
    return 'qwen'

def build_model_and_processor(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but torch.cuda.is_available() is False')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS device requested but torch.backends.mps.is_available() is False')

    args.model_type = resolve_model_type(args.model_path, args.model_type)
    if args.model_type == 'videollava' and args.frames_per_video != 8:
        raise ValueError('Video-LLaVA requires exactly 8 frames; use --frames_per_video 8.')

    chat_target_device = None
    if args.model_type in {'internvl', 'minicpmv', 'deepseekvl'} and args.device == 'auto':
        chat_target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_type == 'qwen' and AutoVisionLanguageModel is None:
        raise RuntimeError(
            'No compatible auto model class was found in current transformers. '
            'Please install the transformers version required by the selected Qwen model.'
        )
    if args.model_type == 'qwen3vl' and Qwen3VLForConditionalGeneration is None:
        raise RuntimeError(
            'Qwen3-VL requires Qwen3VLForConditionalGeneration. '
            'Install: pip install -U "transformers>=4.57.0,<5" accelerate'
        )
    if args.model_type == 'glm' and Glm4vForConditionalGeneration is None:
        raise RuntimeError(
            'GLM-4.6V requires Glm4vForConditionalGeneration. '
            'Use a separate environment and install: pip install -U "transformers>=5.2.0" accelerate'
        )
    if args.model_type == 'videollava' and (
        VideoLlavaForConditionalGeneration is None or VideoLlavaProcessor is None
    ):
        raise RuntimeError(
            'Video-LLaVA support is unavailable in current transformers. '
            'Use a separate environment and install: pip install "transformers==4.46.3" accelerate'
        )
    if args.model_type == 'deepseekvl' and DeepseekVLV2Processor is None:
        raise RuntimeError(
            'DeepSeek-VL2 support package is not installed. Use a separate environment, '
            'install transformers==4.38.2, then install the official DeepSeek-VL2 repository.'
        )
    if args.model_type == 'deepseekvl' and args.load_in_4bit:
        raise ValueError('DeepSeek-VL2 backend does not support --load_in_4bit in this evaluator.')

    kwargs = {'trust_remote_code': True}
    if args.model_type == 'internvl':
        kwargs['low_cpu_mem_usage'] = True
        if args.attn_implementation:
            if args.attn_implementation != 'flash_attention_2':
                raise ValueError(
                    'InternVL supports --attn-implementation flash_attention_2; '
                    'leave the option empty to use its default attention backend.'
                )
            kwargs['use_flash_attn'] = True
    elif args.model_type == 'minicpmv':
        kwargs['low_cpu_mem_usage'] = True
        attention_backend = args.attn_implementation or 'sdpa'
        if attention_backend not in {'sdpa', 'flash_attention_2'}:
            raise ValueError(
                'MiniCPM-V supports --attn-implementation sdpa or flash_attention_2.'
            )
        kwargs['attn_implementation'] = attention_backend
    elif args.model_type == 'deepseekvl':
        kwargs['low_cpu_mem_usage'] = True
        if args.attn_implementation:
            kwargs['attn_implementation'] = args.attn_implementation
    elif args.attn_implementation:
        kwargs['attn_implementation'] = args.attn_implementation

    if args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError(
                'bitsandbytes support is unavailable in current transformers. '
                'Please upgrade transformers and install bitsandbytes.'
            )
        if not (args.device == 'auto' or args.device.startswith('cuda')):
            raise ValueError('--load-in-4bit requires --device auto or cuda[:id]')

        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=get_dtype(args.dtype),
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        if args.model_type in {'internvl', 'minicpmv'}:
            kwargs['device_map'] = {'': 0}
        else:
            kwargs['device_map'] = 'auto'
    else:
        kwargs['torch_dtype'] = get_dtype(args.dtype)
        if args.device == 'auto' and args.model_type not in {'internvl', 'minicpmv', 'deepseekvl'}:
            kwargs['device_map'] = 'auto'

    if args.model_type == 'internvl':
        processor = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(args.model_path, **kwargs)
    elif args.model_type == 'minicpmv':
        processor = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(args.model_path, **kwargs)
    elif args.model_type == 'qwen3vl':
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    elif args.model_type == 'glm':
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = Glm4vForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    elif args.model_type == 'deepseekvl':
        processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    elif args.model_type == 'videollava':
        processor = VideoLlavaProcessor.from_pretrained(args.model_path)
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.padding_side = 'left'
        model = VideoLlavaForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    else:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoVisionLanguageModel.from_pretrained(args.model_path, **kwargs)

    if not args.load_in_4bit:
        target_device = chat_target_device
        if target_device is None and args.device != 'auto':
            target_device = args.device
        if target_device is not None:
            model = model.to(target_device)

    model.eval()
    return processor, model

def internvl_pixel_values(images, model):
    image_size = 448
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    tensors = []
    for image in images:
        resized = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.array(resized, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append((tensor - mean) / std)

    try:
        vision_param = next(model.vision_model.parameters())
    except (AttributeError, StopIteration):
        vision_param = next(model.parameters())
    return torch.stack(tensors).to(device=vision_param.device, dtype=vision_param.dtype)

def judge_internvl(processor, model, system, user, images, max_new_tokens):
    pixel_values = internvl_pixel_values(images, model)
    num_patches_list = [1] * len(images)
    video_prefix = ''.join(f'Frame{i}: <image>\n' for i in range(1, len(images) + 1))
    question = f'{video_prefix}{system}\n\n{user}'
    generation_config = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
    }
    with torch.inference_mode():
        response = model.chat(
            tokenizer=processor,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
    return parse_json(response), response

def judge_minicpmv(processor, model, system, user, images, max_new_tokens):
    question = f'{system}\n\n{user}'
    messages = [{'role': 'user', 'content': images + [question]}]
    with torch.inference_mode():
        response = model.chat(
            image=None,
            msgs=messages,
            tokenizer=processor,
            sampling=False,
            max_new_tokens=max_new_tokens,
            use_image_id=False,
            max_slice_nums=1,
        )
    if not isinstance(response, str):
        response = str(response)
    return parse_json(response), response

def judge_glm(processor, model, system, user, frame_paths, max_new_tokens):
    content = [
        {'type': 'image', 'url': str(path)} for path in frame_paths
    ]
    content.append({'type': 'text', 'text': f'{system}\n\n{user}'})
    messages = [{'role': 'user', 'content': content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )
    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)
    inputs.pop('token_type_ids', None)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_length = inputs['input_ids'].shape[1]
    response = processor.decode(outputs[0][input_length:], skip_special_tokens=False)
    return parse_json(response), response
def judge_deepseekvl(processor, model, system, user, images, frame_paths, max_new_tokens):
    frame_prefix = ''.join(
        f'Frame {index}: <image>\n' for index in range(1, len(images) + 1)
    )
    conversation = [
        {
            'role': '<|User|>',
            'content': f'{frame_prefix}{system}\n\n{user}',
            'images': [str(path) for path in frame_paths],
        },
        {'role': '<|Assistant|>', 'content': ''},
    ]
    prepared = processor(
        conversations=conversation,
        images=images,
        force_batchify=True,
        system_prompt='',
    )
    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device
    prepared = prepared.to(model_device)

    language_model = getattr(model, 'language', None)
    if language_model is None:
        language_model = getattr(model, 'language_model', None)
    if language_model is None:
        raise RuntimeError('DeepSeek-VL2 model exposes neither language nor language_model.')

    tokenizer = processor.tokenizer
    attention_mask = (
        prepared.attention_mask
        if hasattr(prepared, 'attention_mask')
        else prepared['attention_mask']
    )
    with torch.inference_mode():
        inputs_embeds = model.prepare_inputs_embeds(**prepared)
        outputs = language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True)
    return parse_json(response), response

def judge_videollava(processor, model, system, user, images, max_new_tokens):
    if len(images) != 8:
        raise ValueError(f'Video-LLaVA requires exactly 8 decoded frames, got {len(images)}.')
    video = np.stack([np.asarray(image, dtype=np.uint8) for image in images])
    prompt = f'USER: <video>\n{system}\n\n{user}\nASSISTANT:'
    inputs = processor(text=prompt, videos=video, return_tensors='pt')
    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if hasattr(inputs, 'input_ids'):
        outputs = outputs[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_json(response), response

def judge(
    processor,
    model,
    prompt_text: str,
    cat_name: str,
    questions,
    frame_paths,
    max_new_tokens: int,
    disable_thinking: bool,
    model_type: str,
):
    system = (
        'You are a strict video evaluation judge. Return JSON only with schema '
        '{"answers":{"q1":"yes/no","q2":"yes/no","q3":"yes/no"},"brief_reason":"..."}. '
        'If only 2 questions, return q1 and q2 only.'
    )
    user = f'Prompt category: {cat_name}\nOriginal prompt: {prompt_text}\nQuestions:\n' + '\n'.join(questions)

    images = []
    for p in frame_paths:
        with Image.open(p) as img:
            images.append(img.convert('RGB'))

    if model_type == 'internvl':
        return judge_internvl(processor, model, system, user, images, max_new_tokens)
    if model_type == 'minicpmv':
        return judge_minicpmv(processor, model, system, user, images, max_new_tokens)
    if model_type == 'glm':
        return judge_glm(processor, model, system, user, frame_paths, max_new_tokens)
    if model_type == 'deepseekvl':
        return judge_deepseekvl(
            processor, model, system, user, images, frame_paths, max_new_tokens
        )
    if model_type == 'videollava':
        return judge_videollava(processor, model, system, user, images, max_new_tokens)

    if hasattr(processor, 'apply_chat_template'):
        user_content = [{'type': 'image', 'image': str(p)} for p in frame_paths]
        user_content.append({'type': 'text', 'text': user})
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
            {'role': 'user', 'content': user_content},
        ]
        if disable_thinking:
            try:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors='pt')
    else:
        text = system + '\n\n' + user
        inputs = processor(text=[text], images=images, return_tensors='pt')

    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device

    if hasattr(inputs, 'to'):
        inputs = inputs.to(model_device)
    else:
        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if hasattr(inputs, 'input_ids'):
        in_len = inputs.input_ids.shape[1]
        out = out[:, in_len:]
    elif isinstance(inputs, dict) and 'input_ids' in inputs:
        in_len = inputs['input_ids'].shape[1]
        out = out[:, in_len:]

    dec = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return parse_json(dec), dec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-folder', required=True, type=Path)
    ap.add_argument('--prompts-file', required=True, type=Path)
    ap.add_argument('--config-file', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-type', choices=('auto', 'qwen', 'qwen3vl', 'internvl', 'minicpmv', 'glm', 'deepseekvl', 'videollava'), default='auto')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--load-in-4bit', action='store_true')
    ap.add_argument('--attn-implementation', default='')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--frames-per-video', type=int, default=8)
    ap.add_argument('--max-videos-per-prompt', type=int, default=3)
    ap.add_argument('--disable-thinking', action='store_true')
    args = ap.parse_args()

    prompts = read_prompts(args.prompts_file)
    cfg = load_cfg(args.config_file)
    n_prompts = int(cfg.get('expected_prompt_count', 100))
    if len(prompts) != n_prompts:
        raise ValueError(f'Prompt count mismatch: expected {n_prompts}, got {len(prompts)}')

    processor, model = build_model_and_processor(args)

    prompt_slug_to_idx = build_prompt_slug_index(prompts)
    video_map = find_videos(args.video_folder, n_prompts, prompt_slug_to_idx)
    total_videos_planned = 0
    for i in range(1, n_prompts + 1):
        vids_i = video_map[i]
        if args.max_videos_per_prompt > 0:
            vids_i = vids_i[:args.max_videos_per_prompt]
        total_videos_planned += len(vids_i)

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.output_dir / f'run_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'[MotionStress] Start. Planned videos: {total_videos_planned}, prompts: {n_prompts}')

    rows = []
    processed_videos = 0
    start_t = time.time()
    raw_path = run_dir / 'raw_judgements.jsonl'
    with raw_path.open('w', encoding='utf-8') as fout:
        for idx in range(1, n_prompts + 1):
            cat = cat_for_idx(idx, cfg)
            qs = cat['questions']
            qn = len(qs)
            vids = video_map[idx]
            if args.max_videos_per_prompt > 0:
                vids = vids[:args.max_videos_per_prompt]

            for v in vids:
                processed_videos += 1
                with tempfile.TemporaryDirectory(prefix='mybench_frames_') as td:
                    fps = sample_frames(v, args.frames_per_video, Path(td))
                    if not fps:
                        elapsed = time.time() - start_t
                        print(
                            f'[MotionStress] {processed_videos}/{total_videos_planned} prompt={idx} video={v.name} status=failed_to_extract_frames elapsed={elapsed:.1f}s',
                            flush=True,
                        )
                        fout.write(json.dumps({'prompt_idx': idx, 'video_path': str(v), 'error': 'failed_to_extract_frames'}, ensure_ascii=False) + '\n')
                        continue

                    try:
                        obj, raw = judge(
                            processor,
                            model,
                            prompts[idx - 1],
                            cat['name'],
                            qs,
                            fps,
                            args.max_new_tokens,
                            args.disable_thinking,
                            args.model_type,
                        )
                        answers = obj.get('answers', {}) if isinstance(obj, dict) else {}
                        normed = {f'q{i}': norm(str(answers.get(f'q{i}', 'unknown'))) for i in range(1, qn + 1)}
                        yes = sum(1 for i in range(1, qn + 1) if normed.get(f'q{i}') == 'yes')
                        better = 0
                        for i in range(1, qn + 1):
                            expected = expected_answer_for_question(cat, i - 1)
                            if normed.get(f'q{i}') == expected:
                                better += 1
                        score = better / qn if qn else 0.0
                        rec = {
                            'prompt_idx': idx,
                            'category': cat['name'],
                            'video_path': str(v),
                            'question_count': qn,
                            'q1': normed.get('q1', ''),
                            'q2': normed.get('q2', ''),
                            'q3': normed.get('q3', ''),
                            'yes_count': yes,
                            'better_count': better,
                            'score': f'{score:.6f}',
                            'raw_response': raw,
                        }
                        rows.append(rec)
                        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        elapsed = time.time() - start_t
                        print(
                            f'[MotionStress] {processed_videos}/{total_videos_planned} prompt={idx} video={v.name} status=ok score={score:.3f} elapsed={elapsed:.1f}s',
                            flush=True,
                        )
                    except Exception as e:
                        fout.write(json.dumps({'prompt_idx': idx, 'video_path': str(v), 'error': str(e)}, ensure_ascii=False) + '\n')
                        elapsed = time.time() - start_t
                        print(
                            f'[MotionStress] {processed_videos}/{total_videos_planned} prompt={idx} video={v.name} status=error msg={str(e)} elapsed={elapsed:.1f}s',
                            flush=True,
                        )

    with (run_dir / 'video_scores.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['prompt_idx', 'category', 'video_path', 'question_count', 'q1', 'q2', 'q3', 'yes_count', 'better_count', 'score']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    p_scores = {i: [] for i in range(1, n_prompts + 1)}
    for r in rows:
        p_scores[int(r['prompt_idx'])].append(float(r['score']))

    p_rows = []
    for i in range(1, n_prompts + 1):
        c = cat_for_idx(i, cfg)
        vals = p_scores[i]
        p_rows.append({'prompt_idx': i, 'category': c['name'], 'video_count': len(vals), 'prompt_score': f'{statistics.mean(vals):.6f}' if vals else ''})

    with (run_dir / 'prompt_scores.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['prompt_idx', 'category', 'video_count', 'prompt_score'])
        w.writeheader()
        for r in p_rows:
            w.writerow(r)

    cat_scores = {}
    for c in cfg['categories']:
        s, e = c['range']
        vals = [float(x['prompt_score']) for x in p_rows if s <= int(x['prompt_idx']) <= e and x['prompt_score']]
        cat_scores[c['name']] = statistics.mean(vals) if vals else None

    all_scores = [float(r['score']) for r in rows]
    missing = [i for i in range(1, n_prompts + 1) if not p_scores[i]]
    summary = {
        'timestamp': ts,
        'model_path': args.model_path,
        'model_type': args.model_type,
        'device': args.device,
        'dtype': args.dtype,
        'load_in_4bit': args.load_in_4bit,
        'video_folder': str(args.video_folder),
        'disable_thinking': args.disable_thinking,
        'total_prompts': n_prompts,
        'prompts_with_video': n_prompts - len(missing),
        'prompts_missing_video': len(missing),
        'missing_prompt_indices': missing,
        'total_videos_scored': len(rows),
        'overall_video_score': statistics.mean(all_scores) if all_scores else None,
        'category_prompt_scores': cat_scores,
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[MotionStress] Done. Results written to: {run_dir}')

if __name__ == '__main__':
    main()
