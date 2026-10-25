#!/usr/bin/env python3
"""Compute CLIP-SIM and DINO/CLIP frame consistency using local models."""

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute CLIP-SIM and DINO/CLIP frame consistency for folders of videos.")
    p.add_argument("--video-dirs", nargs="+", required=True, help="One or more generated video folders.")
    p.add_argument("--prompts-file", default="", help="Optional prompts file (one prompt per line).")
    p.add_argument(
        "--prompts-extended-file",
        default="",
        help="Optional extended prompts file to map filenames to original prompts by index.",
    )
    p.add_argument("--output-dir", default="outputs/metric_runs", help="Output root directory.")
    p.add_argument("--frames-per-video", type=int, default=8, help="Frames per video for CLIP-SIM.")
    p.add_argument("--clip-model-path", default=str(Path(__file__).resolve().parents[1] / "models/ViT-B-32.pt"), help="Local CLIP model path.")
    p.add_argument("--dino-model", default="dino_vitbase16", help="timm DINO model name (optional).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"], help="dtype")
    return p.parse_args()

def get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

def list_videos(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

def read_prompts(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def parse_prompt_index(video: Path) -> Optional[int]:
    name = video.name.lower()
    parent = video.parent.name.lower()
    m = re.match(r"^(?:prompt[_-]?)?0*(\d{1,4})(?:[_-].*)?\.[a-z0-9]+$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^(?:prompt[_-]?)?0*(\d{1,4})$", parent)
    if m:
        return int(m.group(1))
    return None

def infer_prompt_from_filename(
    video: Path,
    prompts: List[str],
    extended_to_index: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    idx = parse_prompt_index(video)
    if idx is not None and 1 <= idx <= len(prompts):
        return prompts[idx - 1]
    stem = video.stem
    m = re.match(r"^(.*)-\d+$", stem)
    if m:
        stem = m.group(1).strip()
    else:
        stem = stem.strip()

    if extended_to_index is not None and stem in extended_to_index:
        mapped_idx = extended_to_index[stem]
        if 0 <= mapped_idx < len(prompts):
            return prompts[mapped_idx]

    return stem or None

def sample_frames_as_pil(video_path: Path, num_frames: int) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if num_frames <= 0:
        return frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return frames
        idxs = []
        for i in range(num_frames):
            idx = int((i + 0.5) * total / num_frames)
            idx = max(0, min(total - 1, idx))
            idxs.append(idx)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    finally:
        cap.release()
    return frames

def build_clip_from_local(model_path: str, device: torch.device):
    """Load CLIP model from local .pt file using open_clip or TorchScript."""
    try:
        import open_clip

        if "ViT-L-14" in model_path:
            model_id = "ViT-L-14"
        else:
            model_id = "ViT-B-32"

        try:
            model, _, transforms = open_clip.create_model_from_pretrained(model_id, pretrained=model_path)
            model.eval().to(device)
            return model, transforms
        except Exception:
            pass

        try:
            jit_model = torch.jit.load(model_path, map_location="cpu")
            jit_model.eval().to(device)
            _, _, transforms = open_clip.create_model_and_transforms(model_id, pretrained=False)
            return jit_model, transforms
        except Exception:
            pass

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model, _, transforms = open_clip.create_model_and_transforms(model_id, pretrained=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(device)
        return model, transforms
    except Exception as e:
        print(f"[warn] Failed to load CLIP from {model_path}: {e}")
        return None, None

def build_dino(model_name: str, device: torch.device):
    try:
        import timm
        model = timm.create_model(model_name, pretrained=True)
        model.eval().to(device)
        return model
    except Exception as e:
        import timm
        local_dino_path = (Path(__file__).resolve().parents[1] / "models/dino_vitbase16_pretrain.pth")
        if local_dino_path.exists():
            try:
                
                for candidate in ["vit_base_patch16_224", "vit_base_patch16_224_in21k", "vit_base_patch16_224.dino"]:
                    try:
                        model = timm.create_model(candidate, pretrained=False)
                        break
                    except Exception:
                        model = None
                if model is None:
                    raise RuntimeError("No compatible ViT base model found in timm")
                state = torch.load(str(local_dino_path), map_location="cpu", weights_only=False)
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                model.eval().to(device)
                print(f"[info] Loaded DINO from local cache: {local_dino_path}")
                return model
            except Exception as e2:
                print(f"[warn] Failed to load DINO from local cache: {e2}")
        print(f"[warn] DINO model not available: {e}")
        return None

def get_vision_transforms():
    """Standard vision transforms for CLIP-compatible models."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

def clip_frame_embeds(frames: List[Image.Image], model, transforms, device: torch.device) -> Optional[torch.Tensor]:
    if not frames or model is None:
        return None
    frame_tensors = [transforms(f).unsqueeze(0) for f in frames]
    batch = torch.cat(frame_tensors, dim=0).to(device)
    with torch.no_grad():
        feat = model.encode_image(batch)
    feat = torch.nn.functional.normalize(feat.float(), dim=-1)
    return feat.cpu()

def clip_text_embed(text: str, model, device: torch.device) -> torch.Tensor:
    if model is None:
        raise RuntimeError("CLIP model is required for text embeddings")

    try:
        import open_clip
        tokens = open_clip.tokenize([text])
        tokens = tokens.to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        return feat[0].cpu()
    except Exception as e:
        raise RuntimeError(f"Failed to encode CLIP text: {text}") from e

def dino_frame_embeds(frames: List[Image.Image], model, device: torch.device) -> Optional[torch.Tensor]:
    if model is None or not frames:
        return None
    try:
        import torchvision.transforms as T
        preprocess = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except Exception:
        return None

    tensors = [preprocess(f).unsqueeze(0) for f in frames]
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        try:
            out = model.forward(batch)
        except Exception:
            try:
                out = model(batch)
            except Exception:
                return None
    if isinstance(out, tuple):
        feat = out[0]
    else:
        feat = out
    feat = torch.nn.functional.normalize(feat.float(), dim=-1)
    return feat.cpu()

def frame_consistency_from_feats(feats: torch.Tensor) -> float:
    if feats is None:
        return float('nan')
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if feats.shape[0] < 2:
        return float('nan')
    fn = torch.nn.functional.normalize(feats, dim=-1)
    sims = (fn[:-1] * fn[1:]).sum(dim=1)
    return float(sims.mean().item())

def safe_name(p: Path) -> str:
    import hashlib
    base = p.name.replace("/", "_").replace(" ", "_")
    digest = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"

def run() -> None:
    args = parse_args()
    video_dirs = [Path(x).expanduser().resolve() for x in args.video_dirs]
    prompts_file = Path(args.prompts_file).expanduser().resolve() if args.prompts_file else None
    prompts = read_prompts(prompts_file)
    extended_prompts = read_prompts(Path(args.prompts_extended_file).expanduser().resolve()) if args.prompts_extended_file else []
    extended_to_index: Optional[Dict[str, int]] = None
    if prompts and extended_prompts:
        
        extended_to_index = {p: i for i, p in enumerate(extended_prompts)}
    device = get_device(args.device)
    dtype = torch.float32 if device.type == 'cpu' else (torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)
    timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir).expanduser().resolve() / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device}, dtype={dtype}")

    clip_model, clip_transforms = build_clip_from_local(args.clip_model_path, device)
    if clip_model is None:
        print("[warn] CLIP model failed to load; text encoding will use fallback.")
        clip_transforms = get_vision_transforms()
    else:
        print(f"[info] Loaded CLIP from {args.clip_model_path}")

    dino_model = build_dino(args.dino_model, device)
    if dino_model is None:
        print("[warn] DINO model not available; DINO consistency will be skipped.")
    text_cache: Dict[str, torch.Tensor] = {}
    summary_rows = []
    for folder in tqdm(video_dirs, desc="folders"):
        print(f"[info] evaluating folder: {folder}")
        videos = list_videos(folder)
        per_video = []
        clip_scores = []
        clip_cons_scores = []
        dino_cons_scores = []
        for video in tqdm(videos, desc=f"videos/{folder.name}", leave=False):
            prompt = infer_prompt_from_filename(video, prompts, extended_to_index)
            if not prompt:
                continue
            if prompt not in text_cache:
                text_cache[prompt] = clip_text_embed(prompt, clip_model, device)
            txt = text_cache[prompt]
            frames = sample_frames_as_pil(video, args.frames_per_video)
            if not frames:
                continue
            
            img_feats = clip_frame_embeds(frames, clip_model, clip_transforms, device)
            if img_feats is None or img_feats.numel() == 0:
                continue
            
            sim = torch.matmul(img_feats, txt.to(img_feats.device).unsqueeze(-1)).squeeze(-1)
            clip_sim = float(sim.mean().item())
            
            clip_cons = frame_consistency_from_feats(img_feats)
            
            dino_cons = float('nan')
            if dino_model is not None:
                try:
                    dino_feats = dino_frame_embeds(frames, dino_model, device)
                    dino_cons = frame_consistency_from_feats(dino_feats) if dino_feats is not None else float('nan')
                except Exception:
                    dino_cons = float('nan')
            per_video.append({
                "folder": str(folder),
                "video": str(video),
                "prompt": prompt,
                "clip_sim": clip_sim,
                "clip_frame_consistency": clip_cons,
                "dino_frame_consistency": dino_cons,
            })
            clip_scores.append(clip_sim)
            clip_cons_scores.append(clip_cons)
            if not (isinstance(dino_cons, float) and np.isnan(dino_cons)):
                dino_cons_scores.append(dino_cons)
        summary_rows.append({
            "folder": str(folder),
            "n_videos_total": len(videos),
            "n_videos_evaluated": len(per_video),
            "clip_sim_mean": float(np.mean(clip_scores)) if clip_scores else float('nan'),
            "clip_frame_consistency_mean": float(np.mean(clip_cons_scores)) if clip_cons_scores else float('nan'),
            "dino_frame_consistency_mean": float(np.mean(dino_cons_scores)) if dino_cons_scores else float('nan'),
        })
        detail_csv = out_root / f"detail_clip_consistency_{safe_name(folder)}.csv"
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["folder", "video", "prompt", "clip_sim", "clip_frame_consistency", "dino_frame_consistency"])
            writer.writeheader()
            writer.writerows(per_video)
    summary_csv = out_root / "summary_metrics.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_json = out_root / "summary_metrics.json"
    report = {
        "timestamp": timestamp,
        "video_dirs": [str(x) for x in video_dirs],
        "frames_per_video": args.frames_per_video,
        "clip_model_path": args.clip_model_path,
        "dino_model": args.dino_model,
        "device": str(device),
        "dtype": args.dtype,
        "summary": summary_rows,
    }
    summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] results saved to: {out_root}")
    print(f"[done] summary csv: {summary_csv}")
    print(f"[done] summary json: {summary_json}")

if __name__ == "__main__":
    run()
