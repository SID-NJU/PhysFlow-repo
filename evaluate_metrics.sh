#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


PROMPTS_FILE=""
PROMPTS_EXTENDED_FILE=""
OUTPUT_DIR="${SCRIPT_DIR}/outputs/metric_runs"
FRAMES_PER_VIDEO="8"
  CLIP_MODEL_PATH="${SCRIPT_DIR}/models/ViT-B-32.pt"
DINO_MODEL="dino_vitbase16"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float16}"

VIDEO_DIRS=()

usage() {
  cat << USAGE
Usage:
  bash evaluate_metrics.sh --video_dirs <dir1> [dir2 ...] [options]

Required:
  --video_dirs <dir1> [dir2 ...] One or multiple generated video folders.

Optional:
  --prompts_file <path>          Prompt file path for CLIP-SIM text mapping.
  --prompts_extended_file <path> Extended prompts file to map filenames to original prompts by index.
  --output_dir <path>            Output root folder.
  --frames_per_video <int>       Frames per video for CLIP-SIM. Default: 8
  --clip_model_path <path>       Local CLIP model .pt file. Default: ViT-B-32.pt
  --dino_model <name>            timm DINO model name (optional). Default: dino_vitbase16
  --device <auto|cuda|cuda:0|cpu|mps>
  --dtype <float16|float32|bfloat16>

Example:
  bash evaluate_metrics.sh \
    --video_dirs \
      ${SCRIPT_DIR}/videos/no_physics_frame41-fixed_seed42/prompts \
      ${SCRIPT_DIR}/videos/nsg5.0_str1.0_s0.0-0.7_th20.0-decay1.00_frame41-fixed_seed42/prompts \
    --prompts_file ${SCRIPT_DIR}/prompts/prompts.txt
    --prompts_extended_file ${SCRIPT_DIR}/prompts/prompts_extended.txt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video_dirs)
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        VIDEO_DIRS+=("$1")
        shift
      done
      ;;
    --prompts_file)
      PROMPTS_FILE="$2"
      shift 2
      ;;
    --prompts_extended_file)
      PROMPTS_EXTENDED_FILE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --frames_per_video)
      FRAMES_PER_VIDEO="$2"
      shift 2
      ;;
    --clip_model_path)
      CLIP_MODEL_PATH="$2"
      shift 2
      ;;
    --dino_model)
      DINO_MODEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#VIDEO_DIRS[@]} -eq 0 ]]; then
  echo "Error: --video_dirs is required and must contain at least one directory"
  usage
  exit 1
fi

for d in "${VIDEO_DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "Error: video dir not found: $d"
    exit 1
  fi
done

threshold=1 # GPU memory usage threshold (percent)
echo "Checking GPUs automatically (0,1,2,3)..."
while true; do
    for gpu_id in 0 1 2 3; do
        mem_info=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
        if [ -z "$mem_info" ]; then
            echo "nvidia-smi returned nothing for GPU $gpu_id, skipping..."
            continue
        fi
        used=$(echo "$mem_info" | awk -F',' '{print $1}' | tr -d ' ')
        total=$(echo "$mem_info" | awk -F',' '{print $2}' | tr -d ' ')
        if [ -z "$used" ] || [ -z "$total" ] || [ "$total" -eq 0 ]; then
            echo "Bad memory info for GPU $gpu_id: '$mem_info', skipping..."
            continue
        fi

        usage=$((used * 100 / total))
        echo "GPU $gpu_id usage: $usage%  ($used MiB / $total MiB)"

        if [ "$usage" -lt "$threshold" ]; then
            echo "GPU $gpu_id is free (<${threshold}%), selecting it..."
            export CUDA_VISIBLE_DEVICES=$gpu_id
            break 2
        fi
    done
    echo "No free GPU yet, waiting 10s..."
    sleep 10
done

mkdir -p "${OUTPUT_DIR}"

cmd=(
  python3 "${SCRIPT_DIR}/scripts/evaluate_clip_dino.py"
  --video-dirs
)

for d in "${VIDEO_DIRS[@]}"; do
  cmd+=("$d")
done

cmd+=(
  --prompts-file "${PROMPTS_FILE}"
  --prompts-extended-file "${PROMPTS_EXTENDED_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --frames-per-video "${FRAMES_PER_VIDEO}"
  --clip-model-path "${CLIP_MODEL_PATH}"
  --dino-model "${DINO_MODEL}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
)

"${cmd[@]}"
