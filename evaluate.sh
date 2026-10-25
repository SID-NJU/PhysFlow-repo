#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VIDEO_FOLDER=""
PROMPTS_FILE="${SCRIPT_DIR}/prompts/prompts.txt"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
MODEL_PATH="${QWEN_MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
INTERNVL_DEFAULT_MODEL="${INTERNVL_DEFAULT_MODEL:-OpenGVLab/InternVL2_5-8B}"
QWEN3VL_DEFAULT_MODEL="${QWEN3VL_DEFAULT_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MINICPMV_DEFAULT_MODEL="${MINICPMV_DEFAULT_MODEL:-openbmb/MiniCPM-V-2_6}"
GLM_DEFAULT_MODEL="${GLM_DEFAULT_MODEL:-zai-org/GLM-4.6V-Flash}"
DEEPSEEKVL_DEFAULT_MODEL="${DEEPSEEKVL_DEFAULT_MODEL:-deepseek-ai/deepseek-vl2-tiny}"
VIDEOLLAVA_DEFAULT_MODEL="${VIDEOLLAVA_DEFAULT_MODEL:-LanguageBind/Video-LLaVA-7B-hf}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
FRAMES_PER_VIDEO="8"
MAX_VIDEOS_PER_PROMPT="3"

usage() {
    cat << USAGE
Usage:
  bash evaluate.sh --video_folder <path> [--model_path <path_or_hf_id>] [options]

Required:
  --video_folder <path>          Folder containing generated videos to evaluate.

Optional:
  --model_path <path_or_hf_id>   Model path / Hub ID. Qwen3-VL default: ${QWEN3VL_DEFAULT_MODEL}
                                 InternVL default: ${INTERNVL_DEFAULT_MODEL}
                                 MiniCPM-V default: ${MINICPMV_DEFAULT_MODEL}
                                 GLM default: ${GLM_DEFAULT_MODEL}
                                 DeepSeek-VL2 default: ${DEEPSEEKVL_DEFAULT_MODEL}
                                 Video-LLaVA default: ${VIDEOLLAVA_DEFAULT_MODEL}
  --model_type <auto|qwen|qwen3vl|internvl|minicpmv|glm|deepseekvl|videollava>
                                 Model backend. Default: ${MODEL_TYPE}
  --prompts_file <path>          Prompt file path. Default: ${PROMPTS_FILE}
  --output_dir <path>            Output folder. Default: ${OUTPUT_DIR}
  --device <auto|cuda|cuda:0|cpu|mps>  Inference device. Default: ${DEVICE}
  --dtype <bfloat16|float16|float32>  Torch dtype. Default: ${DTYPE}
  --load_in_4bit <true|false>    Enable bitsandbytes 4bit loading. Default: ${LOAD_IN_4BIT}
  --disable_thinking <true|false>  Try to disable model thinking output for cleaner JSON (if supported). Default: ${DISABLE_THINKING}
  --attn_implementation <name>   Optional HF attention backend (e.g., flash_attention_2, sdpa)
  --max_new_tokens <int>         Max generated tokens for judge output. Default: ${MAX_NEW_TOKENS}
  --frames_per_video <int>       Sampled frame count per video. Default: ${FRAMES_PER_VIDEO}
  --max_videos_per_prompt <int>  0 means no limit. Default: ${MAX_VIDEOS_PER_PROMPT}

Example:
  bash evaluate.sh \
    --video_folder "${SCRIPT_DIR}/videos10/no_physics_frame41-fixed_seed42/prompts" \
    --model_path "${SCRIPT_DIR}/models/Qwen2.5-VL-7B-Instruct" \
    --model_type qwen \
    --device auto --dtype bfloat16
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video_folder) VIDEO_FOLDER="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --prompts_file) PROMPTS_FILE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --load_in_4bit) LOAD_IN_4BIT="$2"; shift 2 ;;
    --disable_thinking) DISABLE_THINKING="$2"; shift 2 ;;
    --attn_implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --frames_per_video) FRAMES_PER_VIDEO="$2"; shift 2 ;;
    --max_videos_per_prompt) MAX_VIDEOS_PER_PROMPT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${VIDEO_FOLDER}" ]]; then
  echo "Error: --video_folder is required"
  usage
  exit 1
fi

if [[ -z "${MODEL_PATH}" && ( "${MODEL_TYPE,,}" == "auto" || "${MODEL_TYPE,,}" == "qwen" ) ]]; then
  MODEL_PATH="${SCRIPT_DIR}/models/Qwen2.5-VL-7B-Instruct"
fi

if [[ -z "${MODEL_PATH}" && "${MODEL_TYPE,,}" == "internvl" ]]; then
  MODEL_PATH="${INTERNVL_DEFAULT_MODEL}"
  echo "No --model_path provided; using InternVL default: ${MODEL_PATH}"
fi
if [[ -z "${MODEL_PATH}" && "${MODEL_TYPE,,}" == "qwen3vl" ]]; then
  MODEL_PATH="${QWEN3VL_DEFAULT_MODEL}"
  echo "No --model_path provided; using Qwen3-VL default: ${MODEL_PATH}"
fi
if [[ -z "${MODEL_PATH}" && "${MODEL_TYPE,,}" == "minicpmv" ]]; then
  MODEL_PATH="${MINICPMV_DEFAULT_MODEL}"
  echo "No --model_path provided; using MiniCPM-V default: ${MODEL_PATH}"
fi
if [[ -z "${MODEL_PATH}" && "${MODEL_TYPE,,}" == "glm" ]]; then
  MODEL_PATH="${GLM_DEFAULT_MODEL}"
  echo "No --model_path provided; using GLM default: ${MODEL_PATH}"
fi
if [[ -z "${MODEL_PATH}" && "${MODEL_TYPE,,}" == "deepseekvl" ]]; then
  MODEL_PATH="${DEEPSEEKVL_DEFAULT_MODEL}"
  echo "No --model_path provided; using DeepSeek-VL2 default: ${MODEL_PATH}"
fi
if [[ -z "${MODEL_PATH}" && "${MODEL_TYPE,,}" == "videollava" ]]; then
  MODEL_PATH="${VIDEOLLAVA_DEFAULT_MODEL}"
  echo "No --model_path provided; using Video-LLaVA default: ${MODEL_PATH}"
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: --model_path is required (or set QWEN_MODEL_PATH)"
  usage
  exit 1
fi

if [[ ! -d "${VIDEO_FOLDER}" ]]; then
  echo "Error: video folder not found: ${VIDEO_FOLDER}"
  exit 1
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Error: prompts file not found: ${PROMPTS_FILE}"
  exit 1
fi

threshold=10 # GPU memory usage threshold (percent)
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
  python3 "${SCRIPT_DIR}/scripts/evaluate_motion_qa.py"
  --video-folder "${VIDEO_FOLDER}"
  --prompts-file "${PROMPTS_FILE}"
  --config-file "${SCRIPT_DIR}/config/question_sets.json"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
  --model-type "${MODEL_TYPE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --frames-per-video "${FRAMES_PER_VIDEO}"
  --max-videos-per-prompt "${MAX_VIDEOS_PER_PROMPT}"
)

if [[ "${LOAD_IN_4BIT,,}" == "true" ]]; then
  cmd+=(--load-in-4bit)
fi

if [[ "${DISABLE_THINKING,,}" == "true" ]]; then
  cmd+=(--disable-thinking)
fi

if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

"${cmd[@]}"
