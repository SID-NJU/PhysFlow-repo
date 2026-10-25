#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MULTI_GPU=True
DESIRED_GPUS=${DESIRED_GPUS:-1}
read -ra CANDIDATE_GPUS <<< "${GPU_IDS:-0}"
threshold=110

echo "Checking GPUs automatically (${CANDIDATE_GPUS[*]})..."

while true; do
    free_gpu_ids=()
    normal_gpu_ids=()
    for gpu_id in "${CANDIDATE_GPUS[@]}"; do
        mem_info=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
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
        normal_gpu_ids+=("$gpu_id")
        usage=$((used * 100 / total))
        echo "GPU $gpu_id usage: $usage%  ($used MiB / $total MiB)"
        if [ "$usage" -lt "$threshold" ]; then
            free_gpu_ids+=("$gpu_id")
        fi
    done

    if [ "${#free_gpu_ids[@]}" -eq "${#normal_gpu_ids[@]}" ] && [ "${#normal_gpu_ids[@]}" -ge "$DESIRED_GPUS" ]; then
        if [ "$MULTI_GPU" = "True" ]; then
            selected_gpu_ids=("${free_gpu_ids[@]:0:$DESIRED_GPUS}")
            NUM_PROCS="$DESIRED_GPUS"
        else
            selected_gpu_ids=("${free_gpu_ids[0]}")
            NUM_PROCS=1
        fi
        echo "All ${#normal_gpu_ids[@]} normal GPUs are free. Selected: ${selected_gpu_ids[*]}"
        break
    fi

    busy=$((${#normal_gpu_ids[@]} - ${#free_gpu_ids[@]}))
    echo "Waiting: $busy of ${#normal_gpu_ids[@]} normal GPUs still busy. Retrying in 10s..."
    sleep 10
done

PROMPT_DIR='./prompts'
COMPARE_BASELINE=True
USE_PHYSICS=True

if [[ "$PROMPT_DIR" == *ext* ]]; then
    OUTPUT_TAG_SUFFIX='-ext'
else
    OUTPUT_TAG_SUFFIX='10'
fi

NSG_ALPHA=5.0
STRAIGHT_BETA=1.0
PHYSICS_START=0
PHYSICS_END=0.7
SAFE_THRESHOLD=20.0
physics_decay=1

NUM_VIDEOS_PER_PROMPT=3
BASE_SEED="${BASE_SEED:-}"

if [ ! -d "$PROMPT_DIR" ]; then
    echo "Error: prompt dir not found: $PROMPT_DIR"
    exit 1
fi

for PROMPT_FILE in "$PROMPT_DIR"/*s.txt; do
    filename=$(basename -- "$PROMPT_FILE")
    dimension="${filename%.*}"

    echo "================================================================"
    echo "Start dimension: $dimension"
    echo "Prompt file: $PROMPT_FILE"
    echo "Workers: $NUM_PROCS on GPUs: ${selected_gpu_ids[*]}"
    echo "================================================================"

    pids=()
    for (( worker_rank=0; worker_rank<NUM_PROCS; worker_rank++ )); do
        gpu_id="${selected_gpu_ids[$worker_rank]}"

        (
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            cmd=(
                python generate.py
                --task                t2v-1.3B
                --size                832*480
                --ckpt_dir            ./Wan2.1-T2V-1.3B
                --sample_solver       unipc
                --sample_guide_scale  5.0
                --sample_shift        5.0
                --sample_steps        50
                --frame_num           41
                --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
                --prompt_file         "$PROMPT_FILE"
                --use_physics         "$USE_PHYSICS"
                --compare_physics     "$COMPARE_BASELINE"
                --nsg_alpha           "$NSG_ALPHA"
                --straight_beta       "$STRAIGHT_BETA"
                --physics_decay       "$physics_decay"
                --physics_start_ratio "$PHYSICS_START"
                --physics_end_ratio   "$PHYSICS_END"
                --safe_threshold      "$SAFE_THRESHOLD"
                --output_dir          videos$OUTPUT_TAG_SUFFIX
                --shard_rank          "$worker_rank"
                --shard_world_size    "$NUM_PROCS"
                --log_all_workers     True
            )

            if [ -n "$BASE_SEED" ]; then
                cmd+=(--base_seed "$BASE_SEED")
            fi

            echo "worker=$worker_rank gpu=$gpu_id launch"
            "${cmd[@]}"
        ) &

        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "Dimension failed: $dimension"
        exit 1
    fi

    echo "Done dimension: $dimension"
    echo ""
done

echo "All dimensions finished."
