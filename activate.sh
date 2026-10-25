#!/usr/bin/env bash
# Source this file after creating the environment from environment.yml.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Run: source activate.sh' >&2
    exit 1
fi
if ! declare -F conda >/dev/null; then
    if [[ -n "${CONDA_SH:-}" ]]; then
        source "$CONDA_SH" || return 1
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
    else
        echo 'Conda not found. Initialize conda or set CONDA_SH first.' >&2
        return 1
    fi
fi
conda activate "${PHYSFLOW_CONDA_ENV:-physflow}" || return 1
export TOKENIZERS_PARALLELISM=false
