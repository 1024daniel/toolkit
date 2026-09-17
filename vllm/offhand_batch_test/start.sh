#!/usr/bin/env bash
# Environment overrides and foreground mode are used by the matrix supervisor.
set -u
set -o pipefail

MODEL="${MODEL:-/spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4-Flash/main}"
MODEL_NAME="${MODEL_NAME:-}"
TP="${TP:-8}"
PP="${PP:-1}"
DP="${DP:-1}"
EP="${EP:-false}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG="${LOG:-ds_v4_flash.log}"
START_USE_DEFAULT_ARGS="${START_USE_DEFAULT_ARGS:-1}"
RESET_COMM_ENV="${RESET_COMM_ENV:-1}"

FOREGROUND=0
FOLLOW_LOG=1
while (($#)); do
    case "$1" in
        --foreground) FOREGROUND=1; shift ;;
        --no-tail) FOLLOW_LOG=0; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

for name in TP PP DP PORT; do
    value="${!name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || ((${#value} > 9)); then
        printf 'ERROR: %s must be a positive integer (at most 9 digits).\n' "$name" >&2
        exit 2
    fi
done
if ((PORT > 65535)); then
    printf 'ERROR: PORT must be at most 65535.\n' >&2
    exit 2
fi
case "${EP,,}" in
    true|1) EP=true ;;
    false|0) EP=false ;;
    *) printf 'ERROR: EP must be true, false, 1, or 0.\n' >&2; exit 2 ;;
esac
for name in START_USE_DEFAULT_ARGS RESET_COMM_ENV; do
    if [[ "${!name}" != 0 && "${!name}" != 1 ]]; then
        printf 'ERROR: %s must be 0 or 1.\n' "$name" >&2
        exit 2
    fi
done

if [[ "$RESET_COMM_ENV" == 1 ]]; then
    unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_NVLS_ENABLE
    unset NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE
    unset VLLM_ALLREDUCE_USE_SYMM_MEM VLLM_ALLREDUCE_USE_FLASHINFER
fi

COMMAND=(uv run --no-sync vllm serve "$MODEL"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --data-parallel-size "$DP"
    --host "$HOST"
    --port "$PORT")
if [[ "$EP" == true ]]; then
    COMMAND+=(--enable-expert-parallel)
fi
if [[ -n "$MODEL_NAME" ]]; then
    COMMAND+=(--served-model-name "$MODEL_NAME")
fi
if [[ "$START_USE_DEFAULT_ARGS" == 1 ]]; then
    COMMAND+=(
        --kv-cache-dtype fp8
        --block-size 256
        --max-model-len 30000
        --gpu-memory-utilization 0.9
        --max-num-seqs 256
        --max-num-batched-tokens 16384
        --no-enable-flashinfer-autotune
        --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
        --tokenizer-mode deepseek_v4
        --tool-call-parser deepseek_v4
        --enable-auto-tool-choice
        --reasoning-parser deepseek_v4
        --no-enable-prefix-caching)
fi
COMMAND+=("$@")
if [[ -n "${OFFHAND_CONFIG_ENV_KEYS:-}" ]]; then
    python3 "$(dirname -- "${BASH_SOURCE[0]}")/log_environment.py" 'Server environment' || exit 1
fi
printf 'Server command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'

if ((FOREGROUND)); then
    exec "${COMMAND[@]}"
fi

mkdir -p -- "$(dirname -- "$LOG")" || exit 1
nohup "${COMMAND[@]}" > "$LOG" 2>&1 < /dev/null &
SERVER_PID=$!
printf 'Server PID: %s\nServer log: %s\n' "$SERVER_PID" "$LOG"
if ((FOLLOW_LOG)); then
    tail --pid="$SERVER_PID" -n +1 -F -- "$LOG"
fi
