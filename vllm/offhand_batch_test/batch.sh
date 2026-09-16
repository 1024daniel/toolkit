#!/usr/bin/env bash
# One running server, an explicit input/output matrix, and all concurrencies.
set -u
set -o pipefail

MODEL="${MODEL:-/spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4-Flash/main}"
MODEL_NAME="${MODEL_NAME:-}"
TOKENIZER="${TOKENIZER:-$MODEL}"
STRATEGY_NAME="${STRATEGY_NAME:-auto}"
TP="${TP:-8}"
PP="${PP:-1}"
DP="${DP:-1}"
EP="${EP:-false}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RESULT_DIR="${RESULT_DIR:-./bench_results}"
LOG_DIR="${LOG_DIR:-./bench_logs}"
GROUP_LOG="${GROUP_LOG:-}"
WORKLOADS_JSON="${WORKLOADS_JSON:-[[4096,1024]]}"
CONCURRENCIES_JSON="${CONCURRENCIES_JSON:-[1,4,16,32,64,128,256]}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
PROMPTS_MULTIPLIER="${PROMPTS_MULTIPLIER:-1}"
CASE_TIMEOUT="${CASE_TIMEOUT:-3600}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
EXTRA_ARGS=("$@")

# Parse JSON as data; never evaluate configuration as shell code.
export MODEL MODEL_NAME STRATEGY_NAME TP PP DP EP
export WORKLOADS_JSON CONCURRENCIES_JSON REQUEST_RATE PROMPTS_MULTIPLIER CASE_TIMEOUT CONTINUE_ON_ERROR
if ! MATRIX=$(python3 - <<'PY'
import json
import math
import os
import re
import sys
from pathlib import PurePosixPath

def integer(name, minimum=1):
    value = os.environ[name]
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if not minimum <= result <= 2147483647:
        raise ValueError(f"{name} must be between {minimum} and 2147483647")
    return result

def positive(value):
    return type(value) is int and 1 <= value <= 2147483647

def label(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value[:64] or "unnamed"

try:
    topology = [integer(name) for name in ("TP", "PP", "DP")]
    ep_value = os.environ["EP"].lower()
    if ep_value not in ("true", "false", "1", "0"):
        raise ValueError("EP must be true, false, 1, or 0")
    ep = str(topology[0] * topology[2]) if ep_value in ("true", "1") else "off"
    multiplier = integer("PROMPTS_MULTIPLIER")
    integer("CASE_TIMEOUT", 0)
    if os.environ["CONTINUE_ON_ERROR"] not in ("0", "1"):
        raise ValueError("CONTINUE_ON_ERROR must be 0 or 1")
    rate = os.environ["REQUEST_RATE"]
    if rate != "inf" and (not math.isfinite(float(rate)) or float(rate) <= 0):
        raise ValueError("REQUEST_RATE must be inf or a positive finite number")
    workloads = json.loads(os.environ["WORKLOADS_JSON"])
    concurrencies = json.loads(os.environ["CONCURRENCIES_JSON"])
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("WORKLOADS_JSON must be a nonempty list of [input, output] pairs")
    for pair in workloads:
        if not isinstance(pair, list) or len(pair) != 2 or not all(map(positive, pair)):
            raise ValueError("every workload must be [positive_integer_input, positive_integer_output]")
    if len(set(map(tuple, workloads))) != len(workloads):
        raise ValueError("duplicate input/output pairs are not allowed")
    if not isinstance(concurrencies, list) or not concurrencies or not all(map(positive, concurrencies)):
        raise ValueError("CONCURRENCIES_JSON must be a nonempty list of positive integers")
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("duplicate concurrencies are not allowed")
    if any(concurrency * multiplier > 2147483647 for concurrency in concurrencies):
        raise ValueError("concurrency * PROMPTS_MULTIPLIER must be at most 2147483647")
    model_path = PurePosixPath(os.environ["MODEL"].rstrip("/"))
    default_name = model_path.parent.name if model_path.name == "main" else model_path.name
    model = label(os.environ["MODEL_NAME"] or default_name)
    strategy = label(os.environ["STRATEGY_NAME"])
    tp, pp, dp = topology
    print(f"{model}-{strategy}-tp{tp}-pp{pp}-dp{dp}-ep{ep}")
    for input_len, output_len in workloads:
        for concurrency in concurrencies:
            print(input_len, output_len, concurrency, concurrency * multiplier, sep="\t")
except (ValueError, TypeError, OverflowError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(2)
PY
); then
    exit 2
fi

mapfile -t MATRIX_LINES <<< "$MATRIX"
FILE_PREFIX="${MATRIX_LINES[0]}"
TOTAL_CASES=$((${#MATRIX_LINES[@]} - 1))
RUN_TAG="$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p -- "$RESULT_DIR" "$LOG_DIR" || exit 1
if [[ -n "$GROUP_LOG" ]]; then
    mkdir -p -- "$(dirname -- "$GROUP_LOG")" || exit 1
    # Fail before starting requests if the aggregate destination cannot be opened.
    : >> "$GROUP_LOG" || exit 1
fi

run_case() {
    local index="$1" input_len="$2" output_len="$3" concurrency="$4" num_prompts="$5"
    local case_name="$6" log_file="$7" result_file="$8"
    local start_time end_time elapsed seed ret validation_ret
    local -a command
    start_time=$(date +%s)
    printf '\n============================================================\n'
    printf '[%s/%s] Start benchmark: %s\n' "$index" "$TOTAL_CASES" "$case_name"
    printf 'Model: %s\nTokenizer: %s\nStrategy: %s (TP=%s PP=%s DP=%s EP=%s)\n' \
        "$MODEL" "$TOKENIZER" "$STRATEGY_NAME" "$TP" "$PP" "$DP" "$EP"
    printf 'Input: %s  Output: %s  Concurrency: %s  Prompts: %s\n' \
        "$input_len" "$output_len" "$concurrency" "$num_prompts"
    printf 'Base URL: %s\nResult: %s/%s\nLog: %s\n' "$BASE_URL" "$RESULT_DIR" "$result_file" "$log_file"
    printf '============================================================\n'
    if ! seed=$(python3 -c 'import secrets; print(secrets.randbits(32))'); then
        printf 'FAILURE: unable to generate benchmark seed\n'
        return 1
    fi
    command=(uv run --no-sync vllm bench serve
        --seed "$seed"
        --backend openai-chat
        --endpoint /v1/chat/completions
        --base-url "$BASE_URL"
        --model "$MODEL"
        --tokenizer "$TOKENIZER"
        --dataset-name random
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio 0
        --max-concurrency "$concurrency"
        --num-prompts "$num_prompts"
        --request-rate "$REQUEST_RATE"
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 50,90,95,99
        --save-result
        --save-detailed
        --result-dir "$RESULT_DIR"
        --result-filename "$result_file"
        "${EXTRA_ARGS[@]}")
    printf 'Command: '
    printf '%q ' "${command[@]}"
    printf '\n'
    # Keep each case in its own process group, within the supervisor's session.
    # GNU timeout may exit before killing descendants when its direct child exits
    # on TERM. This wrapper always reaps the leader and cleans the entire group,
    # including descendants that retain the log pipe after their parent exits.
    python3 - "$CASE_TIMEOUT" "${command[@]}" <<'PY'
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

proc = None
exit_code = 1

class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum

def interrupted(signum, _frame):
    raise Interrupted(signum)

def report(message, *, error=False):
    try:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)
    except OSError:
        # A closed log pipe must not prevent process cleanup.
        pass

def live_group():
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) == proc.pid and fields[0] not in ('Z', 'X'):
                members.append(int(path.name))
        except (OSError, ValueError, IndexError):
            continue
    return members

def send_group(signum):
    try:
        os.killpg(proc.pid, signum)
    except ProcessLookupError:
        pass

for signum in (signal.SIGTERM, signal.SIGINT):
    signal.signal(signum, interrupted)
try:
    # preexec_fn is safe here: this small wrapper has no threads. Unlike setsid,
    # setpgrp retains the session owned by run_matrix.py for emergency cleanup.
    proc = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL,
                            preexec_fn=os.setpgrp)
    limit = int(sys.argv[1])
    try:
        code = proc.wait(timeout=limit if limit else None)
        exit_code = code if code >= 0 else 128 - code
    except subprocess.TimeoutExpired:
        report(f'ERROR: benchmark exceeded CASE_TIMEOUT={limit}s')
        exit_code = 124
except Interrupted as exc:
    exit_code = 128 + exc.signum
except OSError as exc:
    report(f'ERROR: could not launch benchmark: {exc}', error=True)
    exit_code = 127
finally:
    # Repeated shutdown signals must not interrupt the guaranteed KILL phase.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if proc is not None:
        proc.poll()
        if live_group():
            send_group(signal.SIGTERM)
            deadline = time.monotonic() + 30
            while live_group() and time.monotonic() < deadline:
                proc.poll()
                time.sleep(0.1)
            if live_group():
                report('Benchmark descendants remain after 30s; sending SIGKILL')
                send_group(signal.SIGKILL)
                deadline = time.monotonic() + 5
                while live_group() and time.monotonic() < deadline:
                    proc.poll()
                    time.sleep(0.1)
            if live_group():
                report('ERROR: benchmark process group could not be cleaned up', error=True)
                exit_code = 125
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            report('ERROR: benchmark leader could not be reaped', error=True)
            exit_code = 125
sys.exit(exit_code)
PY
    ret=$?
    validation_ret=0
    if ((ret == 0)); then
        python3 - "$RESULT_DIR/$result_file" "$num_prompts" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("benchmark result must be a JSON object")
    completed = result.get("completed")
    expected = int(sys.argv[2])
    if type(completed) is not int or completed != expected:
        raise ValueError(f"completed={completed!r}; expected exactly {expected} successful requests")
    if "failed" in result and (type(result["failed"]) is not int or result["failed"] != 0):
        raise ValueError(f"failed={result['failed']!r}; expected zero failed requests")
    print(f"Result verified: {completed}/{expected} requests completed")
except (OSError, ValueError, TypeError) as exc:
    print(f"ERROR: result validation failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
        validation_ret=$?
    fi
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    if ((ret != 0 || validation_ret != 0)); then
        printf 'FAILURE: %s; command_exit=%s; validation_exit=%s; elapsed=%ss\n' \
            "$case_name" "$ret" "$validation_ret" "$elapsed"
        if ((ret == 124 || ret == 125)); then
            return "$ret"
        fi
        return 1
    fi
    printf 'SUCCESS: %s; elapsed=%ss\n' "$case_name" "$elapsed"
}

run_matrix() {
    local current=0 passed=0 failed=0 input_len output_len concurrency num_prompts case_name log_file result_file row
    local -a statuses
    printf '============================================================\n'
    printf 'vLLM Benchmark Matrix\nModel: %s\nTokenizer: %s\nStrategy: %s\n' "$MODEL" "$TOKENIZER" "$STRATEGY_NAME"
    printf 'TP=%s PP=%s DP=%s EP=%s\nBase URL: %s\nRequest rate: %s\n' "$TP" "$PP" "$DP" "$EP" "$BASE_URL" "$REQUEST_RATE"
    printf 'Workloads: %s\nConcurrencies: %s\nTotal cases: %s\n' "$WORKLOADS_JSON" "$CONCURRENCIES_JSON" "$TOTAL_CASES"
    printf 'Timeout per case: %ss\nGroup log: %s\n' "$CASE_TIMEOUT" "${GROUP_LOG:-provided by caller, or stdout only}"
    printf '============================================================\n'
    for row in "${MATRIX_LINES[@]:1}"; do
        read -r input_len output_len concurrency num_prompts <<< "$row"
        current=$((current + 1))
        case_name="${FILE_PREFIX}-in${input_len}-out${output_len}-c${concurrency}-${RUN_TAG}"
        log_file="${LOG_DIR}/${case_name}.log"
        result_file="${case_name}.json"
        run_case "$current" "$input_len" "$output_len" "$concurrency" "$num_prompts" \
            "$case_name" "$log_file" "$result_file" 2>&1 | tee -- "$log_file"
        statuses=("${PIPESTATUS[@]}")
        if ((statuses[0] != 0 || statuses[1] != 0)); then
            failed=$((failed + 1))
            if ((statuses[1] != 0)); then
                printf 'ERROR: could not write complete case log: %s (tee exit %s)\n' "$log_file" "${statuses[1]}"
            fi
            if ((statuses[0] == 124 || statuses[0] == 125)); then
                printf 'Stopping remaining cases after timeout or process cleanup failure.\n'
                break
            fi
            if [[ "$CONTINUE_ON_ERROR" == 0 ]]; then
                break
            fi
        else
            passed=$((passed + 1))
        fi
    done
    printf '\n============================================================\n'
    printf 'Benchmark matrix finished\nTotal: %s  Attempted: %s  Passed: %s  Failed: %s  Skipped: %s\n' \
        "$TOTAL_CASES" "$current" "$passed" "$failed" "$((TOTAL_CASES - current))"
    printf '============================================================\n'
    ((failed == 0))
}

if [[ -n "$GROUP_LOG" ]]; then
    run_matrix 2>&1 | tee -a -- "$GROUP_LOG"
    MATRIX_STATUSES=("${PIPESTATUS[@]}")
    if ((MATRIX_STATUSES[1] != 0)); then
        printf 'ERROR: could not write complete group log: %s\n' "$GROUP_LOG" >&2
        exit 1
    fi
    exit "${MATRIX_STATUSES[0]}"
fi
run_matrix
