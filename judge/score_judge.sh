#!/bin/bash
# Serve a judge model with vLLM, label one model's (or every model's) responses, then stop the server.
#
#   ./score_judge.sh <judge> <scored_model|all> <eval> [eval ...] [--override_results]
#
#   judge         qwen        Qwen/Qwen3-30B-A3B-Instruct-2507  (xstest, sgxstest, or-bench, xsafety, civics)
#                 llama       meta-llama/Llama-3.3-70B-Instruct (same benchmarks)
#                 classifier  the benchmark's own classifier from config.py
#                             (harm_bench, sorry-bench, wild-guard, sg_bench, salad-bench)
#   scored_model  model whose logs to judge, as in logs/ (google_gemma-2-9b-it or google/gemma-2-9b-it),
#                 or "all" for every model in logs/
#   eval          one or more benchmarks (names as in config.py)
#
# Examples:
#   ./score_judge.sh qwen google/gemma-2-9b-it xstest or-bench xsafety
#   TP=4 ./score_judge.sh llama all xstest sgxstest or-bench xsafety civics
#   ./score_judge.sh classifier all harm_bench
#   ./score_judge.sh llama google/gemma-2-9b-it or-bench --override_results   # re-judge everything
#
# Env vars: TP (tensor-parallel size, default 1; Llama-3.3-70B needs ~4x80GB GPUs),
#           PORT (default 8001), VLLM_ARGS (extra args for `vllm serve`).
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

JUDGE=$1
SCORED_MODEL=$2
shift 2
EVALS=()
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --*) EXTRA_ARGS+=("$arg") ;;
        *)   EVALS+=("$arg") ;;
    esac
done
TP=${TP:-1}
PORT=${PORT:-8001}

cd "$(dirname "$0")"              # judge/

# --- which model serves as the judge ----------------------------------------
case "$JUDGE" in
    qwen)  MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 ;;
    llama) MODEL=meta-llama/Llama-3.3-70B-Instruct ;;
    classifier)
        # every eval must use the same classifier, since one server is started
        MODEL=$(python -c '
import sys
from config import eval_to_judge
models = {eval_to_judge[e]["model"] for e in sys.argv[1:]}
if "multiple" in models:
    sys.exit("these evals have no classifier of their own; use qwen or llama")
if len(models) > 1:
    sys.exit("these evals use different classifiers; run them separately: " + ", ".join(sorted(models)))
print(models.pop())' "${EVALS[@]}") ;;
    *) echo "judge must be qwen, llama, or classifier (got: $JUDGE)"; exit 1 ;;
esac
MODEL_NAME=$(echo "$MODEL" | tr '/' '_')

MODEL_ARGS=()
if [ "$SCORED_MODEL" != all ]; then
    MODEL_ARGS=(--model "$SCORED_MODEL")
fi
JUDGE_ARGS=()
if [ "$JUDGE" != classifier ]; then
    JUDGE_ARGS=(--judge "$JUDGE")
fi

# --- start server --------------------------------------------------------------
mkdir -p ../logs
SERVER_LOG=../logs/vllm_judge_${MODEL_NAME}.log
echo "Starting judge $MODEL on port $PORT (server log: $SERVER_LOG)"
nohup vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --swap-space 8 \
  --disable-log-stats \
  ${VLLM_ARGS:-} \
 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "Stopping vLLM server ($SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

# --- wait for it to be ready ------------------------------------------------------
timeout=1500  # 25 minutes, for first-time model downloads
elapsed=0
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vLLM server exited. Last lines of its log:"; tail -n 50 "$SERVER_LOG"; exit 1
    fi
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout waiting for server to start. Last lines of its log:"; tail -n 50 "$SERVER_LOG"; exit 1
    fi
    echo "Server not ready yet, waiting... (${elapsed}s elapsed)"
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "vLLM server is ready!"

# --- judge ------------------------------------------------------------------------
for EVAL in "${EVALS[@]}"; do
    python score_judge.py --eval "$EVAL" --port "$PORT" \
        ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"} ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done
