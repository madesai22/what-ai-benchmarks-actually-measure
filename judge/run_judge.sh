#!/bin/bash
# Start a judge model with vLLM and label the collected responses in logs/.
#
#   bash judge/run_judge.sh qwen          # Qwen3-30B judge for xstest, sgxstest, xsafety, or-bench, civics
#   bash judge/run_judge.sh llama         # same benchmarks with the Llama-3.3-70B judge
#   bash judge/run_judge.sh harm_bench    # a benchmark with its own classifier
#                                         # (harm_bench, sorry-bench, wild-guard, sg_bench, salad-bench)
#
# Extra args after the first are passed to score_judge.py, e.g. --model google/gemma-2-9b-it
# Env vars: TP (tensor-parallel size, default 1), PORT (default 8001), VLLM_ARGS.
#
# The paper reports both qwen and llama verdicts. Llama-3.3-70B needs ~4x80GB GPUs (TP=4).
set -euo pipefail

TARGET=${1:?usage: run_judge.sh <qwen|llama|benchmark> [score_judge.py args]}
shift
TP=${TP:-1}
PORT=${PORT:-8001}

GENERAL_EVALS=(xstest sgxstest xsafety or-bench civics)
case "$TARGET" in
    qwen)        JUDGE=Qwen/Qwen3-30B-A3B-Instruct-2507 ;;
    llama)       JUDGE=meta-llama/Llama-3.3-70B-Instruct ;;
    harm_bench)  JUDGE=cais/HarmBench-Llama-2-13b-cls ;;
    sorry-bench) JUDGE=sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406 ;;
    wild-guard)  JUDGE=allenai/wildguard ;;
    sg_bench)    JUDGE=meta-llama/LlamaGuard-7b ;;
    salad-bench) JUDGE=OpenSafetyLab/MD-Judge-v0.1 ;;
    *) echo "unknown target: $TARGET"; exit 1 ;;
esac

cd "$(dirname "$0")/.."           # repo root
mkdir -p logs
SERVER_LOG=logs/vllm_judge_$(echo "$JUDGE" | tr '/' '_').log

vllm serve "$JUDGE" --port "$PORT" --tensor-parallel-size "$TP" --dtype auto \
    --gpu-memory-utilization 0.85 --disable-log-stats ${VLLM_ARGS:-} > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for judge $JUDGE on port $PORT; server log: $SERVER_LOG"
for _ in $(seq 300); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited:"; tail -n 50 "$SERVER_LOG"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "Timed out"; tail -n 50 "$SERVER_LOG"; exit 1; }

cd judge
if [ "$TARGET" = qwen ] || [ "$TARGET" = llama ]; then
    for EVAL in "${GENERAL_EVALS[@]}"; do
        python score_judge.py --eval "$EVAL" --judge "$TARGET" --port "$PORT" "$@"
    done
    # Non-English XSafety: score_judge.py judges each response's English
    # 'translation' field, which must be added to the logs first, e.g.
    #   python score_judge.py --eval xsafety --judge "$TARGET" --lang fr
else
    python score_judge.py --eval "$TARGET" --port "$PORT" "$@"
fi
