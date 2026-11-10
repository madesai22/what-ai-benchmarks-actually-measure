#!/bin/bash
# Collect responses from one open-weights model with vLLM, end to end.
#
#   bash run_experiments/run_vllm_example.sh google/gemma-2-9b-it            # a few example benchmarks
#   bash run_experiments/run_vllm_example.sh google/gemma-2-9b-it --all      # every benchmark
#
# Env vars: TP (tensor-parallel size, default 1), PORT (default 8000),
#           MAX_CONCURRENT (default 8), VLLM_ARGS (extra args for `vllm serve`).
#
# Afterwards: run judge/run_judge.sh for the judged benchmarks, then score with
#   python run_experiments/make_run_script.py --model <model> --scoring && bash run_<model>_scoring.sh
set -euo pipefail

MODEL=${1:?usage: run_vllm_example.sh <hf-model-name> [--all]}
ALL=${2:-}
TP=${TP:-1}
PORT=${PORT:-8000}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
MODEL_NAME=$(echo "$MODEL" | tr '/' '_')

cd "$(dirname "$0")/.."           # repo root
mkdir -p logs
SERVER_LOG=logs/vllm_${MODEL_NAME}.log

# --- start the server, and stop it however this script exits ---------------
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" --dtype auto \
    --gpu-memory-utilization 0.85 --disable-log-stats ${VLLM_ARGS:-} > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM ($MODEL) on port $PORT; server log: $SERVER_LOG"
for _ in $(seq 300); do                    # up to 25 min for large downloads
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited:"; tail -n 50 "$SERVER_LOG"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "Timed out"; tail -n 50 "$SERVER_LOG"; exit 1; }
echo "Server ready."

# --- collect ----------------------------------------------------------------
# Commands come from each benchmark's *_config.json, so settings match the paper.
if [ "$ALL" = "--all" ]; then
    EVALS=()                       # every benchmark in data_acquisition/data
else
    EVALS=(--evals mmlu gsm8k xstest)   # multiple choice, math, judged open generation
fi
python run_experiments/make_run_script.py --model "$MODEL" --port "$PORT" \
    --max_concurrent "$MAX_CONCURRENT" --out "logs/run_${MODEL_NAME}.sh" ${EVALS[@]+"${EVALS[@]}"}
bash "logs/run_${MODEL_NAME}.sh"

echo "Done. Responses are in logs/$MODEL_NAME/"
