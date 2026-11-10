#!/bin/bash
# Usage: ./score_judge.sh <judge> <scored_model|all> <eval> [eval ...] [--override_results] [--lang=fr]
#   judge: qwen, llama, or classifier (the benchmark's own judge from config.py)
# Examples:
#   ./score_judge.sh qwen google/gemma-2-9b-it xstest or-bench xsafety
#   TP=4 ./score_judge.sh llama all xstest sgxstest or-bench xsafety civics
#   ./score_judge.sh classifier all harm_bench

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your-env>

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
TP=${TP:-1}   # number of GPUs; Llama-3.3-70B needs TP=4

cd "$(dirname "$0")"

if [ "$JUDGE" = "qwen" ]; then
    MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
    JUDGE_ARG="--judge qwen"
elif [ "$JUDGE" = "llama" ]; then
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    JUDGE_ARG="--judge llama"
else
    # the benchmark's own classifier, e.g. cais/HarmBench-Llama-2-13b-cls for harm_bench
    MODEL=$(python -c "from config import eval_to_judge; print(eval_to_judge['${EVALS[0]}']['model'])")
    JUDGE_ARG=""
fi
MODEL_NAME=$(echo $MODEL | tr '/' '_')

if [ "$SCORED_MODEL" = "all" ]; then
    MODEL_ARG=""
else
    MODEL_ARG="--model $SCORED_MODEL"
fi


echo "Current active environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Judge: $MODEL"

rm -f nohup.out
# start server
nohup vllm serve "$MODEL" \
  --tensor-parallel-size $TP \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --swap-space 8 \
  --disable-log-stats \
 > nohup.out 2>&1 &

# Save the PID
echo $! > $MODEL_NAME.pid
echo "Started vLLM server with PID: $(cat $MODEL_NAME.pid)"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
timeout=1500  # 25 minutes timeout
elapsed=0

while true; do
    if grep -q "Application startup complete." nohup.out 2>/dev/null; then
        echo "vLLM server is ready!"
        break
    else
        if [ $elapsed -ge $timeout ]; then
            echo "Timeout waiting for server to start"
            echo "Last few lines of log:"
            tail -n 50 nohup.out 2>/dev/null
            kill "$(cat $MODEL_NAME.pid)"
            rm $MODEL_NAME.pid
            exit 1
        fi

        echo "Server not ready yet, waiting... (${elapsed}s elapsed)"
        echo "Last log line: $(tail -n 1 nohup.out 2>/dev/null)"
        sleep 5
        elapsed=$((elapsed + 5))
    fi
done

for EVAL in "${EVALS[@]}"; do
    python score_judge.py --eval $EVAL $JUDGE_ARG $MODEL_ARG "${EXTRA_ARGS[@]}"
done

kill "$(cat $MODEL_NAME.pid)"
rm $MODEL_NAME.pid
