#!/bin/bash
# Usage: ./run_model.sh <model> <eval> [eval ...]     or     ./run_model.sh <model> all
# Examples:
#   ./run_model.sh google/gemma-2-9b-it xstest mmlu gsm8k
#   TP=4 ./run_model.sh meta-llama/Llama-3.3-70B-Instruct all

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your-env>

MODEL=$1
shift
EVALS="$@"
TP=${TP:-1}   # number of GPUs
MODEL_NAME=$(echo $MODEL | tr '/' '_')

cd "$(dirname "$0")"


echo "Current active environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Running model: $MODEL"

rm -f nohup.out
# start server
nohup vllm serve "$MODEL" \
  --tensor-parallel-size $TP \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --swap-space 8 \
  --max-logprobs 150 \
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

# write the run_experiments_vllm.py commands for these benchmarks (settings come from
# each benchmark's *_config.json), then run them
if [ "$EVALS" = "all" ]; then
    python make_run_script.py --model $MODEL
else
    python make_run_script.py --model $MODEL --evals $EVALS
fi
bash run_$MODEL_NAME.sh

kill "$(cat $MODEL_NAME.pid)"
rm $MODEL_NAME.pid
