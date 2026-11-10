# What AI Benchmarks Actually Measure — data acquisition & scoring code

Code for building the benchmark datasets, collecting and judging model responses, and scoring the model outputs analyzed in
*What AI Benchmarks Actually Measure: Adapting Convergent and Discriminant Validity
to Interrogate Fifty-Six AI Benchmarks* (Desai et al., 2026,
[arxiv.org/abs/2609.08812](https://arxiv.org/abs/2609.08812)).

🤗 The item-level model outputs and scores produced with this code are published
on the Hugging Face Hub:
[madesai/what-ai-benchmarks-actually-measure](https://huggingface.co/datasets/madesai/what-ai-benchmarks-actually-measure).

## Contents

| Path | What it does |
|---|---|
| `data_acquisition/get_data.py` | Downloads each benchmark from its source (Hugging Face, HELM, GitHub releases, …), samples it (fixed seeds), and writes `data/<benchmark>/<benchmark>.jsonl` plus a `*_config.json` with the system prompt, eval type, and scoring functions. |
| `score.py` | Turns raw model response logs (`logs/<model>/<eval>.jsonl`) into benchmark scores — exact match, F1, ROUGE, demographic parity, refusal accuracies from LLM-judge verdicts, etc. |
| `run_experiments/run_experiments_vllm.py` | Sends one benchmark file to a model served by vLLM and writes its responses to `logs/<model>/<eval>.jsonl`. |
| `run_experiments/make_run_script.py` | Writes the collection (or scoring) commands for every benchmark for one model, with the paper's per-benchmark settings. |
| `run_experiments/run_vllm_example.sh` | Starts a vLLM server for a model and collects responses, end to end. |
| `judge/` | Runs the LLM judges whose verdicts the safety and refusal benchmarks are scored from (`score_judge.py`, prompts in `config.py`, launcher `run_judge.sh`). |

## Building datasets

```bash
cd data_acquisition

# list the available benchmark builders
python get_data.py --list

# build one or several
python get_data.py gsm8k xstest

# build everything
python get_data.py --all
```

Outputs land in `./data/` relative to the working directory. Failures are
reported per dataset and don't stop the run.

Caveats:

- Some builders read from HELM's public benchmark output storage
  (`storage.googleapis.com/crfm-helm-public`); a few of those paths have moved
  since collection, so those builders may 404 until pointed at the current
  HELM release.
- `world_values` requires a manual download of
  [WorldValuesBench](https://github.com/Demon702/WorldValuesBench) (the builder
  prints instructions).
- Sampling uses fixed random seeds, so rebuilt datasets match the ones used in
  the paper wherever the upstream source is unchanged.

## Collecting model responses

Responses are collected from models served with [vLLM](https://docs.vllm.ai)
(any model with an OpenAI-compatible chat endpoint works). Build the datasets
first, then, on a GPU machine:

```bash
# a few example benchmarks
bash run_experiments/run_vllm_example.sh google/gemma-2-9b-it

# every benchmark (TP = number of GPUs to shard the model over)
TP=4 bash run_experiments/run_vllm_example.sh google/gemma-2-9b-it --all
```

Responses land in `logs/<model>/`. Sampling uses temperature 1, and each
benchmark's system prompt and token limit come from its `*_config.json`. Failed
requests are reported and skipped; rerunning resumes where the last run stopped.

To run single benchmarks against a server you started yourself:

```bash
vllm serve google/gemma-2-9b-it --port 8000 &
python run_experiments/make_run_script.py --model google/gemma-2-9b-it --evals xstest mmlu
bash run_google_gemma-2-9b-it.sh
```

Each response is stored with its token log-probabilities. Benchmarks scored from
answer-option probabilities (DiscrimEval) get `--log_probs 150` from their
config, which records each option's log-probability as the first generated
token; the server must allow it (`vllm serve ... --max-logprobs 150`, which
`run_vllm_example.sh` sets). API models (OpenAI, Anthropic, …) were queried with
separate clients that are not included here.

## LLM judges

The safety and refusal benchmarks (listed below) are scored from a judge model's verdict on
each response, so run the judges on the collected logs before scoring them.
Each benchmark's judge, prompt, and output column are set in `judge/config.py`:

| Benchmarks | Judge | Command |
|---|---|---|
| XSTest, SGXSTest, OR-Bench, XSafety, Civics | Qwen3-30B-A3B-Instruct-2507 and Llama-3.3-70B-Instruct | `bash judge/run_judge.sh qwen`, `TP=4 bash judge/run_judge.sh llama` |
| HarmBench | `cais/HarmBench-Llama-2-13b-cls` | `bash judge/run_judge.sh harm_bench` |
| SORRY-Bench | `sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406` | `bash judge/run_judge.sh sorry-bench` |
| WildGuardTest | `allenai/wildguard` | `bash judge/run_judge.sh wild-guard` |
| SG-Bench | `meta-llama/LlamaGuard-7b` | `bash judge/run_judge.sh sg_bench` |

Verdicts are written into the log files in place (columns `QWEN`, `LLAMA`,
`LLAMA2_CLS`, …). Already-judged items are skipped, so reruns only fill gaps.
Add `--model <model>` to judge one model's logs only. The non-English XSafety
splits are judged on an English translation of each response, stored in a
`translation` field that must be added to the logs first.

## Scoring

```bash
python score.py \
  --dir <log directory> \
  --eval_log_file <model responses>.jsonl \
  --model_name <model> \
  --scoring_fn <scoring function> \
  --config_fpath <benchmark config>.json
```

To write every scoring command for a model:

```bash
python run_experiments/make_run_script.py --model google/gemma-2-9b-it --scoring
bash run_google_gemma-2-9b-it_scoring.sh
```

The scoring function for each benchmark is listed in the `scoring` field of the
benchmark's `*_config.json` written by `get_data.py`. Toxicity scoring
(RealToxicityPrompts, BOLD) calls the Perspective API and expects a
`PERSPECTIVE_API_KEY` environment variable.

Some scoring functions write per-item results back into the log (DiscrimEval
adds `yes`, `no`, and `item_score`) and reuse them on later runs, so score a log
only once it is complete, or delete those fields before rescoring.

Two scoring functions read a benchmark's built dataset directly rather than
just the log being scored, so `get_data.py` must have been run first:
`decodingtrust_fair` reads `data_acquisition/data/decoding_trust/dt_fair.jsonl`
and `dyck` reads `data_acquisition/data/dyck/dyck.jsonl`. `world_values`
additionally needs `question_metadata.json` from a
[WorldValuesBench](https://github.com/Demon702/WorldValuesBench) checkout,
copied to `world_values_metadata.json` at the repo root.

## Requirements

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

Collecting responses and running the judges also needs
[vLLM](https://docs.vllm.ai) on a GPU machine (`pip install vllm`). Building
datasets and scoring do not.

## Citation

If you use this code or the accompanying dataset, please cite our paper:

```bibtex
@inproceedings{desai2026benchmarks,
  title     = {What {AI} Benchmarks Actually Measure: Adapting Convergent and
               Discriminant Validity to Interrogate Fifty-Six {AI} Benchmarks},
  author    = {Desai, Meera and Truong, Sang T. and Wallach, Hanna and
               Chouldechova, Alex and Cooper, A. Feder and
               Garcia-Gathright, Jean and Ho, Daniel E. and
               Jacobs, Abigail Z. and Koyejo, Sanmi and
               Pangakis, Nicholas and Wang, Angelina},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.08812}
}
```
