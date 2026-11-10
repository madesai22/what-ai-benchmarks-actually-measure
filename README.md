# What AI Benchmarks Actually Measure — benchmark data, model responses, judging & scoring code

Code for building the benchmark datasets, collecting and judging model responses, and scoring the model outputs analyzed in
*What AI Benchmarks Actually Measure: Adapting Convergent and Discriminant Validity
to Interrogate Fifty-Six AI Benchmarks* (Desai et al., 2026,
[arxiv.org/abs/2609.08812](https://arxiv.org/abs/2609.08812)).

🤗 The item-level model outputs and scores produced with this code are published
on the Hugging Face Hub:
[madesai/what-ai-benchmarks-actually-measure](https://huggingface.co/datasets/madesai/what-ai-benchmarks-actually-measure).

## Pipeline

All commands below are run from the top of the repo. The steps, in order:

1. **Build datasets** — `data_acquisition/get_data.py` downloads and samples each
   benchmark into `data_acquisition/data/`.
2. **Collect responses** — `run_experiments/run_model.sh` serves a model with
   vLLM and writes its responses to `logs/<model>/`.
3. **Translate** (non-English XSafety only) — `judge/translate_gemma.py`.
4. **Judge** (safety and refusal benchmarks) — `judge/score_judge.sh` adds a
   judge model's verdict to each response.
5. **Score** — `score.py` turns the logs into benchmark scores in
   `logs/<model>/scores/<model>-scores.csv`.

### Quick start

Try one small model on one benchmark first. This needs vLLM and a GPU; the
Qwen3-30B judge needs about 80GB of GPU memory (use `--tp=2` on smaller GPUs).

```bash
pip install -r requirements.txt vllm
python data_acquisition/get_data.py xstest
./run_experiments/run_model.sh google/gemma-2-2b-it xstest
./judge/score_judge.sh Qwen/Qwen3-30B-A3B-Instruct-2507 google/gemma-2-2b-it xstest
python run_experiments/make_run_script.py --model google/gemma-2-2b-it --evals xstest --scoring
bash run_google_gemma-2-2b-it_scoring.sh
cat logs/google_gemma-2-2b-it/scores/google_gemma-2-2b-it-scores.csv
```

## Contents

| Path | What it does |
|---|---|
| `data_acquisition/get_data.py` | **Building datasets.** Downloads each benchmark from its source (Hugging Face, HELM, GitHub releases, …), samples it (fixed seeds), and writes `data_acquisition/data/<benchmark>/<benchmark>.jsonl` plus a `*_config.json` with the system prompt, eval type, and scoring functions. |
| `run_experiments/run_model.sh` | **Collecting model responses.** Starts a vLLM server for a model, collects its responses, and stops the server. |
| `run_experiments/make_run_script.py` | Writes the collection (or scoring) commands for every benchmark for one model, with the paper's per-benchmark settings. Used by `run_model.sh`. |
| `run_experiments/run_experiments_vllm.py` | Sends one benchmark file to a running vLLM server and saves the responses. Called by the commands `make_run_script.py` writes. |
| `judge/score_judge.sh` | **Judging safety and refusal benchmarks.** Starts a judge model with vLLM and runs `judge/score_judge.py`, which adds its verdict to each response. Judge prompts are in `judge/config.py`. |
| `judge/translate_gemma.py` | Translates the French, German, and Spanish XSafety prompts and responses to English before judging. |
| `score.py` | **Scoring.** Turns response logs (`logs/<model>/<eval>.jsonl`) into benchmark scores — exact match, F1, ROUGE, demographic parity, refusal accuracies from LLM-judge verdicts, etc. |

## Building datasets

```bash
# list the available benchmark builders
python data_acquisition/get_data.py --list

# build one or several
python data_acquisition/get_data.py gsm8k xstest

# build everything
python data_acquisition/get_data.py --all
```

Outputs land in `data_acquisition/data/`, wherever you run the script from.
Failures are reported per dataset and don't stop the run.

Caveats:

- Some builders read from HELM's public benchmark output storage
  (`storage.googleapis.com/crfm-helm-public`); a few of those paths have moved
  since collection, so those builders may 404 until pointed at the current
  HELM release.
- `world_values` requires a manual download of
  [WorldValuesBench](https://github.com/Demon702/WorldValuesBench), cloned into
  the top of this repo (the builder prints instructions if it is missing).
- Sampling uses fixed random seeds, so rebuilt datasets match the ones used in
  the paper wherever the upstream source is unchanged.

## Collecting model responses

Responses are collected from models served with [vLLM](https://docs.vllm.ai)
(any model with an OpenAI-compatible chat endpoint works). Build the datasets
first, then, on a GPU machine:

```bash
./run_experiments/run_model.sh <model> <eval> [eval ...] [--tp=N]     # or: ... <model> all

./run_experiments/run_model.sh google/gemma-2-9b-it xstest mmlu gsm8k
./run_experiments/run_model.sh meta-llama/Llama-3.3-70B-Instruct all --tp=4   # --tp = number of GPUs
```

`run_model.sh` writes the collection commands to `run_experiments/run_<model>.sh` (with
`make_run_script.py`) and runs them, so you can also edit and rerun that file.

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
`run_model.sh` sets). API models (OpenAI, Anthropic, …) were queried with
separate clients that are not included here.

## Judging safety and refusal benchmarks

The safety and refusal benchmarks (listed below) are scored from a judge model's verdict on
each response, so run the judges on the collected logs before scoring them.
Each benchmark's judge, prompt, and output column are set in `judge/config.py`.

`judge/score_judge.sh` starts the judge with vLLM, labels the responses, and
stops the server:

```bash
./judge/score_judge.sh <judge_model> <scored_model|all> <eval> [eval ...] [--tp=N] [--override_results]
```

Use this `judge_model` for each benchmark:

| `eval` | `judge_model` |
|---|---|
| `xstest`, `sgxstest`, `or-bench`, `xsafety`\*, `civics` | `Qwen/Qwen3-30B-A3B-Instruct-2507` **and** `meta-llama/Llama-3.3-70B-Instruct` (run both; the paper reports both) |
| `harm_bench` | `cais/HarmBench-Llama-2-13b-cls` |
| `sorry-bench` | `sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406` |
| `wild-guard` | `allenai/wildguard` |
| `sg_bench` | `meta-llama/LlamaGuard-7b` |

### Examples of using score_judge.sh: 
To score responses from gemma-2-9b-it on xstest, sgxstest, or-bench, xsafety, and civics using Qwen/Qwen3-30B-A3B-Instruct-2507, use:
```
./judge/score_judge.sh Qwen/Qwen3-30B-A3B-Instruct-2507 google/gemma-2-9b-it xstest sgxstest or-bench xsafety civics
```

To score responses from all models on harm_bench using cais/HarmBench-Llama-2-13b-cls,  use:
```
./judge/score_judge.sh cais/HarmBench-Llama-2-13b-cls all harm_bench
```

Benchmarks can share one `score_judge.sh` call only if they use the same
`judge_model`.

Add `--tp=N` to run the judge on N GPUs (Llama-3.3-70B needs about 4×80GB,
so `--tp=4`). Verdicts are written into the log files in place (columns
`QWEN`, `LLAMA`, `LLAMA2_CLS`, …). Already-judged items are skipped, so reruns
only fill gaps; `--override_results` re-judges everything.

### Translation
\*The French, German, and Spanish XSafety splits are judged in English: translate
the prompts and the responses before doing LLM judging with
[TranslateGemma](https://huggingface.co/google/translategemma-12b-it) first,
then pass `--lang`:

```bash
python judge/translate_gemma.py --mode data   # once: translate the prompts
python judge/translate_gemma.py --mode logs   # translate responses (add --model <model> for one)
./judge/score_judge.sh Qwen/Qwen3-30B-A3B-Instruct-2507 all xsafety --lang=fr
./judge/score_judge.sh Qwen/Qwen3-30B-A3B-Instruct-2507 all xsafety --lang=de
./judge/score_judge.sh Qwen/Qwen3-30B-A3B-Instruct-2507 all xsafety --lang=sp
```

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

Scores are written to `logs/<model>/scores/<model>-scores.csv`, one row per
metric (columns `eval_name`, `concept`, `score`, `value`, `eval_type`,
`system_prompt`). Each run appends its rows, so delete that file before
rescoring a model from scratch.

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
