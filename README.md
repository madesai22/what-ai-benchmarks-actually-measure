# What AI Benchmarks Actually Measure — data acquisition & scoring code

Code for building the benchmark datasets and scoring the model outputs analyzed in
*What AI Benchmarks Actually Measure: Adapting Convergent and Discriminant Validity
to Interrogate Fifty-Six AI Benchmarks*.

The item-level model outputs and scores produced with this code are published on
the Hugging Face Hub:
[madesai/what-ai-benchmarks-actually-measure](https://huggingface.co/datasets/madesai/what-ai-benchmarks-actually-measure).

## Layout

| Path | What it does |
|---|---|
| `data_acquisition/get_data.py` | Downloads each benchmark from its source (Hugging Face, HELM, GitHub releases, …), samples it (fixed seeds), and writes `data/<benchmark>/<benchmark>.jsonl` plus a `*_config.json` with the system prompt, eval type, and scoring functions. |
| `data_acquisition/get_data_utils.py` | Shared helpers: sampling (simple, stratified, set-preserving), HELM download, config templates, judge prompt templates. |
| `data_acquisition/raft_prompts.py`, `sg_test_prompts.py` | Prompt templates for RAFT and SG-Bench. |
| `run_experiments/score.py` | Turns raw model response logs (`logs/<model>/<eval>.jsonl`) into benchmark scores — exact match, F1, ROUGE, demographic parity, refusal accuracies from LLM-judge verdicts, etc. |

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

## Scoring

```bash
python run_experiments/score.py \
  --dir <log directory> \
  --eval_log_file <model responses>.jsonl \
  --model_name <model> \
  --scoring_fn <scoring function> \
  --config_fpath <benchmark config>.json
```

The scoring function for each benchmark is listed in the `scoring` field of the
benchmark's `*_config.json` written by `get_data.py`. Toxicity scoring
(RealToxicityPrompts, BOLD) calls the Perspective API and expects a
`PERSPECTIVE_API_KEY` environment variable.

## Requirements

```bash
pip install -r requirements.txt
```

## Citation

See the [Hugging Face dataset card](https://huggingface.co/datasets/madesai/what-ai-benchmarks-actually-measure)
for the citation.
