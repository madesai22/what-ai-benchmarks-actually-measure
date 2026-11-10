#!/usr/bin/env python3
"""Write the shell commands that collect (and then score) every benchmark for one model,
using the same per-benchmark settings as the paper.

Settings come from each benchmark's *_config.json (written by get_data.py):
system prompt, eval type (-> max_tokens), and scoring functions.

    # collection commands (run while a vLLM server for the model is up)
    python make_run_script.py --model google/gemma-2-9b-it

    # scoring commands (run after collection and, for judged benchmarks, after judge/)
    python make_run_script.py --model google/gemma-2-9b-it --scoring

    # just some benchmarks
    python make_run_script.py --model google/gemma-2-9b-it --evals xstest gsm8k

The script is written to run_<model>[_scoring].sh in the current directory and
expects to be run from the repo root (it cd's there itself).
"""
import argparse
import json
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data_acquisition" / "data"

MAX_TOKENS = {"open_generation": 200, "closed_generation": 25, "mcq": 5}
LONG_ANSWER_EVALS = {"gsm8k": 500, "math": 500}

# Benchmarks whose scores come from an LLM judge: run judge/score_judge.sh on the
# collected logs before scoring them.
JUDGED_EVALS = {"harm_bench", "sorry-bench", "salad-bench", "sg_bench", "sgxstest",
                "wild-guard", "xsafety", "or-bench", "xstest", "civics"}


def benchmark_files(bench_dir):
    """Base names of the benchmark files in a directory, e.g. {'bbq_ambig', 'bbq_disambig'}."""
    names = set()
    for f in bench_dir.iterdir():
        name = re.sub(r"\.jsonl?$", "", f.name)
        names.add(re.sub(r"_config|_random|_system_prompt", "", name))
    # sorted so e.g. bbq_ambig is scored before bbq_disambig, which reads it
    return sorted(names)


def commands_for(bench_dir, model, scoring, port, max_concurrent, log_dir):
    eval_name = bench_dir.name
    model_dir = model.replace("/", "_")
    files = {f.name for f in bench_dir.iterdir()}
    lines = []
    for base in benchmark_files(bench_dir):
        cfg_path = bench_dir / f"{base}_config.json"
        if not cfg_path.exists():
            print(f"warning: no {cfg_path.name} in {bench_dir}, skipping {base}")
            continue
        cfg = json.loads(cfg_path.read_text())
        eval_file = f"{base}_random.jsonl" if f"{base}_random.jsonl" in files else f"{base}.jsonl"
        rel_cfg = cfg_path.relative_to(REPO_ROOT) if cfg_path.is_relative_to(REPO_ROOT) else cfg_path

        if scoring:
            fns = cfg.get("scoring") or (["exact_match"] if cfg["eval_type"] == "mcq" else [])
            for fn in fns:
                lines.append(f"python score.py --dir {log_dir}/{model_dir} --eval_log_file {eval_file} "
                             f"--model_name {model_dir} --scoring_fn {fn} --config_fpath {rel_cfg}")
            continue

        max_tokens = LONG_ANSWER_EVALS.get(eval_name, MAX_TOKENS[cfg["eval_type"]])
        cmd = (f"python run_experiments/run_experiments_vllm.py --eval_name {eval_name} --eval_file {eval_file} "
               f"--model {model} --temperature 1 --max_tokens {max_tokens} --log_dir {log_dir} "
               f"--port {port} --max_concurrent {max_concurrent}")
        sp = cfg.get("system_prompt")
        if sp == "see dict":
            cmd += f' --system_prompt_json {base}_system_prompt.json --system_prompt "see dict"'
        elif sp:
            cmd += " --system_prompt " + shlex.quote(sp)
        if cfg.get("log_probs"):  # scored from the answer options' log-probabilities (e.g. DiscrimEval)
            cmd += f" --log_probs {cfg['log_probs']}"
        # --incomplete makes reruns resume instead of starting over
        lines.append(cmd + " --incomplete")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='Model name as served by vLLM')
    p.add_argument('--evals', nargs='*', default=None, help='Benchmark directory names (default: all)')
    p.add_argument('--scoring', action='store_true', help='Write scoring commands instead of collection commands')
    p.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR))
    p.add_argument('--log_dir', default='logs', help='Log directory, relative to the repo root')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--max_concurrent', type=int, default=8)
    p.add_argument('--out', default=None, help='Output script name')
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    names = args.evals or sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    lines = ["#!/bin/bash", "set -u", f'cd "{REPO_ROOT}"', ""]
    for name in names:
        bench_dir = data_dir / name
        if not bench_dir.is_dir():
            print(f"warning: {bench_dir} not found (run get_data.py {name}?)")
            continue
        if args.scoring and name in JUDGED_EVALS:
            lines.append(f"# {name}: run judge/score_judge.sh on these logs first")
        lines += commands_for(bench_dir, args.model, args.scoring, args.port, args.max_concurrent, args.log_dir)

    suffix = "_scoring" if args.scoring else ""
    out = Path(args.out or f"run_{args.model.replace('/', '_')}{suffix}.sh")
    out.write_text("\n".join(lines) + "\n")
    out.chmod(0o755)
    print(f"wrote {out} ({sum(1 for l in lines if l.startswith('python'))} commands)")


if __name__ == "__main__":
    main()
