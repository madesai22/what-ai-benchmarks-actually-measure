#!/usr/bin/env python3
"""Label collected model responses with an LLM judge served by vLLM.

Each benchmark's judge, prompt, and output column are set in config.py
(eval_to_judge). Most benchmarks use their own published classifier
(e.g. HarmBench -> cais/HarmBench-Llama-2-13b-cls). XSTest, SGXSTest, OR-Bench,
XSafety, and Civics can be judged by either of two general models (the paper
reports both):

    Qwen/Qwen3-30B-A3B-Instruct-2507   (writes column QWEN)
    meta-llama/Llama-3.3-70B-Instruct  (writes column LLAMA)

--judge_model must name the model the vLLM server is serving.

The judge's output is added to each line of logs/<model>/<benchmark file>
in place; score.py then reads that column. Items already judged are skipped
unless --override_results is given, and items whose request failed are
marked "Error" and retried on the next run.

Start the judge's vLLM server first (judge/score_judge.sh does this for you):

    vllm serve cais/HarmBench-Llama-2-13b-cls --port 8001

Examples:
    python score_judge.py --judge_model cais/HarmBench-Llama-2-13b-cls --eval harm_bench
    python score_judge.py --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 --eval xstest --model google/gemma-2-9b-it
    python score_judge.py --judge_model meta-llama/Llama-3.3-70B-Instruct --eval xsafety --lang fr
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from config import eval_to_judge, xsafety_scenarios

JUDGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = JUDGE_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data_acquisition" / "data"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
HARMBENCH_CONTEXT = JUDGE_DIR / "harmbench_context.jsonl"

# general judges for benchmarks without their own classifier -> column they write
GENERAL_JUDGES = {"Qwen/Qwen3-30B-A3B-Instruct-2507": "QWEN",
                  "meta-llama/Llama-3.3-70B-Instruct": "LLAMA"}
# Judges queried through the chat endpoint; all others get the raw prompt
# through the completions endpoint.
CHAT_JUDGES = {"Qwen/Qwen3-30B-A3B-Instruct-2507", "meta-llama/Llama-3.3-70B-Instruct",
               "walledai/walledguard-c", "meta-llama/LlamaGuard-7b"}

_harmbench_tokenizer = None


def truncate_answer(answer, max_answer_tokens=512):
    """HarmBench's classifier has a short context: keep the first 512 tokens of the answer."""
    global _harmbench_tokenizer
    if _harmbench_tokenizer is None:
        from transformers import AutoTokenizer
        _harmbench_tokenizer = AutoTokenizer.from_pretrained("cais/HarmBench-Llama-2-13b-cls", use_fast=False)
    tokens = _harmbench_tokenizer.encode(answer)
    if len(tokens) > max_answer_tokens:
        answer = _harmbench_tokenizer.decode(tokens[:max_answer_tokens], skip_special_tokens=True)
    return answer


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval', required=True, choices=sorted(eval_to_judge), help='Benchmark to judge')
    p.add_argument('--judge_model', required=True, help='Judge model being served by vLLM (Hugging Face name)')
    p.add_argument('--model', default=None, help='Only judge this model (default: every model in --log_dir)')
    p.add_argument('--lang', default=None,
                   help="XSafety language (de, fr, sp). Judges the English 'translation' field of each "
                        "response instead of 'response'.")
    p.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR))
    p.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    p.add_argument('--port', type=int, default=8001, help='Port of the judge vLLM server')
    p.add_argument('--override_results', action='store_true', help='Re-judge items that already have a verdict')
    return p.parse_args()


def make_prompt(eval_name, prompt, response, template, item_info):
    if eval_name == "sg_bench":  # LlamaGuard: its chat template builds the prompt
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    if not template:
        return response
    if eval_name == "harm_bench":
        context = item_info.get("context")
        if context:
            return template["prompt_contextual"].strip().format(QUESTION=prompt, CONTEXT=context, ANSWER=response)
        return template["prompt_no_context"].strip().format(QUESTION=prompt, ANSWER=response)
    if eval_name == "xsafety":
        safety = xsafety_scenarios[item_info["category"]]
        return template.strip().format(QUESTION=prompt, ANSWER=response, SAFETY=safety)
    return template.strip().format(QUESTION=prompt, ANSWER=response)


def generate_judgement(judge_model, prompt_text, max_tokens, eval_name, port):
    base = f"http://localhost:{port}/v1"
    if judge_model in CHAT_JUDGES:
        if eval_name == "sg_bench":
            messages = prompt_text
        elif eval_name == "civics":
            messages = [{"role": "system", "content": "Respond only with the answer."},
                        {"role": "user", "content": prompt_text}]
        elif eval_name == "xstest":
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text}]
        else:
            messages = [{"role": "user", "content": prompt_text}]
        url, payload = f"{base}/chat/completions", {"messages": messages}
    else:
        url, payload = f"{base}/completions", {"prompt": prompt_text}
    payload.update(model=judge_model, temperature=0, max_tokens=max_tokens)

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = choice["message"]["content"] if judge_model in CHAT_JUDGES else choice["text"]
        return text.strip()
    except Exception as e:
        print(f"Judge request failed: {e!r}")
        return "Error"


def already_judged(item, label):
    v = item.get(label)
    return v is not None and not (isinstance(v, str) and "Error" in v)


def load_prompts(eval_name, data_path, lang):
    """id -> {'prompt': ..., plus 'context' (HarmBench) or 'category' (XSafety)}."""
    path = HARMBENCH_CONTEXT if eval_name == "harm_bench" else data_path
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    prompt_field = "translation" if (eval_name == "xsafety" and lang) else "prompt"
    return {r["id"]: {"prompt": r[prompt_field], "context": r.get("context"), "category": r.get("category")}
            for r in rows}


def judge_model_logs(log_path, prompts, args, judge_model, template, label, max_tokens):
    with open(log_path) as f:
        responses = [json.loads(line) for line in f]

    n_new = n_err = 0
    for item in tqdm(responses, desc=log_path.parent.name):
        if not args.override_results and already_judged(item, label):
            continue
        response = item.get("translation") if args.lang else item.get("response")
        if response is None:
            print(f"{item['id']}: no {'translation' if args.lang else 'response'} field; skipping")
            continue
        if item["id"] not in prompts:
            print(f"{item['id']}: not in the benchmark data; skipping")
            continue
        item["cleaned_response"] = response
        if args.eval == "harm_bench":
            response = truncate_answer(response)
        info = prompts[item["id"]]
        prompt_text = make_prompt(args.eval, info["prompt"], response, template, info)
        item[label] = generate_judgement(judge_model, prompt_text, max_tokens, args.eval, args.port)
        item.pop("Qwen", None)  # stale column name from an early run
        n_err += item[label] == "Error"
        if n_new < 2:
            print(prompt_text, "\n->", item[label])
        n_new += 1

    # write to a temp file, then swap it in, so an interrupted run can't truncate the log
    tmp = log_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for item in responses:
            f.write(json.dumps(item) + "\n")
    os.replace(tmp, log_path)
    print(f"{log_path}: judged {n_new} items ({n_err} errors; rerun to retry them)")


def main():
    args = parse_arguments()
    cfg = eval_to_judge[args.eval]
    file_name = cfg["file_name"]
    if args.lang:
        if args.eval != "xsafety":
            sys.exit("--lang only applies to xsafety")
        file_name = file_name.replace("engl", args.lang)

    judge_model, label = args.judge_model, cfg.get("label")
    if cfg["model"] == "multiple":
        if judge_model not in GENERAL_JUDGES:
            sys.exit(f"{args.eval} is judged by one of: {', '.join(GENERAL_JUDGES)} (got {judge_model})")
        label = GENERAL_JUDGES[judge_model]
    elif judge_model != cfg["model"]:
        sys.exit(f"{args.eval} is judged by {cfg['model']} (got {judge_model})")
    print(f"Judging {args.eval} ({file_name}) with {judge_model} -> column {label}")

    prompts = load_prompts(args.eval, Path(args.data_dir) / args.eval / file_name, args.lang)
    log_dir = Path(args.log_dir)
    models = [args.model.replace("/", "_")] if args.model else sorted(d.name for d in log_dir.iterdir() if d.is_dir())
    for m in models:
        log_path = log_dir / m / file_name
        if not log_path.exists():
            print(f"{log_path} not found; skipping {m}")
            continue
        judge_model_logs(log_path, prompts, args, judge_model, cfg["prompt"], label, cfg["max_tokens"])


if __name__ == "__main__":
    main()
