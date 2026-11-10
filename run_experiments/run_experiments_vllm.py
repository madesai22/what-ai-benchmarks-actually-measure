#!/usr/bin/env python3
"""Collect model responses for one benchmark file from a running vLLM server.

Start a server first (see run_vllm_example.sh), e.g.

    vllm serve google/gemma-2-9b-it --port 8000

then

    python run_experiments_vllm.py --eval_name xstest --eval_file xstest.jsonl \
        --model google/gemma-2-9b-it --max_tokens 200

Responses are written to <log_dir>/<model with / -> _>/<eval_file>, one JSON
object per line: the benchmark item (minus its prompt) plus a "response" field.
A run log is appended to <log_dir>/<model>/<model>_config.jsonl.

Requests that fail are reported and left out of the output; rerun with
--incomplete to fill them in without redoing finished items.
"""
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import aiofiles
import aiohttp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data_acquisition" / "data"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"

# Models whose chat templates reject a system role: the system prompt is
# prepended to the user turn instead (as was done for the paper).
NO_SYSTEM_ROLE = ("gemma", "bloomz")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval_name', required=True, help='Benchmark directory name under --data_dir (e.g. xstest)')
    p.add_argument('--eval_file', required=True, help='File inside that directory (e.g. xstest.jsonl)')
    p.add_argument('--model', required=True, help='Model name exactly as served by vLLM (e.g. google/gemma-2-9b-it)')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (paper: 1.0)')
    p.add_argument('--max_tokens', type=int, default=250, help='Max tokens per completion')
    p.add_argument('--min_tokens', type=int, default=0, help='Min tokens per completion')
    p.add_argument('--system_prompt', default=None,
                   help='System prompt. "see dict" looks it up per item category in --system_prompt_json.')
    p.add_argument('--system_prompt_json', default=None,
                   help='JSON file (inside the benchmark directory) mapping item category -> system prompt')
    p.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR), help='Where get_data.py wrote the benchmarks')
    p.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR), help='Where to write responses')
    p.add_argument('--outfile', default=None, help='Output file name (default: same as --eval_file)')
    p.add_argument('--port', type=int, default=8000, help='Port of the vLLM server')
    p.add_argument('--max_concurrent', type=int, default=1, help='Max concurrent requests')
    p.add_argument('--incomplete', action='store_true',
                   help='Append to an existing output file, only running items not already in it')
    return p.parse_args()


def format_messages(prompt, system_prompt, model):
    if system_prompt and any(name in model for name in NO_SYSTEM_ROLE):
        prompt, system_prompt = system_prompt + "\n" + prompt, None
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    return messages


async def make_request(session, item, args, system_prompt_dict, url):
    system_prompt = system_prompt_dict[item['category']] if args.system_prompt == "see dict" else args.system_prompt
    payload = {
        "model": args.model,
        "messages": format_messages(item['prompt'], system_prompt, args.model),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
    }
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    out = {k: v for k, v in item.items() if k != 'prompt'}
    out["response"] = data["choices"][0]["message"]["content"]
    out["max_concurrent"] = args.max_concurrent
    return out


async def async_main():
    args = parse_arguments()
    url = f"http://localhost:{args.port}/v1/chat/completions"
    bench_dir = Path(args.data_dir) / args.eval_name
    model_dir = Path(args.log_dir) / args.model.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / (args.outfile or args.eval_file)

    data = read_jsonl(bench_dir / args.eval_file)
    system_prompt_dict = {}
    if args.system_prompt_json:
        with open(bench_dir / args.system_prompt_json) as f:
            system_prompt_dict = json.load(f)

    mode = "w"
    if args.incomplete:
        mode = "a"
        if out_path.exists():
            seen = {d['id'] for d in read_jsonl(out_path)}
            data = [d for d in data if d['id'] not in seen]
            print(f"{len(seen)} items already done, {len(data)} to go")
        else:
            print(f"{out_path} does not exist yet; starting fresh.")

    start = datetime.now()
    run_log = {
        'eval_name': args.eval_name, 'eval_file': args.eval_file, 'model': args.model,
        'temperature': args.temperature, 'system_prompt': args.system_prompt,
        'max_tokens': args.max_tokens, 'timestamp': start.strftime('%Y-%m-%d %H:%M:%S'),
    }

    sem = asyncio.Semaphore(args.max_concurrent)
    n_failed = 0

    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:
        async def bounded(item):
            async with sem:
                return await make_request(session, item, args, system_prompt_dict, url)

        tasks = [bounded(d) for d in data]
        async with aiofiles.open(out_path, mode, encoding='utf-8') as f:
            for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks))):
                try:
                    result = await coro
                except Exception as e:  # network error, timeout, HTTP error, bad JSON
                    n_failed += 1
                    print(f"Request failed: {type(e).__name__} {getattr(e, 'status', '') or e}")
                    continue
                if i < 3:
                    print(result)
                await f.write(json.dumps(result) + '\n')
                await f.flush()

    end = datetime.now()
    run_log.update(end_timestamp=end.strftime('%Y-%m-%d %H:%M:%S'), time_taken=str(end - start), n_failed=n_failed)
    with open(model_dir / f"{args.model.replace('/', '_')}_config.jsonl", "a") as f:
        f.write(json.dumps(run_log) + '\n')
    if n_failed:
        print(f"{n_failed} requests failed; rerun with --incomplete to retry them.")


if __name__ == '__main__':
    asyncio.run(async_main())
