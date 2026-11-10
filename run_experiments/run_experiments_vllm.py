#!/usr/bin/env python3
"""Collect model responses for one benchmark file from a running vLLM server.

Start a server first (run_model.sh does this for you), e.g.

    vllm serve google/gemma-2-9b-it --port 8000 --max-logprobs 150

then

    python run_experiments_vllm.py --eval_name xstest --eval_file xstest.jsonl \
        --model google/gemma-2-9b-it --max_tokens 200

Responses are written to <log_dir>/<model with / -> _>/<eval_file>, one JSON
object per line: the benchmark item (minus its prompt) plus "response" and
"logprobs" fields. A run log is appended to <log_dir>/<model>/<model>_config.jsonl.

"logprobs" holds, by default, the log-probability of each generated token
([{token: logprob}, ...]). With --log_probs N it instead holds, for each of the
item's answer options (its "options" field, e.g. ["Yes", "No"] for DiscrimEval),
the option's log-probability as the first generated token, looked up among the
top N candidates (None if the option is not among them). N must not exceed the
server's --max-logprobs (vLLM's default is 20).

With --prompt_probs, the prompt is sent as raw text (no chat template) and the
log-probability of each prompt token is stored in "prompt_probs".

Requests that fail are reported and left out of the output; rerun with
--incomplete to fill them in without redoing finished items.
"""
import argparse
import asyncio
import json
from datetime import datetime
from functools import lru_cache
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
# Models without a chat template: queried through the completions endpoint.
NO_CHAT_TEMPLATE = ("bloomz",)


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
    p.add_argument('--log_probs', type=int, default=0,
                   help="Record each answer option's first-token log-probability, searching the top N tokens")
    p.add_argument('--prompt_probs', action='store_true', help='Record log-probabilities of the prompt tokens')
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


@lru_cache(maxsize=None)
def get_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def option_logprobs(top_logprobs, options):
    """{option: logprob} for each option found (case-insensitively) among the top first-token candidates."""
    found = {}
    for option in options:
        found[option] = next((c['logprob'] for c in top_logprobs if c['token'].lower() == option.lower()), None)
    return found


def prompt_token_logprobs(model, prompt, prompt_logprobs):
    """Log-probability of each actual prompt token. vLLM returns, per prompt position, a dict
    {token_id: {logprob, rank, decoded_token}} with the actual token plus the top candidates;
    the first position (nothing to condition on) is None."""
    # same tokenization vLLM applies to a completions prompt (special tokens included)
    token_ids = get_tokenizer(model).encode(prompt)
    out = []
    for token_id, candidates in zip(token_ids[1:], prompt_logprobs[1:]):
        if candidates and str(token_id) in candidates:
            out.append(candidates[str(token_id)])
    return out


def build_request(item, args, system_prompt_dict, base_url):
    system_prompt = system_prompt_dict[item['category']] if args.system_prompt == "see dict" else args.system_prompt
    payload = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.min_tokens:
        payload["min_tokens"] = args.min_tokens

    if args.prompt_probs:
        payload.update(prompt=item['prompt'], prompt_logprobs=10, max_tokens=1, logprobs=1)
        return f"{base_url}/completions", payload
    if any(name in args.model for name in NO_CHAT_TEMPLATE):
        prompt = (system_prompt + "\n" if system_prompt else "") + item['prompt']
        payload.update(prompt=prompt, logprobs=max(args.log_probs, 1))
        return f"{base_url}/completions", payload

    payload["messages"] = format_messages(item['prompt'], system_prompt, args.model)
    payload["logprobs"] = True
    if args.log_probs:
        payload["top_logprobs"] = args.log_probs
    return f"{base_url}/chat/completions", payload


def parse_response(data, item, args):
    """-> (response text, logprobs, prompt_probs or None)."""
    choice = data["choices"][0]
    if "message" in choice:  # chat endpoint
        text, tokens = choice["message"]["content"], (choice.get("logprobs") or {}).get("content") or []
        if args.log_probs:
            top = tokens[0]["top_logprobs"] if tokens else []
            return text, option_logprobs(top, item["options"]), None
        return text, [{t["token"]: round(t["logprob"], 6)} for t in tokens], None

    # completions endpoint: logprobs come as parallel lists
    text, lp = choice["text"], choice.get("logprobs") or {}
    if args.prompt_probs:
        return text, lp, prompt_token_logprobs(args.model, item["prompt"], choice["prompt_logprobs"])
    if args.log_probs:
        top = [{"token": t, "logprob": v} for t, v in (lp.get("top_logprobs") or [{}])[0].items()]
        return text, option_logprobs(top, item["options"]), None
    return text, [{t: round(v, 6)} for t, v in zip(lp.get("tokens", []), lp.get("token_logprobs", []))], None


async def make_request(session, item, args, system_prompt_dict, base_url):
    url, payload = build_request(item, args, system_prompt_dict, base_url)
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        data = await resp.json(content_type=None)
    text, logprobs, prompt_probs = parse_response(data, item, args)

    # 'translation' is an English rendering of a non-English prompt; it is dropped so it
    # can't be mistaken for the translated response that the XSafety judge reads.
    out = {k: v for k, v in item.items() if k not in ('prompt', 'translation')}
    out.update(response=text, max_concurrent=args.max_concurrent, max_tokens=args.max_tokens, logprobs=logprobs)
    if args.prompt_probs:
        out["prompt_probs"] = prompt_probs
    return out


async def async_main():
    args = parse_arguments()
    base_url = f"http://localhost:{args.port}/v1"
    bench_dir = Path(args.data_dir) / args.eval_name
    model_dir = Path(args.log_dir) / args.model.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / (args.outfile or args.eval_file)

    data = read_jsonl(bench_dir / args.eval_file)
    if args.log_probs and not all("options" in d for d in data):
        raise SystemExit(f"--log_probs needs an 'options' field on every item of {args.eval_file}")
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
        'max_tokens': args.max_tokens, 'log_probs': args.log_probs, 'prompt_probs': args.prompt_probs,
        'timestamp': start.strftime('%Y-%m-%d %H:%M:%S'),
    }

    sem = asyncio.Semaphore(args.max_concurrent)
    n_failed = 0

    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:
        async def bounded(item):
            async with sem:
                return await make_request(session, item, args, system_prompt_dict, base_url)

        tasks = [bounded(d) for d in data]
        async with aiofiles.open(out_path, mode, encoding='utf-8') as f:
            for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks))):
                try:
                    result = await coro
                except Exception as e:  # network error, timeout, HTTP error, bad JSON
                    n_failed += 1
                    print(f"Request failed: {type(e).__name__}: {e}")
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
