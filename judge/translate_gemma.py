#!/usr/bin/env python3
"""Translate the non-English XSafety splits (fr, de, sp) to English with
google/translategemma-12b-it, adding a "translation" field to each line in place.

The XSafety judge reads English: `score_judge.py --eval xsafety --lang fr` judges
the translated response against the translated prompt. So, before judging:

    # once, after get_data.py xsafety: translate the prompts
    python translate_gemma.py --mode data

    # after collecting responses: translate them (all models, or one)
    python translate_gemma.py --mode logs
    python translate_gemma.py --mode logs --model google/gemma-2-9b-it

Lines that already have a translation are skipped unless --overwrite is given.
Runs vLLM in-process (no server needed); --tp sets the number of GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data_acquisition" / "data"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
MODEL = "google/translategemma-12b-it"

# benchmark file -> source language code expected by TranslateGemma
LANG_MAP = {
    "xsafety_fr_random.jsonl": "fr",
    "xsafety_de_random.jsonl": "de",
    "xsafety_sp_random.jsonl": "es",
}


def build_prompt(processor, text: str, src_lang: str) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "text", "source_lang_code": src_lang, "target_lang_code": "en", "text": text}],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def find_files(mode: str, data_dir: Path, log_dir: Path, model: str | None) -> list[Path]:
    if mode == "data":
        return [data_dir / "xsafety" / f for f in LANG_MAP if (data_dir / "xsafety" / f).exists()]
    root = log_dir / model.replace("/", "_") if model else log_dir
    return sorted(p for f in LANG_MAP for p in root.rglob(f))


def load_file(path: Path) -> list[dict]:
    """Read JSONL, tolerating lines that hold more than one JSON object."""
    decoder = json.JSONDecoder()
    records = []
    with open(path) as f:
        for line in f:
            line, pos = line.strip(), 0
            while pos < len(line):
                obj, pos = decoder.raw_decode(line, pos)
                records.append(obj)
                while pos < len(line) and line[pos] in " \t":
                    pos += 1
    return records


def write_file(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # an interrupted run can't truncate the file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["logs", "data"], required=True,
                   help="'data': translate each prompt in the XSafety data; "
                        "'logs': translate each model's response in logs/")
    p.add_argument("--model", default=None, help="logs mode: only this model (e.g. google/gemma-2-9b-it)")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    p.add_argument("--overwrite", action="store_true", help="Retranslate lines that already have a translation")
    p.add_argument("--tp", type=int, default=1, help="Tensor-parallel size (GPUs)")
    args = p.parse_args()

    source_field = "response" if args.mode == "logs" else "prompt"
    files = find_files(args.mode, Path(args.data_dir), Path(args.log_dir), args.model)
    print(f"Mode: {args.mode} | model: {args.model or 'all'} | field: {source_field} | files: {len(files)}")
    if not files:
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=args.tp)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    for i, path in enumerate(files, 1):
        records = load_file(path)
        todo = [r for r in records
                if isinstance(r.get(source_field), str) and (args.overwrite or not r.get("translation"))]
        print(f"[{i}/{len(files)}] {path}: translating {len(todo)} of {len(records)}")
        if not todo:
            continue
        src_lang = LANG_MAP[path.name]
        outputs = llm.generate([build_prompt(processor, r[source_field], src_lang) for r in todo], sampling_params)
        for record, output in zip(todo, outputs):
            record["translation"] = output.outputs[0].text.strip()
        write_file(path, records)

    print("Done.")


if __name__ == "__main__":
    main()
