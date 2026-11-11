# -=-=-=-=-=-=-
# 11/11/2026
# Makes the GSM8K grade-school maths job for the network, and scores what it trained. `prepare` downloads GSM8K at a
# pinned revision and writes its training problems as the chats file `vibehash train` takes. `score` asks a model the
# test problems and counts how many final answers are right, with or without the adapter the network trained.
# Version: 1.0
# -=-=-=-=-=-=-

"""Prepare the GSM8K fine-tune job, and score the result.

    python scripts/gsm8k_job.py prepare --out gsm8k-train.jsonl
    vibehash train gsm8k-train.jsonl --base qwen3-0.6b --node ...      (then: vibehash job download)
    python scripts/gsm8k_job.py score --base Qwen/Qwen3-0.6B                       # the base model alone
    python scripts/gsm8k_job.py score --base Qwen/Qwen3-0.6B --adapter gsm8k-train-lora

GSM8K (openai/gsm8k, MIT licence) is 7,473 training and 1,319 test problems, each answer a worked solution ending in
"#### <number>". Training teaches that format; scoring reads the number after "####" (or, failing that, the last
number in the reply) and compares it with the test answer's. Needs the `evo` extra (transformers, datasets) and, to
score an adapter, `pip install peft`.
"""
from __future__ import annotations

# -=-=-=- Imports -=-=-=-
# Standard library tools: argparse reads the options, hashlib fingerprints the output file, json writes chats, re finds
# numbers in text, sys/Path find files, time measures how long scoring takes.
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

# -=-=-=- Settings -=-=-=-
DATASET = "openai/gsm8k"  # on Hugging Face; MIT licence
CONFIG = "main"  # the plain problems (the other config, "socratic", adds sub-questions)
CALCULATOR = re.compile(r"<<[^>]*>>")  # GSM8K's calculator notes, like <<48/2=24>>: not part of the written answer
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")  # 1,234 or -5 or 3.75


# -=-=-=- Step 1: download GSM8K at a fixed revision -=-=-=-
def load_split(split: str, revision: str | None):
    """One split of GSM8K, and the dataset commit it came from. With no revision, the newest commit is looked up and
    used, so the result names exactly what it read."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = revision or HfApi().dataset_info(DATASET).sha  # the dataset's current commit, pinned from here on
    return load_dataset(DATASET, CONFIG, split=split, revision=sha), sha


def clean_answer(answer: str) -> str:
    """The worked solution without calculator notes, and with its final line as "#### <number>"."""
    return CALCULATOR.sub("", answer).strip()


def final_number(text: str) -> str | None:
    """The answer a solution gives: the number after "####" if there is one, else the last number in the text.
    Commas are dropped and a trailing ".0" too, so "1,200" and "1200.0" both read as "1200"."""
    tail = text.split("####")[-1] if "####" in text else text
    found = NUMBER.findall(tail)
    if not found:
        return None
    n = found[0] if "####" in text else found[-1]
    n = n.replace(",", "")
    return n[:-2] if n.endswith(".0") else n


# -=-=-=- Step 2: write the training chats -=-=-=-
def prepare(a: argparse.Namespace) -> None:
    rows, sha = load_split("train", a.revision)
    with a.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            chat = {"messages": [{"role": "user", "content": row["question"].strip()},
                                 {"role": "assistant", "content": clean_answer(row["answer"])}]}
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(f"wrote {len(rows):,} chats to {a.out}")
    print(f"  from {DATASET} ({CONFIG}) at revision {sha}")
    print(f"  sha256 {digest}  (record both in the run sheet)")


# -=-=-=- Step 3: ask the model the test problems -=-=-=-
def score(a: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, sha = load_split("test", a.revision)
    if a.limit:
        rows = rows.select(range(min(a.limit, len(rows))))
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    tok.padding_side = "left"  # batched generation needs the prompts lined up at the right edge
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, revision=a.base_revision, torch_dtype=torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.adapter))  # the network's LoRA on top of the same base
    model.to(device).eval()

    def prompt(question: str) -> str:
        messages = [{"role": "user", "content": question.strip()}]
        try:  # Qwen3's template thinks out loud unless told not to; other templates don't take the option
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    right, started, examples = 0, time.perf_counter(), []
    for i in range(0, len(rows), a.batch):
        batch = rows.select(range(i, min(i + a.batch, len(rows))))
        enc = tok([prompt(q) for q in batch["question"]], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():  # greedy: the same answer every time, so the score is reproducible
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        replies = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for question, answer, reply in zip(batch["question"], batch["answer"], replies):
            want, got = final_number(answer), final_number(reply)
            right += got is not None and got == want
            if len(examples) < 3:
                examples.append((question, want, got, reply))
        print(f"  {min(i + a.batch, len(rows))}/{len(rows)}: {right} right so far", flush=True)

    took = time.perf_counter() - started
    which = f"{a.base} + {a.adapter}" if a.adapter else f"{a.base} alone"
    print(f"\n{which}: {right}/{len(rows)} right = {right / len(rows):.1%}  ({took:.0f} s on {device})")
    print(f"  test set {DATASET} ({CONFIG}) at revision {sha}, greedy, at most {a.max_new_tokens} new tokens")
    for question, want, got, reply in examples:
        print(f"\n  Q: {question[:120]}\n  want {want}, got {got}: {reply.strip()[:200]!r}")


# -=-=-=- Run it -=-=-=-
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write GSM8K's training problems as a chats file")
    p.add_argument("--out", type=Path, default=Path("gsm8k-train.jsonl"))
    p.add_argument("--revision", help="the dataset commit to read (default: the newest, printed so you can pin it)")
    p.set_defaults(fn=prepare)

    p = sub.add_parser("score", help="ask a model GSM8K's test problems and count the right answers")
    p.add_argument("--base", required=True, help="the base model, as Hugging Face names it (Qwen/Qwen3-0.6B)")
    p.add_argument("--base-revision", help="the base model's commit (use the one the job pinned)")
    p.add_argument("--adapter", type=Path, help="the folder `vibehash job download` wrote; leave out for the base alone")
    p.add_argument("--revision", help="the dataset commit (use the one `prepare` printed)")
    p.add_argument("--limit", type=int, help="score only the first N test problems (a quick check)")
    p.add_argument("--batch", type=int, default=16, help="problems per batch (default 16)")
    p.add_argument("--max-new-tokens", type=int, default=320, help="longest reply (default 320)")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto (default auto)")
    p.set_defaults(fn=score)

    a = ap.parse_args()
    a.fn(a)


# Only run main() when this file is run directly, not when another file imports it.
if __name__ == "__main__":
    sys.exit(main())
