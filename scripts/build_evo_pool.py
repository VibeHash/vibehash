"""Build an Evo pool task's file (es_tasks.PoolTask): training problems asked the way scripts/vibe_bench.py asks the
benchmarks, each with a whole target reply that a `gold` job scores its members on.

The vibe_reason pool (2026-11-09):
* GSM8K and MATH training problems (es_tasks' gsm8k and math training parts): Qwen3.5-4B's greedy reply, kept when
  it is right and finished (it gets ~93% of GSM8K right, the 0.8B ~49%).
* Multiple choice, MMLU's validation and dev questions (not its test split, which MMLU, MMLU-Redux and MMLU-Pro draw
  on), MedMCQA's and AQuA's training questions: Qwen3.5-0.8B's own replies from 8 copies twisted by Evo's noise
  (sigma 1e-3); a problem is kept when at least 2 copies got it right, its target the right reply of median length.
  (The shortest right reply teaches brevity, not accuracy: on multiple choice it is often a lucky guess.)
* Problems the untwisted 0.8B gets wrong appear twice in "train" (they are what there is to learn).
* "eval": held-out problems of the same sources, no target, for eval_items.

Measured with gradient steps toward such a pool (2026-11-09, LoRA, 670 held-out problems scored paired against the
untwisted model): +4.5 points +/- 3.9 overall, replies shorter on every source, cut-off replies on GSM8K 16% to 8%.

  python scripts/build_evo_pool.py teacher --out pool_teacher.json      # Qwen3.5-4B on the math problems (GPU)
  python scripts/build_evo_pool.py self --out pool_self.json            # Qwen3.5-0.8B's twisted copies (GPU)
  python scripts/build_evo_pool.py write pool_teacher.json pool_self.json --out vibehash/data/evo_pool_vibe_reason.jsonl.gz

Needs the venv with math-verify (MATH's answers are checked with it, as vibe_bench checks MATH-500).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 20261109
_OPT = re.compile(r"^([A-H])\. (.*)$")


def _problems(source: str, n: int, part: str, seed: int = SEED) -> list[dict]:
    """n problems of `source` in the benchmarks' format: {"prompt", "gold", "kind", "src"}. part "train" draws from
    the task's training part, "eval" from what training never sees."""
    import vibe_bench as vb
    from vibehash import es_tasks
    if source == "mmlu":
        from datasets import load_dataset
        d = load_dataset("cais/mmlu", "all")
        rows = [r for s in ("validation", "dev") for r in d[s]]
        idx = random.Random(f"{seed} mmlu").sample(range(len(rows)), len(rows))
        cut = len(rows) * 9 // 10  # the last tenth held out
        pick = idx[:cut][:n] if part == "train" else idx[cut:][:n]
        return [{"prompt": vb._mc_prompt(rows[i]["question"], rows[i]["choices"]),
                 "gold": vb.LETTERS[int(rows[i]["answer"])], "kind": "mc", "src": "mmlu"} for i in pick]
    task = es_tasks.get(source)
    got, seen, k = [], set(), 0
    while len(got) < n and k < 200:
        its = task.items(seed + 7919 * k, n) if part == "train" else task.eval_items()
        for it in its:
            q = _from_task(it, source)
            if q and q["prompt"] not in seen and len(got) < n:
                seen.add(q["prompt"])
                got.append(q)
        if part != "train":
            break
        k += 1
    return got


def _from_task(it, source: str) -> dict | None:
    import vibe_bench as vb
    content = it.messages[-1]["content"]
    if source in ("gsm8k", "math"):
        return {"prompt": vb._math_prompt(content.rsplit("\n", 1)[0]), "gold": it.gold,
                "kind": "number" if source == "gsm8k" else "math", "src": source}
    lines = content.rsplit("\n", 1)[0].split("\n")
    i = len(lines)
    while i > 0 and _OPT.match(lines[i - 1]):
        i -= 1
    opts = [_OPT.match(x) for x in lines[i:]]
    if not opts or [m.group(1) for m in opts] != list(vb.LETTERS[:len(opts)]):
        return None
    return {"prompt": vb._mc_prompt("\n".join(lines[:i]), [m.group(2) for m in opts]), "gold": it.gold, "kind": "mc",
            "src": source}


def _right(q: dict, text: str, finished: bool, math_check) -> bool:
    import vibe_bench as vb
    if not finished:
        return False
    if q["kind"] == "mc":
        return vb.read_letter(text, True)[0] == q["gold"]
    if q["kind"] == "number":
        pred = vb.read_number(text, True)[0]
        g = float(q["gold"])
        return pred is not None and abs(pred - g) <= 1e-6 * max(1.0, abs(g))
    return bool(math_check(q["gold"], text)[0])


def cmd_teacher(a):
    import vibe_bench as vb
    qs = _problems("gsm8k", a.gsm8k, "train") + _problems("math", a.math, "train")
    gen = vb.Generator(vb.resolve(a.teacher), "auto", 8, cap=64, slots=120000)
    t0 = time.time()
    got = gen.answer([gen.ids(q["prompt"]) for q in qs], a.T,
                     lambda d: print(f"  {d}/{len(qs)} [{time.time() - t0:.0f}s]", flush=True) if d % 64 == 0 else None)
    check = vb._math_checker()
    out = []
    for q, toks in zip(qs, got):
        fin = bool(toks) and toks[-1] in gen.eos
        text = gen.text(toks)
        out.append({**q, "teacher": text if _right(q, text, fin, check) else None, "teacher_tokens": len(toks)})
    for src in ("gsm8k", "math"):
        rs = [x for x in out if x["src"] == src]
        print(f"  {src}: {len(rs)} problems, the teacher right on {sum(x['teacher'] is not None for x in rs)}",
              flush=True)
    Path(a.out).write_text(json.dumps(out))


def cmd_self(a):
    import torch
    import vibe_bench as vb
    from vibehash import es2, es_client, es_fast, es_local
    mc = _problems("mmlu", a.mmlu, "train") + _problems("medmcqa", a.medmcqa, "train") + _problems("aqua_rat", a.aqua,
                                                                                                    "train")
    qs = mc + _problems("gsm8k", a.gsm8k, "train") + _problems("math", a.math, "train")  # (math: the plain model only)
    found = es_local.find(a.model, download=False)
    es_local.register(found)
    spec = es_local.spec2_for(found, "gsm8k", tasks=("gsm8k",), pairs=8, problems=16)
    rt = es_client.runtime2(spec, "cuda")
    eos = set(rt.eos)
    rows = es_client.prompt_rows(rt, [type("I", (), {"messages": [{"role": "user", "content": q["prompt"]}]})()
                                      for q in qs])
    members = [(j, s) for j in range(a.copies // 2) for s in (1, -1)]
    fac = es_client.round_factors(rt, spec, SEED)
    fac0 = es_client.round_factors(rt, spec, SEED + 1)
    es2.Change(rt.linears, "cuda").load_into(rt.linears, rt.base)
    t0 = time.time()
    copies, plain = [[] for _ in members], []
    with torch.no_grad():
        for k in range(0, len(rows), 32):
            part = rows[k:k + 32]
            if k < len(mc):
                for m, x in zip(copies, es_fast.answer_members(rt, rt.linears, fac, members, part[:len(mc) - k], a.T,
                                                               spec.sigma, spec.rank, loop=(32, 48))):
                    m += x
            plain += es_fast.answer_members(rt, rt.linears, fac0, [(0, 1)], part, a.T, 0.0, spec.rank)[0]
            torch.cuda.empty_cache()
            print(f"  {min(k + 32, len(rows))}/{len(rows)} [{time.time() - t0:.0f}s]", flush=True)
    check = vb._math_checker()

    def text(t):
        return rt.tokenizer.decode([x for x in t if x not in eos], skip_special_tokens=True)

    def ok(q, t):
        return _right(q, text(t), bool(t) and t[-1] in eos, check)

    out = []
    for i, q in enumerate(qs):
        right = sorted((text(m[i]) for m in copies if ok(q, m[i])), key=len) if i < len(mc) else []
        out.append({**q, "self": right, "plain_right": ok(q, plain[i])})
    for src in sorted({q["src"] for q in qs}):
        rs = [x for x in out if x["src"] == src]
        print(f"  {src}: {len(rs)} problems, plain right {sum(x['plain_right'] for x in rs)}, 2+ copies right "
              f"{sum(len(x['self']) >= 2 for x in rs)}", flush=True)
    Path(a.out).write_text(json.dumps(out))


def cmd_write(a):
    teacher = {x["prompt"]: x for x in json.loads(Path(a.teacher_file).read_text())}
    selfp = {x["prompt"]: x for x in json.loads(Path(a.self_file).read_text())}
    rows, kept = [], {}
    for prompt, x in selfp.items():
        if x["src"] in ("gsm8k", "math"):
            t = teacher.get(prompt)
            target = t and t["teacher"]
        else:
            target = x["self"][(len(x["self"]) - 1) // 2] if len(x["self"]) >= a.min_right else None
        if not target:
            continue
        row = {"prompt": prompt, "target": target, "gold": str(x["gold"]), "kind": x["kind"], "src": x["src"],
               "part": "train"}
        rows += [row] * (2 if not x["plain_right"] else 1)
        kept[x["src"]] = kept.get(x["src"], 0) + 1
    for src, n in (("gsm8k", a.eval_n), ("math", a.eval_n), ("medmcqa", a.eval_n), ("aqua_rat", a.eval_n),
                   ("mmlu", a.eval_n)):
        rows += [{**q, "gold": str(q["gold"]), "target": "", "part": "eval"} for q in _problems(src, n, "eval")]
    random.Random(SEED).shuffle(rows)
    raw = gzip.compress("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode(),
                        mtime=0)
    Path(a.out).write_bytes(raw)
    train = [r for r in rows if r["part"] == "train"]
    print(f"wrote {a.out}: {len(train)} train rows ({kept}), {len(rows) - len(train)} eval; "
          f"{len(raw) / 1e6:.2f} MB, sha256 {hashlib.sha256(raw).hexdigest()}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    t.add_argument("--gsm8k", type=int, default=400)
    t.add_argument("--math", type=int, default=150)
    t.add_argument("--T", type=int, default=1024)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_teacher)
    s = sub.add_parser("self")
    s.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    s.add_argument("--mmlu", type=int, default=250)
    s.add_argument("--medmcqa", type=int, default=100)
    s.add_argument("--aqua", type=int, default=100)
    s.add_argument("--gsm8k", type=int, default=400, help="(the plain model only: which math problems count twice)")
    s.add_argument("--math", type=int, default=150)
    s.add_argument("--copies", type=int, default=8)
    s.add_argument("--T", type=int, default=768)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_self)
    w = sub.add_parser("write")
    w.add_argument("teacher_file")
    w.add_argument("self_file")
    w.add_argument("--min-right", type=int, default=2)
    w.add_argument("--eval-n", type=int, default=100)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_write)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
