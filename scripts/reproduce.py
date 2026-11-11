"""Regenerate the numbers Vibe Hash publishes, on CPU, with no GPU and no PyTorch, and check them.

    python scripts/reproduce.py            # print every section
    python scripts/reproduce.py --check    # also check each against its expected result; exit 1 on any mismatch

Sections:
  detection  the whitepaper's §13 table (docs/whitepaper.html): the chance a worker that skips m of a unit's k segments
             is caught by q spot-checked segments, C(k-m, q) / C(k, q) escaped, for k = 8 and k = 4 with q = 3. Checked
             against the values the paper prints, and against its claim that P(caught) is at least the share skipped.
  deterrence that skipping segments never beats honest work for a risk-neutral worker who forfeits the unit's reward
             when caught, whenever the reward covers the compute, with no stake at all (DESIGN.md §4's inequality).
  tradeoff   replay cost q/k against the chance of catching a skip of one segment, a quarter or half of a unit.
  referee    the op-level referee's sum bound (referee.py), as tests/test_dispute.py BoundTests sets it up: on linear
             tiles at K = 1024, 2560 and 4096, honest float32 products pass every tile, while bf16 activations and TF32
             fail every tile, and the allowance stays under 1e-4 of a value (median).

Not here: the network's catches (every cheater caught, no honest worker slashed), which tests/test_core.py
checks. Not here yet, because they need PyTorch or a GPU: DESIGN.md §8's CPU-vs-GPU replay deltas, the Evo margins
(docs/CROSS-HARDWARE.md, scripts/evo_margin_xgpu.py). referee.py's docstring cites "CalibrationTests" for the
"1.02 of the scale" figure in DESIGN.md; no test of that name exists yet, so that figure has no reproduction.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# docs/whitepaper.html §13, as printed: P(caught) for m skipped of k = 8, q = 3; and the fine-tune case k = 4, q = 3.
PAPER_K8 = {1: 0.3750, 2: 0.6429, 3: 0.8214, 4: 0.9286, 5: 0.9821, 6: 1.0, 7: 1.0, 8: 1.0}
PAPER_K4 = {1: 0.75, 2: 1.0, 3: 1.0, 4: 1.0}


def p_caught(k: int, m: int, q: int) -> float:
    return 1 - comb(k - m, q) / comb(k, q)


def detection() -> tuple[dict, list[str]]:
    out, bad = {}, []
    for k, paper in ((8, PAPER_K8), (4, PAPER_K4)):
        rows = {}
        for m, printed in paper.items():
            p = p_caught(k, m, 3)
            rows[m] = round(p, 4)
            if abs(p - printed) > 5e-5:
                bad.append(f"k={k} m={m}: computed {p:.4f}, the paper prints {printed:.4f}")
            if p + 1e-12 < m / k:
                bad.append(f"k={k} m={m}: P(caught) {p:.4f} is below the share skipped {m / k:.4f}")
        out[f"k={k},q=3"] = rows
    return out, bad


def deterrence() -> tuple[dict, list[str]]:
    """A risk-neutral worker that skips m of k segments and, if caught, forfeits at least that unit's reward R, against
    an honest one: skipping pays (1 - P) R - (1 - m/k) C, honesty pays R - C. Skipping comes out ahead only when
    (m/k) C > P R, and since P >= m/k that needs C > R, a job nobody would mine. Checked with no stake at all."""
    worst, bad = -1.0, []
    for k in (4, 8, 16, 32, 64, 128):
        for q in (1, 2, 3, 4):
            for m in range(1, k + 1):
                for c in (i / 100 for i in range(1, 101)):  # compute cost as a share of the reward, C / R
                    worst = max(worst, (1 - p_caught(k, m, q)) - (1 - m / k) * c - (1 - c))
    if worst > 1e-12:
        bad.append(f"skipping beat honesty by {worst:.3g} of a reward with C <= R")
    return {"max advantage of skipping, C <= R, no stake": round(worst, 12)}, bad


def tradeoff() -> tuple[dict, list[str]]:
    """Replay cost (q/k of a unit) against the chance of catching a worker that skipped one segment, a quarter or half
    of a unit. Catching a one-segment skip with probability p costs replaying p of the work, at any k; finer
    segments make the large skips cheap to catch."""
    out = {}
    for k in (8, 16, 32, 64, 128):
        for q in (3, 4):
            out[f"k={k},q={q}"] = {"replay": round(q / k, 4), "one segment": round(p_caught(k, 1, q), 4),
                                   "quarter": round(p_caught(k, k // 4, q), 4), "half": round(p_caught(k, k // 2, q), 4)}
    bad = [f"{key}: one-segment catch {v['one segment']} != replay share {v['replay']}"
           for key, v in out.items() if abs(v["one segment"] - v["replay"]) > 1e-4]
    return out, bad


def _round_bits(a, drop):
    """float32s rounded to `drop` fewer mantissa bits, ties to even (16: bf16; 13: TF32). As tests/test_dispute.py."""
    b = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    half = (1 << (drop - 1)) - 1
    b = ((b + half + ((b >> drop) & 1)) >> drop) << drop
    return b.astype(np.uint32).view(np.float32)


class _Src:
    def __init__(self, ws):
        self.ws = ws

    def shape(self, name):
        return "F32", tuple(self.ws[name].shape)

    def read(self, name, r0, r1):
        return np.ascontiguousarray(self.ws[name][r0:r1], dtype="<f4").tobytes()


def referee_bound() -> tuple[dict, list[str]]:
    from vibehash import dispute, merkle, referee
    from vibehash.program import Op

    out, bad = {}, []
    rng = np.random.default_rng(0)
    for K, O in ((1024, 1024), (2560, 2560), (4096, 512)):
        T = 4
        x = rng.normal(0, 1, (T, K)).astype(np.float32)
        w = _round_bits(rng.normal(0, 1 / np.sqrt(K), (O, K)), 16)
        op = Op("linear", "y", ("x",), {"weight": "W", "bias": None}, (), (T, O), ((T, K),))
        src = dispute.Outside(None, None, _Src({"W": w}), keep=False)
        cases = {"honest": x @ w.T, "bf16 activations": _round_bits(x, 16) @ w.T,
                 "tf32": _round_bits(x, 13) @ _round_bits(w, 13).T}
        row = {}
        for name, y in cases.items():
            flat = y.reshape(-1)
            passed = [referee.judge(op, dispute.Inputs({"x": (T, K)}, src, full={"x": x}), leaf,
                                    flat[leaf * merkle.LEAF:(leaf + 1) * merkle.LEAF])[0]
                      for leaf in range(merkle.n_leaves(T, O))]
            share = sum(passed) / len(passed)
            row[name] = round(share, 4)
            want = 1.0 if name == "honest" else 0.0
            if share != want:
                bad.append(f"K={K} {name}: {share:.1%} of tiles passed, expected {want:.0%}")
        exact = referee.op_linear(op, dispute.Inputs({"x": (T, K)}, src, full={"x": x}), 0, 0, merkle.LEAF)
        allowed = referee.MARGIN * exact.e + referee.REL * np.abs(exact.v) + referee.FLOOR
        med = float(np.median(allowed / np.abs(exact.v)))
        row["median allowance / |value|"] = float(f"{med:.3g}")
        if not med < 1e-4:
            bad.append(f"K={K}: median allowance {med:.3g} of a value, expected under 1e-4")
        out[f"K={K}"] = row
    return out, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if any section doesn't match what it should")
    ap.add_argument("--json", type=Path, help="also write the results here")
    args = ap.parse_args()

    results, failures = {}, []
    for name, fn in (("detection", detection), ("deterrence", deterrence), ("tradeoff", tradeoff),
                     ("referee", referee_bound)):
        got, bad = fn()
        results[name] = got
        failures += [f"{name}: {b}" for b in bad]
        print(f"== {name}: {'ok' if not bad else f'{len(bad)} mismatch(es)'}")
        print(json.dumps(got, indent=2))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    for f in failures:
        print("MISMATCH", f, file=sys.stderr)
    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
