"""How far honest fine-tune replays drift between GPUs, against how far cheating workers' claims land.

    python scripts/xgpu_drift.py make    --data chats.jsonl --base qwen3-0.6b --units 16 --dir /shared/run
    python scripts/xgpu_drift.py replay  --dir /shared/run --label rtx4090 --modes honest,tf32   (on every GPU)
    python scripts/xgpu_drift.py compare --dir /shared/run                          (once all have replayed)

A checker accepts a fine-tune segment when its own replay of the segment lands within rule 2 of the worker's claim
(verification.adam_reproduces): the adapters B, and Adam's moments m and v, each within a share of their size and of
what the segment changed. Run 1 on a test network found honest Blackwell and Ada workers rejecting each other on the
moments, by up to 2.9 times the limit, near the drift verification.py records for a worker computing in TF32. This
measures that question directly, without a network: every GPU starts each segment from the same bytes.

make     trains a warm round start on this device, then N units from it, and saves each segment's input checkpoint.
replay   trains every segment from those inputs on this device, once per mode, and saves the results.
compare  scores every claim (each device, each mode) against every device's honest replay by rule 2, writes
         compare.csv, and prints the verdicts and each part's share of its limit. Needs no GPU.

The modes are what a worker might do with a segment of c steps on b sequences each. `honest` and `tf32` run the
segment in one call, as a worker does. The rest run it a step at a time, so one step can differ:

    split           honest, a step at a time: the control, which must equal `honest` exactly on the same device
    tf32            every step with TF32 on (faster matmuls that keep 10 bits of mantissa)
    tf32-last       TF32 on for the last step only
    drop-last-step  the last step never run
    half-batch      every step on the first half of its sequences
    drop-one        every step without its last sequence
    drop-one-last   the last step without its last sequence: the smallest cut here, 1/(b*c) of the segment
    other-batches   every step on the next unit's batches (one unit trained, two claimed)
    skip            nothing run: the claim is the input

Checkpoints are float32 .npy files, about 27.5 MB each for Qwen3-0.6B at rank 16, in a folder every pod can read.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

WHOLE = ("honest", "tf32")  # run as one call a segment; the others a step at a time (plan_steps)
STEPPED = ("split", "tf32-last", "drop-last-step", "half-batch", "drop-one", "drop-one-last", "other-batches", "skip")
WARM_SEGMENTS = 4  # segments of training before the round start, so B is a trained adapter, not zeros


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        import torch
        return torch.cuda.get_device_name(torch.device(device))
    return f"CPU {platform.processor() or platform.machine()}".replace(",", "")


def pick_device(asked: str) -> str:
    if asked != "auto":
        return asked
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(a):
    """The job as `vibehash train` would plan it: spec, tokenized data, learning rate, batch and segment length. A
    catalog short name becomes its repository here, as `vibehash train` does, so meta.json records the repository."""
    from vibehash import catalog, finetune

    a.base = catalog.resolve(a.base)[0]
    p = finetune.plan(a.data, a.base, ask=lambda *_: True, max_len=a.max_len)
    return p.job, p.prepared.data


def set_tf32(on: bool) -> None:
    """Make every fine-tune step run with TF32 on (the cheat) or off (honest). finetune.train_segment re-pins full
    float32 before each segment through compute.pin_precision, so the cheat replaces that one function."""
    import torch
    from vibehash import compute

    if not hasattr(set_tf32, "honest"):
        set_tf32.honest = compute.pin_precision

    def cheat():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    compute.pin_precision = cheat if on else set_tf32.honest


def plan_steps(mode: str, c: int, b: int) -> list:
    """What a worker in a stepped `mode` does at each of a segment's c steps: None for a step it never runs, else
    (sequences in the batch, TF32 on, how many units along its batches come from). A smaller batch is the honest
    batch's first sequences (model.batch_indices draws them in order)."""
    honest, rest = (b, False, 0), c - 1
    return {
        "split": [honest] * c,
        "tf32-last": [honest] * rest + [(b, True, 0)],
        "drop-last-step": [honest] * rest + [None],
        "half-batch": [(max(1, b // 2), False, 0)] * c,
        "drop-one": [(b - 1, False, 0)] * c,
        "drop-one-last": [honest] * rest + [(b - 1, False, 0)],
        "other-batches": [(b, False, 1)] * c,
        "skip": [None] * c,
    }[mode]


def skipped(mode: str, c: int, b: int) -> float:
    """The share of a segment's sequence-steps a mode never computes on the claimed data. (TF32 computes them all, at
    lower precision; `other-batches` computes none of the claimed unit's, having trained its neighbour instead.)"""
    if mode in WHOLE:
        return 0.0
    if mode == "other-batches":
        return 1.0
    return 1.0 - sum(step[0] for step in plan_steps(mode, c, b) if step) / (c * b)


def segment(job, data, theta, seed, seg, backend, mode="honest"):
    """Segment `seg` of the unit with `seed`, from `theta`, as a worker in `mode` would claim it."""
    from vibehash import finetune

    c, b = job.checkpoint_every, job.batch_size
    if mode in WHOLE:
        set_tf32(mode == "tf32")
        return job.spec.round(finetune.train_segment(job.spec, data, theta, seed, seg * c, c, job.lr, b, backend))
    for i, step in enumerate(plan_steps(mode, c, b)):
        if step is None:
            continue
        batch, tf32, along = step
        set_tf32(tf32)
        theta = finetune.train_segment(job.spec, data, theta, seed + along, seg * c + i, 1, job.lr, batch, backend)
    return job.spec.round(theta)


def save(path: Path, theta) -> None:
    # Written in place: a RunPod global volume has no atomic rename. Readers only start once meta.json or done.json,
    # written after every file of their step, exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(theta, dtype=np.float32))


def load(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64)


def cmd_make(a) -> None:
    from vibehash import compute

    job, data = build(a)
    device = pick_device(a.device)
    backend = compute.TorchBackend(device)
    spec, d = job.spec, Path(a.dir)
    theta = spec.init_params()
    t0 = time.perf_counter()
    for seg in range(WARM_SEGMENTS):  # a short warm-up on one unit's batches, then Adam's moments zeroed: a round start
        theta = segment(job, data, theta, 1, seg, backend)
    start = spec.fresh(theta)
    save(d / "start.npy", start)
    for u in range(a.units):
        seed = 1000 + u
        theta = start
        for seg in range(job.n_segments):
            save(d / "inputs" / f"u{u:03d}-s{seg}.npy", theta)
            theta = segment(job, data, theta, seed, seg, backend)
        print(f"unit {u + 1}/{a.units} made", flush=True)
    meta = {"base": a.base, "data": str(a.data), "max_len": a.max_len, "units": a.units,
            "segments": job.n_segments, "steps_per_segment": job.checkpoint_every, "lr": job.lr,
            "batch_size": job.batch_size, "spec": spec.to_json(), "made_on": device_name(device),
            "seconds": round(time.perf_counter() - t0, 1)}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"made {a.units} units x {job.n_segments} segments in {meta['seconds']} s on {meta['made_on']}")


def cmd_replay(a) -> None:
    import torch
    from vibehash import compute

    modes = [m for m in a.modes.split(",") if m]
    unknown = [m for m in modes if m not in WHOLE + STEPPED]
    if unknown or "honest" not in modes:
        sys.exit(f"--modes must include honest, and only name {', '.join(WHOLE + STEPPED)}: got {a.modes}")
    d = Path(a.dir)
    while not (d / "meta.json").exists():  # make runs on another pod; wait for it
        time.sleep(15)
    meta = json.loads((d / "meta.json").read_text())
    a.data, a.base, a.max_len = meta["data"], meta["base"], meta["max_len"]
    job, data = build(a)
    device = pick_device(a.device)
    backend = compute.TorchBackend(device)
    out = d / "replays" / a.label
    seconds, counts = {}, {}
    for mode in modes:  # every unit honestly; the cheats on the first --cheat-units of them
        units = meta["units"] if mode == "honest" else min(meta["units"], a.cheat_units or meta["units"])
        t0 = time.perf_counter()
        for u in range(units):
            for seg in range(meta["segments"]):
                theta = load(d / "inputs" / f"u{u:03d}-s{seg}.npy")
                save(out / mode / f"u{u:03d}-s{seg}.npy", segment(job, data, theta, 1000 + u, seg, backend, mode))
        seconds[mode], counts[mode] = round(time.perf_counter() - t0, 1), units * meta["segments"]
        print(f"{a.label} {mode}: {counts[mode]} segments in {seconds[mode]:.0f} s", flush=True)
    set_tf32(False)
    # Proof each mode took effect here: its first claim's distance from the honest one. `split` must be 0; a cheat
    # that's 0 measured nothing on this device (TF32 on a CPU, say).
    honest = load(out / "honest" / "u000-s0.npy")
    effect = {}
    for mode in (m for m in modes if m != "honest"):
        other = load(out / mode / "u000-s0.npy")
        effect[mode] = float(np.linalg.norm(other - honest) / (np.linalg.norm(honest) + 1e-12))
        print(f"{a.label}: {mode} vs honest on the first segment, relative distance {effect[mode]:.3g}", flush=True)
    info = {"label": a.label, "device": device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "driver": _driver(), "seconds": seconds, "segments": counts,
            "effect": effect}
    (out / "done.json").write_text(json.dumps(info, indent=2))
    print(f"replayed on {info['device']}: {json.dumps(info)}")


def _driver() -> str:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def compare_segment(task) -> list[dict]:
    """Every claim on one segment against every device's honest replay of it. One call a process, so `compare` can
    spread the segments over the machine's cores (each comparison sorts a 6.9M-weight array)."""
    from types import SimpleNamespace

    from vibehash import finetune
    from finetune_margins import segment_margins

    d, meta, labels, u, seg = task
    d = Path(d)
    spec = finetune.LoraSpec.from_json(meta["spec"], 11)
    job = SimpleNamespace(spec=spec, checkpoint_every=meta["steps_per_segment"], tolerance=finetune.TOLERANCE,
                          lr=meta["lr"])
    name = f"u{u:03d}-s{seg}.npy"
    before = load(d / "inputs" / name)
    honest = {lab: load(d / "replays" / lab / "honest" / name) for lab in labels}
    rows = []
    for claimant in labels:
        for mode in WHOLE + STEPPED:
            path = d / "replays" / claimant / mode / name
            if not path.exists():  # a mode this device didn't run, or a unit past its --cheat-units
                continue
            claim = honest[claimant] if mode == "honest" else load(path)
            for checker in labels:
                for r in segment_margins(job, before, claim, honest[checker]):
                    rows.append({"unit": u, "segment": seg, "claimant": claimant, "mode": mode, "checker": checker,
                                 **r})
    return rows


def cmd_compare(a) -> None:
    d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    labels = sorted(p.name for p in (d / "replays").iterdir() if (p / "done.json").exists())
    devices = {lab: json.loads((d / "replays" / lab / "done.json").read_text()) for lab in labels}
    tasks = [(str(d), meta, labels, u, seg) for u in range(meta["units"]) for seg in range(meta["segments"])]
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.jobs or max(1, min(10, (os.cpu_count() or 2) - 1))) as pool:
        for k, part in enumerate(pool.map(compare_segment, tasks), 1):
            rows.extend(part)
            if k % 16 == 0 or k == len(tasks):
                print(f"compared {k}/{len(tasks)} segments, {time.perf_counter() - t0:.0f} s", flush=True)
    with open(d / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows, labels, devices, meta)


def _spread(shares) -> str:
    s = np.array(shares)
    return f"{s.min():.3g} {np.median(s):.3g} {np.quantile(s, 0.9):.3g} {s.max():.3g}"


def summarize(rows, labels, devices, meta) -> None:
    """The verdicts of the real check per mode, then each part's share of its limit (1.0 is at the limit): honest
    claims between different devices, by pair and by whether the segment started from zero moments, then each cheat.
    compare.csv keeps every row."""
    c, b = meta["steps_per_segment"], meta["batch_size"]
    print(f"=== SUMMARY: {meta['units']} units x {meta['segments']} segments, {meta['base']}, {c} steps a segment, "
          f"{b} sequences a step ===")
    for lab in labels:
        print(f"device {lab}: {json.dumps(devices[lab])}")
    checks: dict = {}  # one verdict per (claim, checker): every part's row carries it
    for r in rows:
        checks[(r["unit"], r["segment"], r["claimant"], r["mode"], r["checker"])] = (r["check_passes"],
                                                                                    r["fresh_moments"])
    print("=== VERDICTS: mode, share of the segment skipped, checks, passed ===")
    cross = {k: v for k, v in checks.items() if k[3] == "honest" and k[2] != k[4]}
    same = [v[0] for k, v in checks.items() if k[3] == "honest" and k[2] == k[4]]
    print(f"honest same-device 0 {len(same)} {sum(same)}")
    for name, fresh in (("zero-moments", True), ("built-up-moments", False)):
        some = [v[0] for v in cross.values() if v[1] == fresh]
        print(f"honest cross-device {name} 0 {len(some)} {sum(some)}")
    for mode in WHOLE[1:] + STEPPED:
        some = [v[0] for k, v in checks.items() if k[3] == mode]
        if some:
            print(f"{mode} {skipped(mode, c, b):.3g} {len(some)} {sum(some)}")
    print("=== HONEST, by pair: claimant checker moments checks passed, then max share of B m v ===")
    for cl in labels:
        for ch in labels:
            for name, fresh in (("zero", True), ("built-up", False)):
                if cl >= ch:
                    continue
                some = [r for r in rows if r["mode"] == "honest" and r["claimant"] == cl and r["checker"] == ch
                        and r["fresh_moments"] == fresh]
                verdicts = [r["check_passes"] for r in some if r["part"] == "B"]
                top = " ".join(f"{max(r['allowed_share'] for r in some if r['part'] == p):.3g}" for p in "Bmv")
                print(f"{cl} {ch} {name} {len(verdicts)} {sum(verdicts)} {top}")
    print("=== SHARES: mode moments part, then min median p90 max share of the limit ===")
    for mode in WHOLE + STEPPED:
        for name, fresh in (("zero", True), ("built-up", False)):
            for part in "Bmv":
                some = [r["allowed_share"] for r in rows if r["mode"] == mode and r["part"] == part
                        and r["fresh_moments"] == fresh and (mode != "honest" or r["claimant"] != r["checker"])]
                if some:
                    print(f"{mode} {name} {part} {_spread(some)}")
    off = sum(not r["agrees_with_rule"] for r in rows)
    print(f"rows where this script's shares disagree with the real check: {off} of {len(rows)}")
    print("=== END SUMMARY ===")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--data", required=True)
    p.add_argument("--base", default="qwen3-0.6b")
    p.add_argument("--units", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--dir", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("replay")
    p.add_argument("--dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--modes", default="honest,tf32", help="comma-separated, honest first: " + ", ".join(WHOLE + STEPPED))
    p.add_argument("--cheat-units", type=int, default=0, help="run the modes other than honest on the first N units "
                                                              "only (default: all)")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("compare")
    p.add_argument("--dir", required=True)
    p.add_argument("--jobs", type=int, default=0, help="processes (default: the cores but one, at most 10)")
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
