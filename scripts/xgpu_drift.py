"""How far honest fine-tune replays drift between GPU generations, against how far a TF32 cheat drifts.

    python scripts/xgpu_drift.py make    --data chats.jsonl --base qwen3-0.6b --units 16 --dir /shared/run
    python scripts/xgpu_drift.py replay  --dir /shared/run --label rtx4090          (on every GPU)
    python scripts/xgpu_drift.py compare --dir /shared/run                          (once all have replayed)

A checker accepts a fine-tune segment when its own replay of the segment lands within rule 2 of the worker's claim
(verification.adam_reproduces): the adapters B, and Adam's moments m and v, each within a share of their size and of
what the segment changed. Run 1 on a test network found honest Blackwell and Ada workers rejecting each other on the
moments, by up to 2.9 times the limit, near the drift verification.py records for a worker computing in TF32. This
measures that question directly, without a network: every GPU starts each segment from the same bytes.

make     trains a warm round start on this device, then N units from it, and saves each segment's input checkpoint.
replay   trains every segment from those inputs on this device, honestly and with TF32 on, and saves the results.
compare  scores every claim (each device, honest or TF32) against every device's honest replay by rule 2, writes
         compare.csv, and prints the distribution of each part's share of its limit, per pair. Needs no GPU.

Checkpoints are float32 .npy files, about 27.5 MB each for Qwen3-0.6B at rank 16, in a folder every pod can read.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

MODES = ("honest", "tf32")
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
    """The job as `vibehash train` would plan it: spec, tokenized data, learning rate, batch and segment length."""
    from vibehash import finetune

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


def segment(job, data, theta, seed, seg, backend):
    from vibehash import finetune

    c = job.checkpoint_every
    return job.spec.round(finetune.train_segment(job.spec, data, theta, seed, seg * c, c, job.lr, job.batch_size,
                                                 backend))


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

    d = Path(a.dir)
    while not (d / "meta.json").exists():  # make runs on another pod; wait for it
        time.sleep(15)
    meta = json.loads((d / "meta.json").read_text())
    a.data, a.base, a.max_len = meta["data"], meta["base"], meta["max_len"]
    job, data = build(a)
    device = pick_device(a.device)
    backend = compute.TorchBackend(device)
    out = d / "replays" / a.label
    for mode in MODES:
        set_tf32(mode == "tf32")
        t0 = time.perf_counter()
        for u in range(meta["units"]):
            for seg in range(meta["segments"]):
                theta = load(d / "inputs" / f"u{u:03d}-s{seg}.npy")
                save(out / mode / f"u{u:03d}-s{seg}.npy", segment(job, data, theta, 1000 + u, seg, backend))
        print(f"{a.label} {mode}: {meta['units'] * meta['segments']} segments in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    set_tf32(False)
    # Proof the cheat took effect here: on a GPU with TF32, its result must differ from the honest one. (On a CPU, or
    # a GPU without TF32, they're equal, and this device's TF32 rows measure nothing.)
    first = [load(out / mode / "u000-s0.npy") for mode in MODES]
    gap = float(np.linalg.norm(first[1] - first[0]) / (np.linalg.norm(first[0]) + 1e-12))
    print(f"{a.label}: TF32 vs honest on the first segment, relative distance {gap:.3g}"
          f"{'' if gap > 0 else '  (TF32 had no effect on this device)'}", flush=True)
    info = {"label": a.label, "device": device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "driver": _driver(), "tf32_effect": gap}
    (out / "done.json").write_text(json.dumps(info, indent=2))
    print(f"replayed on {info['device']}: {json.dumps(info)}")


def _driver() -> str:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cmd_compare(a) -> None:
    from types import SimpleNamespace

    from vibehash import finetune
    from finetune_margins import segment_margins

    d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    spec = finetune.LoraSpec.from_json(meta["spec"], 11)
    job = SimpleNamespace(spec=spec, checkpoint_every=meta["steps_per_segment"], tolerance=finetune.TOLERANCE,
                          lr=meta["lr"])
    labels = sorted(p.name for p in (d / "replays").iterdir() if (p / "done.json").exists())
    devices = {lab: json.loads((d / "replays" / lab / "done.json").read_text()) for lab in labels}
    rows = []
    for u in range(meta["units"]):
        for seg in range(meta["segments"]):
            before = load(d / "inputs" / f"u{u:03d}-s{seg}.npy")
            honest = {lab: load(d / "replays" / lab / "honest" / f"u{u:03d}-s{seg}.npy") for lab in labels}
            for claimant in labels:
                for mode in MODES:
                    claim = honest[claimant] if mode == "honest" else load(
                        d / "replays" / claimant / mode / f"u{u:03d}-s{seg}.npy")
                    for checker in labels:
                        for r in segment_margins(job, before, claim, honest[checker]):
                            rows.append({"unit": u, "segment": seg, "claimant": claimant, "mode": mode,
                                         "checker": checker, **r})
    with open(d / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows, labels, devices, meta)


def summarize(rows, labels, devices, meta) -> None:
    """One line per (claim, checker, part): how many segments, how many the real check passed, and the share of the
    limit at the median, 90th percentile and maximum; then the gap between the worst honest and the best TF32."""
    print(f"=== SUMMARY: {meta['units']} units x {meta['segments']} segments, {meta['base']}, "
          f"{meta['steps_per_segment']} steps a segment ===")
    for lab in labels:
        print(f"device {lab}: {json.dumps(devices[lab])}")
    print("claim_by mode checker part n passed median p90 max")
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["claimant"], r["mode"], r["checker"], r["part"]), []).append(r)
    for (cl, mode, ch, part), rs in sorted(groups.items()):
        s = np.array([r["allowed_share"] for r in rs])
        passed = sum(r["check_passes"] for r in rs)
        print(f"{cl} {mode} {ch} {part} {len(rs)} {passed} {np.median(s):.3g} {np.quantile(s, 0.9):.3g} "
              f"{s.max():.3g}")
    print("=== SEPARATION: per part, worst honest cross-device share vs best TF32 share ===")
    for part in ("B", "m", "v"):
        honest = [r["allowed_share"] for r in rows if r["part"] == part and r["mode"] == "honest"
                  and r["claimant"] != r["checker"]]
        cheat = [r["allowed_share"] for r in rows if r["part"] == part and r["mode"] == "tf32"]
        if honest and cheat:
            print(f"{part}: honest cross-device max {max(honest):.3g}, TF32 min {min(cheat):.3g}, "
                  f"ratio {min(cheat) / max(max(honest), 1e-12):.3g}")
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
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("compare")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
