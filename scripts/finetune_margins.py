# -=-=-=-=-=-=-
# 11/11/2026
# Replays every checked segment of a network's fine-tune rounds on this computer's GPU (or CPU), and writes how close
# each check came to failing: the distance the comparison measured, as a share of what the rule allows. Run it on each
# kind of GPU in a run to get the cross-hardware table: trained on X, replayed on Y.
# Version: 1.0
# -=-=-=-=-=-=-

"""Measure how close honest fine-tune checks come to their limits, on this computer's hardware.

    python scripts/finetune_margins.py --node https://ADDRESS:8545 --tls-fingerprint FP --out margins-4090.csv
    python scripts/finetune_margins.py --node http://127.0.0.1:8545 --device cpu --labels pods.json

A worker's check returns only pass or fail (verification.adam_reproduces, rule 2). This script runs the same replay
and the same comparison, but keeps the numbers: for each segment, and for each part of the checkpoint (the adapters B,
and Adam's moments m and v), the measured distance divided by the most the rule allows. 1.0 is the limit; an honest
replay should sit well under it, and the gap is the safety margin the paper reports.
"""
from __future__ import annotations

# -=-=-=- Imports -=-=-=-
# Standard library tools: argparse reads the command-line options, csv writes the results table, json reads the node's
# replies and the labels file, math does the rounding-up, platform names this computer's CPU, sys/Path find the project.
import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> the project folder, so `import vibehash` works
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402  (fast maths on whole arrays of weights)

from vibehash.blobs import BlobStore  # noqa: E402  (a folder of downloaded files, each checked against its hash)
from vibehash.client import NodeClient, NodeError  # noqa: E402  (the one way node traffic goes, proxy and TLS aware)
from vibehash.crypto import address_of  # noqa: E402  (a worker's address, from the public key it signed with)
from vibehash.model import relative_distance, train_segment  # noqa: E402  (the same replay a checker runs)
from vibehash.tasks import TrainingJob, WorkUnit, load_dataset  # noqa: E402
from vibehash.verification import (FINETUNE_CHECK, FLIP_SHARE, FRESH_TOLERANCE, MOMENT_TOLERANCE,  # noqa: E402
                                   UPDATE_TOLERANCE, adam_reproduces)
from vibehash.wire import blob_openings_in  # noqa: E402  (decodes a reveal's {segment: (blob id, proof)})

# -=-=-=- Settings -=-=-=-
FIELDS = ["height", "round", "job", "unit", "worker", "trained_on", "replayed_on", "segment", "fresh_moments",
          "part", "distance", "allowed_share", "flips_set_aside", "check_passes", "agrees_with_rule"]


# -=-=-=- Step 1: margins, the same comparisons the checker makes, keeping the numbers -=-=-=-
def margin_reproduces(claim, expected, start, tolerance, spec) -> tuple[float, float]:
    """verification._reproduces, as a share of its limit. It has two tests and passes when both hold:
    the claim within `tolerance` of the replay's size, and within 1% of what the segment changed.
    Returns (the larger of the two shares, the relative distance). Under 1.0 passes."""
    rel = relative_distance(claim, expected)
    err = float(np.linalg.norm(claim - expected))
    limit = max(UPDATE_TOLERANCE * np.linalg.norm(expected - start), np.finfo(spec.dtype).eps * np.linalg.norm(expected))
    second = err / limit if limit > 0 else (0.0 if err == 0 else math.inf)
    return max(rel / tolerance, second), rel


def margin_flips_aside(claim, expected, start, fraction, spec, lr, steps) -> tuple[float, float, int]:
    """verification._flips_aside, as a share of its limit: a unit's first segment, where Adam starts from zero moments
    and a weight whose gradient is nearer zero than two devices' rounding can step the other way. The furthest
    FLIP_SHARE of weights are set aside (each still no further than two runs of Adam can put it), and the rest must be
    within `fraction` of the segment's change. Returns (share of the limit, relative distance of the rest, flips)."""
    d = np.abs(np.asarray(claim, dtype=np.float64) - expected)
    cap = 2 * spec.max_step(lr, steps) * (1 + 1e-6)  # no weight may be further than this
    k = min(d.size, int(np.ceil(FLIP_SHARE * d.size)))  # how many weights may be set aside
    rest = np.partition(d, d.size - k)[:d.size - k]  # everything but the k furthest
    limit = max(fraction * np.linalg.norm(expected - start), np.finfo(spec.dtype).eps * np.linalg.norm(expected))
    rest_share = float(np.linalg.norm(rest)) / limit if limit > 0 else 0.0
    flips = int(np.count_nonzero(d > lr / 2))  # a flipped first step lands about lr away; honest noise is far smaller
    rel = float(np.linalg.norm(rest)) / (float(np.linalg.norm(expected)) + 1e-12)
    return max(float(d.max(initial=0.0)) / cap, rest_share), rel, flips


def segment_margins(job: TrainingJob, before, after, expected) -> list[dict]:
    """One row per part (B, m, v) for one replayed segment, by rule 2 (verification.adam_reproduces)."""
    spec, c = job.spec, job.checkpoint_every
    b, m, v = spec.parts
    fresh = not (np.any(before[m]) or np.any(before[v]))  # a unit's first segment: Adam's moments start at zero
    rows = []
    for name, part in (("m", m), ("v", v)):  # the moments: their size, or the job's tolerance if that's larger
        share, rel = margin_reproduces(after[part], expected[part], before[part], max(job.tolerance, MOMENT_TOLERANCE),
                                       spec)
        rows.append({"part": name, "distance": rel, "allowed_share": share, "flips_set_aside": 0})
    if fresh:  # B from zero moments: flips set aside
        share, rel, flips = margin_flips_aside(after[b], expected[b], before[b], FRESH_TOLERANCE, spec, job.lr, c)
        rows.append({"part": "B", "distance": rel, "allowed_share": share, "flips_set_aside": flips})
    else:  # B from built-up moments: the job's tolerance
        share, rel = margin_reproduces(after[b], expected[b], before[b], job.tolerance, spec)
        rows.append({"part": "B", "distance": rel, "allowed_share": share, "flips_set_aside": 0})
    passes = bool(adam_reproduces(after, expected, before, job.tolerance, spec, job.lr, c))
    ours = all(r["allowed_share"] <= 1.0 for r in rows)
    for r in rows:  # the real check's answer, and whether this script's numbers agree with it
        r |= {"fresh_moments": fresh, "check_passes": passes, "agrees_with_rule": passes == ours}
    return rows


# -=-=-=- Step 2: what the node has: a round's record, its job, its dataset and checkpoints -=-=-=-
class Fetcher:
    """Downloads from the node into a scratch folder, each file checked against its hash on arrival."""

    def __init__(self, node: NodeClient, scratch: Path):
        self.node = node
        self.blobs = BlobStore(scratch / "blobs")
        self.datasets = BlobStore(scratch / "datasets")
        self.jobs: dict[str, TrainingJob] = {}

    def record(self, height: int) -> dict | None:
        try:
            return self.node.get(f"/round/{height}/record")
        except NodeError:
            return None  # a block with no round, or one the node no longer keeps

    def job(self, job_id: str) -> TrainingJob:
        """The job as workers see it, with its dataset attached (as client.WorkerClient.job does)."""
        if job_id not in self.jobs:
            info = self.node.get(f"/job/{job_id}")
            job = TrainingJob.from_spec_json(json.dumps(info["spec"]), info["owner"], 0)
            sha = job.data_source["sha256"]
            job.attach_data(load_dataset(self.node.download("dataset", sha, self.datasets)), sha)
            self.jobs[job_id] = job
        return self.jobs[job_id]

    def array(self, blob_id: str, job: TrainingJob) -> np.ndarray:
        self.node.download("blob", blob_id, self.blobs)
        return np.array(self.blobs.load(blob_id, job.wire_dtype, job.spec.n_params), dtype=np.float64)


# -=-=-=- Step 3: which hardware this is -=-=-=-
def describe(device: str) -> str:
    """A name for this computer's device, for the replayed_on column."""
    if device.startswith("cuda"):
        import torch
        return torch.cuda.get_device_name(torch.device(device))
    return f"CPU {platform.processor() or platform.machine()}".replace(",", "")  # no commas: it's a CSV column


def pick_device(asked: str) -> str:
    if asked != "auto":
        return asked
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# -=-=-=- Step 4: replay every checked segment of every fine-tune round -=-=-=-
def measure(fetch: Fetcher, heights: range, device: str, labels: dict, writer: csv.DictWriter) -> tuple[int, int]:
    """Returns (segments replayed, rows written)."""
    from vibehash import compute, finetune

    backend = compute.TorchBackend(device)  # TF32 off and deterministic kernels, as a checker's
    here = describe(device)
    segments = rows = 0
    for height in heights:
        record = fetch.record(height)
        if not record or "games" not in record:  # only fine-tune rounds (record version 4) have disputes
            continue
        if record.get("rules", {}).get("finetune_check", 1) != FINETUNE_CHECK:
            print(f"block {height}: checked by rule {record['rules'].get('finetune_check', 1)}, skipped")
            continue
        job = fetch.job(record["job_id"])
        finetune.runtime(job.spec, job.data, device)  # the base model, fetched and checked against its manifest once
        units = {u.unit_id: u for u in (WorkUnit.make(job, record["round"], i, bytes.fromhex(record["start_hash"]))
                                         for i in range(job.units_per_round))}
        c = job.checkpoint_every
        # Which GPU each worker said it had, from this round's own signed registrations; --labels wins when given.
        said = {address_of(bytes.fromhex(e["pubkey"])): e.get("body", {}).get("hardware", {}).get("label", "?")
                for e in record.get("registrations", []) if isinstance(e, dict) and e.get("pubkey")}
        for rv in record.get("reveals", []):
            if "openings" not in rv or rv.get("unit_id") not in units:
                continue  # a reveal without checkpoints (rejected before upload) or for another kind of unit
            opened = blob_openings_in(rv["openings"], job.n_segments)
            for seg in sorted(i for i in opened if i + 1 in opened):  # segments with both ends revealed
                before = fetch.array(opened[seg][0], job)
                after = fetch.array(opened[seg + 1][0], job)
                expected = train_segment(job.spec, job.data, before, units[rv["unit_id"]].seed, seg * c, c, job.lr,
                                         job.batch_size, backend)
                segments += 1
                for r in segment_margins(job, before, after, expected):
                    writer.writerow({"height": height, "round": record["round"], "job": record["job_id"],
                                     "unit": rv["unit_id"], "worker": rv["worker"],
                                     "trained_on": labels.get(rv["worker"], said.get(rv["worker"], "?")),
                                     "replayed_on": here,
                                     "segment": seg, **r})
                    rows += 1
        print(f"block {height}: {segments} segments so far")
    return segments, rows


# -=-=-=- Run it -=-=-=-
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", required=True, help="the node's address, https:// (or http://127.0.0.1 for a local one)")
    ap.add_argument("--tls-fingerprint", help="the node's certificate fingerprint, for a self-signed one")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first block height (default 1)")
    ap.add_argument("--to", dest="end", type=int, help="last block height (default: the node's tip)")
    ap.add_argument("--device", default="auto", help="cuda, cpu or auto (default auto)")
    ap.add_argument("--labels", type=Path, help='JSON {"worker address": "what it trained on"}, for trained_on')
    ap.add_argument("--scratch", type=Path, default=Path("margins-scratch"), help="where downloads go")
    ap.add_argument("--out", type=Path, default=Path("margins.csv"), help="the results table (default margins.csv)")
    a = ap.parse_args()

    node = NodeClient(a.node, fingerprint=a.tls_fingerprint)
    end = a.end or node.status()["height"]
    labels = json.loads(a.labels.read_text()) if a.labels else {}
    device = pick_device(a.device)
    a.scratch.mkdir(parents=True, exist_ok=True)

    with a.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        segments, rows = measure(Fetcher(node, a.scratch), range(a.start, end + 1), device, labels, writer)
    print(f"replayed {segments} segments on {describe(device)}; {rows} rows in {a.out}")


# Only run main() when this file is run directly, not when another file imports it.
if __name__ == "__main__":
    main()
