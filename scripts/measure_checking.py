# -=-=-=-=-=-=-
# 11/11/2026
# Measures how often the protocol's spot-checks catch a cheating worker, and what the checking costs,
# using the protocol's own training, challenge and replay code. Compares 8 segments per unit (today) with 128.
# Version: 1.0
# -=-=-=-=-=-=-

"""Measure what checking a unit catches and what it costs.

    python scripts/measure_checking.py
"""
from __future__ import annotations

# -=-=-=- Imports -=-=-=-
# Standard library tools: sys controls where Python looks for code, time measures how long things take,
# and Path works out file paths.
import sys
import time
from pathlib import Path

# Go up two folders (scripts/ -> project folder) and tell Python to look there, so `import vibehash` works.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The project's own code. These are the exact functions a real worker and verifier run.
from vibehash.crypto import KeyPair, h256  # noqa: E402  (h256 makes a fingerprint; KeyPair is a wallet key)
from vibehash.coordinator import CoordinatorConfig  # noqa: E402  (a node's settings: q, replication, canaries)
from vibehash.ledger import Chain  # noqa: E402  (a fresh, in-memory blockchain)
from vibehash.model import ModelSpec, params_hash, train_segment  # noqa: E402
from vibehash.tasks import SYN_PER_VIBE, TrainingJob, WorkUnit  # noqa: E402
from vibehash.verification import challenge_segments, replay_cost, replay_segment  # noqa: E402
from vibehash.worker import OFFICE_DESKTOP, HardwareProfile, Worker  # noqa: E402
from tests.protocol import Protocol  # noqa: E402  (runs whole rounds in one process, no network needed)

# -=-=-=- Settings -=-=-=-
STEPS = 128  # training steps per unit; every setting does this same total work, so the comparison is fair
TRIALS = 20_000  # how many random challenges to draw per setting
SETTINGS = [(8, 3), (128, 3), (128, 4)]  # (segments per unit k, segments checked q); (8, 3) is today's default
SKIPS = {"1 segment": lambda _: 1, "a quarter": lambda k: k // 4, "half": lambda k: k // 2}  # how much a cheater fakes


# -=-=-=- Step 1: a job with k segments -=-=-=-
def make_job(k: int, spec: ModelSpec | None = None) -> TrainingJob:
    """A job whose units are STEPS steps long, with a checkpoint every STEPS // k steps, so each unit has k segments."""
    extra = {"spec": spec} if spec else {}  # use the small default model unless we're given another one
    return TrainingJob("measure", "req", budget=0, fee_per_round=0, steps_per_unit=STEPS, checkpoint_every=STEPS // k,
                       **extra)


# -=-=-=- Step 2: train a unit, honestly or not -=-=-=-
def train(job: TrainingJob, unit: WorkUnit, start, skip: int = 0) -> list:
    """Train a unit and return its checkpoints. The last `skip` segments are faked, not trained: the cheater guesses
    each one by repeating the previous segment's change (the "shortcut" cheater in vibehash/worker.py)."""
    if not 0 <= skip < job.n_segments:  # the guess needs one real segment before it to copy the change from
        raise ValueError(f"skip must be between 0 and {job.n_segments - 1}")
    checkpoints = [start]  # checkpoint 0 is the starting weights
    for seg in range(job.n_segments):  # seg = 0, 1, 2 ... k-1
        if seg >= job.n_segments - skip:
            # Faked: last checkpoint plus the last change again. No training, so no compute spent.
            theta = checkpoints[-1] + (checkpoints[-1] - checkpoints[-2])
        else:
            # Honest: run this segment's training steps from the last checkpoint, exactly as a real worker does.
            theta = train_segment(job.spec, job.data, checkpoints[-1], unit.seed,
                                  seg * job.checkpoint_every, job.checkpoint_every, job.lr, job.batch_size)
        checkpoints.append(job.spec.round(theta))  # round to the job's precision, as the worker does
    return checkpoints


# -=-=-=- Step 3: which segments fail a replay? -=-=-=-
def failing_segments(job: TrainingJob, unit: WorkUnit, checkpoints: list) -> set[int]:
    """Replay every segment the way a verifier does, and return the ones that don't reproduce within tolerance."""
    bad = set()
    for seg in range(job.n_segments):
        if not replay_segment(job, unit, seg, checkpoints[seg], checkpoints[seg + 1]):
            bad.add(seg)
    return bad


# -=-=-=- Step 4: how often does a random challenge land on a bad segment? -=-=-=-
def catch_rate(job: TrainingJob, unit: WorkUnit, worker: str, bad: set[int], q: int) -> float:
    """Draw TRIALS beacons, let the protocol pick q segments from each (challenge_segments, the real selection code),
    and count how often at least one picked segment is a bad one."""
    caught = 0
    for i in range(TRIALS):
        beacon = h256(b"trial", i.to_bytes(8, "big"))  # a different, unpredictable-looking beacon each trial
        picked = challenge_segments(beacon, unit.unit_id, worker, job.n_segments, q)
        if bad & set(picked):  # `&` keeps what's in both sets; an empty set counts as False
            caught += 1
    return caught / TRIALS


# -=-=-=- Step 5: what does a check cost, in time? -=-=-=-
def replay_share(job: TrainingJob, unit: WorkUnit, checkpoints: list, q: int) -> float:
    """Time training the whole unit against replaying q segments of it. Returns replay time / training time."""
    t0 = time.perf_counter()
    train(job, unit, checkpoints[0])  # the worker's job: all k segments
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    for seg in range(q):  # the verifier's job: q segments (which ones doesn't matter for the time)
        replay_segment(job, unit, seg, checkpoints[seg], checkpoints[seg + 1])
    t_replay = time.perf_counter() - t0
    return t_replay / t_train


# -=-=-=- Step 6: what does checking cost a whole round? -=-=-=-
def round_overhead(k: int, q: int, replication: float, rounds: int = 5) -> tuple[float, int, int, int]:
    """Run whole rounds with 18 honest workers and 12 units a round through the in-process network. Returns the extra
    compute (checks, canaries and second copies of units) divided by the useful training compute, plus the counts it
    came from: units trained, second copies, and checks.

    18 workers for 12 units, because a new worker is capped at one unit a round: with fewer workers than that, no one
    has room for a second copy, and the replication setting never happens."""
    reliable = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb,
                               reliability=1.0)  # never drops offline, so every unit gets done
    workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), reliable, "honest") for i in range(18)]
    requester = KeyPair.generate(b"req")
    chain = Chain({requester.address: 1000 * SYN_PER_VIBE},
                  stakes={w.address: 10 * SYN_PER_VIBE for w in workers})  # everyone staked, so they can work
    proto = Protocol(chain, workers, CoordinatorConfig(challenge_segments=q, replication_rate=replication), seed=3)
    job = TrainingJob("job", requester.address, budget=100 * SYN_PER_VIBE, fee_per_round=10 * SYN_PER_VIBE,
                      units_per_round=12, steps_per_unit=STEPS, checkpoint_every=STEPS // k)
    proto.post_job(requester, job)

    units = copies = 0
    checked = 0  # compute the coordinator counted for checking (replays and canaries)
    for _ in range(rounds):
        report = proto.run_round()
        distinct = len({res.unit.unit_id for res in report.results})  # each unit counted once
        units += distinct
        copies += len(report.results) - distinct  # results beyond one per unit are second copies
        checked += sum(report.verifier_flops.values())
    checks = checked // replay_cost(job, q)  # each check replays q segments, so this many checks were paid for
    # Count what actually happened, not what the settings asked for: a second copy is only made when someone has room.
    extra = checked + copies * job.unit_flops
    return extra / (units * job.unit_flops), units, copies, checks


# -=-=-=- Run it -=-=-=-
def main() -> None:
    print("== Catch rate: share of units where a random check finds a faked segment")
    print(f"   ({TRIALS:,} random challenges per row; small test model, {STEPS} steps per unit)\n")
    for k, q in SETTINGS:
        job = make_job(k)
        start = job.spec.init_params(job.init_seed)
        unit = WorkUnit.make(job, 1, 0, params_hash(start))
        worker = KeyPair.generate(b"cheater").address  # the challenge depends on who the worker is

        honest = train(job, unit, start)
        false_alarms = failing_segments(job, unit, honest)  # should be empty: honest work must always pass
        print(f"k={k:<4} q={q}  honest segments failing replay: {len(false_alarms)}"
              f"   replay time / training time: {replay_share(job, unit, honest, q):.1%}"
              f"   (math says {q / k:.1%})")

        for name, how_many in SKIPS.items():
            m = how_many(k)
            bad = failing_segments(job, unit, train(job, unit, start, skip=m))
            rate = catch_rate(job, unit, worker, bad, q)
            print(f"           skips {name:<10} ({m:>3} of {k}): {len(bad):>3} faked segments fail replay,"
                  f" caught {rate:6.1%}")
        print()

    print("== Whole-round overhead: checking compute / useful training compute (coordinator's accounting)\n")
    for k, q, replication in [(8, 3, 0.2), (8, 3, 0.0), (128, 4, 0.0)]:
        share, units, copies, checks = round_overhead(k, q, replication)
        print(f"k={k:<4} q={q}  replication {replication:.0%}:  {share:6.1%}"
              f"   ({units} units, {copies} second copies, {checks} checks incl. canaries)")

    print("\n== The upload problem: minimum compute phase for a 34M-weight model (the code's own timing model)\n")
    big = ModelSpec(8, (4096, 4096, 4096), 4, "float32")
    for k in (8, 128):
        print(f"k={k:<4} compute phase {make_job(k, big).phase_lengths()['compute'] / 60:6.0f} minutes")


# Only run main() when this file is run directly, not when another file imports it.
if __name__ == "__main__":
    main()
