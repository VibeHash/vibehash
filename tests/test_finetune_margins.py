"""scripts/finetune_margins.py measures how close a fine-tune check came to failing. Its numbers are only worth
reporting if they say pass exactly when the real check (verification.adam_reproduces) does, so these tests move a
replay's result by known amounts, from bit-identical to plainly wrong, and hold the script to the check's verdict."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vibehash import finetune, selftest
from vibehash.verification import adam_reproduces

_path = Path(__file__).resolve().parent.parent / "scripts" / "finetune_margins.py"
_spec = importlib.util.spec_from_file_location("finetune_margins", _path)
margins = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(margins)


def tiny_job(lr=5e-4, every=4):
    """A job-shaped object with a real LoraSpec for the selftest's tiny Qwen3: what segment_margins reads."""
    spec = finetune.LoraSpec(base="hf:selftest/tiny@" + "a" * 40, manifest="b" * 64,
                             dims=finetune.arch_dims(selftest.TINY), seq_len=32)
    return SimpleNamespace(spec=spec, checkpoint_every=every, tolerance=finetune.TOLERANCE, lr=lr)


def segment(job, fresh: bool, seed=0):
    """A segment's start and an honest replay's end: B moved by about lr a step, and Adam's moments built up."""
    rng = np.random.default_rng(seed)
    b, m, v = job.spec.parts
    before = np.zeros(job.spec.n_params)
    if not fresh:
        before[b] = rng.normal(0, 1e-2, job.spec.n_adapter)
        before[m] = rng.normal(0, 1e-3, job.spec.n_adapter)
        before[v] = rng.uniform(1e-8, 1e-6, job.spec.n_adapter)
    expected = before.copy()
    expected[b] += rng.normal(0, job.lr * job.checkpoint_every, job.spec.n_adapter)
    expected[m] += rng.normal(0, 1e-3, job.spec.n_adapter)
    expected[v] += rng.uniform(1e-8, 1e-6, job.spec.n_adapter)
    return before, expected


class MarginTests(unittest.TestCase):
    def test_an_identical_replay_uses_none_of_the_limit(self):
        job = tiny_job()
        for fresh in (True, False):
            before, expected = segment(job, fresh)
            rows = margins.segment_margins(job, before, expected.copy(), expected)
            self.assertEqual({r["part"] for r in rows}, {"B", "m", "v"})
            for r in rows:
                self.assertEqual(r["allowed_share"], 0.0)
                self.assertTrue(r["check_passes"])
                self.assertTrue(r["agrees_with_rule"])

    def test_the_share_grows_with_the_error_and_agrees_with_the_check_at_every_size(self):
        """Noise from far below the limit to far above it: the script's pass/fail is the check's, every time, and its
        share of the limit only grows as the error does."""
        job = tiny_job()
        rng = np.random.default_rng(1)
        for fresh in (True, False):
            before, expected = segment(job, fresh)
            b = job.spec.parts[0]
            shares = []
            for scale in (1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                claim = expected.copy()
                claim += rng.normal(0, scale, claim.size) * np.abs(expected)
                rows = margins.segment_margins(job, before, claim, expected)
                for r in rows:
                    self.assertTrue(r["agrees_with_rule"], f"fresh={fresh} scale={scale} part={r['part']}")
                shares.append(next(r["allowed_share"] for r in rows if r["part"] == "B"))
                self.assertEqual(rows[0]["check_passes"],
                                 bool(adam_reproduces(claim, expected, before, job.tolerance, job.spec, job.lr,
                                                      job.checkpoint_every)))
            self.assertEqual(shares, sorted(shares), f"fresh={fresh}: {shares}")
            self.assertLess(shares[0], 1.0)  # tiny noise passes...
            self.assertGreater(shares[-1], 1.0)  # ...and 1% of every weight fails

    def test_a_flip_from_zero_moments_is_set_aside_but_too_many_fail(self):
        """A unit's first segment may have FLIP_SHARE of B's weights a sign flip apart (two devices' rounding on a
        near-zero gradient). One such weight of this tiny model's 5,120 passes; several don't."""
        job = tiny_job()
        before, expected = segment(job, fresh=True)
        b = job.spec.parts[0]
        one = expected.copy()
        one[b.start] += 2 * job.lr  # a first step taken the other way
        rows = {r["part"]: r for r in margins.segment_margins(job, before, one, expected)}
        self.assertEqual(rows["B"]["flips_set_aside"], 1)
        self.assertTrue(rows["B"]["check_passes"])
        self.assertTrue(rows["B"]["agrees_with_rule"])

        several = expected.copy()
        several[b.start:b.start + 5] += 2 * job.lr
        rows = {r["part"]: r for r in margins.segment_margins(job, before, several, expected)}
        self.assertEqual(rows["B"]["flips_set_aside"], 5)
        self.assertFalse(rows["B"]["check_passes"])
        self.assertTrue(rows["B"]["agrees_with_rule"])

    def test_a_device_name_never_breaks_the_csv(self):
        self.assertNotIn(",", margins.describe("cpu"))


if __name__ == "__main__":
    unittest.main()
