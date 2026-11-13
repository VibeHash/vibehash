"""Rules that came with block version 5: fine-tune check rule 3 (verification.FINETUNE_CHECK).

Honest fine-tune workers on two GPU generations, an RTX PRO 4500 (Blackwell) and an RTX 4090 (Ada) on the same PyTorch
and driver, rejected each other's segments on Adam's moments, and honest twins of a unit that disputed each other were
marked as missing (GitHub issue 85). Rule 3 lets B decide a segment, holds the moments to a looser bound, and makes a
replica that withdrew blameless (tests/test_finetune.py, NodeFreeRoundTests). Older rounds replay by their own rule.
"""
from __future__ import annotations

import unittest

import numpy as np

from vibehash import ledger
from vibehash.verification import FINETUNE_CHECK, MOMENT_BOUND, _flips_aside, _reproduces, adam_reproduces

try:
    import torch
except ImportError:  # the checks on real training need PyTorch; the rest don't
    torch = None
if torch is not None:
    from tests.finetune_fixtures import TinyModel
    from tests.test_finetune import tmpdir
    from vibehash import finetune as ft

needs_torch = unittest.skipIf(torch is None, "needs PyTorch")
# Honest m and v across the two GPU generations, at most, in 101 cross-type checks of the issue's run (relative
# distance), played here a little wider.
CROSS_GPU_M, CROSS_GPU_V = 1.3e-3, 3e-3


class VersionTests(unittest.TestCase):
    def test_block_version_5_brings_rule_3(self):
        """A node from before would refuse a rule-3 round and settle one differently, so the block version moves with
        the rule: such a node refuses the block, and its workers are told to upgrade."""
        self.assertEqual((ledger.VERSION, FINETUNE_CHECK), (5, 3))
        self.assertGreater(MOMENT_BOUND, CROSS_GPU_V * 3, "the bound keeps 3 times the measured noise")


def _tf32(t):
    """t rounded to TF32's 10-bit mantissa, to nearest even: what an NVIDIA tensor core multiplies with."""
    b = t.contiguous().view(torch.int32).to(torch.int64)
    return (((b + 0xFFF + ((b >> 13) & 1)) >> 13) << 13).to(torch.int32).view(torch.float32)


@needs_torch
class Rule3Tests(unittest.TestCase):
    """Rule 3 on the tiny model's real training, at the learning rate of the issue's run (5e-4, the largest
    finetune.settings_for gives), 4 steps a segment of batch 8 as there."""

    def setUp(self):
        self.tiny = TinyModel(tmpdir(self)).use(self)
        self.spec, self.data, self.lr = self.tiny.spec(), self.tiny.dataset(), 5e-4
        s, d = self.spec, self.data
        self.start = s.init_params()
        self.seg0 = self.train(self.start, 0)  # from zero moments: a unit's first segment
        self.mid = self.train(self.seg0, 1)
        self.seg2 = self.train(self.mid, 2)  # from moments built up

    def train(self, theta, seg, seed=7):
        return self.spec.round(ft.train_segment(self.spec, self.data, theta, seed, 4 * seg, 4, self.lr, 8))

    def check(self, claim, expected, before, rule=3):
        return adam_reproduces(self.spec.round(claim), expected, before, ft.TOLERANCE, self.spec, self.lr, 4, rule)

    def segments(self):
        return ((self.start, self.seg0), (self.mid, self.seg2))

    def test_the_moments_two_gpu_generations_put_apart_pass_rule_3_and_not_rule_2(self):
        b, m, v = self.spec.parts
        rng = np.random.default_rng(0)
        for before, honest in self.segments():
            noisy = honest.copy()
            noisy[m] *= 1 + CROSS_GPU_M * rng.standard_normal(self.spec.n_adapter)
            noisy[v] *= 1 + CROSS_GPU_V * rng.standard_normal(self.spec.n_adapter)
            self.assertFalse(self.check(noisy, honest, before, rule=2), "rule 2 rejected honest work")
            self.assertTrue(self.check(noisy, honest, before))
            self.assertTrue(self.check(honest, self.spec.round(noisy), before), "either way round")
            self.assertTrue(self.check(honest, honest, before))

    def test_cheats_still_fail_rule_3(self):
        """What rule 2 failed, rule 3 fails: a freezer, another unit's batches (a canary's), signs flipped across B,
        a step past Adam's reach, a flip from moments built up; and moments forged by more than the bound: scaled,
        turned around, or left where they started while B trained."""
        s = self.spec
        b, m, v = s.parts
        rng = np.random.default_rng(1)
        for before, honest in self.segments():
            seg = 0 if before is self.start else 2
            cheats = {"freezer": before, "other batches": self.train(before, seg, seed=8)}
            signs = honest.copy()
            signs[b] = np.where(rng.random(s.n_adapter) < 0.02, -honest[b], honest[b])
            far = honest.copy()
            far[b.start] += 3 * s.max_step(self.lr, 4)
            scaled, turned, stale = honest.copy(), honest.copy(), honest.copy()
            scaled[m] *= 1 + 2 * MOMENT_BOUND
            turned[m] = -honest[m]
            stale[m], stale[v] = before[m], before[v]
            cheats |= {"signs": signs, "far": far, "m scaled": scaled, "m turned": turned, "moments stale": stale}
            if seg:
                one = honest.copy()
                one[b.start] += 2 * self.lr
                cheats["flip from built-up moments"] = one
            for name, claim in cheats.items():
                with self.subTest(segment=seg, cheat=name):
                    self.assertFalse(self.check(claim, honest, before))

    def test_a_tf32_worker_fails_rule_3_on_b_alone(self):
        """A worker multiplying in TF32 (every projection's input rounded to 10 bits of mantissa) is a cheaper, less
        precise one, and its moments land as far off as two GPU generations' honest ones (here 4e-4 to 1.8e-3). Rule 3
        still fails it, on B alone. Measured on this model: from zero moments, B 1.28e-3 of the segment's change off
        (FRESH_TOLERANCE is 1e-3); from moments built up, 1.7e-4 of its size (the job's tolerance is 1e-4). Qwen3-0.6B's
        margins are wider (DESIGN §8)."""
        from vibehash import tracer
        from vibehash.tracer import Executor

        class TF32(Executor):
            def run(self, op, st):
                if op.kind in ("linear", "linear_bwd"):
                    view = dict(st)
                    view[op.ins[0]] = _tf32(st[op.ins[0]])
                    st[op.out] = tracer.RUN[op.kind](self, op, view).contiguous()
                    for d in op.drops:
                        del st[d]
                else:
                    super().run(op, st)

        s = self.spec
        b = s.parts[0]
        old = tracer.FastExecutor  # what finetune.train_segment trains with
        tracer.FastExecutor = TF32
        try:
            cheap = [self.train(before, seg) for seg, (before, _) in zip((0, 2), self.segments())]
        finally:
            tracer.FastExecutor = old
        (start, seg0), (mid, seg2) = self.segments()
        self.assertFalse(_flips_aside(cheap[0][b], seg0[b], start[b], 1e-3, s, self.lr, 4))
        self.assertFalse(_reproduces(cheap[1][b], seg2[b], mid[b], ft.TOLERANCE, s))
        for claim, (before, honest) in zip(cheap, self.segments()):
            self.assertFalse(self.check(claim, honest, before))


if __name__ == "__main__":
    unittest.main()
