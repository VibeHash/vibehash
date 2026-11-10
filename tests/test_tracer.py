"""tracer.FastExecutor against tracer.Executor: the same bits, tensor for tensor, from non-zero adapters and moments
(zero moments hide differences in them), on a tiny model on the CPU. Workers with and without the speed-ups must agree
bit for bit, and a dispute's re-run must reach the states the training went through."""
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tests.finetune_fixtures import TinyModel
    from tests.test_finetune import tmpdir
    from vibehash import finetune as ft
    from vibehash import tracer
    from vibehash.program import Program, split_theta

needs_torch = unittest.skipIf(torch is None, "needs PyTorch")
ALL = ("q", "k", "v", "o", "gate", "up", "down")
LR, BATCH, SEED = 1e-2, 3, 5


class Counting(tracer.FastExecutor if torch is not None else object):
    """A FastExecutor that counts what it reused, so a test can tell the reuse ran."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.replays = self.logits = 0

    def _replayed(self, op, st):
        hit = super()._replayed(op, st)
        self.replays += hit
        return hit

    def _head_grad(self, op, st):
        hit = super()._head_grad(op, st)
        self.logits += hit
        return hit


@needs_torch
class FastExecutorTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel(tmpdir(self), "qwen3").use(self)
        self.spec = self.model.spec(targets=ALL, rank=4)
        self.data = self.model.dataset()
        self.rt = ft.runtime(self.spec, self.data, "cpu")
        self.prog = Program(self.spec, self.data, LR, BATCH, SEED)
        rng = np.random.default_rng(0)
        theta = np.zeros(self.spec.n_params)
        b, m, v = self.spec.parts
        theta[b] = rng.normal(0, 0.05, b.stop - b.start)
        theta[m] = rng.normal(0, 1e-3, m.stop - m.start)
        theta[v] = np.abs(rng.normal(0, 1e-5, v.stop - v.start))
        self.theta = self.spec.round(theta)

    def state(self):
        return {k: torch.tensor(a) for k, a in split_theta(self.spec, self.theta).items()}

    def steps(self, ex, n=3):
        st = self.state()
        with torch.no_grad():
            for t in range(n):
                node = self.prog.step(t)
                ex.run_atoms(node, 0, len(node.atoms), st, (node.label,))
        return st

    def assertSameBits(self, a, b):
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), f"{k} differs")

    def test_the_same_bits_as_executor_with_everything_reused(self):
        plain = self.steps(tracer.Executor(self.rt, self.spec, self.prog))
        ex = Counting(self.rt, self.spec, self.prog, room=1 << 30)
        self.assertSameBits(self.steps(ex), plain)
        self.assertGreater(ex.replays, 0, "layer forwards were reused in the backward")
        self.assertGreater(ex.logits, 0, "head logits were reused")
        self.assertTrue(ex._w32, "projection weights were kept as float32")
        self.assertEqual(ex.used, sum(t.numel() * 4 for t in ex._w32.values()), "all but the weights given back")

    def test_the_same_bits_with_no_room_and_nothing_kept(self):
        plain = self.steps(tracer.Executor(self.rt, self.spec, self.prog))
        ex = Counting(self.rt, self.spec, self.prog, room=0)
        self.assertSameBits(self.steps(ex), plain)
        self.assertEqual((ex.replays, ex.logits, ex._w32, ex.used), (0, 0, {}, 0))

    def test_a_cheating_worker_keeps_nothing(self):
        ex = Counting(self.rt, self.spec, self.prog, tamper=lambda where, op: None)
        self.assertEqual(ex.room, 0)
        self.steps(ex, 1)
        self.assertEqual((ex.replays, ex.logits), (0, 0))

    def test_a_rerun_from_inside_a_backward_recomputes_and_matches(self):
        """A dispute can start a re-run at a layer's backward, or partway into one: nothing was recorded for it there,
        so the layer is recomputed, and the bits match a plain re-run's."""
        step = self.prog.step(0)
        micro = step.atoms[0]
        where = (step.label, micro.label)
        b = next(i for i, a in enumerate(micro.atoms) if getattr(a, "label", "").endswith(" backward"))
        st = self.state()
        with torch.no_grad():
            tracer.Executor(self.rt, self.spec, self.prog).run_atoms(micro, 0, b, st, where)
            for lo_node, lo in ((micro, b), (micro.atoms[b], 3)):
                start = dict(st)
                if lo_node is not micro:  # partway into the layer's backward: its first ops run plainly first
                    tracer.Executor(self.rt, self.spec, self.prog).run_atoms(lo_node, 0, lo, start, where)
                a, c = dict(start), dict(start)
                tracer.Executor(self.rt, self.spec, self.prog).run_atoms(lo_node, lo, len(lo_node.atoms), a, where)
                ex = Counting(self.rt, self.spec, self.prog, room=1 << 30)
                ex.run_atoms(lo_node, lo, len(lo_node.atoms), c, where)
                self.assertSameBits(c, a)
                self.assertEqual(ex.replays, 0, "nothing recorded to reuse")

    def test_train_segment_gives_the_plain_executors_result(self):
        fast = ft.train_segment(self.spec, self.data, self.theta, SEED, 0, 3, LR, BATCH)
        st = self.steps(tracer.Executor(self.rt, self.spec, Program(self.spec, self.data, LR, BATCH, SEED)))
        from vibehash.program import join_theta
        self.assertTrue(np.array_equal(fast, join_theta(self.spec, {k: v.numpy() for k, v in st.items()})))


@needs_torch
class HeadGradTests(unittest.TestCase):
    def old_grad(self, lo, w, tgt, op, st):
        """r_head_grad before 2026-11-10: rows picked out with .nonzero()."""
        q = torch.exp(lo - st[op.ins[1]])
        b0, b1 = op.attrs["b0"], op.attrs["b1"]
        rows = ((tgt >= b0) & (tgt < b1)).nonzero()[:, 0]
        if len(rows):
            q[rows, tgt[rows] - b0] -= 1
        return tracer._acc(st, op, 2, (q * op.attrs["ic"]) @ w)

    def test_without_nonzero_gives_the_same_bits(self):
        from vibehash.program import Op
        g = torch.Generator().manual_seed(0)
        for b0, b1, tgt in ((0, 16, [3, 15, 0]), (16, 32, [3, 40, 7]), (16, 32, [20, 31, 16]), (0, 16, [])):
            P = len(tgt)
            lo = torch.randn(P, b1 - b0, generator=g)
            w = torch.randn(b1 - b0, 8, generator=g)
            st = {"h": torch.zeros(P, 8), "lse": torch.randn(P, 1, generator=g), "acc": torch.randn(P, 8, generator=g)}
            for ins in (("h", "lse"), ("h", "lse", "acc")):
                op = Op("head_grad", "out", ins, {"b0": b0, "b1": b1, "ic": 0.25})
                t = torch.tensor(tgt, dtype=torch.long)
                self.assertTrue(torch.equal(tracer._grad(lo, w, t, op, st), self.old_grad(lo, w, t, op, st)))


@needs_torch
class NoFillTests(unittest.TestCase):
    def test_off_inside_and_put_back_after_even_on_an_error(self):
        det = torch.utils.deterministic
        old = det.fill_uninitialized_memory
        try:
            det.fill_uninitialized_memory = True
            with tracer.no_fill():
                self.assertFalse(det.fill_uninitialized_memory)
            self.assertTrue(det.fill_uninitialized_memory)
            with self.assertRaises(ValueError), tracer.no_fill():
                raise ValueError
            self.assertTrue(det.fill_uninitialized_memory)
        finally:
            det.fill_uninitialized_memory = old


if __name__ == "__main__":
    unittest.main()
