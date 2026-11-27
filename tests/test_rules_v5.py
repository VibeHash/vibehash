"""Rules that came with block version 5: fine-tune check rule 3 (verification.FINETUNE_CHECK).

Honest fine-tune workers on two GPU generations, an RTX PRO 4500 (Blackwell) and an RTX 4090 (Ada) on the same PyTorch
and driver, rejected each other's segments on Adam's moments, and honest twins of a unit that disputed each other were
marked as missing (GitHub issue 85). Rule 3 lets B decide a segment, holds the moments to a looser bound, and makes a
replica that withdrew blameless (tests/test_finetune.py, NodeFreeRoundTests). Older rounds replay by their own rule.
"""
from __future__ import annotations

import json
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


# --- the job kinds that were cut: Evo and DeepSeek -------------------------------------------------------------------

from vibehash.coordinator import Coordinator  # noqa: E402
from vibehash.crypto import KeyPair, h256  # noqa: E402
from vibehash.ledger import SYSTEM, Block, Chain, InvalidBlock, RoundSummary, Tx, apply_block, canonical  # noqa: E402
from vibehash.replay import replay_block  # noqa: E402
from vibehash.tasks import SYN_PER_VIBE, RetiredJob, TrainingJob, retired_kind  # noqa: E402

V = SYN_PER_VIBE
LEAD, MINER = KeyPair.generate(b"rules v5 lead"), KeyPair.generate(b"rules v5 miner")
# think-shorter-pilot-2's spec as vibehash-public-1 holds it (posted at height 7,529), under another id.
DEEPSEEK_SPEC = {
    "base": "hf:deepseek-ai/DeepSeek-V4.1-Flash@314113f8a7dadb388215396ba5fbee2c610a04d6", "batch_size": 8,
    "checkpoint_every": 4, "data": {"heldback": [1500, 2000], "kind": "deepseek", "train": [0, 1500],
                                    "sha256": "f4d7f822cf54a9f5a7f21441759bee63e2d1294d4a747b89588f7fb762523a3a"},
    "deepseek": {"chunk_tokens": 4096, "cut": 36, "pieces": 4, "row_ranges": 15}, "fee_per_round": 16 * V // 10,
    "id": "think-shorter-test", "init_seed": 11, "lora": {"alpha": 32, "rank": 16, "targets": ["wq_a", "wq_b", "wkv", "wo_b"]},
    "lr": 0.0001, "manifest": "5d069a8a464c59210d35973f5c52b0adaf630eb6a013af0b91267382e73c6f29", "max_update_ratio": 1.0,
    "memory_mb": 8000, "model": {"config": "flash", "family": "deepseek_v41"}, "optimizer": "adam",
    "precision": "float32", "seq_len": 512, "steps_per_unit": 16, "tolerance": 0.0001, "units_per_round": 16}
EVO_SPEC = {"id": "evo-test", "fee_per_round": V, "base": "hf:Qwen/Qwen2.5-0.5B-Instruct@" + "0" * 40, "manifest": "0" * 64,
            "model": {"family": "qwen2"}, "es": {"pairs": 8, "sigma": 0.001}, "init_seed": 11, "units_per_round": 8,
            "memory_mb": 4000}


def _memo(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _chain() -> Chain:
    return Chain({LEAD.address: 1000 * V, MINER.address: 10 * V}, {LEAD.address: 10 * V, MINER.address: 10 * V},
                 chain_id="rules-v5-retired")


def _seal(c: Chain, txs=(), summary=None, version: int = ledger.VERSION) -> Block:
    block, dropped = c.build_block(summary, list(txs), c.tip.timestamp + 1, version=version)
    if dropped:
        raise InvalidBlock(f"dropped {len(dropped)} tx")
    block.sign(LEAD)
    c.append(block)
    return block


def _post(c: Chain, spec: dict, budget: int = 50 * V) -> Tx:
    return Tx.signed(LEAD, "post_job", SYSTEM, budget, c.next_nonce(LEAD.address), _memo(spec), c.chain_id)


def _round(job_id: str, record: dict | None = None, units: int = 6) -> RoundSummary:
    flops = {MINER.address: 10**15}
    return RoundSummary(job_id, units, flops, dict(flops), [], [MINER.address], [], "00" * 32, LEAD.address,
                        h256(canonical(record)).hex() if record is not None else "")



class RetiredJobTests(unittest.TestCase):
    """Evo and DeepSeek were cut after 0.5.28 (DESIGN.md, "Evo and DeepSeek, cut"). vibehash-public-1 holds two DeepSeek
    jobs and their rounds, which must apply exactly as before; from block version 5 nothing new of either kind."""

    def test_specs_are_told_apart_as_the_old_code_did(self):
        self.assertEqual(retired_kind(DEEPSEEK_SPEC), "deepseek")
        self.assertEqual(retired_kind(EVO_SPEC), "evo")
        self.assertIsNone(retired_kind({**DEEPSEEK_SPEC, "model": {"n_in": 3}}))  # an MLP's model: not one
        self.assertIsNone(retired_kind({k: v for k, v in DEEPSEEK_SPEC.items() if k != "deepseek"}))  # a fine-tune's
        self.assertIsNone(retired_kind({**EVO_SPEC, "lora": {}}))
        for spec, kind in ((DEEPSEEK_SPEC, "deepseek"), (EVO_SPEC, "evo")):
            with self.assertRaises(RetiredJob) as e:
                TrainingJob.from_spec_json(_memo(spec), LEAD.address, 1)
            self.assertEqual((e.exception.kind, e.exception.job_id), (kind, spec["id"]))
            self.assertIn("removed from VibeHash after 0.5.28", str(e.exception))
        self.assertEqual((e.exception.fee_per_round, e.exception.units_per_round), (V, 8))

    def test_an_old_block_posting_a_deepseek_job_applies_as_it_did(self):
        c = _chain()
        _seal(c, [_post(c, DEEPSEEK_SPEC)], version=4)
        s = c.state
        self.assertEqual((s.escrow["think-shorter-test"], s.job_owner["think-shorter-test"]), (50 * V, LEAD.address))
        self.assertEqual(s.balance(LEAD.address), 950 * V)
        # Its rounds settle from their summaries: the fee pro rata to the units accepted (6 of 16), and pay.
        _seal(c, summary=_round("think-shorter-test"), version=4)
        fee = [t.amount for t in c.tip.txs if t.kind == "fee_release"]
        self.assertEqual(fee, [DEEPSEEK_SPEC["fee_per_round"] * 6 // 16])
        self.assertEqual(c.state.escrow["think-shorter-test"], 50 * V - fee[0])
        self.assertTrue(any(t.kind == "reward" and t.recipient == MINER.address for t in c.tip.txs))
        # Its owner can still cancel it and get the rest back.
        cancel = Tx.signed(LEAD, "cancel_job", SYSTEM, 0, c.next_nonce(LEAD.address), "think-shorter-test", c.chain_id)
        _seal(c, [cancel], version=4)
        self.assertEqual(c.state.escrow["think-shorter-test"], 0)
        self.assertTrue(c.validate())

    def test_nothing_new_of_a_cut_kind_from_version_5(self):
        c = _chain()
        with self.assertRaises(InvalidBlock):
            c.check_tx(_post(c, DEEPSEEK_SPEC), [])  # (the mempool checks at the current version)
        with self.assertRaises(InvalidBlock):
            _seal(c, [_post(c, DEEPSEEK_SPEC)], version=5)
        for version in (3, 4, 5):  # an Evo job was never on a chain this code runs
            d = _chain()
            with self.assertRaises(InvalidBlock, msg=version):
                d.check_tx(_post(d, EVO_SPEC), [])
            with self.assertRaises(InvalidBlock, msg=version):
                _seal(d, [_post(d, EVO_SPEC)], version=version)
        _seal(c, [_post(c, DEEPSEEK_SPEC)], version=4)  # (an old post, which still applies)
        block, _ = c.build_block(_round("think-shorter-test", units=0), [], c.tip.timestamp + 1, version=5)
        block.sign(LEAD)
        with self.assertRaisesRegex(InvalidBlock, "a round of job think-shorter-test"):
            c.append(block)
        _seal(c, summary=_round("think-shorter-test", units=0), version=4)  # (in an old block it applied, and does)

    def test_an_old_round_of_a_cut_kind_is_taken_as_certified_on_replay(self):
        c = _chain()
        _seal(c, [_post(c, DEEPSEEK_SPEC)], version=4)
        record = {"job_id": "think-shorter-test", "prep": {"units": {}}, "version": 4}
        block, _ = c.build_block(_round("think-shorter-test", record), [], c.tip.timestamp + 1, version=4)
        block.sign(LEAD)
        stats = {}
        self.assertEqual(replay_block(c, block, record, stats=stats), [])
        self.assertEqual(stats, {"retired": 1})
        self.assertEqual(replay_block(c, block, {**record, "x": 1}), ["the record doesn't match the hash the block commits to"])
        new, _ = c.build_block(_round("think-shorter-test", record), [], c.tip.timestamp + 1)
        self.assertIn("is of a kind that was cut", replay_block(c, new, record)[0])

    def test_the_coordinator_never_runs_one(self):
        c = _chain()
        _seal(c, [_post(c, DEEPSEEK_SPEC)], version=4)
        coord = Coordinator(c, LEAD)
        self.assertIsNone(coord.active_job_id())
        self.assertIn("removed from VibeHash", coord.waiting["think-shorter-test"])
        self.assertIn("owner can cancel it", coord.waiting["think-shorter-test"])
        cancel = Tx.signed(LEAD, "cancel_job", SYSTEM, 0, c.next_nonce(LEAD.address), "think-shorter-test", c.chain_id)
        _seal(c, [cancel], version=4)
        self.assertIsNone(coord.active_job_id())
        self.assertNotIn("think-shorter-test", coord.waiting, "a cancelled job waits no more (/status stops saying so)")

    def test_an_evo_genesis_is_refused_with_a_clear_message(self):
        g = {"chain_id": "an-evo-net", "allocations": {LEAD.address: 1}, "es": True}
        with self.assertRaisesRegex(InvalidBlock, "Evo .* removed from VibeHash after 0.5.28"):
            Chain.from_genesis(g)
        genesis = Block(0, b"\x00" * 32, 0, None, [Tx("genesis", SYSTEM, SYSTEM, 0, memo=ledger.ES_GATE, chain_id="x")], "x")
        with self.assertRaisesRegex(InvalidBlock, "run that chain with 0.5.28"):
            apply_block(ledger.State(), genesis, "x", None)
        Chain.from_genesis({"chain_id": "a-plain-net", "allocations": {LEAD.address: 1}})  # (one without it is fine)
