"""Evo across machine kinds (docs/CROSS-HARDWARE.md, es2.Spec2.cross_margin): a tight margin between machines with the
same number setup and a wider one across, a referee of its worker's setup added to a claim whose drawn checkers don't
share it, and a same-setup approval overruling a cross-setup rejection on gaps alone up to the job's referee floor
(twice its cross_margin)."""
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import es2
from vibehash.coordinator import CHECKERS, Coordinator, Registration, VerifyTask

CUDA = {"backend": "cuda", "dtype": "bf16", "arch": "sm_89", "units": 24, "torch": "2.15.0", "toolkit": "cuda13.2"}
MAC = {"backend": "mps", "dtype": "float32", "arch": "mps", "units": 0, "torch": "2.15.0", "toolkit": ""}
SPEC = dict(base="hf:Qwen/Qwen3.5-0.8B@" + "0" * 40, manifest="0" * 64, dims={}, tensors=("x",), n=1,
            tensors_sha="0" * 64, check=2, margin=0.1)


def spec(**kw) -> es2.Spec2:
    return es2.Spec2(**(SPEC | kw))


class SpecTests(unittest.TestCase):
    def test_off_by_default_and_left_out_of_the_spec(self):
        s = spec()
        self.assertEqual(s.cross_margin, 0.0)
        self.assertNotIn("cross_margin", s.to_json()["es"])  # a job's hash is what it was before the field
        self.assertEqual((s.margin_for(True), s.margin_for(False)), (0.1, 0.1))

    def test_round_trip_and_the_margin_by_pair(self):
        s = spec(cross_margin=0.5)
        self.assertEqual(s.to_json()["es"]["cross_margin"], 0.5)
        back = es2.Spec2.from_json(s.to_json())
        self.assertEqual(back.cross_margin, 0.5)
        self.assertEqual((back.margin_for(True), back.margin_for(False)), (0.1, 0.5))
        self.assertIsInstance(spec(cross_margin=1).cross_margin, float)

    def test_bounds(self):
        ok = lambda **kw: not any("cross_margin" in p for p in spec(**kw).problems_list())  # noqa: E731
        self.assertTrue(ok(cross_margin=0.0))
        self.assertTrue(ok(cross_margin=0.1))  # the same as margin
        self.assertTrue(ok(cross_margin=1.5))
        self.assertFalse(ok(cross_margin=0.05))  # under margin: tighter across machines than between alike ones
        self.assertTrue(ok(cross_margin=es2.MAX_CROSS_MARGIN))
        self.assertFalse(ok(cross_margin=1.75))  # its floor, 3.5, is where a cheat's gaps start on Qwen3.5-0.8B
        self.assertFalse(ok(cross_margin=-1.0))


def fake_round(regs: dict, holders=None):
    """A Coordinator with just a round's registrations, enough for _setup, _same_setup and _overruled."""
    c = Coordinator.__new__(Coordinator)
    c.r = SimpleNamespace(regs={a: Registration(a, 1e12, 8192, None, n) for a, n in regs.items()},
                          holders=holders or {}, reveals={}, units={}, beacon=b"b" * 32, secret=b"s" * 32)
    return c


def task(verifier, ok, gap=None, submitter="w", paired=False):
    t = VerifyTask("t-" + verifier, verifier, None, submitter, [0], {}, kind="es", paired=paired)
    t.verdict, t.gap = (ok, None if ok else 0), gap
    return t


class RefereeTests(unittest.TestCase):
    def setUp(self):
        self.c = fake_round({"w": CUDA, "alike": CUDA, "mac": MAC, "mac2": MAC, "unsaid": None})

    def test_a_same_setup_approval_overrules_a_small_cross_setup_gap(self):
        self.assertEqual(spec(cross_margin=0.5).referee_floor(), 1.0)
        self.assertTrue(self.c._overruled("w", [task("alike", True), task("mac", False, gap=0.75)], 1.0))
        self.assertTrue(self.c._overruled("w", [task("alike", True), task("mac", False, gap=1.0)], 1.0))

    def test_what_always_stands(self):
        c = self.c
        # over the floor (twice cross_margin 0.5): a cheat's gaps
        self.assertFalse(c._overruled("w", [task("alike", True), task("mac", False, gap=1.01)], 1.0))
        # not on gaps alone (no gap sent: tokens that don't decode, an answer that ends wrong, a log-probability off)
        self.assertFalse(c._overruled("w", [task("alike", True), task("mac", False)], 1.0))
        # a checker of the worker's own setup rejecting: that's the strict check
        self.assertFalse(c._overruled("w", [task("alike", False, gap=0.2), task("mac", True)], 1.0))
        # no same-setup approval to stand on
        self.assertFalse(c._overruled("w", [task("mac2", True), task("mac", False, gap=0.75)], 1.0))
        self.assertFalse(c._overruled("w", [task("mac", False, gap=0.75)], 1.0))
        # nothing rejected: nothing to overrule
        self.assertFalse(c._overruled("w", [task("alike", True), task("mac", True)], 1.0))

    def test_no_setup_is_never_the_same_setup(self):
        c = fake_round({"w": None, "unsaid": None, "mac": MAC})
        self.assertFalse(c._same_setup("w", "unsaid"))
        self.assertFalse(c._overruled("w", [task("unsaid", True), task("mac", False, gap=0.75)], 1.0))


class JudgeTests(unittest.TestCase):
    """Coordinator._judge_es2: a rule-2 claim's result from the verdicts that came in."""
    def setUp(self):
        self.c = fake_round({"w": CUDA, "friend": CUDA, "alike": CUDA, "mac": MAC})
        self.c.r.units = {"u1": SimpleNamespace(unit_id="u1")}

    def judge(self, got, cross_margin=0.5):
        return self.c._judge_es2(("u1", "w"), got, spec(cross_margin=cross_margin))

    def test_an_overruled_rejection_is_accepted_and_says_why(self):
        from vibehash.tasks import Verdict
        res = self.judge([task("alike", True), task("mac", False, gap=0.75)])
        self.assertEqual((res.verdict, res.verifier), (Verdict.ACCEPTED, "alike"))
        self.assertIn("rejected it on gaps alone", res.detail)
        self.assertEqual(self.judge([task("alike", True), task("mac", False, gap=1.5)]).verdict, Verdict.UNCHECKED)
        # without a cross_margin, any rejection leaves it unchecked, as before
        self.assertEqual(self.judge([task("alike", True), task("mac", False, gap=0.75)], 0.0).verdict, Verdict.UNCHECKED)

    def test_the_paired_checker_cant_carry_a_claim_alone(self):
        """A worker naming a setup only it and a friend share gets the friend as referee: if the checkers drawn from
        everyone don't answer, the friend's approval alone pays nothing."""
        from vibehash.tasks import Verdict
        self.assertEqual(self.judge([task("friend", True, paired=True)]).verdict, Verdict.UNCHECKED)
        self.assertEqual(self.judge([task("friend", True, paired=True), task("mac", True)]).verdict, Verdict.ACCEPTED)
        self.assertEqual(self.judge([task("friend", True, paired=True)], 0.0).verdict, Verdict.ACCEPTED)  # (no field)
        self.assertEqual(self.judge([]).verdict, Verdict.UNCHECKED)

    def test_a_referees_approval_doesnt_outweigh_a_cheats_gap(self):
        """The friend approves, one drawn checker is a friend too and the other honest: the honest one's rejection
        with a cheat's gap stands, and so does one for anything but gaps."""
        from vibehash.tasks import Verdict
        for honest in (task("mac", False, gap=3.5), task("mac", False)):
            got = [task("friend", True, paired=True), task("alike", True), honest]
            self.assertEqual(self.judge(got).verdict, Verdict.UNCHECKED)
        # an honest machine of another kind a little over the cross margin: the referee overrules it
        got = [task("friend", True, paired=True), task("mac", False, gap=0.75)]
        self.assertEqual(self.judge(got).verdict, Verdict.ACCEPTED)


class PairingTests(unittest.TestCase):
    def tasks(self, cross_margin, regs, worker="w", uid="u1"):
        c = fake_round(regs, holders={uid: {worker}})
        c.r.reveals = {(uid, worker): None}
        c.r.units = {uid: SimpleNamespace(unit_id=uid)}
        job = SimpleNamespace(spec=spec(cross_margin=cross_margin))
        pool = lambda exclude: sorted(a for a in regs if a not in exclude)  # noqa: E731
        with mock.patch("vibehash.coordinator.es.sample", return_value=[0, 1]):
            got = c._es_tasks(job, SimpleNamespace(shuffle=lambda xs: list(xs)), pool, lambda u, s, v: f"{u}-{v}")
        self.paired = [t.verifier for t in got.values() if t.paired]
        return [t.verifier for t in got.values()]

    @staticmethod
    def old_draw(regs, worker="w", uid="u1"):
        from vibehash.coordinator import Draw, _pick
        c = fake_round(regs)
        mine = Draw(b"check", c.r.beacon, c.r.secret, uid.encode(), worker.encode())
        return sorted(_pick(mine, sorted(a for a in regs if a != worker), CHECKERS))

    def test_a_referee_of_the_workers_setup_is_added(self):
        """The CHECKERS drawn from everyone, as without pairing, and the only other CUDA machine as a third, unless
        it was drawn anyway."""
        regs = {"w": CUDA, "alike": CUDA, **{f"mac{i}": MAC for i in range(12)}}
        cases = set()
        for n in range(40):
            got = self.tasks(0.5, regs, uid=f"u{n}")
            drawn = sorted(v for v in got if v not in self.paired)
            self.assertEqual(drawn, self.old_draw(regs, uid=f"u{n}"))
            self.assertIn("alike", got)
            self.assertNotIn("w", got)  # never its own unit's holder
            if "alike" in drawn:
                self.assertEqual((len(got), self.paired), (CHECKERS, []))  # drawn: it referees as it checks
            else:
                self.assertEqual((len(got), self.paired), (CHECKERS + 1, ["alike"]))  # can't carry it (_judge_es2)
            cases.add(bool(self.paired))
        self.assertEqual(cases, {True, False})

    def test_a_ring_gets_no_more_of_the_deciding_checkers(self):
        """Five friends naming a made-up setup only they registered, among sixteen others: one always referees a
        friend's claim, but the two checkers that decide it are the draw from everyone, so both are friends as
        rarely as without pairing."""
        ring = dict(CUDA, units=57)
        regs = {**{f"h{i:02d}": (CUDA if i % 2 else MAC) for i in range(16)}, **{f"r{i}": ring for i in range(5)}}
        for n in range(40):
            got = self.tasks(0.5, regs, worker="r0", uid=f"u{n}")
            drawn = sorted(v for v in got if v not in self.paired)
            self.assertEqual(drawn, self.old_draw(regs, worker="r0", uid=f"u{n}"))
            if not any(v.startswith("r") for v in drawn):
                self.assertEqual(len(self.paired), 1)
                self.assertTrue(self.paired[0].startswith("r"))

    def test_without_a_match_its_the_draw_as_before(self):
        """A paired job whose worker's setup nobody else registered (or that has none): the old draw exactly."""
        from vibehash.coordinator import Draw, _pick
        for regs in ({"w": CUDA, **{f"mac{i}": MAC for i in range(12)}},
                     {"w": None, "x": None, **{f"mac{i}": MAC for i in range(12)}}):
            c = fake_round(regs)
            mine = Draw(b"check", c.r.beacon, c.r.secret, b"u1", b"w")
            self.assertEqual(sorted(self.tasks(0.5, regs)), sorted(_pick(mine, sorted(a for a in regs if a != "w"),
                                                                         CHECKERS)))
            self.assertEqual(self.paired, [])

    def test_without_a_cross_margin_the_checkers_are_the_old_draw(self):
        """Jobs posted before the field: the same checkers, so their records replay exactly."""
        from vibehash.coordinator import Draw, _pick
        regs = {"w": CUDA, "alike": CUDA, **{f"mac{i}": MAC for i in range(12)}}
        c = fake_round(regs)
        mine = Draw(b"check", c.r.beacon, c.r.secret, b"u1", b"w")
        self.assertEqual(sorted(self.tasks(0.0, regs)), sorted(_pick(mine, sorted(a for a in regs if a != "w"), CHECKERS)))


class RegistrationTests(unittest.TestCase):
    def test_a_paired_job_takes_only_workers_that_give_their_setup(self):
        """Held to the wider cross margin by everyone, a worker that left its setup out would get looser checks."""
        from vibehash.coordinator import ProtocolError
        from tests.test_record_v2 import network
        proto, _ = network(3)
        c = proto.coord
        c.open_round()
        r = c.r
        w1, w2, _ = proto.workers.values()
        job = c.job(r.job_id)

        def register(w, extra):
            body = {"round": r.round_no, "flops_per_sec": w.hardware.flops_per_sec, "memory_mb": w.hardware.memory_mb,
                    **extra}
            return c.register(w.address, w.hardware.flops_per_sec, w.hardware.memory_mb,
                              envelope=proto._signed(w, "register", body))
        with mock.patch.object(c, "job", return_value=SimpleNamespace(spec=SimpleNamespace(cross_margin=0.5),
                                                                      memory_mb=job.memory_mb)):
            with self.assertRaisesRegex(ProtocolError, "number setup"):
                register(w1, {})
            register(w2, {"numerics": CUDA})
        self.assertNotIn(w1.address, r.regs)
        self.assertEqual(r.regs[w2.address].numerics, CUDA)
        register(w1, {})  # an ordinary job: as before
        self.assertIn(w1.address, r.regs)


if __name__ == "__main__":
    unittest.main()
