"""Evo across machine kinds (docs/CROSS-HARDWARE.md, es2.Spec2.cross_margin): a tight margin between machines with the
same number setup and a wider one across, each claim's first checker paired with its worker's setup, and a same-setup
approval overruling a cross-setup rejection on gaps alone up to the job's referee floor (twice its cross_margin)."""
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


def task(verifier, ok, gap=None, submitter="w"):
    t = VerifyTask("t-" + verifier, verifier, None, submitter, [0], {}, kind="es")
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


class PairingTests(unittest.TestCase):
    def tasks(self, cross_margin, regs, worker="w"):
        c = fake_round(regs, holders={"u1": {worker}})
        c.r.reveals = {("u1", worker): None}
        c.r.units = {"u1": SimpleNamespace(unit_id="u1")}
        job = SimpleNamespace(spec=spec(cross_margin=cross_margin))
        pool = lambda exclude: sorted(a for a in regs if a not in exclude)  # noqa: E731
        with mock.patch("vibehash.coordinator.es.sample", return_value=[0, 1]):
            got = c._es_tasks(job, SimpleNamespace(shuffle=lambda xs: list(xs)), pool, lambda u, s, v: f"{u}-{v}")
        return [t.verifier for t in got.values()]

    def test_the_first_checker_shares_the_workers_setup(self):
        regs = {"w": CUDA, "alike": CUDA, **{f"mac{i}": MAC for i in range(12)}}
        got = self.tasks(0.5, regs)
        self.assertEqual(len(got), CHECKERS)
        self.assertIn("alike", got)  # the only other CUDA machine, among 13 candidates: always picked
        self.assertNotIn("w", got)  # never its own unit's holder

    def test_without_a_match_or_a_cross_margin_its_the_draw_as_before(self):
        regs = {"w": CUDA, "alike": CUDA, **{f"mac{i}": MAC for i in range(12)}}
        drawn = [self.tasks(0.0, regs | {"w": CUDA})]
        self.assertEqual(len(drawn[0]), CHECKERS)
        no_match = self.tasks(0.5, {"w": CUDA, **{f"mac{i}": MAC for i in range(12)}})
        self.assertEqual(len(no_match), CHECKERS)
        unsaid = self.tasks(0.5, {"w": None, "x": None, **{f"mac{i}": MAC for i in range(12)}})
        self.assertEqual(len(unsaid), CHECKERS)

    def test_without_a_cross_margin_the_checkers_are_the_old_draw(self):
        """Jobs posted before the field: the same checkers, so their records replay exactly."""
        from vibehash.coordinator import Draw, _pick
        regs = {"w": CUDA, "alike": CUDA, **{f"mac{i}": MAC for i in range(12)}}
        c = fake_round(regs)
        mine = Draw(b"check", c.r.beacon, c.r.secret, b"u1", b"w")
        self.assertEqual(sorted(self.tasks(0.0, regs)), sorted(_pick(mine, sorted(a for a in regs if a != "w"), CHECKERS)))


if __name__ == "__main__":
    unittest.main()
