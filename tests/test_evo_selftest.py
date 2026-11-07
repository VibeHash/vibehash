"""The Evo self-check (vibehash/evo_selftest.py): a device's own answers on a tiny Qwen3.5, checked by itself, are
exactly greedy, or the worker takes no Evo work on it and `vibehash selftest` says why."""
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import evo_selftest, selftest

from tests.test_es import HAVE_TORCH


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class CheckTests(unittest.TestCase):
    def test_the_cpu_checks_its_own_answers_exactly_and_catches_a_twin_swap(self):
        got = evo_selftest.check("cpu")
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["worst"], 0.0)
        self.assertGreater(got["swap"], 0.0)

    def test_a_cuda_gpu_does_too(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("no CUDA GPU")
        got = evo_selftest.check("cuda:0")  # bf16, through CUDA graphs: still exactly its own
        self.assertTrue(got["ok"], got)


class ProblemTests(unittest.TestCase):
    def setUp(self):
        evo_selftest._done.clear()
        self.addCleanup(evo_selftest._done.clear)

    def test_a_device_that_passes_has_no_problem_and_is_checked_once(self):
        with mock.patch.object(evo_selftest, "check", return_value={"ok": True, "worst": 0.0, "swap": 0.2,
                                                                     "seconds": 1.0}) as check:
            self.assertIsNone(evo_selftest.problem("cuda:0"))
            self.assertIsNone(evo_selftest.problem("cuda:0"))
        check.assert_called_once_with("cuda:0")

    def test_an_error_is_tried_again_after_a_while_a_result_never(self):
        """A check that errored (the card briefly full) stands RETRY_S, then is tried again; a failed result stays."""
        clock = [1000.0]
        with mock.patch.object(evo_selftest.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(evo_selftest, "check", side_effect=RuntimeError("out of memory")) as check:
            self.assertIn("out of memory", evo_selftest.problem("cuda:0"))
            clock[0] += evo_selftest.RETRY_S - 1
            evo_selftest.problem("cuda:0")
            self.assertEqual(check.call_count, 1)
            check.side_effect, check.return_value = None, {"ok": False, "worst": 0.5, "swap": 0.2, "seconds": 1.0}
            clock[0] += 2
            self.assertIn("came back 0.5", evo_selftest.problem("cuda:0"))
            clock[0] += 10 * evo_selftest.RETRY_S
            evo_selftest.problem("cuda:0")
            self.assertEqual(check.call_count, 2)

    def test_what_a_failure_says(self):
        with mock.patch.object(evo_selftest, "check", return_value={"ok": False, "worst": 0.125, "swap": 0.2,
                                                                     "seconds": 1.0}):
            self.assertIn("came back 0.125 below the greedy choice instead of exactly 0", evo_selftest.problem("x"))
        with mock.patch.object(evo_selftest, "check", side_effect=RuntimeError("no kernel image is available")):
            self.assertIn("RuntimeError: no kernel image", evo_selftest.problem("y"))
        self.assertIn("wasn't caught", evo_selftest.problem_of("z", {"ok": False, "worst": 0.0, "swap": 0.0}))

    def test_a_worker_takes_no_evo_work_on_a_device_that_fails(self):
        from vibehash.es_client import Es2Work
        work = SimpleNamespace(device=lambda: "cuda:0")
        job = SimpleNamespace(job_id="evo", spec=SimpleNamespace(tasks=(), task="arithmetic"))
        with mock.patch("vibehash.es_local.find_base"), mock.patch("vibehash.es_client.runtime2"), \
                mock.patch("vibehash.es_tasks.get"), \
                mock.patch.object(evo_selftest, "problem", return_value="Evo's self-check on cuda:0: bad") as p:
            self.assertEqual(Es2Work(work).why_not(job), "job evo: Evo's self-check on cuda:0: bad")
            p.return_value = None
            self.assertIsNone(Es2Work(work).why_not(job))


class SelftestRowTests(unittest.TestCase):
    def row(self, **patches):
        rep = selftest.Report(quiet=True, out=io.StringIO())
        gpu = SimpleNamespace(id="cuda:0", problem=None)
        with mock.patch("vibehash.compute.torch_available", return_value=True), \
                mock.patch.multiple(evo_selftest, **patches):
            selftest.evo_check(rep, gpu, [gpu])
        return rep.get("evo")

    def test_ok_and_fail(self):
        ok = self.row(check=mock.Mock(return_value={"ok": True, "worst": 0.0, "swap": 0.22, "seconds": 1.2}))
        self.assertEqual(ok.status, "ok")
        self.assertIn("gap 0", ok.text)
        bad = self.row(check=mock.Mock(return_value={"ok": False, "worst": 0.5, "swap": 0.2, "seconds": 1.2}))
        self.assertEqual(bad.status, "fail")
        self.assertIn("won't take Evo jobs", bad.text)
        broke = self.row(check=mock.Mock(side_effect=RuntimeError("graph capture failed")))
        self.assertEqual(broke.status, "fail")
        self.assertIn("graph capture failed", broke.text)

    def test_skipped_without_a_gpu(self):
        rep = selftest.Report(quiet=True, out=io.StringIO())
        with mock.patch("vibehash.compute.torch_available", return_value=True):
            selftest.evo_check(rep, None, [SimpleNamespace(id="cpu", problem=None)])
        self.assertEqual(rep.get("evo").status, "skip")


if __name__ == "__main__":
    unittest.main()
