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

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")  # es_client imports torch for the work
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

    def test_only_an_error_says_when_to_ask_again(self):
        clock = [1000.0]
        self.assertIsNone(evo_selftest.retry_in("cuda:0"))  # never checked
        with mock.patch.object(evo_selftest.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(evo_selftest, "check", side_effect=RuntimeError("out of memory")):
            evo_selftest.problem("cuda:0")
            clock[0] += 100
            self.assertEqual(evo_selftest.retry_in("cuda:0"), evo_selftest.RETRY_S - 100)
            clock[0] += evo_selftest.RETRY_S
            self.assertEqual(evo_selftest.retry_in("cuda:0"), 0.0)
        for result in ({"ok": True, "worst": 0.0, "swap": 0.2, "seconds": 1.0},  # passed, or failed for good
                       {"ok": False, "worst": 0.5, "swap": 0.2, "seconds": 1.0}):
            evo_selftest._done.clear()
            with mock.patch.object(evo_selftest, "check", return_value=result):
                evo_selftest.problem("cuda:0")
            self.assertIsNone(evo_selftest.retry_in("cuda:0"))


class WorkerRecheckTests(unittest.TestCase):
    """A worker refusing an Evo job remembers it (client.Client.job): over an errored self-check only until
    evo_selftest would try the check again, or the job was refused until the worker restarted."""

    def setUp(self):
        import tempfile

        from vibehash import client as client_module
        from vibehash.crypto import KeyPair

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.client_module = client_module
        self.c = client_module.WorkerClient("http://127.0.0.1:9", KeyPair.generate(b"recheck"), device="cpu",
                                            cache_dir=tmp.name)
        self.c.node = mock.Mock(**{"get.return_value": {"spec": {}, "owner": "someone"}})
        self.job = SimpleNamespace(job_id="evo", es=True, deepseek=False, data_source={"kind": "es"},
                                   spec=SimpleNamespace(repo="tiny", rule=2))
        self.clock = [1000.0]

    def patched(self, why, wait):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(mock.patch.object(self.client_module.time, "monotonic", side_effect=lambda: self.clock[0]))
        stack.enter_context(mock.patch.object(self.client_module.TrainingJob, "from_spec_json", return_value=self.job))
        why_not = stack.enter_context(mock.patch.object(self.c.es, "why_not", side_effect=lambda j: why[0]))
        stack.enter_context(mock.patch.object(self.c.es, "retry_in", side_effect=lambda j: wait))
        stack.enter_context(self.assertLogs("vibehash.worker", "INFO"))
        return stack, why_not

    def test_a_job_refused_over_an_errored_self_check_is_asked_about_again_when_it_is_due(self):
        why = ["job evo: Evo's self-check on cpu failed (RuntimeError: out of memory)"]
        stack, why_not = self.patched(why, evo_selftest.RETRY_S)
        with stack:
            self.c.job("evo")
            self.assertIn("out of memory", self.c.cannot["evo"])
            self.clock[0] += evo_selftest.RETRY_S - 1
            self.c.job("evo")
            self.assertEqual(why_not.call_count, 1)  # not due yet
            self.clock[0] += 2
            why[0] = None  # the check passes this time
            self.assertIs(self.c.job("evo"), self.job)
            self.assertEqual(why_not.call_count, 2)
            self.assertNotIn("evo", self.c.cannot)
            self.assertIn("evo", self.c.speed)  # taken on as any job the worker can run

    def test_a_refusal_that_stands_is_never_asked_about_again(self):
        stack, why_not = self.patched(["job evo: Evo's self-check on cpu: came back 0.5 below"], None)
        with stack:
            self.c.job("evo")
            self.clock[0] += 10 * evo_selftest.RETRY_S
            self.c.job("evo")
            self.assertEqual(why_not.call_count, 1)
            self.assertIn("evo", self.c.cannot)


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
