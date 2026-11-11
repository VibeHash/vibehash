"""Faster Evo rounds (2026-11-07): a job's per-task answer caps (es2.Spec2.task_tokens), and engine changes that keep
every number: noise streams made on threads (es.noise_many), teacher forcing longest row first
(es_hybrid.longest_first), and answering the prompts expected longest first (es_hybrid.answer_order). The spec field
is left out when off, so earlier jobs' specs and hashes are the same; the engine's answers, gaps and log-probabilities
are the same in any order, on test_es_hybrid's tiny random model (the GPU check is scratch work, in the PR)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import es, es2, es2_rounds

from tests.test_es import HAVE_TORCH
from tests.test_loop_stop import Spec, spec_json


def mixed(**es_):
    """A mixed job's spec JSON with three tasks."""
    return spec_json(task="arithmetic", tasks=["arithmetic", "countdown", "trivia_qa"], mix=True, problems=6, **es_)


class TaskTokensSpecTests(unittest.TestCase):
    def test_off_is_left_out_of_the_spec(self):
        s = es2.Spec2.from_json(mixed())
        self.assertEqual(s.task_tokens, ())
        self.assertNotIn("task_tokens", s.to_json()["es"])
        self.assertEqual(s.tokens_for("countdown"), s.max_tokens)
        self.assertEqual(es2_rounds.caps(s, 3), [s.max_tokens] * s.problems)

    def test_on_round_trips_and_is_checked(self):
        s = es2.Spec2.from_json(mixed(task_tokens={"trivia_qa": 16, "countdown": 48}))
        self.assertEqual(s.task_tokens, (("countdown", 48), ("trivia_qa", 16)), "kept by task name")
        self.assertEqual(s.to_json()["es"]["task_tokens"], {"countdown": 48, "trivia_qa": 16})
        self.assertEqual(es2.Spec2.from_json(s.to_json()), s)
        self.assertFalse([p for p in s.problems_list() if "task_tokens" in p])
        self.assertEqual((s.tokens_for("countdown"), s.tokens_for("trivia_qa"), s.tokens_for("arithmetic")), (48, 16, 64))
        for bad in ({"math": 48}, {"countdown": 0}, {"countdown": 65}, {"countdown": 48.0}, {"countdown": "48"}):
            got = es2.Spec2.from_json(mixed(task_tokens=bad)).problems_list()
            self.assertTrue([p for p in got if "task_tokens" in p], bad)
        with self.assertRaises(TypeError):
            es2.Spec2.from_json(mixed(task_tokens=[["countdown", 48]]))

    def test_each_problem_gets_its_tasks_cap(self):
        s = es2.Spec2.from_json(mixed(task_tokens={"countdown": 48, "trivia_qa": 16}))
        for step in range(4):
            names = es2_rounds.task_names(s, step)
            self.assertEqual(names, [s.tasks[(step * s.problems + i) % 3] for i in range(s.problems)])
            self.assertEqual(es2_rounds.caps(s, step), [{"countdown": 48, "trivia_qa": 16}.get(t, 64) for t in names])
        one = es2.Spec2.from_json(spec_json(task_tokens={"arithmetic": 20}))  # one task: every problem has its cap
        self.assertEqual(es2_rounds.caps(one), [20] * one.problems)

    def test_an_answer_ends_at_its_cap(self):
        eos = [99]
        off, on = Spec(10), Spec(10, (2, 3))
        self.assertTrue(es2.ends_right(list(range(10)), off, eos))
        self.assertTrue(es2.ends_right(list(range(4)), off, eos, 4), "at its cap")
        self.assertFalse(es2.ends_right(list(range(5)), off, eos, 4), "past its cap")
        self.assertFalse(es2.ends_right(list(range(3)), off, eos, 4), "neither at the cap nor an end-of-turn token")
        self.assertTrue(es2.ends_right([1, 99], off, eos, 4))
        self.assertTrue(es2.ends_right([5, 5, 5, 5], on, eos, 4), "loops at its last token, also its cap")
        self.assertFalse(es2.ends_right([5, 5, 5, 5, 5], on, eos, 6), "ran on past where it loops")
        self.assertEqual(es2.ends_right(list(range(10)), off, eos, None), es2.ends_right(list(range(10)), off, eos))

    def test_the_node_refuses_an_answer_past_its_tasks_cap(self):
        s = es2.Spec2.from_json(mixed(task_tokens={"countdown": 4}, max_tokens=8, pairs=4, pairs_per_unit=2))
        names = es2_rounds.task_names(s, 0)
        q = names.index("countdown")

        def blob(n):
            answers = [[{"ids": [5] * 3, "text": "<answer>1</answer>"} for _ in range(s.problems)]
                       for _ in range(s.members)]
            answers[1][q] = {"ids": [5] * n, "text": "<answer>1</answer>"}
            return es.claim_bytes("u1", answers)

        with mock.patch("vibehash.es2_rounds.items", lambda spec, seed, step=0: [_Right()] * spec.problems):
            self.assertIsNotNone(es2_rounds.rescore(s, 1, blob(4), "u1"), "at the cap")
            self.assertIsNone(es2_rounds.rescore(s, 1, blob(5), "u1"), "one past countdown's cap")
            other = next(i for i, t in enumerate(names) if t != "countdown")
            raw = json.loads(blob(3))
            raw["answers"][1][other] = {"ids": [5] * 8, "text": "x"}  # another task's answer may run to max_tokens
            self.assertIsNotNone(es2_rounds.rescore(s, 1, json.dumps(raw).encode(), "u1"))


class _Right:
    def check(self, text):
        return True


class NoiseThreadsTests(unittest.TestCase):
    def test_noise_many_is_each_stream_alone(self):
        seeds = [es.pair_seed(77, j) for j in range(6)]
        alone = np.stack([es.noise(s, 10_001) for s in seeds])
        with mock.patch.object(es, "NOISE_THREADS", 4):
            many = es.noise_many(seeds, 10_001)
        self.assertTrue(np.array_equal(alone.view(np.uint32), many.view(np.uint32)))
        self.assertEqual(es.noise_many([], 5).shape, (0, 5))
        with mock.patch.object(es, "NOISE_THREADS", 1):
            self.assertTrue(np.array_equal(es.noise_many(seeds[:2], 9), alone[:2, :9]))


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")  # es_hybrid imports torch
class OrderTests(unittest.TestCase):
    def test_longest_first(self):
        from vibehash.es_hybrid import longest_first
        self.assertEqual(longest_first([3, 0, 9, 3, 1]), [2, 0, 3, 4], "empty rows out, ties in row order")

    def test_answer_order(self):
        from vibehash.es_hybrid import answer_order
        self.assertEqual(answer_order(3, 2), [0, 1, 2, 3, 4, 5])
        self.assertEqual(answer_order(3, 2, [5, 40, 9]), [1, 4, 2, 5, 0, 3], "prompt 1, then 2, then 0")

    def test_a_worker_learns_its_tasks_lengths(self):
        from vibehash.es_client import expected_lengths, learn_lengths
        seen = {}
        tasks, caps = ["math", "boolq", "math2"], [448, 512, 512]
        self.assertIsNone(expected_lengths(seen, tasks, caps), "nothing seen yet: member by member")
        learn_lengths(seen, tasks[:2], [[[1] * 400, [1] * 2], [[1] * 200, [1] * 2]])
        self.assertEqual(seen, {"math": 300, "boolq": 2})
        self.assertEqual(expected_lengths(seen, tasks, caps), [300, 2, 512.0], "an unseen task: its cap")
        self.assertIsNone(expected_lengths(seen, ["math"] * 3, caps), "one task: nothing to order by")
        learn_lengths(seen, ["math"], [[[1] * 100]])
        self.assertEqual(seen["math"], 0.75 * 300 + 0.25 * 100)


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class EngineTests(unittest.TestCase):
    """Qwen3.5's engine (eager on the CPU: the steps a CUDA graph records) on test_es_hybrid's tiny random model."""

    @classmethod
    def setUpClass(cls):
        from vibehash import es_local
        from vibehash.es_client import prompt_rows, round_factors, runtime2

        from tests.test_es_hybrid import tiny_hybrid
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        found = es_local.find(str(tiny_hybrid(Path(cls.tmp.name) / "tiny35", seed=3)))
        es_local.register(found)
        cls.spec = es_local.spec2_for(found, "arithmetic", pairs=4, pairs_per_unit=2, problems=5, max_tokens=40,
                                      sigma=0.05, check=2, margin=1e-3, lp_tol=0.02)
        cls.rt = runtime2(cls.spec, "cpu")
        cls.rows = prompt_rows(cls.rt, es2_rounds.items(cls.spec, 777))
        cls.fac = round_factors(cls.rt, cls.spec, 777)
        cls.members = [(j, s) for j in range(cls.spec.pairs) for s in (1, -1)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def answer(self, **kw):
        from vibehash import es_fast
        return es_fast.answer_members(self.rt, self.rt.linears, self.fac, self.members, self.rows,
                                      self.spec.max_tokens, self.spec.sigma, **kw)

    def test_capped_answers_are_the_answers_cut_at_their_caps(self):
        whole = self.answer()
        caps = [40, 7, 12, 3, 40]
        cut = self.answer(caps=caps)
        n = 0
        for k, member in enumerate(cut):
            for q, a in enumerate(member):
                self.assertEqual(a, whole[k][q][:caps[q]], (k, q))
                self.assertTrue(es2.ends_right(a, self.spec, self.rt.eos, caps[q]), (k, q, a))
                n += len(whole[k][q]) > caps[q]
        self.assertGreater(n, 0, "no answer reached a cap: the test shows nothing")
        self.assertEqual(self.answer(), whole, "a capped call leaves nothing behind for the next")

    def test_the_order_answers_are_taken_in_changes_no_answer(self):
        whole = self.answer()
        self.assertEqual(self.answer(expect=[1, 50, 3, 40, 2]), whole)
        self.assertEqual(self.answer(caps=[40, 7, 12, 3, 40], expect=[40, 7, 12, 3, 40]),
                         self.answer(caps=[40, 7, 12, 3, 40]))

    def test_teacher_forcing_longest_first_changes_no_number(self):
        import torch

        from vibehash import es_fast, es_hybrid
        whole = self.answer()
        items = [(r, a) for r, a in zip(self.rows, whole[0])]
        new = (es_fast.gaps_members(self.rt, self.rt.linears, self.fac, self.members, self.rows, whole,
                                    self.spec.sigma, max_tokens=self.spec.max_tokens),
               es_fast.score_items(self.rt, self.rt.linears, self.fac, self.members, items, self.spec.sigma))
        with mock.patch.object(es_hybrid, "longest_first", lambda lens: [r for r, n in enumerate(lens) if n > 0]):
            old = (es_fast.gaps_members(self.rt, self.rt.linears, self.fac, self.members, self.rows, whole,
                                        self.spec.sigma, max_tokens=self.spec.max_tokens),
                   es_fast.score_items(self.rt, self.rt.linears, self.fac, self.members, items, self.spec.sigma))
        self.assertTrue(torch.equal(new[0], old[0]) and torch.equal(new[1], old[1]))
        self.assertEqual(float(new[0].max()), 0.0)


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class TemplateTests(unittest.TestCase):
    def test_the_vibe_template_caps_the_long_tasks(self):
        from vibehash import cli, es_tasks

        from tests.test_es import tiny_model
        with tempfile.TemporaryDirectory() as d:
            folder = tiny_model(Path(d) / "tiny", seed=3)
            for name, extra in (("vibe", []), ("mine", ["--task-tokens", "math=300"]), ("off", ["--task-tokens", "off"]),
                                ("curated", []), ("short", ["--max-tokens", "300"])):
                with mock.patch("sys.stdout"):
                    cli.main(["job", "template", "--evo", str(folder), "--id", name, "--out", str(Path(d) / name),
                              "--tasks", "curated" if name == "curated" else "vibe", *extra])
            got = {n: json.loads((Path(d) / n).read_text())["es"] for n in ("vibe", "mine", "off", "curated", "short")}
            with mock.patch("sys.stdout"), self.assertRaisesRegex(SystemExit, "--task-tokens is TASK=N"):
                cli.main(["job", "template", "--evo", str(folder), "--out", str(Path(d) / "x"), "--tasks", "vibe",
                          "--task-tokens", "math:300"])
        self.assertEqual(got["vibe"]["task_tokens"], es_tasks.VIBE_TASK_TOKENS)
        self.assertEqual(got["mine"]["task_tokens"], {"math": 300})
        self.assertNotIn("task_tokens", got["off"])
        self.assertNotIn("task_tokens", got["curated"], "only the vibe mix has the long tasks")
        self.assertEqual(got["short"]["task_tokens"], {"ifeval": 256}, "a default cap past max_tokens is left out")


if __name__ == "__main__":
    unittest.main()
