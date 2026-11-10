"""Evo pool tasks (es_tasks.PoolTask): problems with whole target replies, a `gold` job scoring its members on them
(es2_rounds.gold_text / gold_caps, Spec2.target_tokens), the template that makes such a job, and a local network
running one end to end."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from vibehash import es, es2, es2_rounds, es_tasks
from vibehash.crypto import KeyPair

from tests.test_es import HAVE_TORCH, fake_reward, tiny_model
from tests.test_es2 import VIBE, job_of, tiny_spec2

ROWS = [
    {"prompt": "What is 2 + 3? Please reason step by step, and put your final answer within \\boxed{}.",
     "target": "2 + 3 = 5. \\boxed{5}", "gold": "5", "kind": "number", "src": "gsm8k", "part": "train"},
    {"prompt": "Pick one.\n\nOptions:\nA. red\nB. blue", "target": "Blue is right. The answer is B", "gold": "B",
     "kind": "mc", "src": "mmlu", "part": "train"},
    {"prompt": "Simplify 2/4. Please reason step by step, and put your final answer within \\boxed{}.",
     "target": "2/4 = 1/2. \\boxed{\\frac{1}{2}}", "gold": "\\frac{1}{2}", "kind": "math", "src": "math",
     "part": "train"},
    {"prompt": "What is 1 + 1?", "target": "", "gold": "2", "kind": "number", "src": "gsm8k", "part": "eval"},
]


def write_pool(folder: Path, rows=ROWS) -> str:
    raw = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode(), mtime=0)
    (folder / "pool.jsonl.gz").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def use_pool(test: unittest.TestCase, folder: Path) -> es_tasks.PoolTask:
    """vibe_reason backed by a small file in `folder`, for the length of the test."""
    task = es_tasks.PoolTask("vibe_reason", "pool.jsonl.gz", write_pool(folder), folder=folder)
    p = mock.patch.dict(es_tasks._TASKS, {"vibe_reason": task})
    p.start()
    test.addCleanup(p.stop)
    return task


class PoolTaskTests(unittest.TestCase):
    def test_its_file_is_pinned_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            sha = write_pool(Path(d))
            task = es_tasks.PoolTask("t", "pool.jsonl.gz", sha, folder=d)
            self.assertEqual(len(task.items(1, 5)), 5)
            self.assertEqual([i.target for i in task.items(7, 4)], [i.target for i in task.items(7, 4)])
            self.assertTrue(all(i.target for i in task.items(3, 20)), "training draws only problems with targets")
            self.assertEqual([i.gold for i in task.eval_items()], ["2"])
            with self.assertRaises(es_tasks.TaskUnavailable):
                es_tasks.PoolTask("t", "pool.jsonl.gz", "0" * 64, folder=d).items(1, 1)
            with self.assertRaises(es_tasks.TaskUnavailable):
                es_tasks.PoolTask("t", "missing.jsonl.gz", sha, folder=d).items(1, 1)

    def test_its_checks_read_the_benchmarks_way_and_lenient_keeps_the_target(self):
        with tempfile.TemporaryDirectory() as d:
            task = es_tasks.PoolTask("t", "pool.jsonl.gz", write_pool(Path(d)), folder=d)
            num, mc, math = (task._data()["train"][i] for i in range(3))
        self.assertTrue(num.check("so \\boxed{5}"))
        self.assertTrue(num.check("it comes to 5"))
        self.assertFalse(num.check("\\boxed{6}"))
        self.assertTrue(mc.check("Thinking... The answer is B"))
        self.assertFalse(mc.check("The answer is A"))
        self.assertTrue(math.check("\\boxed{\\frac12}"))
        self.assertFalse(math.check("\\boxed{2}"))
        for it in (num, mc, math):
            self.assertEqual(es_tasks.lenient(it).target, it.target)
            self.assertEqual(es2_rounds.gold_text(it), it.target)

    def test_other_tasks_are_unchanged(self):
        it = es_tasks.Item([{"role": "user", "content": "q"}], lambda t: True, "B")
        self.assertEqual(it.target, "")
        self.assertEqual(es2_rounds.gold_text(it), "<answer>B</answer>")
        self.assertNotIn("vibe_reason", es_tasks.CURATED + es_tasks.VIBE)


class PoolSpecTests(unittest.TestCase):
    def test_target_tokens_round_trips_and_is_checked(self):
        from tests.test_loop_stop import spec_json
        plain = es2.Spec2.from_json(spec_json())
        self.assertNotIn("target_tokens", plain.to_json()["es"])
        self.assertEqual(plain.target_tokens, 0)
        d = spec_json()
        d["es"] |= {"task": "vibe_reason", "tasks": ["vibe_reason"], "gold": True, "target_tokens": 512}
        spec = es2.Spec2.from_json(d)
        self.assertEqual(spec.problems_list(), [])
        self.assertEqual(es2.Spec2.from_json(spec.to_json()), spec)
        with mock.patch.object(es2.Spec2, "base_params", new_callable=mock.PropertyMock, return_value=1000):
            plain = es2.Spec2.from_json(spec_json())
            self.assertEqual(spec.flops_per_step() - plain.flops_per_step(),  # the targets' scoring, priced in
                             2 * 1000 * 512 * spec.problems * spec.members)
        for bad, said in (({"gold": False}, "needs gold"), ({"target_tokens": 0}, "needs gold and target_tokens"),
                          ({"target_tokens": 8}, "0 or 16-2048"), ({"target_tokens": 4096}, "0 or 16-2048")):
            got = es2.Spec2.from_json({**d, "es": {**d["es"], **bad}}).problems_list()
            self.assertTrue(any(said in p for p in got), (bad, got))
        stray = spec_json()
        stray["es"]["target_tokens"] = 512
        self.assertTrue(any("for pool tasks" in p for p in es2.Spec2.from_json(stray).problems_list()))

    def test_gold_caps_give_pool_problems_their_room(self):
        from tests.test_loop_stop import spec_json
        d = spec_json()
        d["es"] |= {"task": "vibe_reason", "tasks": ["vibe_reason", "countdown"], "mix": True, "gold": True,
                    "target_tokens": 512, "problems": 4}
        spec = es2.Spec2.from_json(d)
        names = es2_rounds.task_names(spec, 0)
        want = [512 if t == "vibe_reason" else spec.max_tokens for t in names]
        self.assertEqual(es2_rounds.gold_caps(spec, 0), want)


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class PoolTemplateTests(unittest.TestCase):
    def test_naming_a_pool_task_sets_up_its_job(self):
        from vibehash import cli
        with tempfile.TemporaryDirectory() as d:
            folder = tiny_model(Path(d) / "tiny", seed=3)
            with mock.patch("sys.stdout"):
                cli.main(["job", "template", "--evo", str(folder), "--tasks", "vibe_reason", "--out", str(Path(d) / "p")])
                cli.main(["job", "template", "--evo", str(folder), "--tasks", "vibe_reason", "--target-tokens", "512",
                          "--out", str(Path(d) / "q")])
                cli.main(["job", "template", "--evo", str(folder), "--tasks", "all", "--out", str(Path(d) / "all")])
            with mock.patch("sys.stdout"), self.assertRaises(SystemExit):
                cli.main(["job", "template", "--evo", str(folder), "--target-tokens", "512", "--out", str(Path(d) / "x")])
            p, q, every = (json.loads((Path(d) / n).read_text())["es"] for n in ("p", "q", "all"))
        self.assertEqual((p["gold"], p["target_tokens"], p["max_tokens"], p["replay"]),
                         (True, es_tasks.POOL_TARGET_TOKENS, es_tasks.POOL_ANSWER_TOKENS, 0))
        self.assertEqual(q["target_tokens"], 512)
        self.assertNotIn("vibe_reason", every["tasks"])

    def test_a_worker_and_a_checker_score_the_whole_target(self):
        from vibehash import es_client
        with tempfile.TemporaryDirectory() as d:
            use_pool(self, Path(d))
            folder = tiny_model(Path(d) / "tiny", seed=1)
            spec = tiny_spec2(folder, "vibe_reason", tasks=("vibe_reason",), gold=True, target_tokens=24, replay=0)
            rt = es_client.runtime2(spec, "cpu")
            ps = es2_rounds.items(spec, 5, 0)
            rows = es_client.prompt_rows(rt, ps)
            got = es_client.gold_items(rt, spec, ps, rows, 0)
            self.assertEqual(len(got), spec.problems)
            for (row, ids), p in zip(got, ps):
                whole = rt.tokenizer.encode(p.target, add_special_tokens=False)
                self.assertEqual(ids, whole[:23] + [rt.eos[0]])
                self.assertGreater(len(ids), spec.max_tokens, "the target is scored past what members write")


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class PoolNetworkTests(unittest.TestCase):
    def test_a_pool_job_runs_turns_down_a_liar_and_replays(self):
        from vibehash import es_client
        from vibehash.client import NodeClient, WorkerClient
        from vibehash.coordinator import CoordinatorConfig
        from vibehash.ledger import Block, Chain
        from vibehash.node import Node, PhaseTimes
        from vibehash.replay import replay_block
        from tests.test_network import run_until

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        use_pool(self, root)
        spec = tiny_spec2(tiny_model(root / "tiny", seed=1), "vibe_reason", tasks=("vibe_reason",), gold=True,
                          target_tokens=24, replay=0)
        for p in (mock.patch.object(es, "reward", fake_reward),
                  mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(root / "home")}),
                  mock.patch.object(es_client, "SAVE_EVERY", 1), mock.patch("vibehash.es_local.V2_COMPUTE_S", 30)):
            p.start()
            self.addCleanup(p.stop)
        node_keys, req = KeyPair.generate(b"pool node"), KeyPair.generate(b"pool requester")
        workers = [KeyPair.generate(f"pool worker {i}".encode()) for i in range(3)]
        genesis = {"chain_id": "vibehash-pool-test", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}, "es": True}
        node = Node(root / "node", genesis, node_keys, port=0,
                    times=PhaseTimes(register_s=1.0, compute_s=20.0, reveal_s=3.0, verify_s=15.0, idle_s=0.3),
                    config=CoordinatorConfig(newcomer_units=4)).start()
        self.addCleanup(node.stop)
        api = NodeClient(node.url)
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e12, poll_s=0.1, device="cpu",
                                cache_dir=root / f"cache{i}") for i, k in enumerate(workers)]
        clients[2].es.cheat = "lp-liar"
        job = job_of(spec, "evo2-pool", req.address, 6 * VIBE)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        stop = threading.Event()
        self.addCleanup(stop.set)
        with self.assertLogs("vibehash.worker", level="WARNING"):
            run_until(node, clients, stop, 5, seconds=300)
        self.assertGreaterEqual(node.chain.tip.height, 5, "the network didn't make its rounds in time")

        blocks = [Block.from_dict(d) for d in api.get(f"/blocks?from=0&to={node.chain.tip.height}")]
        records = {b.height: api.get(f"/round/{b.height}/record") for b in blocks if b.summary is not None}
        mine = [b for b in blocks if b.summary is not None and b.summary.job_id == "evo2-pool"]
        self.assertTrue(mine)
        es_of = [records[b.height]["es"] for b in mine]
        self.assertEqual({h["fitness"] for h in es_of}, {"replay"})
        self.assertTrue([h for h in es_of if h.get("update")], "no round took a step")
        liar = workers[2].address
        self.assertTrue(all(liar not in b.summary.accepted for b in mine), "made-up log-probabilities were accepted")
        self.assertTrue(any(w.address in b.summary.accepted for b in mine for w in workers[:2]),
                        "no honest claim was accepted")
        state, history, rounds = es2_rounds.rebuild_chain(blocks, records.__getitem__, "evo2-pool")
        self.assertEqual(es.state_hash(state), mine[-1].summary.params_hash)
        audit = Chain.from_genesis(genesis)
        for b in blocks[1:]:
            if b.summary is not None:
                self.assertEqual(replay_block(audit, b, records[b.height]), [], f"block {b.height}")
            audit.append(b)


if __name__ == "__main__":
    unittest.main()
