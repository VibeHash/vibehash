"""Right-answer scoring (es2.Spec2.gold, es2_rounds.gold_problems, es_client.gold_items): a round's members scored on
its own problems' gold answers, so a job learns from what its model gets wrong too, instead of only on answers the line
already got right. Off, the field is left out of the spec and every round is what it always was; on, a local network's
rounds are scored on log-probabilities from the first, a worker lying about them is turned down by its checkers, and
every round replays from its record."""
from __future__ import annotations

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
from tests.test_es2 import VIBE, AnchorTests, job_of, tiny_spec2
from tests.test_loop_stop import spec_json


class GoldSpecTests(unittest.TestCase):
    def test_off_is_left_out_of_the_spec(self):
        s = es2.Spec2.from_json(spec_json())
        self.assertFalse(s.gold)
        self.assertNotIn("gold", s.to_json()["es"])  # a job's hash is what it was before the field
        self.assertEqual(s.to_json(), es2.Spec2.from_json(s.to_json()).to_json())
        self.assertEqual(es2_rounds.gold_problems(s, 3), [])

    def test_on_round_trips_and_is_checked(self):
        s = es2.Spec2.from_json(spec_json(gold=True))
        self.assertTrue(s.gold)
        self.assertIs(s.to_json()["es"]["gold"], True)
        self.assertEqual(es2.Spec2.from_json(s.to_json()), s)
        self.assertFalse([p for p in s.problems_list() if "gold" in p])
        for bad in (1, "yes", None):
            self.assertTrue([p for p in es2.Spec2.from_json(spec_json(gold=bad)).problems_list() if "gold" in p], bad)

    def test_which_problems_are_scored_on_their_gold_answers(self):
        one = es2.Spec2.from_json(spec_json(gold=True))  # one task, a short-answer one: every problem
        self.assertEqual(es2_rounds.gold_problems(one, 0), list(range(one.problems)))
        # A reasoned task's right reply isn't its gold answer alone (es_tasks.WRITTEN): its problems keep replay targets.
        written = es2.Spec2.from_json(spec_json(gold=True, task="gsm8k"))
        self.assertEqual(es2_rounds.gold_problems(written, 0), [])
        mixed = AnchorTests().spec(task="arithmetic", tasks=("arithmetic", "gsm8k", "countdown"), mix=True, gold=True)
        for step in (0, 1, 5):
            names = es2_rounds.task_names(mixed, step)
            got = es2_rounds.gold_problems(mixed, step)
            self.assertEqual(got, [i for i, t in enumerate(names) if t != "gsm8k"])
            self.assertTrue(0 < len(got) < mixed.problems, "a mixed round has both kinds")
        self.assertEqual(set(es_tasks.WRITTEN), {"ifeval", "gsm8k", "math"})

    def test_the_scored_text_is_a_right_answer(self):
        for task in ("countdown", "arithmetic"):
            spec = AnchorTests().spec(task=task, gold=True)
            for it in es2_rounds.items(spec, 77, 0):
                text = es2_rounds.gold_text(it)
                self.assertEqual(text, f"<answer>{it.gold}</answer>")
                self.assertTrue(it.check(text), (task, text))

    def test_the_settings(self):
        from vibehash import es_local
        self.assertLess(es_local.V2_GOLD_REPLAY, es_local.V2_SETTINGS["problems"])  # the replay targets stay a part


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class GoldTemplateTests(unittest.TestCase):
    def test_the_template_turns_it_on_only_when_asked(self):
        from vibehash import cli, es_local
        with tempfile.TemporaryDirectory() as d:
            folder = tiny_model(Path(d) / "tiny", seed=3)
            for name, extra in (("plain", []), ("gold", ["--gold"])):
                with mock.patch("sys.stdout"):
                    cli.main(["job", "template", "--evo", str(folder), "--id", name, "--out", str(Path(d) / name),
                              *extra])
            got = {n: json.loads((Path(d) / n).read_text())["es"] for n in ("plain", "gold")}
        self.assertNotIn("gold", got["plain"])
        self.assertEqual(got["plain"]["replay"], es_local.V2_SETTINGS["replay"])
        self.assertIs(got["gold"]["gold"], True)
        self.assertEqual(got["gold"]["replay"], es_local.V2_GOLD_REPLAY)
        self.assertEqual({k: v for k, v in got["gold"].items() if k not in ("gold", "replay")},
                         {k: v for k, v in got["plain"].items() if k != "replay"})

    def test_the_items_a_worker_and_a_checker_score(self):
        from vibehash import es_client
        with tempfile.TemporaryDirectory() as d:
            folder = tiny_model(Path(d) / "tiny", seed=1)
            spec, plain = tiny_spec2(folder, gold=True), tiny_spec2(folder)
            rt = es_client.runtime2(spec, "cpu")
            ps = es2_rounds.items(spec, 5, 0)
            rows = es_client.prompt_rows(rt, ps)
            got = es_client.gold_items(rt, spec, ps, rows, 0)
            self.assertEqual(es_client.gold_items(rt, plain, ps, rows, 0), [])
            self.assertEqual(len(got), spec.problems)
            for (row, ids), want_row, p in zip(got, rows, ps):
                self.assertEqual(row, want_row)
                whole = rt.tokenizer.encode(es2_rounds.gold_text(p), add_special_tokens=False)
                self.assertEqual(ids, whole[:spec.max_tokens - 1] + [rt.eos[0]])  # a reply a member could have given
                self.assertTrue(es2.ends_right(ids, spec, rt.eos, spec.max_tokens))
            self.assertEqual(got, es_client.gold_items(rt, spec, es2_rounds.items(spec, 5, 0), rows, 0))


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class GoldNetworkTests(unittest.TestCase):
    def test_rounds_are_scored_from_the_first_a_liar_is_turned_down_and_they_replay(self):
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
        spec = tiny_spec2(tiny_model(root / "tiny", seed=1), gold=True)
        for p in (mock.patch.object(es, "reward", fake_reward),
                  mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(root / "home")}),
                  mock.patch.object(es_client, "SAVE_EVERY", 1), mock.patch("vibehash.es_local.V2_COMPUTE_S", 30)):
            p.start()
            self.addCleanup(p.stop)
        node_keys, req = KeyPair.generate(b"gold node"), KeyPair.generate(b"gold requester")
        workers = [KeyPair.generate(f"gold worker {i}".encode()) for i in range(3)]
        genesis = {"chain_id": "vibehash-gold-test", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}, "es": True}
        node = Node(root / "node", genesis, node_keys, port=0,
                    times=PhaseTimes(register_s=1.0, compute_s=20.0, reveal_s=3.0, verify_s=15.0, idle_s=0.3),
                    config=CoordinatorConfig(newcomer_units=4)).start()
        self.addCleanup(node.stop)
        api = NodeClient(node.url)
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e12, poll_s=0.1, device="cpu",
                                cache_dir=root / f"cache{i}") for i, k in enumerate(workers)]
        clients[2].es.cheat = "lp-liar"  # log-probabilities its members didn't give the gold answers
        job = job_of(spec, "evo2-gold", req.address, 6 * VIBE)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        stop = threading.Event()
        self.addCleanup(stop.set)
        with self.assertLogs("vibehash.worker", level="WARNING") as said:
            run_until(node, clients, stop, 5, seconds=300)
        self.assertGreaterEqual(node.chain.tip.height, 5, "the network didn't make its rounds in time")

        blocks = [Block.from_dict(d) for d in api.get(f"/blocks?from=0&to={node.chain.tip.height}")]
        records = {b.height: api.get(f"/round/{b.height}/record") for b in blocks if b.summary is not None}
        mine = [b for b in blocks if b.summary is not None and b.summary.job_id == "evo2-gold"]
        self.assertTrue(mine)
        es_of = [records[b.height]["es"] for b in mine]
        # Scored on log-probabilities from the job's first round: a plain job's has no targets yet and takes "binary".
        self.assertEqual({h["fitness"] for h in es_of}, {"replay"})
        self.assertEqual(int(es_of[0]["step"]), 0)
        stepped = [h for h in es_of if h.get("update")]
        self.assertTrue(stepped, "no round took a step")
        self.assertEqual({h["used"] for h in stepped}, {"replay"})
        # Every problem here is a gold one, so none becomes a replay target (they'd be scored twice).
        self.assertEqual([h.get("targets") or [] for h in es_of], [[]] * len(es_of))
        liar = workers[2].address
        self.assertTrue(all(liar not in b.summary.accepted for b in mine), "made-up log-probabilities were accepted")
        self.assertTrue(any(w.address in b.summary.accepted for b in mine for w in workers[:2]),
                        "no honest claim was accepted")
        # (its answers are its members' own, so the checkers' teacher forcing passes them: the scores alone gave it away)
        self.assertRegex("\n".join(said.output), r"fails the check: replay log-probability -?[\d.]+, worked out")

        state, history, rounds = es2_rounds.rebuild_chain(blocks, records.__getitem__, "evo2-gold")
        self.assertEqual(rounds, len(mine))
        self.assertEqual(es.state_hash(state), mine[-1].summary.params_hash)
        audit = Chain.from_genesis(genesis)
        for b in blocks[1:]:
            if b.summary is not None:
                self.assertEqual(replay_block(audit, b, records[b.height]), [], f"block {b.height}")
            audit.append(b)


if __name__ == "__main__":
    unittest.main()
