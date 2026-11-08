"""GET /job/<id>/model: node-local policy (--model-access), public by default, owner-only when asked."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from vibehash import node as node_module
from vibehash.client import NodeClient
from vibehash.crypto import KeyPair
from vibehash.ledger import SYSTEM, canonical
from vibehash.model import ModelSpec
from vibehash.node import Node, PhaseTimes
from vibehash.tasks import SYN_PER_VIBE, TrainingJob
from vibehash.wire import WireError
from tests.test_dos import raw

OWNER, STRANGER, VALIDATOR = (KeyPair.generate(n) for n in (b"mg-owner", b"mg-stranger", b"mg-validator"))
GENESIS = {"chain_id": "mg-net", "timestamp": 0, "allocations": {OWNER.address: 5 * SYN_PER_VIBE},
           "stakes": {VALIDATOR.address: 10 * SYN_PER_VIBE}}
FAST = PhaseTimes(register_s=0.5, compute_s=1.0, reveal_s=0.5, verify_s=0.5, idle_s=0.2)
WEIGHTS = b"w" * 64


def signed(keys, job_id, ts=None, chain_id="mg-net"):
    ts = int(time.time()) if ts is None else ts
    return {"X-VibeHash-Pubkey": keys.public.hex(), "X-VibeHash-Ts": str(ts),
            "X-VibeHash-Sig": keys.sign(canonical([chain_id, "model", job_id, 0, ts])).hex()}


class ModelGateTests(unittest.TestCase):
    def node(self, **kw) -> Node:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        # The validator's key is the node's own: it is in the validator set, and peers sync as validators do.
        n = Node(Path(tmp.name) / "n", GENESIS, VALIDATOR, port=0, times=FAST, **kw).start()
        self.addCleanup(n.stop)
        st = n.chain.state
        st.validators[VALIDATOR.address] = n.url  # (a staked address is a validator once registered)
        mlp = TrainingJob("mine", OWNER.address, 0, 0, spec=ModelSpec(4, (4,), 2, "float32"))
        pub = TrainingJob("open", OWNER.address, 0, 0, spec=ModelSpec(4, (4,), 2, "float32"), public=True)
        starter = TrainingJob("starter", SYSTEM, 0, 0, spec=ModelSpec(4, (4,), 2, "float32"))
        for job in (mlp, pub, starter):
            st.jobs[job.job_id] = job.spec_json()
            st.job_owner[job.job_id] = job.requester
        patch = mock.patch.object(n, "_model_bytes", return_value=(0, WEIGHTS, "x"))
        patch.start()
        self.addCleanup(patch.stop)
        return n

    def test_owner_with_a_valid_signature_is_served_under_owner(self):
        n = self.node(model_access="owner")
        self.assertEqual(n.job_model("mine", 0, signed(OWNER, "mine")), WEIGHTS)

    def test_a_stranger_is_refused_under_owner_and_served_under_public(self):
        n = self.node(model_access="owner")
        with self.assertRaisesRegex(WireError, "private to its owner"):
            n.job_model("mine", 0, signed(STRANGER, "mine"))
        self.assertEqual(self.node().job_model("mine", 0, signed(STRANGER, "mine")), WEIGHTS)

    def test_unsigned_is_refused_under_owner_and_served_under_public(self):
        with self.assertRaisesRegex(WireError, "private to its owner"):
            self.node(model_access="owner").job_model("mine", 0, {})
        self.assertEqual(self.node().job_model("mine", 0, {}), WEIGHTS)

    def test_a_starter_job_is_served_unsigned_under_owner(self):
        self.assertEqual(self.node(model_access="owner").job_model("starter", 0, {}), WEIGHTS)

    def test_a_job_that_says_public_is_served_unsigned_under_owner(self):
        self.assertEqual(self.node(model_access="owner").job_model("open", 0, {}), WEIGHTS)

    def test_a_validator_is_served_under_owner(self):
        n = self.node(model_access="owner")
        self.assertIn(VALIDATOR.address, n.chain.state.validator_set())
        self.assertEqual(n.job_model("mine", 0, signed(VALIDATOR, "mine")), WEIGHTS)

    def test_a_stale_timestamp_is_an_error(self):
        n = self.node(model_access="owner")
        with self.assertRaisesRegex(WireError, "too far"):
            n.job_model("mine", 0, signed(OWNER, "mine", ts=int(time.time()) - 121))

    def test_a_bad_signature_or_one_for_another_job_is_an_error(self):
        n = self.node(model_access="owner")
        with self.assertRaisesRegex(WireError, "bad"):
            n.job_model("mine", 0, signed(OWNER, "open"))
        with self.assertRaisesRegex(WireError, "malformed"):
            n.job_model("mine", 0, {**signed(OWNER, "mine"), "X-VibeHash-Sig": "zz"})

    def test_an_unknown_job_is_not_found(self):
        with self.assertRaises(KeyError):
            self.node(model_access="owner").job_model("nope", 0, {})

    def test_a_refusal_is_http_403_and_a_malformed_signature_stays_400(self):
        n = self.node(model_access="owner")
        status, body = raw(n.url, "GET", "/job/mine/model?offset=0")
        self.assertEqual(status, 403, body)
        self.assertIn("private to its owner", json.loads(body)["error"])
        self.assertEqual(raw(n.url, "GET", "/job/mine/model?offset=0", headers=signed(OWNER, "mine"))[0], 200)
        stale = signed(OWNER, "mine", ts=int(time.time()) - 121)
        self.assertEqual(raw(n.url, "GET", "/job/mine/model?offset=0", headers=stale)[0], 400)

    def test_public_logs_once_per_job_and_address_what_owner_would_refuse(self):
        n = self.node()
        with self.assertLogs("vibehash.node", "INFO") as seen:
            for _ in range(3):
                n.job_model("mine", 0, signed(STRANGER, "mine"))
            n.job_model("mine", 0, signed(OWNER, "mine"))
        lines = [m for m in seen.output if "model_access=owner" in m]
        self.assertEqual(len(lines), 1, seen.output)

    def test_status_reports_model_access(self):
        self.assertEqual(self.node().status()["model_access"], "public")
        self.assertEqual(self.node(model_access="owner").status()["model_access"], "owner")

    def test_a_bad_setting_is_refused(self):
        with self.assertRaises(ValueError):
            self.node(model_access="nobody")


class PeerSyncTests(unittest.TestCase):
    def test_the_node_downloads_peers_models_with_its_own_keys(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        n = Node(Path(tmp.name) / "n", GENESIS, VALIDATOR, port=0, times=FAST, peers=["http://127.0.0.1:1"])
        self.addCleanup(n.store.close)
        self.addCleanup(n.peers.close)
        self.addCleanup(n.server.server_close)
        job = TrainingJob("mine", OWNER.address, 0, 0, spec=ModelSpec(4, (4,), 2, "float32"))
        n.chain.state.jobs["mine"] = job.spec_json()
        n.chain.state.job_owner["mine"] = OWNER.address
        client = mock.Mock()
        client.download_model.side_effect = node_module.NodeError("no")
        with mock.patch.object(n.coord, "expected_start_hash", return_value=b"\x01" * 32), \
                mock.patch.object(n.peers, "client", return_value=client):
            n._ensure_params("mine")
        client.download_model.assert_called_once()
        self.assertIs(client.download_model.call_args.kwargs["keys"], VALIDATOR)


class ClientKeysTests(unittest.TestCase):
    def test_download_model_signs_every_chunk_without_upload_only_headers(self):
        job = TrainingJob("mine", OWNER.address, 0, 0, spec=ModelSpec(4, (4,), 2, "float32"))
        data = b"\x01" * job.checkpoint_bytes
        import hashlib
        info = {"spec": job.to_spec(), "owner": OWNER.address, "model": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
        api, sent = NodeClient("http://127.0.0.1:1"), []
        api._status = {"chain_id": "mg-net"}

        def req(method, path, body=None, headers=None, **kw):
            sent.append(headers)
            return data
        with mock.patch.object(api, "get", return_value=info), mock.patch.object(api, "_req", side_effect=req):
            api.download_model("mine", keys=OWNER)
            api.download_model("mine")
        self.assertEqual(set(sent[0]), {"X-VibeHash-Pubkey", "X-VibeHash-Sig", "X-VibeHash-Ts"})
        self.assertEqual(sent[0]["X-VibeHash-Pubkey"], OWNER.public.hex())
        self.assertFalse(sent[1])


class SpecPublicTests(unittest.TestCase):
    def test_public_round_trips_and_absent_stays_absent(self):
        for kw in ({}, {"public": True}, {"public": False}):
            job = TrainingJob("j", OWNER.address, 0, 0, spec=ModelSpec(4, (4,), 2), **kw)
            spec = job.to_spec()
            self.assertEqual("public" in spec, bool(kw), spec)
            back = TrainingJob.from_spec_json(job.spec_json(), OWNER.address, 0)
            self.assertEqual(back.public, kw.get("public"))
            self.assertEqual(back.spec_json(), job.spec_json())

    def test_a_public_that_is_not_a_bool_is_refused(self):
        d = TrainingJob("j", OWNER.address, 0, 0, spec=ModelSpec(4, (4,), 2)).to_spec()
        from vibehash.tasks import InvalidJob
        with self.assertRaises(InvalidJob):
            TrainingJob.from_spec_json(json.dumps({**d, "public": "yes"}), OWNER.address, 0)


if __name__ == "__main__":
    unittest.main()
