"""The operators' side: the node's admin routes and job policy (nodeadmin.py), the admin app's server, places and
orders (admin.py), and the globe's land dots (scripts/make_globe.py)."""
import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vibehash import admin, nodeadmin, settings
from vibehash import wallet as wallets
from vibehash.cli import template_job
from vibehash.client import NodeClient, NodeError
from vibehash.coordinator import Coordinator, CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import SYSTEM
from vibehash.storage import Store
from vibehash.tasks import SYN_PER_VIBE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_globe  # noqa: E402


TOKEN = "t" * nodeadmin.MIN_TOKEN
NODE_TOKEN = "t0k3n-" + "long-enough-for-the-node"
KEY = nodeadmin.new_mfa_key()


class FakeStore:
    def __init__(self):
        self.meta = {}

    def save_meta(self, key, value):
        self.meta[key] = json.loads(json.dumps(value))

    def load_meta(self, key):
        return self.meta.get(key)


class AdminPolicyTests(unittest.TestCase):
    def test_no_token_means_no_routes_and_a_wrong_one_is_refused(self):
        a = nodeadmin.Admin(FakeStore(), None)
        self.assertFalse(a.enabled)
        with self.assertRaises(KeyError):
            a.check("Bearer anything")
        a = nodeadmin.Admin(FakeStore(), "s3cret-and-long-enough-to-use")
        for bad in (None, "", "Bearer", "Bearer wrong", "s3cret-and-long-enough-to-us"):
            with self.assertRaises(nodeadmin.Forbidden):
                a.check(bad)
        a.check("Bearer s3cret-and-long-enough-to-use")
        a.check("bearer s3cret-and-long-enough-to-use")
        a.check("s3cret-and-long-enough-to-use")  # the bare token does too

    def test_short_tokens_and_keys_that_arent_one_are_refused(self):
        with self.assertRaises(ValueError):
            nodeadmin.Admin(FakeStore(), "short")
        with self.assertRaises(ValueError):
            nodeadmin.Admin(FakeStore(), TOKEN, mfa="not base32!")
        with self.assertRaises(ValueError):
            nodeadmin.Admin(FakeStore(), None, mfa=KEY)  # a code with no token makes no sense
        with self.assertRaises(ValueError):
            nodeadmin.mfa_key("MFRGGZDF")  # 40 bits: too short to be a key
        self.assertGreaterEqual(len(nodeadmin.new_token()), nodeadmin.MIN_TOKEN)
        self.assertEqual(len(nodeadmin.mfa_key(KEY)), 20)
        spaced = " ".join(KEY[i:i + 4] for i in range(0, len(KEY), 4)).lower()  # as a person types it
        self.assertEqual(nodeadmin.mfa_key(spaced), nodeadmin.mfa_key(KEY))
        uri = nodeadmin.otpauth_uri(KEY)
        self.assertTrue(uri.startswith("otpauth://totp/Vibe%20Hash%3Aadmin?secret=" + KEY))
        self.assertIn("issuer=Vibe%20Hash", uri)

    def test_codes_are_rfc_6238s(self):
        key = b"12345678901234567890"  # the RFC's own test key and times (SHA-1)
        for t, code in ((59, "94287082"), (1111111109, "07081804"), (1234567890, "89005924"), (2000000000, "69279037")):
            self.assertEqual(nodeadmin.totp(key, t // 30, digits=8), code)
        self.assertEqual(nodeadmin.totp(key, 1), "287082")

    def test_with_a_key_the_token_alone_opens_only_the_login(self):
        clock = [1_000_000.0]
        a = nodeadmin.Admin(FakeStore(), TOKEN, now=lambda: clock[0], mfa=KEY)
        key, step = nodeadmin.mfa_key(KEY), int(clock[0] // 30)
        with self.assertRaises(nodeadmin.NeedCode):
            a.check("Bearer " + TOKEN)
        with self.assertRaises(nodeadmin.Forbidden):
            a.login("Bearer wrong", nodeadmin.totp(key, step))  # the code alone opens nothing either
        with self.assertRaises(nodeadmin.Forbidden):
            a.login("Bearer " + TOKEN, "000000" if nodeadmin.totp(key, step) != "000000" else "111111")
        out = a.login("Bearer " + TOKEN, nodeadmin.totp(key, step))
        self.assertTrue(out["mfa"])
        self.assertEqual(out["expires"], clock[0] + nodeadmin.SESSION_S)
        a.check("Bearer " + TOKEN, out["session"])
        with self.assertRaises(nodeadmin.NeedCode):
            a.check("Bearer " + TOKEN, "a-made-up-sign-in")
        with self.assertRaises(nodeadmin.Forbidden):
            a.check("Bearer wrong", out["session"])  # a sign-in without the token is nothing
        # a code is good once, and the one before it no longer, but neither counts as a guess
        strikes = len(a.fails.get("?", []))
        for c in (step, step - 1):
            with self.assertRaisesRegex(nodeadmin.Forbidden, "just used"):
                a.login("Bearer " + TOKEN, nodeadmin.totp(key, c))
        self.assertEqual(len(a.fails.get("?", [])), strikes)
        clock[0] += 30  # the next code; a clock a step behind or ahead is forgiven
        self.assertTrue(a.login("Bearer " + TOKEN, " " + nodeadmin.totp(key, step + 2) + " ")["session"])
        clock[0] += nodeadmin.SESSION_S + 1
        with self.assertRaises(nodeadmin.NeedCode):
            a.check("Bearer " + TOKEN, out["session"])  # over
        plain = nodeadmin.Admin(FakeStore(), TOKEN)  # without a key, the login is just the token
        self.assertEqual(plain.login("Bearer " + TOKEN, ""), {"ok": True, "mfa": False, "session": None, "expires": None})

    def test_wrong_tries_turn_an_address_away_for_a_while(self):
        clock = [5000.0]
        a = nodeadmin.Admin(FakeStore(), TOKEN, now=lambda: clock[0])
        for _ in range(nodeadmin.MAX_FAILS):
            with self.assertRaises(nodeadmin.Forbidden):
                a.check("Bearer guess", ip="93.184.216.34")
        with self.assertRaises(nodeadmin.TooMany) as caught:
            a.check("Bearer " + TOKEN, ip="93.184.216.34")  # even the right token, for now
        self.assertEqual(caught.exception.retry_after, nodeadmin.FAIL_WINDOW_S)
        a.check("Bearer " + TOKEN, ip="198.51.100.20")  # another address isn't affected
        clock[0] += nodeadmin.FAIL_WINDOW_S
        a.check("Bearer " + TOKEN, ip="93.184.216.34")

    def test_many_wrong_codes_from_anywhere_stop_codes_for_a_while(self):
        clock = [2_000_000.0]
        a = nodeadmin.Admin(FakeStore(), TOKEN, now=lambda: clock[0], mfa=KEY)
        key = nodeadmin.mfa_key(KEY)
        good = a.login("Bearer " + TOKEN, nodeadmin.totp(key, int(clock[0] // 30)))["session"]
        right = {nodeadmin.totp(key, int(clock[0] // 30) + d) for d in (-1, 0, 1)}
        wrong = next(f"{n:06d}" for n in range(1000) if f"{n:06d}" not in right)
        for i in range(nodeadmin.MAX_CODE_FAILS):  # each from its own address, so no single address is turned away
            with self.assertRaises(nodeadmin.Forbidden):
                a.login("Bearer " + TOKEN, wrong, ip=f"10.9.{i // 250}.{i % 250}")
        clock[0] += 30
        with self.assertRaisesRegex(nodeadmin.TooMany, "make a new token"):
            a.login("Bearer " + TOKEN, nodeadmin.totp(key, int(clock[0] // 30)), ip="8.8.4.4")
        a.check("Bearer " + TOKEN, good)  # a sign-in made before stands
        clock[0] += nodeadmin.FAIL_WINDOW_S
        self.assertTrue(a.login("Bearer " + TOKEN, nodeadmin.totp(key, int(clock[0] // 30)), ip="8.8.4.4")["session"])

    def test_a_fresh_check_needs_a_new_code_on_top_of_the_sign_in(self):
        clock = [3_000_000.0]
        a = nodeadmin.Admin(FakeStore(), TOKEN, now=lambda: clock[0], mfa=KEY)
        key, step = nodeadmin.mfa_key(KEY), lambda: int(clock[0] // 30)
        session = a.login("Bearer " + TOKEN, nodeadmin.totp(key, step()))["session"]
        a.check("Bearer " + TOKEN, session)  # a plain check wants no code
        with self.assertRaisesRegex(nodeadmin.Forbidden, nodeadmin.NEED_FRESH):
            a.check("Bearer " + TOKEN, session, fresh=True)
        with self.assertRaisesRegex(nodeadmin.Forbidden, nodeadmin.NEED_FRESH):
            a.check("Bearer " + TOKEN, session, code="  ", fresh=True)
        with self.assertRaisesRegex(nodeadmin.Forbidden, "just used"):  # the sign-in's own code
            a.check("Bearer " + TOKEN, session, code=nodeadmin.totp(key, step()), fresh=True)
        with self.assertRaises(nodeadmin.NeedCode):  # no sign-in: that first
            a.check("Bearer " + TOKEN, "nope", code=nodeadmin.totp(key, step() + 1), fresh=True)
        clock[0] += 30
        a.check("Bearer " + TOKEN, session, code=nodeadmin.totp(key, step()), fresh=True)
        with self.assertRaisesRegex(nodeadmin.Forbidden, "just used"):  # and once
            a.check("Bearer " + TOKEN, session, code=nodeadmin.totp(key, step()), fresh=True)
        clock[0] += 30
        a.check("Bearer " + TOKEN, session, code=" " + nodeadmin.totp(key, step()) + " ", fresh=True)

    def test_a_wrong_fresh_code_is_a_strike_like_a_wrong_login_code(self):
        clock = [4_000_000.0]
        a = nodeadmin.Admin(FakeStore(), TOKEN, now=lambda: clock[0], mfa=KEY)
        key = nodeadmin.mfa_key(KEY)
        session = a.login("Bearer " + TOKEN, nodeadmin.totp(key, int(clock[0] // 30)))["session"]
        clock[0] += 30
        right = {nodeadmin.totp(key, int(clock[0] // 30) + d) for d in (-1, 0, 1)}
        wrong = next(f"{n:06d}" for n in range(1000) if f"{n:06d}" not in right)
        ip = "93.184.216.34"
        for _ in range(nodeadmin.MAX_FAILS):
            with self.assertRaisesRegex(nodeadmin.Forbidden, "code is wrong"):
                a.check("Bearer " + TOKEN, session, ip=ip, code=wrong, fresh=True)
        self.assertEqual(len(a.code_fails), nodeadmin.MAX_FAILS)
        with self.assertRaises(nodeadmin.TooMany):  # the address is turned away, as after wrong logins
            a.check("Bearer " + TOKEN, session, ip=ip, code=nodeadmin.totp(key, int(clock[0] // 30)), fresh=True)

    def test_a_node_without_a_key_has_no_fresh_codes_to_ask_for(self):
        a = nodeadmin.Admin(FakeStore(), TOKEN)
        a.check("Bearer " + TOKEN)
        with self.assertRaisesRegex(nodeadmin.Forbidden, "--admin-mfa"):
            a.check("Bearer " + TOKEN, fresh=True)

    def test_holds_and_review_survive_a_restart_and_say_why(self):
        store = FakeStore()
        a = nodeadmin.Admin(store, TOKEN, now=lambda: 100.0)
        self.assertIsNone(a.reason("j1", "vibe1aaa"))
        a.hold("j1", "spam")
        self.assertEqual(a.reason("j1", "vibe1aaa"), "an admin holds it")  # the reason given isn't in it: /status is public
        self.assertEqual(a.held["j1"]["why"], "spam")
        a.hold("j2")
        self.assertEqual(a.reason("j2", "vibe1aaa"), "an admin holds it")
        b = nodeadmin.Admin(store, TOKEN)  # a restart
        self.assertEqual(b.reason("j1", "vibe1aaa"), "an admin holds it")
        self.assertEqual(b.held["j1"]["why"], "spam")
        b.release("j1")
        self.assertIsNone(b.reason("j1", "vibe1aaa"))
        self.assertIsNone(nodeadmin.Admin(store, TOKEN).reason("j1", "vibe1aaa"))
        # review: jobs already on chain are confirmed; new ones wait; starter jobs never do
        b.set_review(True, ["old"])
        self.assertIsNone(b.reason("old", "vibe1aaa"))
        self.assertEqual(b.reason("new", "vibe1bbb"), "it awaits an admin's review")
        self.assertIsNone(b.reason("starter", SYSTEM))
        b.release("new")
        self.assertIsNone(b.reason("new", "vibe1bbb"))
        self.assertTrue(nodeadmin.Admin(store, TOKEN).review)
        b.set_review(False)
        self.assertIsNone(b.reason("newer", "vibe1ccc"))

    def test_a_release_tells_the_node_to_drop_the_waiting_line(self):
        a = nodeadmin.Admin(FakeStore(), TOKEN)
        dropped = []
        a.changed = dropped.append
        a.hold("j1")
        a.release("j1")
        self.assertEqual(dropped, ["j1", "j1"])

    def test_devices_are_kept_with_where_they_came_from_and_bounded(self):
        clock = [1000.0]
        a = nodeadmin.Admin(FakeStore(), TOKEN, now=lambda: clock[0])
        a.saw("vibe1w1", "203.0.113.5", {"round": 3, "flops_per_sec": 1e12, "memory_mb": 8192,
                                           "hardware": {"vendor": "nvidia", "arch": "ada", "memory_gb": 12, "label": "RTX 4070"}}, 3)
        a.saw("vibe1w2", "198.51.100.9", {"round": 3, "flops_per_sec": 2e12, "memory_mb": 16384, "hardware": {"made": "up"}}, 3)
        clock[0] += nodeadmin.ONLINE_S + 1
        a.saw("vibe1w3", None, {"round": 4, "flops_per_sec": 3e12, "memory_mb": 1024}, 4)
        devs = {d["address"]: d for d in a.device_list()}
        self.assertEqual(devs["vibe1w1"]["ip"], "203.0.113.5")
        self.assertEqual(devs["vibe1w1"]["hardware"]["label"], "RTX 4070")
        self.assertIsNone(devs["vibe1w2"]["hardware"])  # not a canonical class
        self.assertFalse(devs["vibe1w1"]["online"])
        self.assertTrue(devs["vibe1w3"]["online"])
        self.assertEqual([d["address"] for d in a.device_list()][0], "vibe1w3")  # newest first
        with mock.patch.object(nodeadmin, "MAX_DEVICES", 2):
            a.saw("vibe1w4", "1.2.3.4", {"round": 5, "flops_per_sec": 1.0, "memory_mb": 1}, 5)
        self.assertEqual(len(a.devices), 2)
        self.assertNotIn("vibe1w1", a.devices)  # the oldest went

    def test_the_client_ip_trusts_a_forwarded_header_only_from_a_local_proxy(self):
        self.assertEqual(nodeadmin.client_ip("93.184.216.34", "8.8.8.8"), "93.184.216.34")  # from the internet: ignored
        self.assertEqual(nodeadmin.client_ip("127.0.0.1", "8.8.8.8"), "8.8.8.8")  # the proxy's own entry
        self.assertEqual(nodeadmin.client_ip("127.0.0.1", "8.8.8.8, 10.0.0.1"), "8.8.8.8")  # a second local proxy
        self.assertEqual(nodeadmin.client_ip("127.0.0.1", "1.1.1.1, 8.8.8.8"), "8.8.8.8")  # the client's own claim is ignored
        self.assertEqual(nodeadmin.client_ip("10.0.0.2", "8.8.8.8"), "8.8.8.8")
        self.assertEqual(nodeadmin.client_ip("127.0.0.1", "not-an-ip"), "127.0.0.1")
        self.assertEqual(nodeadmin.client_ip("127.0.0.1", "8.8.8.8, junk"), "127.0.0.1")
        self.assertEqual(nodeadmin.client_ip("127.0.0.1", None), "127.0.0.1")


class CoordinatorPolicyTests(unittest.TestCase):
    def test_a_held_job_takes_no_turn_and_says_why(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        req, node_keys = KeyPair.generate(b"req"), KeyPair.generate(b"node")
        genesis = {"chain_id": "policy-test", "timestamp": 0, "allocations": {req.address: 500 * SYN_PER_VIBE},
                   "stakes": {node_keys.address: 10 * SYN_PER_VIBE}}
        store = Store(Path(tmp.name) / "node")
        self.addCleanup(store.close)
        chain = store.open_chain(genesis)
        c = Coordinator(chain, node_keys, CoordinatorConfig(), store=store)
        s = chain.state
        for job_id in ("j1", "j2"):
            job = template_job(job_id, False, req.address, 100 * SYN_PER_VIBE)
            s.jobs[job_id], s.escrow[job_id], s.job_owner[job_id] = job.spec_json(), job.budget, req.address
        self.assertEqual(c.active_job_id(), "j1")
        a = nodeadmin.Admin(FakeStore(), TOKEN)
        c.policy = a.reason
        a.changed = lambda job_id: c.waiting.pop(job_id, None)
        a.hold("j1", "looks wrong")
        self.assertEqual(c.active_job_id(), "j2")
        self.assertEqual(c.waiting["j1"], "job j1 waits: an admin holds it")  # what /status shows: no reason text
        self.assertFalse(c.open_round() and c.r.job_id == "j1")  # the round that opens isn't the held job's
        self.assertEqual(c.r.job_id, "j2")
        c.phase, c.r = type(c.phase).IDLE, None
        a.hold("j2")
        self.assertIsNone(c.active_job_id())
        a.release("j1")
        self.assertNotIn("j1", c.waiting)
        self.assertEqual(c.active_job_id(), "j1")


class NodeRouteTests(unittest.TestCase):
    """A real node with a token: the routes answer 401 without it and the orders take effect."""

    def setUp(self):
        from vibehash.node import Node, PhaseTimes

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.req, keys = KeyPair.generate(b"req"), KeyPair.generate(b"node")
        genesis = {"chain_id": "admin-route-test", "timestamp": 0, "allocations": {self.req.address: 500 * SYN_PER_VIBE},
                   "stakes": {keys.address: 10 * SYN_PER_VIBE}}
        self.node = Node(Path(tmp.name) / "node", genesis, keys, port=0, admin_token=NODE_TOKEN,
                         times=PhaseTimes(register_s=1.0, compute_s=4.0, reveal_s=2.0, verify_s=3.0, idle_s=0.3)).start()
        self.addCleanup(self.node.stop)

    def call(self, path, body=None, token=None, headers=None, raw=False):
        req = urllib.request.Request(self.node.url + path, data=json.dumps(body).encode() if body is not None else None,
                                     method="POST" if body is not None else "GET",
                                     headers={**({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return (e.code, json.loads(e.read()), e.headers) if raw else (e.code, json.loads(e.read()))

    def test_with_a_key_the_node_wants_a_code_and_turns_guessers_away(self):
        self.node.admin.mfa = nodeadmin.mfa_key(KEY)
        code, out = self.call("/admin/overview", token=NODE_TOKEN)
        self.assertEqual(code, 401)
        self.assertIn(nodeadmin.NEED_CODE, out["error"])
        key = nodeadmin.mfa_key(KEY)
        clock = [time.time()]
        self.node.admin.now = lambda: clock[0]
        code, out = self.call("/admin/login", {"code": nodeadmin.totp(key, int(clock[0] // 30))}, token=NODE_TOKEN)
        self.assertEqual(code, 200)
        signed = {nodeadmin.LOGIN_HEADER: out["session"]}
        self.assertEqual(self.call("/admin/overview", token=NODE_TOKEN, headers=signed)[0], 200)
        clock[0] += 30  # (an order takes a code of its own, a later one than the sign-in's)
        fresh = {**signed, nodeadmin.CODE_HEADER: nodeadmin.totp(key, int(clock[0] // 30))}
        self.assertEqual(self.call("/admin/review", {"on": False}, token=NODE_TOKEN, headers=fresh)[0], 200)
        self.assertEqual(self.call("/admin/review", {"on": False}, token=NODE_TOKEN)[0], 401)
        for _ in range(nodeadmin.MAX_FAILS):
            self.call("/admin/overview", token="a-guess")
        code, out, headers = self.call("/admin/overview", token=NODE_TOKEN, headers=signed, raw=True)
        self.assertEqual(code, 403)  # not 429: a client would only wait and send it again
        self.assertIn("too many wrong tries", out["error"])
        self.assertGreater(int(headers["Retry-After"]), 0)

    def test_the_routes_need_the_token_and_hold_release_and_review_work(self):
        self.assertEqual(self.call("/admin/overview")[0], 401)
        self.assertEqual(self.call("/admin/overview", token="nope")[0], 401)
        code, view = self.call("/admin/overview", token=NODE_TOKEN)
        self.assertEqual(code, 200)
        self.assertEqual(view["node"]["url"], self.node.url)
        self.assertFalse(view["review"])
        self.assertEqual(view["jobs"], [])
        # a job on chain, then held and released
        job = template_job("j1", False, self.req.address, 100 * SYN_PER_VIBE)
        NodeClient(self.node.url).send_tx(self.req, "post_job", "system", job.budget, job.spec_json())
        deadline = time.time() + 30
        while "j1" not in self.node.chain.state.jobs and time.time() < deadline:
            time.sleep(0.2)
        self.assertIn("j1", self.node.chain.state.jobs)
        self.assertEqual(self.call("/admin/jobs/nope/hold", {"why": "x"}, token=NODE_TOKEN)[0], 404)
        self.assertEqual(self.call("/admin/jobs/j1/hold", {"why": "x"})[0], 401)
        code, out = self.call("/admin/jobs/j1/hold", {"why": "not now"}, token=NODE_TOKEN)
        self.assertEqual((code, out["held"]["why"]), (200, "not now"))
        self.assertEqual(self.node.admin.reason("j1", self.req.address), "an admin holds it")
        view = self.call("/admin/overview", token=NODE_TOKEN)[1]
        j1 = next(j for j in view["jobs"] if j["id"] == "j1")
        self.assertEqual(j1["policy"], "an admin holds it")
        self.assertEqual(j1["held"]["why"], "not now")
        self.assertEqual(j1["owner"], self.req.address)
        self.assertNotIn("not now", json.dumps(self.call("/status")[1]))  # the reason is the admins' alone
        self.assertEqual(self.call("/admin/jobs/j1/release", {}, token=NODE_TOKEN)[1]["held"], None)
        self.assertIsNone(self.node.admin.reason("j1", self.req.address))
        self.assertEqual(self.call("/admin/review", {"on": True}, token=NODE_TOKEN)[1]["review"], True)
        self.assertIsNone(self.node.admin.reason("j1", self.req.address))  # on chain already: confirmed
        self.assertEqual(self.node.admin.reason("j9", self.req.address), "it awaits an admin's review")
        self.assertTrue(self.call("/admin/overview", token=NODE_TOKEN)[1]["review"])

    def test_with_a_key_hold_release_and_review_need_a_fresh_code_and_the_overview_does_not(self):
        self.node.admin.mfa = key = nodeadmin.mfa_key(KEY)
        clock = [time.time()]
        self.node.admin.now = lambda: clock[0]
        step = lambda: int(clock[0] // 30)  # noqa: E731
        session = self.call("/admin/login", {"code": nodeadmin.totp(key, step())}, token=NODE_TOKEN)[1]["session"]
        signed = {nodeadmin.LOGIN_HEADER: session}
        job = template_job("j1", False, self.req.address, 100 * SYN_PER_VIBE)
        NodeClient(self.node.url).send_tx(self.req, "post_job", "system", job.budget, job.spec_json())
        deadline = time.time() + 30
        while "j1" not in self.node.chain.state.jobs and time.time() < deadline:
            time.sleep(0.2)
        self.assertEqual(self.call("/admin/overview", token=NODE_TOKEN, headers=signed)[0], 200)  # no code
        for path, body in (("/admin/jobs/j1/hold", {"why": "x"}), ("/admin/jobs/j1/release", {}), ("/admin/review", {"on": True})):
            code, out = self.call(path, body, token=NODE_TOKEN, headers=signed)
            self.assertEqual(code, 401, path)
            self.assertIn(nodeadmin.NEED_FRESH, out["error"])
        self.assertIsNone(self.node.admin.reason("j1", self.req.address))  # and they did nothing
        self.assertFalse(self.node.admin.review)
        clock[0] += 30
        sent = {**signed, nodeadmin.CODE_HEADER: nodeadmin.totp(key, step())}
        self.assertEqual(self.call("/admin/jobs/j1/hold", {"why": "x"}, token=NODE_TOKEN, headers=sent)[0], 200)
        self.assertEqual(self.node.admin.reason("j1", self.req.address), "an admin holds it")
        self.assertEqual(self.call("/admin/jobs/j1/release", {}, token=NODE_TOKEN, headers=sent)[0], 401)  # used
        clock[0] += 30
        sent = {**signed, nodeadmin.CODE_HEADER: nodeadmin.totp(key, step())}
        self.assertEqual(self.call("/admin/jobs/j1/release", {}, token=NODE_TOKEN, headers=sent)[0], 200)
        clock[0] += 30
        sent = {**signed, nodeadmin.CODE_HEADER: nodeadmin.totp(key, step())}
        self.assertEqual(self.call("/admin/review", {"on": True}, token=NODE_TOKEN, headers=sent)[1]["review"], True)

    def test_a_node_without_a_token_has_no_admin_routes(self):
        self.node.admin.token = None
        self.assertEqual(self.call("/admin/overview", token=NODE_TOKEN)[0], 404)
        self.assertEqual(self.call("/admin/jobs/j1/hold", {}, token=NODE_TOKEN)[0], 404)


class OpsRouteTests(NodeRouteTests):
    """The server's health in the overview, and the operations on it, each wanting a fresh authenticator code."""

    def setUp(self):
        super().setUp()
        self.clock = [time.time()]
        self.node.admin.now = lambda: self.clock[0]
        self.key = nodeadmin.mfa_key(KEY)
        self.backups = Path(self.node.store.dir).resolve().parent / "backup"

    def code(self):
        """A code of a step no code was taken in yet: each code is good once."""
        self.clock[0] += nodeadmin.CODE_STEP
        return nodeadmin.totp(self.key, int(self.clock[0] // nodeadmin.CODE_STEP))

    def signed_in(self):
        self.node.admin.mfa = self.key
        code, out = self.call("/admin/login", {"code": self.code()}, token=NODE_TOKEN)
        self.assertEqual(code, 200)
        return {nodeadmin.LOGIN_HEADER: out["session"]}

    def test_the_overview_says_how_the_server_is(self):
        ops = self.call("/admin/overview", token=NODE_TOKEN)[1]["ops"]
        from vibehash import __version__
        self.assertEqual(ops["version"], __version__)
        self.assertGreaterEqual(ops["uptime_s"], 0)
        self.assertGreater(ops["disk"]["total_bytes"], 0)
        self.assertTrue(0 <= ops["disk"]["free_pct"] <= 100)
        self.assertIsInstance(ops["blobs_bytes"], int)
        self.assertEqual((ops["backup"], ops["backups_set_up"], ops["halted"]), (None, False, False))
        q = ops["quorum"]  # this chain has no validators: the node seals alone, so a restart stops blocks
        self.assertEqual((q["bft"], q["safe_to_restart"]), (False, False))
        self.backups.mkdir()
        (self.backups / "last_backup.json").write_text(json.dumps(
            {"timestamp": int(time.time()) - 7200, "chain_bytes": 1234, "remote_ok": True, "error": None}))
        (self.backups.parent / "health.json").write_text(json.dumps({"checked_at": 1, "height": 3, "tip_time": 1, "ok": True, "reason": None}))
        ops = self.call("/admin/overview", token=NODE_TOKEN)[1]["ops"]
        self.assertTrue(ops["backups_set_up"])
        self.assertEqual(ops["backup"]["chain_bytes"], 1234)
        self.assertAlmostEqual(ops["backup"]["age_hours"], 2.0, delta=0.1)
        self.assertTrue(ops["health"]["ok"])

    def test_operations_want_a_fresh_code(self):
        for op in ("prune", "backup", "restart"):
            code, out = self.call(f"/admin/{op}", {}, token=NODE_TOKEN)  # no authenticator key on this node: refused
            self.assertEqual(code, 401, op)
            self.assertIn(nodeadmin.NEED_FRESH, out["error"])
        signed = self.signed_in()
        for op in ("prune", "backup", "restart"):
            self.assertEqual(self.call(f"/admin/{op}", {}, token=NODE_TOKEN, headers=signed)[0], 401, op)  # a sign-in alone
            bad = "000000" if self.code() != "000000" else "111111"
            self.assertEqual(self.call(f"/admin/{op}", {}, token=NODE_TOKEN, headers={**signed, nodeadmin.CODE_HEADER: bad})[0], 401)
            self.assertEqual(self.call(f"/admin/{op}", {}, token="nope", headers={**signed, nodeadmin.CODE_HEADER: self.code()})[0], 401)
        used = self.code()
        fresh = {**signed, nodeadmin.CODE_HEADER: used}
        while self.node.coord.r is not None:  # (no job on this chain: no round opens)
            time.sleep(0.1)
        code, out = self.call("/admin/prune", {}, token=NODE_TOKEN, headers=fresh)
        self.assertEqual((code, out["ok"]), (200, True))
        self.assertEqual(self.call("/admin/prune", {}, token=NODE_TOKEN, headers=fresh)[0], 401)  # that code is used

    def test_a_backup_is_asked_of_the_backup_service(self):
        signed = self.signed_in()
        code, out = self.call("/admin/backup", {}, token=NODE_TOKEN, headers={**signed, nodeadmin.CODE_HEADER: self.code()})
        self.assertEqual(code, 409)  # no backup service on this server
        self.assertIn("setup-node.sh", out["error"])
        self.backups.mkdir()
        code, out = self.call("/admin/backup", {}, token=NODE_TOKEN, headers={**signed, nodeadmin.CODE_HEADER: self.code()})
        self.assertEqual((code, out["already"]), (200, False))
        self.assertEqual(json.loads((self.backups / "request").read_text())["requested_at"], out["requested_at"])
        code, out = self.call("/admin/backup", {}, token=NODE_TOKEN, headers={**signed, nodeadmin.CODE_HEADER: self.code()})
        self.assertTrue(out["already"])  # asked a moment ago and not begun: not asked again

    def test_a_restart_that_would_stop_the_chain_is_refused_unless_forced(self):
        signed = self.signed_in()
        with mock.patch.dict(os.environ, {"INVOCATION_ID": "", "VIBEHASH_SUPERVISED": ""}):
            code, out = self.call("/admin/restart", {}, token=NODE_TOKEN, headers={**signed, nodeadmin.CODE_HEADER: self.code()})
            self.assertEqual(code, 409)
            self.assertIn("systemd", out["error"])  # nothing would start it again
        with mock.patch.dict(os.environ, {"INVOCATION_ID": "abc"}):
            code, out = self.call("/admin/restart", {}, token=NODE_TOKEN, headers={**signed, nodeadmin.CODE_HEADER: self.code()})
            self.assertEqual(code, 409)  # it seals this chain's blocks alone
            self.assertIn("force", out["error"])
            self.assertIsNone(self.node.restart_requested)
            code, out = self.call("/admin/restart", {"force": True}, token=NODE_TOKEN,
                                  headers={**signed, nodeadmin.CODE_HEADER: self.code()})
            self.assertEqual((code, out["restarting"], out["safe"]), (200, True, False))
            self.assertIn("forced", self.node.restart_requested)


class QuorumTests(unittest.TestCase):
    """Whether finality survives a validator's restart, from who signed the last blocks' certificates."""

    def view(self, me, signed_by, stakes=None):
        keys = [KeyPair.generate(bytes([i])) for i in range(4)]
        vals = {k.address: (stakes or [100] * 4)[i] for i, k in enumerate(keys)}
        certs = {h: {"votes": [{"pubkey": keys[i].public.hex(), "sig": "00"} for i in signed_by]} for h in range(95, 101)}
        node = SimpleNamespace(
            chain=SimpleNamespace(tip=SimpleNamespace(height=100),
                                  state=SimpleNamespace(validator_set=lambda: vals, validators={a: None for a in vals})),
            store=SimpleNamespace(certificates=lambda lo, hi: {h: c for h, c in certs.items() if lo <= h <= hi}),
            keys=keys[me] if me is not None else KeyPair.generate(b"follower"))
        return nodeadmin.quorum_view(node), keys

    def test_four_equal_validators_survive_one_restart(self):
        q, _ = self.view(0, [0, 1, 2, 3])
        self.assertEqual((q["online"], q["online_pct"], q["safe_to_restart"]), (4, 100.0, True))

    def test_with_one_already_down_restarting_another_would_stop_blocks(self):
        q, keys = self.view(0, [0, 1, 2])
        self.assertTrue(q["quorum_now"])  # 3 of 4: blocks are final
        self.assertFalse(q["safe_to_restart"])  # 2 of 4 without this one: not more than two thirds
        self.assertIn("50%", q["why"])
        self.assertFalse(q["signers"][keys[3].address]["online"])
        self.assertEqual(q["signers"][keys[1].address]["last_signed"], 100)

    def test_two_validators_is_a_trap_and_a_follower_is_always_safe(self):
        q, _ = self.view(0, [0, 1], stakes=[100, 100, 0, 0])
        self.assertFalse(q["safe_to_restart"])
        q, _ = self.view(None, [0, 1, 2])
        self.assertTrue(q["safe_to_restart"])


class HaltExitTests(unittest.TestCase):
    """`vibehash node` ends with HALT_EXIT after a halt (systemd leaves it down: RestartPreventExitStatus), and with
    RESTART_EXIT after an admin's restart (systemd starts it again)."""

    def run_node(self, **state):
        from vibehash import cli
        from vibehash import node as node_module
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        genesis = Path(tmp.name) / "genesis.json"
        genesis.write_text(json.dumps({"chain_id": "x", "timestamp": 0, "allocations": {}}))
        fake = SimpleNamespace(**{"halted": None, "restart_requested": None, "stop": lambda: None, **state})
        fake.start = lambda: fake
        a = cli.parser().parse_args(["node", "--genesis", str(genesis), "--wallet", "w.json", "--data-dir", str(Path(tmp.name) / "d")])
        with mock.patch.object(node_module, "Node", lambda *args, **kw: fake), \
             mock.patch.object(cli, "load_wallet", lambda *args, **kw: KeyPair.generate(b"n")), \
             mock.patch("sys.stderr"), self.assertRaises(SystemExit) as ctx:
            cli.cmd_node(a)
        return ctx.exception.code

    def test_a_halt_exits_42_and_a_restart_75(self):
        from vibehash.node import HALT_EXIT, RESTART_EXIT
        self.assertEqual(HALT_EXIT, 42)
        self.assertEqual(self.run_node(halted="block 9 couldn't be saved"), HALT_EXIT)
        self.assertEqual(self.run_node(restart_requested="an admin asked"), RESTART_EXIT)

    def test_the_service_never_restarts_a_halted_node(self):
        unit = (ROOT / "deploy" / "setup-node.sh").read_text(encoding="utf-8")
        from vibehash.node import HALT_EXIT
        self.assertIn(f"RestartPreventExitStatus={HALT_EXIT}", unit)
        self.assertIn("--trust-proxy-dns", unit)


class GeoTests(unittest.TestCase):
    def test_places_are_looked_up_in_the_background_cached_and_never_for_private_addresses(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        asked = []

        def lookup(ips):
            asked.append(list(ips))
            return {ip: {"lat": 1.0, "lon": 2.0, "city": "Somewhere", "country": "Land"} if ip == "8.8.8.8" else None for ip in ips}

        path = Path(tmp.name) / "geo.json"
        g = admin.Geo(path, lookup=lookup)
        self.assertIsNone(g.place("10.0.0.1"))
        self.assertIsNone(g.place("127.0.0.1"))
        self.assertIsNone(g.place(None))
        self.assertIsNone(g.place("8.8.8.8"))  # queued
        g.place("203.0.113.9")  # a reserved range: never asked
        deadline = time.time() + 5
        while g.waiting() and time.time() < deadline:
            time.sleep(0.05)
        g.thread.join(5)
        self.assertEqual(asked, [["8.8.8.8"]])
        self.assertEqual(g.place("8.8.8.8")["city"], "Somewhere")
        self.assertEqual(asked, [["8.8.8.8"]])  # from the cache now
        again = admin.Geo(path, lookup=lookup)  # a restart reads the file
        self.assertEqual(again.place("8.8.8.8")["country"], "Land")
        self.assertEqual(asked, [["8.8.8.8"]])
        off = admin.Geo(Path(tmp.name) / "other.json", lookup=lookup, enabled=False)
        self.assertIsNone(off.place("1.1.1.1"))
        self.assertEqual(off.waiting(), 0)  # nothing queued: nothing leaves
        self.assertEqual(asked, [["8.8.8.8"]])  # and the fetcher was never called for it

    def test_a_failed_lookup_is_tried_again_next_time(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls = []

        def lookup(ips):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("down")
            return {ip: {"lat": 0.0, "lon": 0.0, "city": "", "country": "X"} for ip in ips}

        g = admin.Geo(Path(tmp.name) / "geo.json", lookup=lookup)
        g.place("8.8.8.8")
        g.thread.join(5)
        self.assertIsNone(g.place("8.8.8.8"))  # queued again
        g.thread.join(5)
        self.assertEqual(g.place("8.8.8.8")["country"], "X")

    @staticmethod
    def _resp(data):
        class Resp:
            def read(self, n=-1):
                return json.dumps(data).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Resp()

    def test_ip_api_answers_are_read_carefully(self):
        answers = [{"status": "success", "query": "8.8.8.8", "lat": 1, "lon": 2, "city": "A", "country": "B"},
                   {"status": "fail", "query": "1.1.1.1"}, {"status": "success", "query": "9.9.9.9", "lat": "bad"}, "junk"]
        with mock.patch.object(admin.urllib.request, "urlopen", return_value=self._resp(answers)):
            out = admin.lookup_batch("https://pro.ip-api.com/batch", ["8.8.8.8", "1.1.1.1", "9.9.9.9"])
        self.assertEqual(out, {"8.8.8.8": {"lat": 1.0, "lon": 2.0, "city": "A", "country": "B"}, "1.1.1.1": None, "9.9.9.9": None})

    def test_ipinfo_answers_are_read_carefully(self):
        answers = {"8.8.8.8": {"city": "Mountain View", "country": "US", "loc": "37.4,-122.1"},
                   "1.1.1.1": {"city": "X", "country": "AU"}, "9.9.9.9": "unknown", "5.5.5.5": {"loc": "bad"}}
        with mock.patch.object(admin.urllib.request, "urlopen", return_value=self._resp(answers)):
            out = admin.lookup_batch("https://ipinfo.io/batch", ["8.8.8.8", "1.1.1.1", "9.9.9.9", "5.5.5.5"])
        self.assertEqual(out, {"8.8.8.8": {"lat": 37.4, "lon": -122.1, "city": "Mountain View", "country": "US"},
                               "1.1.1.1": None, "9.9.9.9": None, "5.5.5.5": None})

    def test_the_token_is_sent_in_the_request_url(self):
        seen = []

        def urlopen(req, timeout=None):
            seen.append(req.full_url)
            return self._resp({})

        with mock.patch.object(admin.urllib.request, "urlopen", urlopen):
            admin.lookup_batch("https://ipinfo.io/batch", ["8.8.8.8"], token="SEKRET")
            admin.lookup_batch("https://ipinfo.io/batch", ["8.8.8.8"])
            admin.lookup_batch("https://pro.ip-api.com/batch?fields=query", ["8.8.8.8"], token="K")
        self.assertEqual(seen, ["https://ipinfo.io/batch?token=SEKRET", "https://ipinfo.io/batch",
                                "https://pro.ip-api.com/batch?fields=query&token=K"])

    def test_a_plain_http_service_is_refused(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "geo.json"
        with self.assertRaises(ValueError):
            admin.Geo(path, enabled=True, url="http://ip-api.com/batch")
        with self.assertRaises(ValueError):
            admin.lookup_batch("http://ip-api.com/batch", ["8.8.8.8"])
        admin.Geo(path, enabled=False, url="http://ip-api.com/batch")  # off: nothing is ever sent, so nothing to refuse
        admin.Geo(path, enabled=True, url="https://ipinfo.io/batch")

    def test_the_url_and_token_come_from_the_environment(self):
        with mock.patch.dict(admin.os.environ, {"VIBEHASH_GEO_URL": "https://pro.ip-api.com/batch", "VIBEHASH_GEO_TOKEN": "T"}):
            self.assertEqual(admin.geo_url(), "https://pro.ip-api.com/batch")
            self.assertEqual(admin.geo_token(), "T")
        with mock.patch.dict(admin.os.environ, {}, clear=True):
            self.assertEqual(admin.geo_url(), admin.GEO_URL)
            self.assertTrue(admin.GEO_URL.startswith("https://"))
            self.assertIsNone(admin.geo_token())

    def test_geo_is_off_by_default_and_under_a_proxy(self):
        class Args:
            pass

        self.assertFalse(admin.geo_wanted({}, Args(), {}))  # off unless turned on
        self.assertTrue(admin.geo_wanted({"geo": True}, Args(), {}))
        self.assertFalse(admin.geo_wanted({"geo": True}, Args(), {"proxy": "socks5://127.0.0.1:9050"}))
        tor = Args()
        tor.tor = True
        self.assertFalse(admin.geo_wanted({"geo": True}, tor, {}))
        proxied = Args()
        proxied.proxy = "socks5://x:1"
        self.assertFalse(admin.geo_wanted({"geo": True}, proxied, {}))

    def test_a_refused_service_turns_geo_off_instead_of_crashing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with self.assertLogs("vibehash.admin", "WARNING") as logs:
            g = admin.make_geo(Path(tmp.name) / "geo.json", True, url="http://ip-api.com/batch")
        self.assertFalse(g.enabled)
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(admin.make_geo(Path(tmp.name) / "geo.json", True, url="https://ipinfo.io/batch").enabled)


class FakeNode:
    """What a NodeClient answers, as a dict of path -> value (or an exception), and what it was sent."""

    def __init__(self, answers, token="t"):
        self.answers, self.token, self.sent = answers, token, []
        self.url = "https://node.test:8545"

    def _req(self, method, path, data=None, headers=None, limit=None):
        auth = (headers or {}).get("Authorization")
        if path.startswith("/admin/"):
            if auth != f"Bearer {self.token}":
                raise NodeError(f"{method} {path}: the admin token is missing or wrong", 401)
            self.sent.append((method, path, json.loads(data) if data else None))
        value = self.answers.get(path.split("?")[0])
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise NodeError(f"{method} {path}: not found", 404)
        return json.dumps(value).encode()

    def get(self, path):
        return json.loads(self._req("GET", path))

    def post(self, path, body):
        self.sent.append(("POST", path, body))
        return json.loads(self._req("POST", path, json.dumps(body).encode()))

    def account(self, addr):
        return {"address": addr, "balance": 7 * SYN_PER_VIBE, "stake": 0, "nonce": 0}

    def status(self):
        return self.answers["/status"]

    @property
    def chain_id(self):
        return "fake"

    def min_fee(self):
        return 0

    def send_tx(self, keys, kind, recipient, amount, memo="", fee=None):
        self.sent.append(("tx", kind, recipient, amount, memo, keys.address))
        return "ab" * 32


def overview_answer(**more):
    return {"now": 1000.0, "review": False, "node": {"url": "https://node.example.org", "address": "vibe1node", "version": "0.5.22",
                                                        "chain_id": "fake", "height": 12, "tip": "ff", "tip_time": 990.0,
                                                        "round": {"no": 13, "phase": "compute", "job_id": "j1", "registered": 2},
                                                        "halted": None, "validator": True, "supply": {"vibe": {"total": "10", "circulating": "9"}},
                                                        "subsidy": 50},
            "validators": {"vibe1node": {"url": "https://node.example.org", "stake": 10 * SYN_PER_VIBE, "votes": True}},
            "peers": [{"url": "http://peer:8545", "alive": False}], "mempool": [],
            "jobs": [{"id": "j1", "owner": "vibe1req", "starter": False, "escrow": 5 * SYN_PER_VIBE, "model": "m", "status": "training",
                      "fee_per_round": 1, "rounds_trained": 2, "units": 8, "metric": "accuracy", "start": 0.1, "now": 0.5,
                      "last_trained": None, "held": None, "policy": None, "waiting": None, "spec_bytes": 100}],
            "devices": [{"address": "vibe1w1", "ip": "8.8.8.8", "when": 999.0, "round": 13, "flops_per_sec": 1e12, "memory_mb": 8192,
                         "hardware": None, "online": True}],
            "miners": {"online": 1}, "hardware": {}} | more


class MfaNode(FakeNode):
    """A node started with an authenticator key: its overview wants a sign-in, which code 123456 gives."""

    def _req(self, method, path, data=None, headers=None, limit=None):
        headers = headers or {}
        if path.startswith("/admin/") and headers.get("Authorization") == f"Bearer {self.token}":
            if path == "/admin/login":
                if json.loads(data).get("code") != "123456":
                    raise NodeError(f"{method} {path}: the authenticator code is wrong", 401)
                return json.dumps({"ok": True, "mfa": True, "session": "sess", "expires": 2000.0}).encode()
            if headers.get(nodeadmin.LOGIN_HEADER) != "sess":
                raise NodeError(f"{method} {path}: {nodeadmin.NEED_CODE}", 401)
        return super()._req(method, path, data, headers, limit)


class FreshNode(MfaNode):
    """An MfaNode whose hold, release and review also want a fresh code (654321) in X-Admin-Code, as the real one's do."""

    def __init__(self, answers, token="t"):
        super().__init__(answers, token)
        self.codes = []

    def _req(self, method, path, data=None, headers=None, limit=None):
        headers = headers or {}
        if method == "POST" and (path == "/admin/review" or path.endswith(("/hold", "/release"))):
            if headers.get("Authorization") == f"Bearer {self.token}" and headers.get(nodeadmin.LOGIN_HEADER) == "sess":
                code = headers.get(nodeadmin.CODE_HEADER)
                if not code:
                    raise NodeError(f"{method} {path}: {nodeadmin.NEED_FRESH} ({nodeadmin.CODE_HEADER})", 401)
                if code != "654321":
                    raise NodeError(f"{method} {path}: the authenticator code is wrong", 401)
                self.codes.append(code)
        return super()._req(method, path, data, headers, limit)


class NodeStartTests(unittest.TestCase):
    """`vibehash node` refuses to serve the admin routes to the network without an authenticator key."""

    def start(self, *flags, host=None, mfa_env=None):
        from vibehash import cli
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        genesis = Path(tmp.name) / "genesis.json"
        genesis.write_text("{}")

        class Stop(Exception):
            pass

        made = []

        def fake(*a, **kw):
            made.append(kw)
            raise Stop
        args = ["node", "--genesis", str(genesis), "--wallet", "w", "--data-dir", tmp.name, *flags]
        if host:
            args += ["--host", host]
        env = {k: v for k, v in os.environ.items() if not k.startswith("VIBEHASH_ADMIN")}
        env.update(mfa_env or {})
        self.addCleanup(logging.disable, logging.NOTSET)
        logging.disable(logging.INFO)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("vibehash.node.Node", side_effect=fake), \
                mock.patch.object(cli, "load_wallet", return_value=KeyPair.generate(b"n")):
            try:
                cli.main(args)
            except Stop:
                return "started", made[0]
            except SystemExit as e:
                return str(e.code), None

    def test_a_public_bind_with_a_token_and_no_key_is_refused_naming_the_fix(self):
        for host in ("0.0.0.0", "::", "203.0.113.5", "10.0.0.7", "example.org"):
            out, _ = self.start("--admin-token", NODE_TOKEN, host=host)
            self.assertIn("--admin-mfa", out, host)
            self.assertIn("vibehash admin --new-mfa", out, host)
            self.assertIn("VIBEHASH_ADMIN_MFA", out, host)

    def test_loopback_binds_need_no_key(self):
        for host in (None, "127.0.0.1", "127.0.0.2", "::1", "localhost", "LOCALHOST"):
            self.assertEqual(self.start("--admin-token", NODE_TOKEN, host=host)[0], "started", host)

    def test_a_key_or_the_unsafe_flag_lets_a_public_bind_start(self):
        out, kw = self.start("--admin-token", NODE_TOKEN, "--admin-mfa", KEY, host="0.0.0.0")
        self.assertEqual((out, kw["admin_mfa"]), ("started", KEY))
        out, _ = self.start("--admin-token", NODE_TOKEN, host="0.0.0.0", mfa_env={"VIBEHASH_ADMIN_MFA": KEY})
        self.assertEqual(out, "started")
        out, kw = self.start("--admin-token", NODE_TOKEN, "--admin-allow-no-mfa", host="0.0.0.0")
        self.assertEqual((out, kw["admin_mfa"]), ("started", None))

    def test_a_public_bind_without_a_token_has_no_admin_to_protect(self):
        self.assertEqual(self.start(host="0.0.0.0")[0], "started")

    def test_the_flag_says_it_is_unsafe(self):
        from vibehash import cli
        text = Path(cli.__file__).read_text(encoding="utf-8")
        self.assertRegex(text, r"(?s)--admin-allow-no-mfa.{0,400}UNSAFE")


class AdminAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("VIBEHASH_ADMIN_TOKEN", None)
        self.req = KeyPair.generate(b"req")
        self.wallet = Path(self.tmp.name) / "wallet.json"
        wallets.save(self.req, self.wallet, None)
        self.fake = FakeNode({"/admin/overview": overview_answer(), "/blocks": [{"height": 12}], "/status": {"height": 12}})
        link = admin.NodeLink(self.fake.url, "t", client=self.fake)
        geo = admin.Geo(Path(self.tmp.name) / "geo.json", lookup=lambda ips: {ip: {"lat": 1.0, "lon": 2.0, "city": "C", "country": "K"} for ip in ips})
        self.app = admin.AdminApp([link], self.wallet, geo)

    def test_the_overview_merges_nodes_places_and_the_admin_wallet(self):
        d = self.app.overview()
        self.assertEqual(d["primary"], self.fake.url)
        self.assertEqual(d["nodes"][0]["admin"], True)
        self.assertEqual(d["nodes"][0]["height"], 12)
        self.assertEqual(d["nodes"][0]["announce"], "https://node.example.org")
        self.assertEqual(d["jobs"][0]["held_on"], [])
        self.assertEqual(d["devices"][0]["node"], self.fake.url)
        self.app.geo.thread.join(5)
        d = self.app.overview()
        self.assertEqual(d["devices"][0]["place"]["city"], "C")
        self.assertEqual(d["me"]["address"], self.req.address)
        self.assertEqual(d["me"]["balance"], 7 * SYN_PER_VIBE)
        self.assertFalse(d["me"]["unlocked"])
        self.assertEqual(d["settings"], {"nodes": [{"url": self.fake.url, "token": True}], "wallet": str(self.wallet), "geo": True})

    def test_the_token_never_crosses_the_internet_in_the_clear(self):
        for url, ok in (("https://node.example.org", True), ("http://127.0.0.1:8545", True), ("http://localhost:8545", True),
                        ("http://10.0.0.5:8545", True), ("http://[::1]:8545", True), ("http://100.101.102.103:8545", True),
                        ("http://node.example.org", False), ("http://93.184.216.34:8545", False)):
            self.assertEqual(admin.NodeLink(url, "t").secure, ok, url)
        fake = FakeNode({"/admin/overview": overview_answer(), "/status": {"height": 3}, "/network": {"jobs": []}, "/validators": {}})
        link = admin.NodeLink("http://node.example.org:8545", "t", client=fake)
        v = link.overview()
        self.assertEqual((v["admin"], v["error"]), (False, admin.INSECURE))
        self.assertEqual(v["node"]["height"], 3)  # its public side still shows
        with self.assertRaisesRegex(admin.AdminError, "https"):
            link.admin_post("/admin/review", {"on": True})
        self.assertEqual(fake.sent, [])  # nothing with the token went out
        app = admin.AdminApp([link], None, self.app.geo)
        self.assertEqual(app.review(True)["nodes"][0], {"url": link.url, "ok": False, "error": admin.INSECURE})

    def test_a_refused_token_isnt_sent_on_every_refresh(self):
        """Each refusal is a strike against this computer's address: a stale token polled every few seconds would lock
        the admin out within a minute, so a refused node is asked again only every RETRY_REFUSED_S."""
        fake = FakeNode({"/status": {"height": 3}, "/network": {"jobs": []}, "/validators": {}}, token="other")
        asked = []
        real = fake._req
        fake._req = lambda method, path, *a, **k: (asked.append(path) if path.startswith("/admin/") else None) or real(method, path, *a, **k)
        link = admin.NodeLink(fake.url, "t", client=fake)
        for _ in range(5):
            v = link.overview()
            self.assertIn("token is missing or wrong", v["error"])
            self.assertEqual(v["node"]["height"], 3)
        self.assertEqual(asked, ["/admin/overview"])
        link.refused_at -= admin.RETRY_REFUSED_S
        link.overview()
        self.assertEqual(len(asked), 2)
        locked = FakeNode({"/admin/overview": NodeError("GET /admin/overview: too many wrong tries from this address: try "
                                                        "again in 9 minutes", 403), "/status": {"height": 4}, "/network": {"jobs": []},
                           "/validators": {}})
        v = admin.NodeLink(locked.url, "t", client=locked).overview()
        self.assertIn("9 minutes", v["error"])
        self.assertEqual(v["node"]["height"], 4)  # still its public side, not "unreachable"

    def test_a_redirect_never_takes_the_token_along(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        seen = []

        class Target(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                body = b'{"height": 9}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        target = ThreadingHTTPServer(("127.0.0.1", 0), Target)

        class Redirector(Target):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_address[1]}{self.path}")
                self.send_header("Content-Length", "0")
                self.end_headers()

        hop = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
        for s in (target, hop):
            threading.Thread(target=s.serve_forever, daemon=True).start()
            self.addCleanup(s.server_close)
            self.addCleanup(s.shutdown)
        link = admin.NodeLink(f"http://127.0.0.1:{hop.server_address[1]}", "t")
        with self.assertRaisesRegex(NodeError, "not followed"):
            link.admin_get("/admin/overview")
        self.assertEqual(seen, [])  # the token went nowhere
        self.assertEqual(link.client.get("/status"), {"height": 9})  # a public route still follows it
        self.assertEqual(seen, [("/status", None)])

    def test_a_node_that_wants_a_code_is_signed_in_with_one(self):
        node = MfaNode({"/admin/overview": overview_answer(), "/admin/jobs/j1/hold": {"ok": True}, "/status": {"height": 3},
                        "/network": {"jobs": []}, "/validators": {}})
        app = admin.AdminApp([admin.NodeLink(node.url, "t", client=node)], None, self.app.geo)
        d = app.overview()
        self.assertEqual((d["nodes"][0]["admin"], d["nodes"][0]["mfa"], d["nodes"][0]["error"]), (False, True, None))
        self.assertEqual(d["nodes"][0]["height"], 3)
        out = app.sign_in("654321")
        self.assertFalse(out["ok"])
        self.assertIn("code is wrong", out["nodes"][0]["error"])
        self.assertTrue(app.sign_in(" 123456 ")["ok"])
        d = app.overview()
        self.assertEqual((d["nodes"][0]["admin"], d["nodes"][0]["mfa"], d["nodes"][0]["login_until"]), (True, False, 2000.0))
        self.assertEqual(app.hold("j1", "x")["nodes"][0]["ok"], True)  # orders carry the sign-in
        with self.assertRaisesRegex(admin.AdminError, "no node is asking"):
            app.sign_in("123456")
        app.links[0].login = "over"  # the node forgot it (a restart, or 12 hours on)
        self.assertTrue(app.overview()["nodes"][0]["mfa"])
        self.assertIsNone(app.links[0].login)

    def test_new_token_and_new_mfa_print_what_a_node_takes(self):
        import contextlib
        import io
        for flag in ("new_token", "new_mfa"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                admin.main(SimpleNamespace(**{flag: True}))
            printed = out.getvalue()
            if flag == "new_token":
                nodeadmin.check_token(printed.strip())
                self.assertGreaterEqual(len(printed.strip()), nodeadmin.MIN_TOKEN)
            else:
                key = printed.split("secret=")[1].split("&")[0]
                self.assertEqual(len(nodeadmin.mfa_key(key)), 20)
                self.assertIn("VIBEHASH_ADMIN_MFA=" + key, printed)

    def test_a_node_without_admin_routes_shows_its_public_side(self):
        fake = FakeNode({"/status": {"height": 3, "version": "0.5.20", "chain_id": "fake", "round": None, "phase": "idle", "waiting": {"j1": "w"}},
                         "/network": {"time": 1.0, "jobs": [{"id": "j1", "status": "open", "escrow": 1}], "miners": {"online": 0}},
                         "/validators": {}}, token="other")
        link = admin.NodeLink(fake.url, "t", client=fake)  # the wrong token: 401, so the public routes
        v = link.overview()
        self.assertFalse(v["admin"])
        self.assertIn("token", v["error"])
        self.assertEqual(v["node"]["height"], 3)
        self.assertEqual(v["jobs"][0]["waiting"], "w")
        self.assertEqual(admin.NodeLink(fake.url, None, client=fake).overview()["error"], None)
        down = FakeNode({"/admin/overview": NodeError("can't reach node", None)})
        self.assertIn("can't reach", admin.NodeLink(down.url, "t", client=down).overview()["error"])
        garbled = FakeNode({"/admin/overview": "<html>a proxy's error page</html>"})  # json.loads of a str answer still parses...
        garbled._req = lambda *a, **k: b"<html>not json</html>"  # ...so give it bytes that don't
        v = admin.NodeLink(garbled.url, "t", client=garbled).overview()
        self.assertFalse(v["admin"])
        self.assertIn("wasn't JSON", v["error"])
        app = admin.AdminApp([admin.NodeLink(garbled.url, "t", client=garbled), admin.NodeLink(self.fake.url, "t", client=self.fake)],
                             None, self.app.geo)
        d = app.overview()  # one garbled node is its own error, not the view's
        self.assertEqual([n["error"] is None for n in d["nodes"]], [False, True])
        self.fake.answers["/admin/jobs/j1/hold"] = {"ok": True}
        self.assertEqual([x["ok"] for x in app.hold("j1")["nodes"]], [False, True])

    def test_hold_release_and_review_carry_a_fresh_code_to_the_nodes_that_want_one(self):
        node = FreshNode({"/admin/jobs/j1/hold": {"ok": True}, "/admin/jobs/j1/release": {"ok": True},
                          "/admin/review": {"ok": True}, "/admin/overview": overview_answer()})
        plain = FakeNode({"/admin/jobs/j1/hold": {"ok": True}})
        plain.url = "https://plain.test:8545"
        app = admin.AdminApp([admin.NodeLink(node.url, "t", client=node), admin.NodeLink(plain.url, "t", client=plain)],
                             None, self.app.geo)
        app.links[0].sign_in("123456")
        out = app.hold("j1", "x")  # no code: the node that wants one says so, the other doesn't mind
        self.assertEqual([x["ok"] for x in out["nodes"]], [False, True])
        self.assertIn(nodeadmin.NEED_FRESH, out["nodes"][0]["error"])
        self.assertEqual(node.codes, [])
        out = app.hold("j1", "x", code=" 654321 ")
        self.assertEqual([x["ok"] for x in out["nodes"]], [True, True])
        self.assertEqual(node.codes, ["654321"])  # trimmed, in X-Admin-Code
        self.assertTrue(app.release("j1", code="654321")["nodes"][0]["ok"])
        self.assertTrue(app.review(True, code="654321")["nodes"][0]["ok"])
        self.assertEqual(node.codes, ["654321"] * 3)
        only = app.hold("j1", "x", code="654321", urls=[node.url])  # a retry for the nodes that refused
        self.assertEqual([x["url"] for x in only["nodes"]], [node.url])
        with self.assertRaisesRegex(admin.AdminError, "6-digit"):
            app.hold("j1", "x", code="12ab")  # (a code that can't be one is not sent)

    def test_the_apis_hold_takes_a_code_from_the_page(self):
        node = FreshNode({"/admin/jobs/j1/hold": {"ok": True}, "/admin/overview": overview_answer()})
        app = admin.AdminApp([admin.NodeLink(node.url, "t", client=node)], None, self.app.geo)
        app.links[0].sign_in("123456")
        self.addCleanup(app.stop)
        port = app.serve(0)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/jobs/j1/hold", method="POST",
                                     data=json.dumps({"why": "x", "code": "654321"}).encode(),
                                     headers={"X-Admin-Session": app.session, "Origin": f"http://127.0.0.1:{port}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertTrue(json.loads(r.read())["nodes"][0]["ok"])
        self.assertEqual(node.codes, ["654321"])

    def test_the_page_asks_for_a_code_on_those_three_orders_and_asks_again_once(self):
        html = (Path(admin.__file__).parent / "data" / "admin.html").read_text(encoding="utf-8")
        self.assertIn(nodeadmin.NEED_FRESH, html)  # the words the page looks for in a node's refusal
        self.assertIn("fresh code", html)

    def test_orders_go_to_every_node_with_a_token_and_transactions_sign_with_the_wallet(self):
        self.fake.answers["/admin/jobs/j1/hold"] = {"ok": True, "job_id": "j1", "held": {"why": "why", "when": 1.0}}
        self.assertEqual(self.app.hold("j1", "why")["nodes"], [{"url": self.fake.url, "ok": True, "result": self.fake.answers["/admin/jobs/j1/hold"]}])
        self.assertIn(("POST", "/admin/jobs/j1/hold", {"why": "why"}), self.fake.sent)
        self.fake.answers["/admin/jobs/j1/release"] = self.fake.answers["/admin/review"] = {"ok": True}
        self.app.release("j1")
        self.assertIn(("POST", "/admin/jobs/j1/release", {}), self.fake.sent)
        self.app.review(True)
        self.assertIn(("POST", "/admin/review", {"on": True}), self.fake.sent)
        with self.assertRaises(admin.AdminError):
            self.app.cancel("j1")  # locked
        self.assertEqual(self.app.unlock("")["address"], self.req.address)
        self.assertEqual(self.app.cancel("j1")["txid"], "ab" * 32)
        self.assertIn(("tx", "cancel_job", "system", 0, "j1", self.req.address), self.fake.sent)
        spec = template_job("j2", False).to_spec()
        out = self.app.post_job(spec, "12.5")
        self.assertEqual(out["job_id"], "j2")
        self.assertEqual([s for s in self.fake.sent if s[0] == "tx"][-1][1:4], ("post_job", "system", int(12.5 * SYN_PER_VIBE)))
        with self.assertRaises(admin.AdminError):
            self.app.post_job({"job_id": "bad id!"}, 1)
        with self.assertRaises(admin.AdminError):
            self.app.post_job(spec, "0")
        self.assertEqual(self.app.template()["id"], "job-1")
        no = admin.AdminApp([admin.NodeLink("http://x", None, client=self.fake)], None, self.app.geo)
        with self.assertRaises(admin.AdminError):
            no.hold("j1")

    def test_an_operation_goes_to_one_node_with_a_fresh_code(self):
        seen = []

        class OpsNode(FakeNode):
            def _req(self, method, path, data=None, headers=None, limit=None):
                if path.startswith("/admin/") and path != "/admin/overview":
                    seen.append((self.url, path, (headers or {}).get(nodeadmin.CODE_HEADER), json.loads(data)))
                    if path == "/admin/restart" and not json.loads(data).get("force"):
                        raise NodeError(f"{method} {path}: blocks stop until it's back. Send force to restart it anyway", 409)
                return super()._req(method, path, data, headers, limit)

        one, two = OpsNode({"/admin/prune": {"ok": True, "removed": 3, "freed_bytes": 10},
                            "/admin/restart": {"ok": True, "restarting": True, "safe": False}}), OpsNode({})
        two.url = "https://two.test:8545"
        app = admin.AdminApp([admin.NodeLink(one.url, "t", client=one), admin.NodeLink(two.url, "t", client=two)],
                             None, admin.Geo(Path(self.tmp.name) / "geo.json", lookup=lambda ips: {}))
        out = app.operate(one.url, "prune", " 123456 ")
        self.assertTrue(out["ok"])
        self.assertEqual(seen, [(one.url, "/admin/prune", "123456", {})])  # that node only, with the code
        out = app.operate(one.url, "restart", "234567")
        self.assertEqual((out["ok"], out["status"]), (False, 409))  # the page offers to force it
        out = app.operate(one.url, "restart", "345678", force=True)
        self.assertTrue(out["ok"])
        self.assertEqual(seen[-1][3], {"force": True})
        for bad in [(one.url, "format", "123456"), (one.url, "prune", "12345"), ("https://elsewhere", "prune", "123456")]:
            with self.assertRaises(admin.AdminError):
                app.operate(*bad)
        self.assertEqual(len(seen), 3)

    def test_settings_keep_a_saved_token_when_the_page_leaves_it_empty(self):
        saved = {}
        self.app.save = lambda **kw: saved.update(kw)
        out = self.app.set_settings([{"url": self.fake.url, "token": ""}, {"url": "10.0.0.5:8545", "token": "new"}], "", False)
        self.assertEqual(out["settings"]["nodes"], [{"url": self.fake.url, "token": True}, {"url": "http://10.0.0.5:8545", "token": True}])
        self.assertEqual(self.app.links[0].token, "t")
        self.assertEqual(self.app.links[1].token, "new")
        self.assertIsNone(self.app.wallet)
        self.assertFalse(self.app.geo.enabled)
        self.app.set_settings([], str(self.wallet), True)
        self.app.unlock("")
        self.app.set_settings([], str(self.wallet), False)  # the same wallet: stays unlocked
        self.assertIsNotNone(self.app.keys)
        self.app.set_settings([], str(self.wallet) + ".other", False)
        self.assertIsNone(self.app.keys)
        self.assertEqual(saved["admin"]["nodes"][1], {"url": "http://10.0.0.5:8545", "token": "new"})

    def test_links_come_from_flags_the_saved_settings_or_the_chosen_node(self):
        cfg = {"nodes": [{"url": "http://a:1/", "token": "ta"}, {"url": "http://b:1"}]}
        links = admin.links_from(cfg, ["http://c:1"], "tc")
        self.assertEqual([(l.url, l.token) for l in links], [("http://c:1", "tc"), ("http://a:1", "ta"), ("http://b:1", None)])
        links = admin.links_from(cfg, ["http://a:1"])  # the saved token for a flagged node
        self.assertEqual((links[0].url, links[0].token), ("http://a:1", "ta"))
        with mock.patch.dict(os.environ, {"VIBEHASH_ADMIN_TOKEN": "env"}):
            links = admin.links_from({}, [])
            self.assertEqual([(l.url, l.token) for l in links], [(settings.node_url().rstrip("/"), "env")])
            links = admin.links_from(cfg, ["http://c:1"])  # the environment's token goes to the node named, never to a saved one without
            self.assertEqual([(l.url, l.token) for l in links], [("http://c:1", "env"), ("http://a:1", "ta"), ("http://b:1", None)])

    def test_the_page_opens_the_docs_by_name_only(self):
        port = self.app.serve(0)
        self.addCleanup(self.app.stop)

        def post(body, session):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/open", data=json.dumps(body).encode(), method="POST",
                                         headers={"X-Admin-Session": session})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        opened = []
        with mock.patch.object(admin, "OPEN", side_effect=opened.append):
            self.assertEqual(post({"name": "docs"}, self.app.session), (200, {"ok": True, "url": admin.LINKS["docs"]}))
            for bad in ("https://evil.example", "javascript:alert(1)", None, ["docs"]):
                self.assertEqual(post({"name": bad}, self.app.session)[0], 400, bad)
            self.assertEqual(post({"name": "docs"}, "wrong")[0], 403)  # only this page's session
        self.assertEqual(opened, [admin.LINKS["docs"]])
        self.assertTrue(all(u.startswith("https://docs.vibehash.org") for u in admin.LINKS.values()))
        self.assertIn('id="open-docs"', admin.PAGE.read_text(encoding="utf-8"))

    def test_the_server_serves_the_page_and_guards_the_api_with_the_session(self):
        port = self.app.serve(0)
        self.addCleanup(self.app.stop)
        base = f"http://127.0.0.1:{port}"

        def call(path, body=None, session=None, origin=None, host=None):
            headers = {}
            if session:
                headers["X-Admin-Session"] = session
            if origin:
                headers["Origin"] = origin
            if host:
                headers["Host"] = host
            req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         method="POST" if body is not None else "GET", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()

        code, page = call("/")
        self.assertEqual(code, 200)
        self.assertIn(b"window.ADMIN = ", page)
        self.assertIn(self.app.session.encode(), page)
        self.assertIn(b'id="globe"', page)
        code, dots = call("/globe.json")
        self.assertEqual(code, 200)
        self.assertGreater(len(json.loads(dots)["rows"]), 100)
        self.assertEqual(call("/api/overview")[0], 403)
        self.assertEqual(call("/api/overview", session="wrong")[0], 403)
        self.assertEqual(call("/api/overview", session=self.app.session, origin="http://evil.test")[0], 403)
        # DNS rebinding: a page on evil.test pointed at 127.0.0.1 asks with its own name as Host, and gets nothing
        self.assertEqual(call("/", host=f"evil.test:{port}")[0], 403)
        self.assertEqual(call("/globe.json", host="evil.test")[0], 403)
        self.assertEqual(call("/api/overview", session=self.app.session, host=f"evil.test:{port}", origin=f"http://evil.test:{port}")[0], 403)
        self.assertEqual(call("/api/jobs/j1/hold", {"why": "x"}, session=self.app.session, host=f"evil.test:{port}")[0], 403)
        self.assertEqual(call("/", host=f"localhost:{port}")[0], 200)
        self.assertEqual(call("/api/overview", session=self.app.session, host=f"LOCALHOST:{port}", origin=f"http://localhost:{port}")[0], 200)
        self.assertIn(b"frame-ancestors 'none'", urllib.request.urlopen(base + "/").headers.get("Content-Security-Policy", "").encode())
        self.assertEqual(urllib.request.urlopen(base + "/").headers.get("X-Frame-Options"), "DENY")
        code, body = call("/api/overview", session=self.app.session)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["primary"], self.fake.url)
        code, body = call("/api/blocks?from=0&to=12", session=self.app.session)
        self.assertEqual((code, json.loads(body)), (200, [{"height": 12}]))
        self.assertEqual(call("/api/jobs/j1/hold", {"why": "x"})[0], 403)
        code, body = call("/api/jobs/j1/hold", {"why": "x"}, session=self.app.session)
        self.assertEqual(code, 200)
        self.assertIn(("POST", "/admin/jobs/j1/hold", {"why": "x"}), self.fake.sent)
        self.assertEqual(call("/api/jobs/not%20a%20job/hold", {}, session=self.app.session)[0], 400)
        code, body = call("/api/unlock", {"password": ""}, session=self.app.session)
        self.assertEqual(json.loads(body)["address"], self.req.address)
        self.assertEqual(call("/api/nothing", session=self.app.session)[0], 404)
        self.assertEqual(call("/nothing")[0], 404)

    def test_the_window_falls_back_to_the_browser(self):
        opened, said = [], []
        with mock.patch.dict(sys.modules, {"webview": None}):
            self.assertEqual(admin.show("http://127.0.0.1:1/", fallback=opened.append, say=said.append), "browser")
        self.assertEqual(opened, ["http://127.0.0.1:1/"])
        self.assertIn("pywebview", said[0])
        wv = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"webview": wv}):
            self.assertEqual(admin.show("http://127.0.0.1:1/", fallback=opened.append), "window")
        self.assertEqual(wv.create_window.call_args.kwargs["url"], "http://127.0.0.1:1/")
        self.assertEqual(admin.show("http://127.0.0.1:2/", browser=True, fallback=opened.append), "browser")
        self.assertEqual(opened[-1], "http://127.0.0.1:2/")


class ShortcutTests(unittest.TestCase):
    """`vibehash admin --install-shortcut`: the admin app kept as an app, never on this machine's real menus here."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(self.home / ".vibehash")})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("XDG_DATA_HOME", None)

    def test_it_runs_this_python_without_a_console_on_windows(self):
        (self.home / "python.exe").write_text("")
        self.assertEqual(admin.shortcut_command(str(self.home / "python.exe"))[1:], ["-m", "vibehash", "admin"])
        self.assertTrue(admin.shortcut_command(str(self.home / "python.exe"))[0].endswith("python.exe"))  # no pythonw beside it
        (self.home / "pythonw.exe").write_text("")
        self.assertEqual(admin.shortcut_command(str(self.home / "python.exe"))[0], str(self.home / "pythonw.exe"))
        self.assertEqual(admin.shortcut_command("/usr/bin/python3")[0], "/usr/bin/python3")

    def test_windows_makes_a_start_menu_link_with_the_admin_icon(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="C:\\Users\\x\\Start Menu\\Programs\\Vibe Hash Admin.lnk\n", stderr="")

        where = admin.install_shortcut("win32", run=run, home=self.home, python="C:\\it's\\pythonw.exe")
        self.assertTrue(str(where).endswith("Vibe Hash Admin.lnk"))  # (a Windows path: on Linux it has no "name" to split)
        script = calls[0][-1]
        self.assertEqual(calls[0][:2], ["powershell", "-NoProfile"])
        self.assertIn("GetFolderPath('Programs')", script)
        self.assertIn("$s.TargetPath = 'C:\\it''s\\pythonw.exe'", script)  # a quote in a path is escaped
        self.assertIn("$s.Arguments = '-m vibehash admin'", script)
        self.assertIn(str(admin.ICON) + ",0", script)
        with self.assertRaises(admin.AdminError):
            admin.install_shortcut("win32", run=lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="denied"),
                                   home=self.home)
        gone = admin.remove_shortcut("win32", run=lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="C:\\x\\Vibe Hash Admin.lnk\n", stderr=""))
        self.assertTrue(str(gone).endswith("Vibe Hash Admin.lnk"))
        self.assertIsNone(admin.remove_shortcut("win32", run=lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr="")))

    def test_linux_writes_a_desktop_entry_and_takes_it_away(self):
        where = admin.install_shortcut("linux", home=self.home, python="/opt/my env/bin/python3")
        self.assertEqual(where, self.home / ".local" / "share" / "applications" / "vibehash-admin.desktop")
        text = where.read_text(encoding="utf-8")
        self.assertIn("Name=Vibe Hash Admin", text)
        self.assertIn("Exec='/opt/my env/bin/python3' -m vibehash admin", text)
        self.assertIn(f"Icon={admin.ICON_PNG}", text)
        self.assertEqual(admin.remove_shortcut("linux", home=self.home), where)
        self.assertFalse(where.exists())
        self.assertIsNone(admin.remove_shortcut("linux", home=self.home))

    def test_a_mac_gets_an_app_in_its_applications_folder(self):
        app = admin.install_shortcut("darwin", home=self.home, python="/usr/local/bin/python3")
        self.assertEqual(app, self.home / "Applications" / "Vibe Hash Admin.app")
        program = app / "Contents" / "MacOS" / "admin"
        self.assertEqual(program.read_text(encoding="utf-8"), "#!/bin/sh\nexec /usr/local/bin/python3 -m vibehash admin\n")
        self.assertIn("<string>admin</string>", (app / "Contents" / "Info.plist").read_text(encoding="utf-8"))
        self.assertEqual(admin.remove_shortcut("darwin", home=self.home), app)
        self.assertFalse(app.exists())

    def test_flags_given_with_it_are_saved_for_it_to_open_with(self):
        store = {"admin": {"nodes": [{"url": "http://old:1", "token": "keep"}, {"url": "http://b:1"}], "geo": False}}
        save = lambda **kw: store.update(kw)  # noqa: E731
        load = lambda: store  # noqa: E731
        self.assertFalse(admin.save_flags(SimpleNamespace(node=None, token="t", wallet=None), save, load))
        a = SimpleNamespace(node=["http://b:1/", "http://new:1"], token="tok", wallet=str(self.home / "w.json"))
        self.assertTrue(admin.save_flags(a, save, load))
        self.assertEqual(store["admin"]["nodes"], [{"url": "http://b:1", "token": "tok"}, {"url": "http://new:1", "token": "tok"},
                                                   {"url": "http://old:1", "token": "keep"}])
        self.assertEqual(store["admin"]["wallet"], str((self.home / "w.json").resolve()))
        self.assertFalse(store["admin"]["geo"])  # the rest kept

    def test_the_command_installs_or_removes_and_opens_nothing(self):
        with mock.patch.object(admin, "install_shortcut", return_value=Path("X.lnk")) as put, \
                mock.patch.object(admin, "build") as build:
            admin.main(SimpleNamespace(install_shortcut=True, remove_shortcut=False, node=None, token=None, wallet=None))
        put.assert_called_once()
        build.assert_not_called()
        with mock.patch.object(admin, "remove_shortcut", return_value=None) as take, mock.patch.object(admin, "build") as build:
            admin.main(SimpleNamespace(install_shortcut=False, remove_shortcut=True))
        take.assert_called_once()
        build.assert_not_called()

    def test_a_taken_port_means_another_one(self):
        app = mock.MagicMock()
        app.serve.side_effect = [OSError("in use"), 51234]
        app.links = []
        with mock.patch.object(admin, "build", return_value=app), mock.patch.object(admin, "show", return_value="window") as show:
            admin.main(SimpleNamespace(port=admin.DEFAULT_PORT, browser=False))
        self.assertEqual(app.serve.call_args_list[1].args, (0,))
        self.assertEqual(show.call_args.args[0], "http://127.0.0.1:51234/")

    def test_the_window_gets_its_icon_and_its_own_taskbar_button(self):
        wv = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"webview": wv}), mock.patch.object(admin, "own_taskbar_button") as own:
            self.assertEqual(admin.show("http://127.0.0.1:1/"), "window")
        own.assert_called_once()
        self.assertEqual(wv.start.call_args.kwargs["icon"], str(admin.window_icon()))
        self.assertFalse(admin.own_taskbar_button("linux"))
        # WinForms' Icon takes only an .ico: a PNG there killed the process from inside .NET
        self.assertEqual(admin.window_icon("win32"), admin.ICON)
        self.assertEqual(admin.window_icon("linux"), admin.ICON_PNG)
        self.assertEqual(admin.ICON.read_bytes()[:4], b"\x00\x00\x01\x00")  # an .ico header

    def test_the_icons_ship_with_the_package_and_are_the_scripts(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_icon
        self.assertEqual(admin.ICON.read_bytes(), make_icon.ico(ground=make_icon.VIOLET))
        self.assertEqual(admin.ICON_PNG.read_bytes(), make_icon.png(256, make_icon.VIOLET))
        self.assertIn('"data/*.ico"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class GlobeTests(unittest.TestCase):
    def test_dots_fall_only_on_land(self):
        square = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]  # (lon, lat)
        hole = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
        rows = make_globe.dots([square, hole], step=1.0)
        self.assertTrue(rows)
        for lat, lons in rows:
            self.assertTrue(-100 <= lat <= 100)
            for lon in lons:
                self.assertTrue(-100 <= lon <= 100)
                self.assertFalse(-19 <= lat <= 19 and -19 <= lon <= 19, "a dot in the lake")  # (a dot may sit on its shore)
        self.assertGreater(sum(len(l) for _, l in rows), 300)

    def test_the_shipped_file_is_what_the_script_makes_of_its_input(self):
        g = json.loads((ROOT / "vibehash" / "data" / "globe.json").read_text(encoding="utf-8"))
        self.assertEqual(g["step"], make_globe.STEP)
        self.assertGreater(sum(len(r[1]) for r in g["rows"]), 5000)
        self.assertTrue(all(-900 <= r[0] <= 900 for r in g["rows"]))


if __name__ == "__main__":
    unittest.main()
