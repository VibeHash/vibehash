"""Tor / SOCKS5h support: the proxy URL rules, how an opener gets the proxy, and the split between the control plane
(JSON, always proxied) and the data plane (big transfers: direct under control-only, refused under strict)."""
import logging
import pathlib
import socket
import threading
import unittest
import urllib.request
from types import SimpleNamespace
from unittest import mock

from vibehash import cli, hub, proxy, settings, tls
from vibehash.client import BLOB_DIRECT_BYTES, NodeClient, NodeError

TOR = "socks5h://127.0.0.1:9050"
MB = 1024 * 1024


def args(**kw):
    return SimpleNamespace(**{"tor": False, "proxy": None, "tor_mode": None, **kw})


class CheckTests(unittest.TestCase):
    def test_socks5h_and_socks4a_pass(self):
        self.assertEqual(proxy.check(TOR), TOR)
        self.assertEqual(proxy.check("socks4a://127.0.0.1:9050"), "socks4a://127.0.0.1:9050")

    def test_local_dns_schemes_are_refused_with_the_leak_message(self):
        for bad in ("socks5://127.0.0.1:9050", "socks4://127.0.0.1:9050", "http://127.0.0.1:8118"):
            with self.assertRaises(ValueError) as cm:
                proxy.check(bad)
            self.assertIn("DNS", str(cm.exception))
        with self.assertRaisesRegex(ValueError, "socks5h"):
            proxy.check("socks5://127.0.0.1:9050")

    def test_it_needs_a_host_and_a_port(self):
        for bad in ("socks5h://", "socks5h://127.0.0.1", "127.0.0.1:9050"):
            with self.assertRaises(ValueError):
                proxy.check(bad)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        proxy.forget_tor()
        self.addCleanup(proxy.forget_tor)

    def test_tor_takes_9050_when_it_listens(self):
        with mock.patch.object(proxy, "_listening", lambda port: port in (9050, 9150)):
            self.assertEqual(proxy.resolve(args(tor=True)), TOR)

    def test_tor_falls_back_to_tor_browser(self):
        with mock.patch.object(proxy, "_listening", lambda port: port == 9150):
            self.assertEqual(proxy.resolve(args(tor=True)), "socks5h://127.0.0.1:9150")

    def test_tor_with_nothing_listening_still_names_9050(self):
        with mock.patch.object(proxy, "_listening", lambda port: False):
            self.assertEqual(proxy.resolve(args(tor=True)), TOR)

    def test_proxy_flag_wins_over_tor_and_settings(self):
        with mock.patch.object(proxy, "_listening", lambda port: True):
            got = proxy.resolve(args(tor=True, proxy="socks5h://10.0.0.1:1080"), {"proxy": "socks5h://x:1"})
        self.assertEqual(got, "socks5h://10.0.0.1:1080")

    def test_settings_then_nothing(self):
        self.assertEqual(proxy.resolve(args(), {"proxy": "socks5h://h:1"}), "socks5h://h:1")
        self.assertIsNone(proxy.resolve(args(), {}))
        self.assertIsNone(proxy.resolve())

    def test_a_bad_proxy_flag_is_refused(self):
        with self.assertRaises(ValueError):
            proxy.resolve(args(proxy="socks5://127.0.0.1:9050"))


class ModeTests(unittest.TestCase):
    def test_defaults_by_command(self):
        cfg = {"proxy": TOR}
        for cmd in ("wallet", "send", "stake", "unstake", "job post"):
            self.assertEqual(proxy.tor_mode(args(), cfg, cmd), "strict", cmd)
        for cmd in ("worker", "app", "node", "admin"):
            self.assertEqual(proxy.tor_mode(args(), cfg, cmd), "control-only", cmd)

    def test_no_proxy_no_mode(self):
        self.assertIsNone(proxy.tor_mode(args(), {}, "worker"))

    def test_the_flag_wins_and_a_flag_counts_as_a_proxy(self):
        self.assertEqual(proxy.tor_mode(args(tor=True, tor_mode="strict"), {}, "worker"), "strict")
        self.assertEqual(proxy.tor_mode(args(tor=True), {}, "send"), "strict")
        self.assertEqual(proxy.tor_mode(args(), {"proxy": TOR, "tor_mode": "strict"}, "worker"), "strict")


class OnionTests(unittest.TestCase):
    def test_is_onion(self):
        self.assertTrue(proxy.is_onion("http://abcdefghijklmnop.onion:8545"))
        self.assertTrue(proxy.is_onion("http://ABC.ONION/x"))
        self.assertFalse(proxy.is_onion("https://node.vibehash.org"))
        self.assertFalse(proxy.is_onion("http://onion.example"))


class OpenerTests(unittest.TestCase):
    def socks(self, opener):
        import sockshandler
        return [h for h in opener.handlers if isinstance(h, sockshandler.SocksiPyHandler)]

    def test_a_proxy_puts_the_socks_handler_in_the_opener(self):
        for url in ("http://node.example:8545", "https://node.example"):
            op = tls.opener_for(url, proxy=TOR)
            self.assertEqual(len(self.socks(op)), 1, url)

    def test_no_proxy_no_change(self):
        self.assertIsNone(tls.opener_for("http://node.example:8545"))

    def test_an_onion_over_http_is_allowed(self):
        op = tls.opener_for("http://abcdefghijklmnop.onion:8545", proxy=TOR)
        self.assertEqual(len(self.socks(op)), 1)
        self.assertTrue(tls.transport_ok("http://abcdefghijklmnop.onion:8545"))

    def test_plain_http_to_a_public_host_is_not_ok(self):
        self.assertFalse(tls.transport_ok("http://node.example:8545"))
        self.assertTrue(tls.transport_ok("https://node.example"))

    def test_a_pinned_https_node_is_checked_through_the_proxy(self):
        op = tls.opener_for("https://node.example", pin="aa" * 32, proxy=TOR)
        self.assertEqual(len(self.socks(op)), 1)
        self.assertFalse([h for h in op.handlers if isinstance(h, tls._PinnedHandler)])  # (it would bypass the proxy)


class PinTests(unittest.TestCase):
    def test_a_certificate_that_is_not_the_pinned_one_is_refused(self):
        import ssl
        cert = b"not the pinned certificate"
        sock = mock.Mock()
        sock.getpeercert.return_value = cert
        proxy.check_pin(sock, tls.fingerprint(cert))  # the right one passes
        with self.assertRaisesRegex(ssl.SSLCertVerificationError, "isn't the pinned one"):
            proxy.check_pin(sock, "aa" * 32)


class FakeSocks:
    """A SOCKS5 server on loopback that records what it is asked to connect to, then says no."""

    def __init__(self):
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        self.asked = []
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            with c:
                c.recv(3)
                c.sendall(b"\x05\x00")
                head = c.recv(4)
                if len(head) < 4:
                    continue
                if head[3] == 3:
                    host = c.recv(c.recv(1)[0]).decode()
                else:
                    host = socket.inet_ntoa(c.recv(4))
                c.recv(2)
                self.asked.append((head[3], host))
                c.sendall(b"\x05\x05\x00\x01" + b"\x00" * 6)

    def close(self):
        self.srv.close()


class RelayingSocks:
    """A SOCKS5 server on loopback that records what it is asked to connect to, then relays to `upstream_port`."""

    def __init__(self, upstream_port):
        self.upstream_port = upstream_port
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        self.asked = []
        threading.Thread(target=self.run, daemon=True).start()

    @staticmethod
    def relay(a, b):
        try:
            while True:
                x = a.recv(65536)
                if not x:
                    break
                b.sendall(x)
        except OSError:
            pass
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def serve(self, c):
        c.recv(3)
        c.sendall(b"\x05\x00")
        head = c.recv(4)
        if head[3] == 3:
            host = c.recv(c.recv(1)[0]).decode()
        else:
            host = socket.inet_ntoa(c.recv(4))
        c.recv(2)
        self.asked.append((head[3], host))
        up = socket.create_connection(("127.0.0.1", self.upstream_port))
        c.sendall(b"\x05\x00\x00\x01" + b"\x00" * 6)
        back = threading.Thread(target=self.relay, args=(up, c), daemon=True)
        back.start()
        self.relay(c, up)
        back.join(5)
        c.close()
        up.close()

    def run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self.serve, args=(c,), daemon=True).start()

    def close(self):
        self.srv.close()


class PinnedOverSocksTests(unittest.TestCase):
    """End to end on loopback: a pinned https node reached through a socks5h proxy (PySocks' https connection reads
    _check_hostname, which Python 3.12's HTTPSConnection no longer has)."""

    def setUp(self):
        import http.server
        import json
        import tempfile

        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        cert, key = str(pathlib.Path(d, "c.pem")), str(pathlib.Path(d, "k.pem"))
        self.fp = tls.make_self_signed(cert, key, ["node.example"])

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"height": 7}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        srv.socket = tls.server_context(cert, key).wrap_socket(srv.socket, server_side=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        self.socks = RelayingSocks(srv.server_address[1])
        self.addCleanup(self.socks.close)
        self.proxy = f"socks5h://127.0.0.1:{self.socks.port}"

    def test_a_pinned_node_answers_through_the_proxy(self):
        client = NodeClient("https://node.example:443", timeout=10, fingerprint=self.fp, proxy=self.proxy)
        self.assertEqual(client.get("/status"), {"height": 7})
        self.assertEqual(self.socks.asked, [(3, "node.example")])  # ATYP 3: the proxy was given the name

    def test_a_wrong_pin_is_refused_through_the_proxy(self):
        client = NodeClient("https://node.example:443", timeout=10, fingerprint="aa" * 32, proxy=self.proxy)
        with self.assertRaisesRegex(NodeError, "pinned"):
            client.get("/status")
        self.assertEqual(self.socks.asked, [(3, "node.example")])


class LeakTests(unittest.TestCase):
    def tearDown(self):
        proxy.activate(None, None)

    def test_a_control_request_hands_the_proxy_the_name_and_resolves_nothing_itself(self):
        fake = FakeSocks()
        self.addCleanup(fake.close)
        resolved, connected = [], []
        real_gai, real_cc = socket.getaddrinfo, socket.create_connection

        def gai(host, *a, **k):
            resolved.append(host)
            return real_gai(host, *a, **k)

        def cc(addr, *a, **k):
            connected.append(addr)
            return real_cc(addr, *a, **k)

        with mock.patch("socket.getaddrinfo", gai), mock.patch("socket.create_connection", cc):
            client = NodeClient("http://secret-node.example:8545", timeout=5, proxy=f"socks5h://127.0.0.1:{fake.port}")
            with self.assertRaises(NodeError):
                client.get("/status")
        self.assertEqual(fake.asked, [(3, "secret-node.example")])  # ATYP 3: the proxy was given the name
        self.assertNotIn("secret-node.example", resolved)
        self.assertFalse([a for a in connected if a[0] == "secret-node.example"])

    def test_the_tor_port_probe_runs_once_however_many_onion_clients(self):
        proxy.activate(None, None)
        proxy.forget_tor()
        self.addCleanup(proxy.forget_tor)
        probes = []

        def cc(addr, *a, **k):
            probes.append(addr)
            return mock.Mock()

        with mock.patch("socket.create_connection", cc):
            clients = [NodeClient("http://abcdefghijklmnop.onion:8545") for _ in range(2)]
        self.assertEqual([c.proxy for c in clients], [TOR, TOR])
        self.assertEqual(probes, [("127.0.0.1", 9050)])

    def test_an_explicit_proxy_never_probes(self):
        proxy.forget_tor()
        with mock.patch.object(proxy, "_listening", side_effect=AssertionError("probed")):
            self.assertEqual(proxy.resolve(args(proxy="socks5h://h:1", tor=True)), "socks5h://h:1")

    def test_an_onion_routes_through_the_proxy_without_being_told_to(self):
        proxy.activate(None, None)
        proxy.forget_tor()
        self.addCleanup(proxy.forget_tor)
        with mock.patch.object(proxy, "_listening", lambda port: port == 9050):
            client = NodeClient("http://abcdefghijklmnop.onion:8545")
        self.assertEqual(client.proxy, TOR)

    def test_a_clearnet_node_is_direct_without_a_proxy(self):
        proxy.activate(None, None)
        client = NodeClient("http://node.example:8545")
        self.assertIsNone(client.proxy)
        self.assertIsNone(client._opener)

    def test_the_active_proxy_is_the_default(self):
        proxy.activate(TOR, "strict")
        client = NodeClient("http://node.example:8545")
        self.assertEqual((client.proxy, client.tor_mode), (TOR, "strict"))


class SplitPlaneTests(unittest.TestCase):
    def client(self, mode):
        return NodeClient("https://node.example", proxy=TOR, tor_mode=mode)

    def run_model(self, c, size):
        planes = []

        def req(method, path, data=None, headers=None, limit=None, plane="proxy"):
            planes.append(plane)
            raise NodeError("stop here", 404)

        job = SimpleNamespace(checkpoint_bytes=size)
        info = {"spec": {}, "owner": "x", "model": {"sha256": "ab", "bytes": size}}
        with mock.patch.object(c, "get", return_value=info), mock.patch.object(c, "_req", req):
            with self.assertRaises(NodeError) as cm:
                c.download_model("j1", job)
        return planes, cm.exception

    def test_control_only_sends_a_big_checkpoint_direct_with_one_warning(self):
        c = self.client("control-only")
        with self.assertLogs("vibehash.worker", logging.WARNING) as logs:
            planes, _ = self.run_model(c, 60 * MB)
        self.assertEqual(planes, ["direct"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("fetching 60 MB directly, not through Tor", logs.output[0])

    def test_control_only_keeps_a_small_one_on_the_proxy(self):
        planes, _ = self.run_model(self.client("control-only"), 1 * MB)
        self.assertEqual(planes, ["proxy"])

    def test_strict_refuses_a_big_one(self):
        planes, err = self.run_model(self.client("strict"), 60 * MB)
        self.assertEqual(planes, [])
        self.assertIn("too large for strict Tor mode", str(err))

    def test_strict_lets_a_small_one_through(self):
        planes, _ = self.run_model(self.client("strict"), 1 * MB)
        self.assertEqual(planes, ["proxy"])

    def test_an_onion_is_never_sent_direct(self):
        c = NodeClient("http://abcdefghijklmnop.onion:8545", proxy=TOR, tor_mode="control-only")
        planes, _ = self.run_model(c, 60 * MB)
        self.assertEqual(planes, ["proxy"])

    def test_the_direct_opener_is_built_once_without_the_proxy(self):
        c = self.client("control-only")
        with mock.patch.object(tls, "opener_for", wraps=tls.opener_for) as of:
            a, b = c._direct_opener(), c._direct_opener()
        self.assertIs(a, b)
        self.assertEqual(of.call_count, 1)
        self.assertIsNone(of.call_args.kwargs.get("proxy"))

    def test_the_open_picks_the_opener_for_the_plane(self):
        c = self.client("control-only")
        proxied, direct = mock.Mock(), mock.Mock()
        c._opener, c._direct = proxied, direct
        c._open(urllib.request.Request("https://node.example/x"), "direct")
        direct.open.assert_called_once()
        c._open(urllib.request.Request("https://node.example/x"))
        proxied.open.assert_called_once()

    def test_the_limit_is_50_mb(self):
        self.assertEqual(BLOB_DIRECT_BYTES, 50 * MB)

    def test_an_upload_is_planned_by_its_size(self):
        c = self.client("strict")
        with mock.patch("pathlib.Path.stat", return_value=SimpleNamespace(st_size=60 * MB)):
            with self.assertRaisesRegex(NodeError, "too large for strict"):
                c.upload(mock.Mock(), "blob", pathlib.Path("x.bin"), "id")

    def test_peers_pass_the_node_s_proxy_and_mode(self):
        from vibehash.p2p import Peers
        peers = Peers("http://me", "addr", [], lambda: {}, proxy=TOR, tor_mode="control-only")
        c = peers.client("http://peer.example:8545")
        self.assertEqual((c.proxy, c.tor_mode), (TOR, "control-only"))


class HubTests(unittest.TestCase):
    def tearDown(self):
        proxy.activate(None, None)

    def test_direct_by_default_and_under_control_only(self):
        for url, mode in ((None, None), (TOR, "control-only")):
            proxy.activate(url, mode)
            self.assertFalse(proxy.is_proxied(hub.Hub()._opener))

    def test_strict_routes_the_hub_through_the_proxy(self):
        proxy.activate(TOR, "strict")
        self.assertTrue(proxy.is_proxied(hub.Hub()._opener))

    def test_an_opener_can_be_given(self):
        mine = urllib.request.build_opener()
        self.assertIs(hub.Hub(opener=mine)._opener, mine)


class CliTests(unittest.TestCase):
    def test_the_flags_are_on_every_command_that_talks_to_a_node(self):
        p = cli.parser()
        for argv in (["worker"], ["app"], ["node", "--genesis", "g", "--wallet", "w", "--data-dir", "d"], ["admin"],
                     ["send", "--to", "x", "--amount", "1"], ["setup"], ["job", "status"]):
            a = p.parse_args([*argv, "--tor", "--proxy", TOR, "--tor-mode", "strict"])
            self.assertTrue(a.tor, argv)
            self.assertEqual((a.proxy, a.tor_mode), (TOR, "strict"), argv)

    def test_admin_tor_turns_geo_off(self):
        from vibehash import admin
        a = cli.parser().parse_args(["admin", "--tor"])
        self.assertFalse(admin.geo_wanted({"geo": True}, a, {}))

    def test_the_settings_key_is_the_one_admin_reads(self):
        self.assertEqual(settings.PROXY_KEY, "proxy")


if __name__ == "__main__":
    unittest.main()
