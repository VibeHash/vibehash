"""The app's window (window.py): the dashboard page in a web view of its own, the app behind it through Api, the
browser as the fallback."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vibehash import app, settings, window


def fake_webview(start_raises=None):
    """A stand-in for the pywebview package: records the window asked for and how start was called."""
    m = mock.MagicMock()
    m.windows = []

    def create_window(title, html=None, **kw):
        w = SimpleNamespace(title=title, html=html, kw=kw)
        m.windows.append(w)
        return w

    def start(**kw):
        m.started = kw
        if start_raises:
            raise start_raises

    m.create_window, m.start = create_window, start
    return m


class Home(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name, "DISPLAY": ":0",
                                              "APPDATA": self.tmp.name + "/AppData"})  # never this machine's Startup folder
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("VIBEHASH_LAUNCHER", None)

    def alive(self, **more):
        app.app_dir().mkdir(parents=True, exist_ok=True)
        state = {"beat": time.time(), "started": time.time() - 5, "pid": 1, "paused": False, "watch": "http://127.0.0.1:7545/",
                 "worker": {"running": True}, "setup": {"step": "mining", "fraction": 1.0}, "events": []} | more
        app.state_path().write_text(json.dumps(state))
        return state


class WindowTests(Home):
    def test_the_command_is_this_python_or_the_launcher(self):
        self.assertEqual(window.command(7000, env={}), [sys.executable, "-m", "vibehash", "window", "--watch", "7000"])
        exe = Path(self.tmp.name) / "VibeHash.exe"
        exe.write_text("")
        self.assertEqual(window.command(7000, env={"VIBEHASH_LAUNCHER": str(exe)}), [str(exe), "--window", "7000"])
        self.assertEqual(window.command(7000, env={"VIBEHASH_LAUNCHER": str(exe) + ".gone"})[1:3], ["-m", "vibehash"])  # not there: python
        self.assertEqual(window.dashboard_url(7000), "http://127.0.0.1:7000/dashboard")

    def test_the_page_is_the_dashboard_told_where_the_feed_is(self):
        text = window.page(7000)
        self.assertIn('<script>window.FEED = "http://127.0.0.1:7000";</script>', text)
        self.assertIn('fetch(FEED + "/"', text)
        self.assertIn("pywebviewready", text)
        self.assertIn('id="welcome"', text)
        self.assertIn('id="set-avatar"', text)

    def test_the_window_is_the_page_with_the_api_served_from_a_port_of_its_own(self):
        wv = fake_webview()
        said = []
        with mock.patch.dict(sys.modules, {"webview": wv}):
            how = window.show(7000, fallback=lambda url: said.append(url), say=said.append)
        self.assertEqual(how, "window")
        self.assertEqual(said, [])
        w = wv.windows[0]
        self.assertEqual(w.title, window.TITLE)
        self.assertIsNone(w.html)  # a file pywebview serves, not a string: WebView2 gives a string the origin "null"
        served = Path(w.kw["url"])
        self.assertEqual(served.parent, app.app_dir() / "window")
        self.assertEqual(os.listdir(served.parent), [served.name])  # its folder holds only the page
        self.assertIn('window.FEED = "http://127.0.0.1:7000"', served.read_text(encoding="utf-8"))
        self.assertIsInstance(w.kw["js_api"], window.Api)
        self.assertEqual(w.kw["js_api"].port, 7000)
        self.assertEqual((w.kw["width"], w.kw["height"], w.kw["min_size"]), (window.SIZE[0], window.SIZE[1], window.MIN_SIZE))
        self.assertTrue(wv.started["http_server"])  # an http origin, which the feed's CORS allows
        self.assertTrue(window.Lock(7000).acquire())  # the lock was let go when the window closed

    def test_a_changed_page_gets_a_new_address_and_the_old_file_goes(self):
        """The web view caches by address: an updated app under the old name could go on showing the old page."""
        first = window.page_file(7000)
        self.assertEqual(window.page_file(7000), first)  # the same page, the same address
        with mock.patch.object(window, "page", return_value="<html>changed</html>"):
            second = window.page_file(7000)
        self.assertNotEqual(second, first)
        self.assertFalse(first.exists())
        self.assertEqual(os.listdir(second.parent), [second.name])

    def test_without_pywebview_the_browser_gets_the_feeds_page(self):
        opened, said = [], []
        with mock.patch.dict(sys.modules, {"webview": None}):  # import webview -> ImportError
            how = window.show(7545, fallback=opened.append, say=said.append)
        self.assertEqual(how, "browser")
        self.assertEqual(opened, ["http://127.0.0.1:7545/dashboard"])
        self.assertIn("pywebview", said[0])

    def test_a_web_view_that_cannot_start_falls_back_to_the_browser(self):
        opened, said = [], []
        wv = fake_webview(start_raises=RuntimeError("You must have either QT or GTK with Python extensions installed"))
        with mock.patch.dict(sys.modules, {"webview": wv}):
            how = window.show(7545, fallback=opened.append, say=said.append)
        self.assertEqual(how, "browser")
        self.assertEqual(opened, ["http://127.0.0.1:7545/dashboard"])
        self.assertIn("QT or GTK", said[0])

    def test_one_window_per_feed(self):
        first = window.Lock(7545)
        self.assertTrue(first.acquire())
        self.assertFalse(window.Lock(7545).acquire())  # the same feed: no
        self.assertTrue(window.Lock(7546).acquire())  # another feed: fine
        opened, said = [], []
        with mock.patch.dict(sys.modules, {"webview": fake_webview()}):
            self.assertEqual(window.show(7545, fallback=opened.append, say=said.append), "open")
        self.assertEqual((opened, said), ([], ["the window is already open"]))
        first.release()
        self.assertTrue(window.Lock(7545).acquire())  # let go: the next one gets it

    def test_no_screen_on_linux_means_no_window_and_no_console_browser(self):
        opened, said = [], []
        with mock.patch.object(window.sys, "platform", "linux"), \
                mock.patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}), \
                mock.patch.dict(sys.modules, {"webview": None}):
            self.assertEqual(window.show(7545, fallback=opened.append, say=said.append), "none")
        self.assertEqual(opened, [])
        self.assertIn("needs a screen", said[0])

    def test_the_cli_has_it(self):
        from vibehash import cli
        a = cli.parser().parse_args(["window", "--watch", "7000"])
        self.assertEqual((a.fn, a.watch), (cli.cmd_window, 7000))
        self.assertEqual(cli.parser().parse_args(["window"]).watch, window.DEFAULT_PORT)
        self.assertTrue(cli.parser().parse_args(["app", "--no-window"]).no_window)
        self.assertFalse(cli.parser().parse_args(["app", "start"]).no_window)


class ApiTests(Home):
    def setUp(self):
        super().setUp()
        self.api = window.Api(7545)

    def test_state_is_the_apps_with_running_added(self):
        self.assertEqual(self.api.state(), {"running": False})
        self.alive()
        s = self.api.state()
        self.assertTrue(s["running"])
        self.assertEqual(s["setup"]["step"], "mining")
        self.assertNotIn("pid", s)

    def test_mfa_status_reflects_the_credential_store_for_the_apps_wallet(self):
        try:
            import keyring
            import keyring.backend
        except ImportError:
            self.skipTest("keyring is not installed")

        class Memory(keyring.backend.KeyringBackend):
            priority = 1
            data: dict = {}

            def set_password(self, service, username, password):
                self.data[(service, username)] = password

            def get_password(self, service, username):
                return self.data.get((service, username))

            def delete_password(self, service, username):
                self.data.pop((service, username), None)

        old = keyring.get_keyring()
        Memory.data = {}
        keyring.set_keyring(Memory())
        self.addCleanup(keyring.set_keyring, old)
        self.assertEqual(self.api.mfa_status(), {"on": False})  # no app, no wallet
        self.alive(wallet={"address": "vibe1" + "ab" * 18})
        self.assertEqual(self.api.mfa_status(), {"on": False})
        keyring.set_password("vibehash-totp", "vibe1" + "ab" * 18, "KEY")
        self.assertEqual(self.api.mfa_status(), {"on": True})
        with mock.patch("vibehash.credstore.get", return_value=None):
            self.assertEqual(self.api.mfa_status(), {"on": False})
        from vibehash import credstore
        with mock.patch("vibehash.credstore.get", side_effect=credstore.Unavailable("locked")):
            out = self.api.mfa_status()
        self.assertIsNone(out["on"])  # unknown, not off
        self.assertIn("locked", out["why"])

    def test_orders_go_to_the_running_app_only(self):
        self.assertEqual(self.api.order("pause")["ok"], False)  # not running
        self.alive()
        self.assertEqual(self.api.order("pause"), {"ok": True, "order": "pause"})
        self.assertEqual(app.take_order(), "pause")
        self.assertEqual(self.api.order("avatar-off")["ok"], True)
        self.assertEqual(app.take_order(), "avatar-off")
        self.assertFalse(self.api.order("format-disk")["ok"])
        self.assertIsNone(app.take_order())

    def test_the_wallet_and_its_words_come_from_setup(self):
        self.assertEqual(self.api.wallet(), {"words": []})
        words = Path(self.tmp.name) / "words.txt"
        words.write_text("Vibe Hash recovery words for 0xabc\nwallet file: w.json\n\n 1. apple\n 2. brick\n 3. cloud\n\nKeep them.\n")
        self.alive(wallet={"address": "0xabc", "path": "w.json", "words_file": str(words)})
        w = self.api.wallet()
        self.assertEqual((w["address"], w["path"], w["words"]), ("0xabc", "w.json", ["apple", "brick", "cloud"]))
        words.unlink()  # the user deleted the file, as told: no words, no error
        w = self.api.wallet()
        self.assertEqual((w["words"], w["words_file"]), ([], None))

    def test_new_words_come_from_the_apps_listener_and_words_seen_clears_them(self):
        words = " ".join(f"w{i}" for i in range(1, 25))
        seen = []
        port = app.serve_words(words, "secret-token", on_seen=lambda: seen.append(1))
        self.addCleanup(app.stop_words)
        self.alive(wallet={"address": "0xabc", "path": "w.json", "words_pending": True, "words_port": port,
                           "words_token": "wrong"})
        self.assertEqual(self.api.wallet()["words"], [])  # the token in the state is what the app checks
        self.assertFalse(self.api.words_seen()["ok"])
        self.assertEqual(seen, [])
        self.alive(wallet={"address": "0xabc", "path": "w.json", "words_pending": True, "words_port": port,
                           "words_token": "secret-token"})
        w = self.api.wallet()
        self.assertEqual(w["words"], words.split())
        self.assertEqual(self.api.wallet()["words"], words.split())  # a restarted window still gets them
        self.assertNotIn("w1 ", Path(app.state_path()).read_text())  # (the state holds the port, not the words)
        with mock.patch.object(app, "_winreg", return_value=None):
            self.assertTrue(self.api.settings()["words_pending"])
        self.assertEqual(self.api.words_seen(), {"ok": True})
        self.assertEqual(seen, [1])
        self.assertEqual(self.api.wallet()["words"], [])  # the app answers 404 now

    def test_words_pending_with_nobody_listening_is_no_words_not_an_error(self):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.alive(wallet={"address": "0xabc", "path": "w.json", "words_pending": True, "words_port": port})
        self.assertEqual(self.api.wallet()["words"], [])
        self.assertFalse(self.api.words_seen()["ok"])
        self.alive(wallet={"address": "0xabc", "path": "w.json"})
        self.assertFalse(self.api.words_seen()["ok"])  # nothing was waiting

    def test_the_window_only_asks_this_computer_whatever_the_state_says(self):
        self.alive(wallet={"address": "0xabc", "words_pending": True, "words_port": "80@attacker.example"})
        with mock.patch("urllib.request.urlopen") as opened:
            self.assertEqual(self.api.wallet()["words"], [])
            self.assertFalse(self.api.words_seen()["ok"])
        opened.assert_not_called()

    def test_deepseek_is_remembered_and_a_running_app_restarts_its_worker(self):
        with mock.patch.object(app, "_winreg", return_value=None):
            self.assertEqual(self.api.settings()["deepseek"], "no")
            self.assertEqual(self.api.set_deepseek(True), {"ok": True, "deepseek": "yes"})
            self.assertEqual(settings.load()["deepseek"], "yes")  # what `vibehash worker` reads as it starts
            self.assertIsNone(app.take_order())  # no app running: the setting is enough
            self.alive()
            self.api.set_deepseek(False)
            self.assertEqual(self.api.settings()["deepseek"], "no")
            self.assertEqual(app.take_order(), "restart")
            self.assertFalse(self.api.set_deepseek("yes")["ok"])

    def test_settings_the_avatar_and_the_welcome(self):
        with mock.patch.object(app, "_winreg", return_value=None):
            c = self.api.settings()
            self.assertEqual((c["avatar"], c["welcomed"], c["startup"]), (True, False, False))
            self.assertEqual(c["home"], self.tmp.name)
            self.assertEqual(self.api.set_avatar(False), {"ok": True, "avatar": False})
            self.assertIs(settings.load()["avatar"], False)
            self.assertIsNone(app.take_order())  # no app running: the setting is enough
            self.alive()
            self.api.set_avatar(True)
            self.assertEqual(app.take_order(), "avatar-on")  # running: told now as well
            self.api.welcomed()
            self.assertTrue(self.api.settings()["welcomed"])
            self.api.welcome_again()
            self.assertFalse(self.api.settings()["welcomed"])

    def test_start_at_login_through_the_window(self):
        calls = []
        with mock.patch.object(Path, "home", return_value=Path(self.tmp.name)), \
                mock.patch.object(app, "_winreg", return_value=None), \
                mock.patch.object(app.platform, "system", return_value="Linux"), \
                mock.patch.object(app.subprocess, "run", side_effect=lambda cmd, **kw: calls.append(cmd)):
            on = self.api.set_startup(True)
            self.assertTrue(on["ok"])
            self.assertTrue(self.api.settings()["startup"])
            self.assertTrue(Path(on["where"]).exists())
            self.assertTrue(self.api.set_startup(False)["ok"])
            self.assertFalse(self.api.settings()["startup"])

    def test_options_take_only_their_own_names_and_kinds(self):
        with mock.patch.object(app, "_winreg", return_value=None):
            for name in window.OPTIONS:
                self.assertEqual(self.api.set_option(name, True), {"ok": True, name: True})
                self.assertIs(self.api.settings()[name], True)
            self.assertFalse(self.api.set_option("pause_on_battery", "yes")["ok"])  # a bool, not a truthy string
            self.assertFalse(self.api.set_option("node", "https://evil.example")["ok"])  # not a setting the page may change
            self.assertNotIn("node", settings.load())
            self.assertEqual(self.api.set_option("idle_minutes", 10), {"ok": True, "idle_minutes": 10})
            self.assertEqual(self.api.settings()["idle_minutes"], 10)
            for bad in (7, True, "5", -1):
                self.assertFalse(self.api.set_option("idle_minutes", bad)["ok"], bad)
            self.assertEqual(self.api.settings()["idle_minutes"], 10)

    def test_links_open_by_name_only_in_the_browser(self):
        opened = []
        with mock.patch.object(window, "OPEN", side_effect=opened.append):
            self.assertEqual(self.api.open_link("source"), {"ok": True, "url": window.LINKS["source"]})
            for bad in ("https://evil.example", "javascript:alert(1)", "file:///etc/passwd", None, ["site"]):
                self.assertFalse(self.api.open_link(bad)["ok"], bad)
        self.assertEqual(opened, [window.LINKS["source"]])
        self.assertTrue(all(u.startswith("https://") for u in self.api.links().values()))

    def test_the_words_file_is_deleted_on_request_and_the_wallet_stays(self):
        self.assertFalse(self.api.delete_words()["ok"])  # no app state, no file named
        words, wallet_file = Path(self.tmp.name) / "words.txt", Path(self.tmp.name) / "w.json"
        words.write_text(" 1. apple\n"), wallet_file.write_text("{}")
        self.alive(wallet={"address": "0xabc", "path": str(wallet_file), "words_file": str(words)})
        with mock.patch.object(app, "_winreg", return_value=None):
            self.assertTrue(self.api.settings()["words_file"])
            self.assertEqual(self.api.delete_words(), {"ok": True, "deleted": True})
            self.assertFalse(words.exists())
            self.assertTrue(wallet_file.exists())
            self.assertFalse(self.api.settings()["words_file"])
        self.assertEqual(self.api.delete_words(), {"ok": True, "deleted": False})

    def test_updates_compare_release_numbers_and_ask_pypi_seldom(self):
        self.assertTrue(window.newer("0.5.23", "0.5.22"))
        self.assertTrue(window.newer("0.10.0", "0.9.9"))  # numbers, not text
        self.assertFalse(window.newer("0.5.22", "0.5.22"))
        self.assertFalse(window.newer("0.5.21", "0.5.22"))
        self.assertFalse(window.newer(None, "0.5.22"))
        self.assertFalse(window.newer("0.5.23", "unknown"))  # a version it can't read: no banner on every start
        asked = []
        with mock.patch.object(window, "PYPI_LATEST", side_effect=lambda: asked.append(1) or "99.0.0"), \
                mock.patch.object(window, "_version", return_value="0.5.22"), mock.patch.object(window.Api, "_update", None):
            self.assertEqual(self.api.update(), {"current": "0.5.22", "latest": "99.0.0", "newer": True})
            self.api.update()
            self.assertEqual(len(asked), 1)  # cached
        with mock.patch.object(window, "PYPI_LATEST", side_effect=OSError("offline")), mock.patch.object(window.Api, "_update", None):
            self.assertFalse(self.api.update()["newer"])  # offline: no banner, no error

    def test_the_round_history_is_this_wallets_sealed_rounds_newest_first(self):
        from vibehash import records
        self.alive(wallet={"address": "vibe1me"})
        log = records.logs_dir() / "rounds.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"worker": "vibe1me", "round": 1, "sealed": True, "earned_vibe": 0.2, "job": "j"},
                {"worker": "vibe1other", "round": 2, "sealed": True, "earned_vibe": 9},
                {"worker": "vibe1me", "round": 3},  # not sealed yet
                {"worker": "vibe1me", "round": 4, "sealed": True, "lost_vibe": 2.5, "secret_stuff": "x"}]
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
        got = self.api.history(10)["rows"]
        self.assertEqual([r["round"] for r in got], [4, 1])
        self.assertNotIn("secret_stuff", got[0])
        self.assertEqual(len(self.api.history(1)["rows"]), 1)

    def test_the_model_store_clears_only_while_nothing_mines(self):
        from vibehash import modelstore
        folder = Path(self.tmp.name) / "models"
        settings.save(models={"folder": str(folder), "cap_gb": 40})
        c = self.api.cache()
        self.assertTrue(c["ok"])
        self.assertEqual((c["used"], c["cap"]), (0, 40 * modelstore.GB))
        self.assertTrue(self.api.set_cache_cap(100)["ok"])
        self.assertEqual(self.api.cache()["cap"], 100 * modelstore.GB)
        for bad in (1, 5000, True, "50"):
            self.assertFalse(self.api.set_cache_cap(bad)["ok"], bad)
        self.alive()  # mining
        self.assertFalse(self.api.clear_cache()["ok"])
        self.alive(paused=True)
        self.assertEqual(self.api.clear_cache(), {"ok": True, "freed": 0})

    def test_the_check_runs_with_the_environments_python_under_the_launcher(self):
        """VibeHash.exe --window hosts this module in the frozen launcher, whose sys.executable doesn't know -m."""
        self.assertEqual(window.python("C:/py/python.exe", frozen=False), Path("C:/py/python.exe"))
        env = Path(self.tmp.name) / "VibeHash" / "env"
        package = env / "Lib" / "site-packages" / "vibehash"
        package.mkdir(parents=True)
        (env / "Scripts").mkdir()
        (env / "Scripts" / "python.exe").write_text("")
        self.assertEqual(window.python("C:/VibeHash/VibeHash.exe", package=package, frozen=True), env / "Scripts" / "python.exe")
        posix = Path(self.tmp.name) / "share" / "env"
        (posix / "lib" / "python3.12" / "site-packages" / "vibehash").mkdir(parents=True)
        (posix / "bin").mkdir()
        (posix / "bin" / "python").write_text("")
        self.assertEqual(window.python("/opt/VibeHash", package=posix / "lib" / "python3.12" / "site-packages" / "vibehash",
                                       frozen=True), posix / "bin" / "python")
        # not in an environment: the launcher itself, never some python further up the tree (/bin/python on Linux)
        nowhere = Path(self.tmp.name) / "loose" / "vibehash"
        nowhere.mkdir(parents=True)
        self.assertEqual(window.python("C:/X/VibeHash.exe", package=nowhere, frozen=True), Path("C:/X/VibeHash.exe"))

    def test_a_support_report_carries_no_secret(self):
        home = str(Path(self.tmp.name))
        report = {"ok": True, "checks": [{"name": "wallet", "text": f"read {home}/wallet.json for vibe1qqqqabcdefghijklmnop1234",
                                          "data": {"words_file": "x", "password": "hunter2", "mnemonic": "a b c",
                                                   "admin_token": "t", "private_key": "k", "size": 3}}]}
        out = window.redact(report, home=home)
        data = out["checks"][0]["data"]
        self.assertEqual(data, {"size": 3})
        self.assertEqual(out["checks"][0]["text"], "read ~/wallet.json for vibe1qqqq...1234")
        self.assertNotIn("hunter2", json.dumps(out))


class PageTests(unittest.TestCase):
    """The page as a whole, read as text: every control has a handler, nothing comes from the wider web, and what the
    feed, node or app say goes in as text, never as markup."""

    def setUp(self):
        self.page = window.DASHBOARD.read_text(encoding="utf-8")
        self.script = self.page.split("<script>", 1)[1]

    def test_every_button_and_input_is_wired(self):
        import re
        ids = re.findall(r'<(?:button|input|select)[^>]*\bid="([^"]+)"', self.page)
        self.assertGreater(len(ids), 25)
        loose = [i for i in ids if f'$("{i}")' not in self.script and f'"{i}"' not in self.script]
        self.assertEqual(loose, [], "controls nothing in the script refers to")
        for data_attr in ("data-tab", "data-level", "data-link"):  # the rest are wired by attribute
            self.assertIn(f'dataset.{data_attr[5:]}', self.script)

    def test_the_tour_and_the_word_check_have_every_step(self):
        for needed in ('id="w-next-1"', 'id="w-next-2"', 'id="w-done"', 'id="w-skip-1"', 'id="w-skip-2"',
                       'id="cer-next"', 'id="cer-done"', 'id="cer-delete"', 'id="cer-cancel"', "function wipe()"):
            self.assertIn(needed, self.page)
        # the words are cleared from the page whenever the check closes, however it closes
        self.assertIn('function closeCeremony() { wipe();', self.script)
        self.assertIn('$("cer-done").addEventListener("click", function () { wipe();', self.script)
        # the word check opens from the welcome's second step, so it must sit above the welcome, not under it
        order = [self.page.index(f'<div class="scrim" id="{i}"') for i in ("ceremony", "ask", "welcome")]
        self.assertIn("#welcome.scrim { z-index: 25; }", self.page)
        self.assertIn("z-index: 30", self.page)
        self.assertTrue(all(o > 0 for o in order))
        self.assertIn("!welcomeOffered", self.script)  # a settings reload mid-tour doesn't restart it

    def test_nothing_from_the_wider_web_and_no_markup_from_data(self):
        self.assertNotIn("https://", self.page.replace("https://vibehash.org", ""))
        self.assertNotIn("innerHTML", self.script)
        self.assertNotIn("window.open", self.script)  # links leave through Api.open_link only
        self.assertIn('http-equiv="Content-Security-Policy"', self.page)
        self.assertIn("default-src 'none'", self.page)


if __name__ == "__main__":
    unittest.main()
