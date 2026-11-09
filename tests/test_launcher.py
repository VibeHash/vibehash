"""The downloadable launcher (launcher/vibehash_launcher.py): where it installs, how it checks uv's download, what
it runs, and that it never touches the wallet. Nothing here goes online."""
import hashlib
import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # (the repo's own vibehash package, to sign test manifests with)


def load():
    spec = importlib.util.spec_from_file_location("vibehash_launcher", ROOT / "launcher" / "vibehash_launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = load()
from vibehash import crypto  # noqa: E402
from tests.test_app import FakeWinreg  # noqa: E402  (the in-memory registry)


class WhereTests(unittest.TestCase):
    def test_a_folder_per_system(self):
        local = r"C:\Users\me\AppData\Local"  # (joined the same way on every system: CI runs this on Linux)
        self.assertEqual(L.app_dir("Windows", {"LOCALAPPDATA": local}), Path(local) / "VibeHash")
        self.assertEqual(L.app_dir("Darwin", {}), Path.home() / "Library" / "Application Support" / "VibeHash")
        self.assertEqual(L.app_dir("Linux", {"XDG_DATA_HOME": "/data"}), Path("/data/vibehash"))
        self.assertEqual(L.app_dir("Linux", {}), Path.home() / ".local" / "share" / "vibehash")
        self.assertEqual(L.app_dir("Linux", {"VIBEHASH_LAUNCHER_DIR": "/tmp/x"}), Path("/tmp/x"))

    def test_the_uv_build_for_the_machine(self):
        self.assertEqual(L.uv_asset("Windows", "AMD64"), "uv-x86_64-pc-windows-msvc.zip")
        self.assertEqual(L.uv_asset("Darwin", "arm64"), "uv-aarch64-apple-darwin.tar.gz")
        self.assertEqual(L.uv_asset("Linux", "x86_64"), "uv-x86_64-unknown-linux-gnu.tar.gz")
        with self.assertRaises(SystemExit):
            L.uv_asset("Linux", "mips")
        self.assertEqual(L.uv_path(Path("f"), "Windows").name, "uv.exe")
        self.assertEqual(L.env_python(Path("e"), "Windows"), Path("e") / "Scripts" / "python.exe")
        self.assertEqual(L.env_python(Path("e"), "Linux"), Path("e") / "bin" / "python")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_checksum_lines(self):
        data = b"uv"
        hexd = hashlib.sha256(data).hexdigest()
        self.assertTrue(L.sha256_ok(data, f"{hexd}  uv-x86_64-pc-windows-msvc.zip\n"))
        self.assertTrue(L.sha256_ok(data, hexd.upper()))
        self.assertFalse(L.sha256_ok(data, "0" * 64))
        self.assertFalse(L.sha256_ok(data, ""))

    def pin(self, payload: bytes, asset: str, good=True):
        """Stand the pinned hash (UV_SHA256) in for the fake download's."""
        hexd = hashlib.sha256(payload).hexdigest() if good else "f" * 64
        patcher = mock.patch.dict(L.UV_SHA256, {asset: hexd})
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_fetch(self, payload: bytes, good=True):
        def fetch(url):
            if url.endswith(".sha256"):
                return (hashlib.sha256(payload).hexdigest() if good else "0" * 64).encode() + b"  asset\n"
            return payload
        return fetch

    def test_uv_from_a_zip_once(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("uv-x86_64-pc-windows-msvc/uv.exe", b"MZ fake uv")
        self.pin(buf.getvalue(), "uv-x86_64-pc-windows-msvc.zip")
        fetch = mock.Mock(side_effect=self.fake_fetch(buf.getvalue()))
        exe = L.ensure_uv(self.folder, "Windows", "AMD64", fetch=fetch)
        self.assertEqual(exe.read_bytes(), b"MZ fake uv")
        self.assertEqual((self.folder / "uv.version").read_text().strip(), L.UV_VERSION)
        self.assertEqual(fetch.call_count, 2)
        self.assertIn(f"/{L.UV_VERSION}/uv-x86_64-pc-windows-msvc.zip", fetch.call_args_list[0].args[0])
        L.ensure_uv(self.folder, "Windows", "AMD64", fetch=fetch)  # there already: nothing fetched
        self.assertEqual(fetch.call_count, 2)

    def test_uv_from_a_tarball(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
            info.size = 7
            t.addfile(info, io.BytesIO(b"ELF uv!"))
        self.pin(buf.getvalue(), "uv-x86_64-unknown-linux-gnu.tar.gz")
        exe = L.ensure_uv(self.folder, "Linux", "x86_64", fetch=self.fake_fetch(buf.getvalue()))
        self.assertEqual(exe.read_bytes(), b"ELF uv!")
        if os.name != "nt":
            self.assertTrue(os.access(exe, os.X_OK))

    def test_a_download_that_fails_its_checksum_is_not_installed(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("uv.exe", b"tampered")
        self.pin(buf.getvalue(), "uv-x86_64-pc-windows-msvc.zip")
        with self.assertRaises(SystemExit) as ctx:
            L.ensure_uv(self.folder, "Windows", "AMD64", fetch=self.fake_fetch(buf.getvalue(), good=False))
        self.assertIn("checksum", str(ctx.exception))
        self.assertFalse((self.folder / "uv.exe").exists())

    def test_a_download_that_is_not_the_pinned_one_is_refused_even_when_its_release_vouches_for_it(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("uv.exe", b"swapped at the origin")
        self.pin(buf.getvalue(), "uv-x86_64-pc-windows-msvc.zip", good=False)  # the constant says otherwise
        with self.assertRaises(SystemExit) as ctx:
            L.ensure_uv(self.folder, "Windows", "AMD64", fetch=self.fake_fetch(buf.getvalue(), good=True))  # .sha256 matches
        self.assertIn("checksum", str(ctx.exception))
        self.assertFalse((self.folder / "uv.exe").exists())

    def test_every_uv_asset_has_a_pinned_hash(self):
        self.assertEqual(set(L.UV_SHA256), set(L.UV_ASSETS.values()))
        for asset, hexd in L.UV_SHA256.items():
            self.assertRegex(hexd, r"^[0-9a-f]{64}$", asset)


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = Path(self.tmp.name) / "env"
        self.uv = Path(self.tmp.name) / "uv"

    def test_the_environment_is_made_once_with_pip(self):
        calls = []
        python = L.ensure_env(self.uv, self.env, "Linux", run=lambda cmd, **kw: calls.append(cmd) or 0)
        self.assertEqual(python, self.env / "bin" / "python")
        self.assertEqual(calls[0][:3], [str(self.uv), "venv", str(self.env)])
        self.assertIn("--seed", calls[0])  # pip inside, for `vibehash install-gpu`
        self.assertIn(L.PYTHON, calls[0])
        python.parent.mkdir(parents=True)
        python.write_text("")
        L.ensure_env(self.uv, self.env, "Linux", run=lambda cmd, **kw: calls.append(cmd) or 0)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(SystemExit):
            L.ensure_env(self.uv, Path(self.tmp.name) / "other", "Linux", run=lambda cmd, **kw: 1)

    def test_the_app_is_run_without_a_password_question_and_waited_for(self):
        child = mock.Mock()
        child.wait.return_value = 7
        with mock.patch.object(L.subprocess, "Popen", return_value=child) as popen, \
                mock.patch.object(L.signal, "signal") as sig:
            code = L.run_app(Path("e") / "bin" / "py", ["--device", "cpu"], child_env={"X": "1"})
        self.assertEqual(code, 7)
        self.assertEqual(popen.call_args.args[0], [str(Path("e") / "bin" / "py"), "-m", "vibehash", "app", "--no-password", "--avatar", "--device", "cpu"])
        self.assertEqual(popen.call_args.kwargs["cwd"], "e")  # the environment's folder, not wherever it was started
        self.assertEqual(popen.call_args.kwargs["env"], {"X": "1"})
        self.assertNotIn("stdout", popen.call_args.kwargs)  # a console: the app talks there
        self.assertEqual(sig.call_args_list[0].args, (L.signal.SIGINT, L.signal.SIG_IGN))  # Ctrl+C is the app's to handle

    def test_with_no_console_the_app_gets_a_hidden_one_and_a_log(self):
        child = mock.Mock()
        child.wait.return_value = 0
        log = Path(tempfile.mkdtemp()) / "app.log"
        with mock.patch.object(L.subprocess, "Popen", return_value=child) as popen, mock.patch.object(L.signal, "signal"), \
                mock.patch.object(L.sys, "stdout", None):  # the windowed .exe
            L.run_app(Path("e") / "bin" / "py", [], log=log)
        kw = popen.call_args.kwargs
        self.assertEqual(kw["creationflags"], L.NO_WINDOW)
        self.assertIs(kw["stdout"], kw["stderr"])
        self.assertTrue(kw["stdout"].closed)  # closed after the wait
        self.assertTrue(log.exists())

    def test_the_children_keep_everything_in_the_launchers_folder(self):
        folder = Path("f")
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/_MEI123", "LD_LIBRARY_PATH_ORIG": "/usr/lib/x"}):
            env = L.child_env(folder)
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(folder / "python"))
            self.assertEqual(env["UV_CACHE_DIR"], str(folder / "cache"))
            self.assertEqual(env["UV_PYTHON_PREFERENCE"], "only-managed")
            self.assertNotIn("VIBEHASH_LAUNCHER", env)  # a script, not the frozen .exe
            self.assertEqual(env["LD_LIBRARY_PATH"], "/tmp/_MEI123")  # as a script: left alone
            with mock.patch.object(L.sys, "frozen", True, create=True), \
                    mock.patch.dict(os.environ, {"TCL_LIBRARY": "/tmp/_MEI123/_tcl_data", "TK_LIBRARY": "/tmp/_MEI123/_tk_data"}):
                env = L.child_env(folder)
                self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/lib/x")  # the .exe's unpacked bundle taken back out
                self.assertEqual(env["VIBEHASH_LAUNCHER"], L.sys.executable)
                self.assertNotIn("TCL_LIBRARY", env)  # the avatar's tkinter would look in the .exe's bundle otherwise
                self.assertNotIn("TK_LIBRARY", env)
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/_MEI123"}, clear=False), \
                mock.patch.object(L.sys, "frozen", True, create=True):
            os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
            self.assertNotIn("LD_LIBRARY_PATH", L.child_env(folder))


class Rfc8032Tests(unittest.TestCase):
    """launcher/ed25519_verify.py, the launcher's stdlib-only verifier, against RFC 8032's vectors."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location("ed25519_verify", ROOT / "launcher" / "ed25519_verify.py")
        self.V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.V)

    def test_vector_1_empty_message(self):
        pub = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        sig = bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
        self.assertTrue(self.V.verify(pub, b"", sig))
        self.assertFalse(self.V.verify(pub, b"x", sig))

    def test_vector_2_one_byte(self):
        pub = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
        sig = bytes.fromhex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00")
        self.assertTrue(self.V.verify(pub, b"\x72", sig))
        self.assertFalse(self.V.verify(pub, b"\x73", sig))

    def test_malformed_input_is_false_not_an_error(self):
        k = crypto.KeyPair.generate(b"v")
        sig = k.sign(b"m")
        self.assertTrue(self.V.verify(k.public, b"m", sig))
        self.assertFalse(self.V.verify(k.public, b"m", sig[:-1]))
        self.assertFalse(self.V.verify(k.public[:-1], b"m", sig))
        self.assertFalse(self.V.verify(k.public, b"m", sig[:32] + b"\xff" * 32))  # s >= the group order
        self.assertFalse(self.V.verify(b"\xff" * 32, b"m", sig))  # not a point


NOW = 1_000_000.0  # (a day after the stamp the tests write, 1.0)


class VerifiedUpdateTests(unittest.TestCase):
    """ensure_vibehash: the signed manifest, then a --require-hashes install of exactly what it lists."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = Path(self.tmp.name) / "env"
        self.env.mkdir()
        self.uv = Path(self.tmp.name) / "uv"
        self.python = self.env / "bin" / "python"
        self.key = crypto.KeyPair.generate(b"release-test")
        keys = mock.patch.object(L, "RELEASE_PUBKEYS", [self.key.public.hex()])
        keys.start()
        self.addCleanup(keys.stop)
        dev = mock.patch.dict(os.environ)
        dev.start()
        self.addCleanup(dev.stop)
        os.environ.pop("VIBEHASH_DEV", None)
        self.calls = []
        self.said = []
        say = mock.patch.object(L, "say", side_effect=self.said.append)
        say.start()
        self.addCleanup(say.stop)
        self.reqs = ("numpy==2.0.0 \\\n    --hash=sha256:" + "a" * 64 + "\nvibehash==9.9.9 \\\n    --hash=sha256:" + "b" * 64 + "\n").encode()

    def manifest(self, **over):
        m = {"version": "9.9.9", "wheel": "vibehash-9.9.9-py3-none-any.whl", "wheel_sha256": "b" * 64,
             "requirements": "requirements-9.9.9.txt", "requirements_sha256": hashlib.sha256(self.reqs).hexdigest(),
             "requirements_evo": "requirements-9.9.9-evo.txt", "requirements_evo_sha256": "e" * 64,
             "published": 1700000000, "min_launcher": "1.2"}
        return m | over

    def files(self, manifest=None, sig=None, reqs=None):
        manifest = self.manifest() if manifest is None else manifest
        body = L.json.dumps(manifest, indent=2).encode()  # (the signature is over the canonical form, not these bytes)
        sig = self.key.sign(L.canonical(manifest)).hex().encode() if sig is None else sig
        files = {f"{L.RELEASES}/release.json": body, f"{L.RELEASES}/release.sig": sig}
        files[f"{L.RELEASES}/{manifest['requirements']}"] = self.reqs if reqs is None else reqs
        return files

    def fetcher(self, files):
        def fetch(url, *a, **kw):
            if url not in files:
                raise OSError(f"offline: {url}")
            return files[url]
        return fetch

    def go(self, files, installed="0.5.27", now=NOW, force=False, stamp=False):
        if stamp:
            (self.env / ".vibehash-checked").write_text("1.0\n")
        run = lambda cmd, **k: self.calls.append(cmd) or 0  # noqa: E731
        return L.ensure_vibehash(self.uv, self.python, self.env, force=force, now=now, run=run,
                                 fetch=self.fetcher(files), installed_version=lambda python: installed)

    def stamped(self):
        return (self.env / ".vibehash-checked").exists() and (self.env / ".vibehash-checked").read_text().strip() != "1.0"

    def test_a_good_manifest_installs_exactly_what_it_lists_with_hashes_required(self):
        self.assertTrue(self.go(self.files()))
        req = self.env / "requirements-9.9.9.txt"
        self.assertEqual(self.calls, [[str(self.uv), "pip", "install", "--python", str(self.python), "--require-hashes",
                                       "-r", str(req), "--quiet"]])
        self.assertEqual(req.read_bytes(), self.reqs)
        self.assertEqual((self.env / ".vibehash-checked").read_text().strip(), "1000000")

    def test_first_install_goes_through_the_same_path(self):
        self.assertTrue(self.go(self.files(), installed=None))
        self.assertIn("--require-hashes", self.calls[0])

    def test_checked_once_a_day_and_update_forces_it(self):
        self.assertTrue(self.go(self.files(), installed=None))
        self.assertFalse(self.go(self.files(), now=NOW + 3600))
        self.assertTrue(self.go(self.files(), now=NOW + L.UPDATE_EVERY_S + 1, installed="0.5.27"))
        self.assertTrue(self.go(self.files(), now=NOW + L.UPDATE_EVERY_S + 2, force=True, installed="0.5.27"))
        self.assertEqual(len(self.calls), 3)

    def test_a_tampered_signature_installs_nothing(self):
        files = self.files()
        sig = bytearray(bytes.fromhex(files[f"{L.RELEASES}/release.sig"].decode()))
        sig[0] ^= 1
        files[f"{L.RELEASES}/release.sig"] = bytes(sig).hex().encode()
        self.assertFalse(self.go(files, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())
        self.assertEqual(len(self.said), 2)  # "checking..." and one line saying why not
        self.assertIn("signature", self.said[-1])

    def test_a_manifest_changed_after_signing_installs_nothing(self):
        files = self.files()
        files[f"{L.RELEASES}/release.json"] = L.json.dumps(self.manifest(requirements_sha256="c" * 64)).encode()
        self.assertFalse(self.go(files, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())

    def test_a_signature_by_another_key_installs_nothing(self):
        other = crypto.KeyPair.generate(b"someone else")
        files = self.files(sig=other.sign(L.canonical(self.manifest())).hex().encode())
        self.assertFalse(self.go(files, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())

    def test_any_key_in_the_list_will_do_so_a_key_can_be_rotated(self):
        with mock.patch.object(L, "RELEASE_PUBKEYS", ["00" * 32, "zz-not-hex", self.key.public.hex()]):
            self.assertTrue(self.go(self.files()))

    def test_no_key_configured_installs_nothing(self):
        with mock.patch.object(L, "RELEASE_PUBKEYS", []):
            self.assertFalse(self.go(self.files(), stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())
        self.assertIn("no release key: update skipped", self.said)

    def test_no_key_on_a_first_install_stops_with_the_reason(self):
        with mock.patch.object(L, "RELEASE_PUBKEYS", []), self.assertRaises(SystemExit) as ctx:
            self.go(self.files(), installed=None)
        self.assertEqual(str(ctx.exception), "This installer can't verify Vibe Hash releases yet. "
                         "Download the latest installer from vibehash.org.")
        self.assertEqual(self.calls, [])

    def test_an_older_or_same_version_installs_nothing(self):
        self.assertFalse(self.go(self.files(), installed="10.0.0", stamp=True))  # a replayed old manifest: not even a stamp
        self.assertFalse(self.stamped())
        self.assertFalse(self.go(self.files(), installed="9.9.9", stamp=True))  # current: checked, so not asked again today
        self.assertTrue(self.stamped())
        self.assertEqual(self.calls, [])

    def test_versions_compare_as_numbers_not_text(self):
        m = self.manifest(version="0.10.0", requirements="requirements-0.10.0.txt")
        self.assertTrue(self.go(self.files(manifest=m), installed="0.9.9"))

    def test_a_manifest_that_is_not_json_installs_nothing(self):
        files = self.files()
        files[f"{L.RELEASES}/release.json"] = b"<html>not found</html>"
        self.assertFalse(self.go(files, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())

    def test_requirements_that_do_not_match_the_manifest_install_nothing(self):
        files = self.files(reqs=self.reqs + b"evil==1.0 --hash=sha256:" + b"d" * 64 + b"\n")
        self.assertFalse(self.go(files, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())
        self.assertFalse((self.env / "requirements-9.9.9.txt").exists())

    def test_a_manifest_cannot_name_a_requirements_file_outside_the_release(self):
        files = self.files(manifest=self.manifest(requirements="../../evil.txt"))
        self.assertFalse(self.go(files, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())

    def test_a_launcher_older_than_the_release_wants_installs_nothing(self):
        self.assertFalse(self.go(self.files(manifest=self.manifest(min_launcher="99.0")), stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())

    def test_offline_after_a_first_install_carries_on(self):
        self.assertFalse(self.go({}, stamp=True))
        self.assertEqual(self.calls, [])
        self.assertIn("carrying on", self.said[-1])
        self.assertFalse(self.stamped())

    def test_offline_on_a_first_install_stops_and_names_the_no_manifest_case(self):
        with self.assertRaises(SystemExit) as ctx:
            self.go({}, installed=None)
        self.assertIn("no manifest yet", str(ctx.exception))
        self.assertNotIn("carrying on", str(ctx.exception))

    def test_an_installed_version_that_cannot_be_compared_installs_nothing_unless_it_is_the_release_itself(self):
        self.assertFalse(self.go(self.files(), installed="0.5.28.dev0", stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())
        self.assertIn("can't compare", self.said[-1])
        self.assertFalse(self.go(self.files(manifest=self.manifest(version="9.9.9")), installed="9.9.9", stamp=True))  # (comparable: equal)
        self.assertEqual(self.calls, [])

    def test_extra_manifest_fields_are_tolerated_but_missing_ones_are_not(self):
        self.assertTrue(self.go(self.files(manifest=self.manifest(future_field="x"))))
        m = self.manifest()
        del m["requirements_evo"]
        self.calls.clear()
        self.assertFalse(self.go(self.files(manifest=m), stamp=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.stamped())

    def test_the_launchers_fields_and_canonical_form_are_the_packages(self):
        from vibehash import update
        self.assertEqual(L.FIELDS, update.FIELDS)
        sample = {"b": [1, {"é": "naïve \u2603", "a": None}], "a": 1.5, "c": "x y"}
        self.assertEqual(L.canonical(sample), crypto.canonical(sample))

    def test_the_launcher_installs_the_base_file_never_the_evo_one(self):
        self.go(self.files())
        self.assertNotIn("-evo", self.calls[0][-2])

    def test_the_unverified_package_is_for_developers_only(self):
        line = next(ln for ln in (ROOT / "launcher" / "vibehash_launcher.py").read_text(encoding="utf-8").splitlines()
                    if ln.startswith("PACKAGE ="))
        self.assertIn("DEV ONLY", line)

    def test_a_tag_build_is_refused_while_the_release_keys_are_empty(self):
        text = (ROOT / ".github" / "workflows" / "launcher.yml").read_text(encoding="utf-8")
        self.assertIn("if: github.ref_type == 'tag'", text)

    def test_a_failed_install_after_a_first_one_carries_on_and_a_first_one_stops(self):
        run = lambda cmd, **k: 1  # noqa: E731
        kw = dict(fetch=self.fetcher(self.files()), installed_version=lambda p: "0.5.27", now=NOW, run=run)
        (self.env / ".vibehash-checked").write_text("1.0\n")
        self.assertFalse(L.ensure_vibehash(self.uv, self.python, self.env, **kw))
        self.assertFalse(self.stamped())
        (self.env / ".vibehash-checked").unlink()
        with self.assertRaises(SystemExit):
            L.ensure_vibehash(self.uv, self.python, self.env, **kw)

    def test_the_dev_escape_hatch_is_the_old_unverified_command_and_says_so_loudly(self):
        os.environ["VIBEHASH_DEV"] = "1"
        with mock.patch.object(L, "RELEASE_PUBKEYS", []):
            self.assertTrue(self.go({}))
        self.assertEqual(self.calls, [[str(self.uv), "pip", "install", "--python", str(self.python), "--upgrade-package",
                                       "vibehash", "--quiet", L.PACKAGE]])
        self.assertTrue(any("VIBEHASH_DEV" in line and "NOT verified" in line for line in self.said), self.said)

    def test_nothing_but_exactly_one_bypasses_verification(self):
        for value in ("0", "", "true", "yes"):
            os.environ["VIBEHASH_DEV"] = value
            self.calls.clear()
            self.assertFalse(self.go(self.files(sig=b"00" * 64), stamp=True), value)
            self.assertEqual(self.calls, [], value)
            self.assertFalse(self.stamped(), value)

    def test_the_launcher_reports_its_version_to_what_it_starts(self):
        self.assertEqual(L.VERSION, "1.2")
        with mock.patch.object(L.sys, "frozen", True, create=True):
            env = L.child_env(Path("f"))
        self.assertEqual(env["VIBEHASH_LAUNCHER_VERSION"], L.VERSION)

    def test_the_verifier_is_bundled_into_the_exe(self):
        text = (ROOT / ".github" / "workflows" / "launcher.yml").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("--hidden-import ed25519_verify"), 2)  # the Windows and the console builds

    def test_the_launcher_has_no_third_party_imports(self):
        import ast
        std = set(sys.stdlib_module_names) | {"ed25519_verify", "webview", "winreg", "tkinter", "vibehash"}
        for name in ("vibehash_launcher.py", "ed25519_verify.py"):
            tree = ast.parse((ROOT / "launcher" / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                else:
                    mods = []
                for mod in mods:
                    self.assertIn(mod.split(".")[0], std, f"{name} imports {mod}")


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Never this machine's Start menu, desktop or registry: a home of its own, and winreg stood in for
        self.env = mock.patch.dict(os.environ, {"APPDATA": self.tmp.name + "/AppData", "USERPROFILE": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.winreg = FakeWinreg()
        self.reg = mock.patch.object(L, "_winreg", return_value=self.winreg)
        self.reg.start()
        self.addCleanup(self.reg.stop)
        self.ask = mock.patch.object(L.screen, "ask", return_value=True)
        self.ask.start()
        self.addCleanup(self.ask.stop)

    def test_uninstall_removes_the_launchers_folder_shortcuts_and_entry_only(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": self.tmp.name + "/vh"}), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            tmp = self.tmp.name
            folder = Path(tmp) / "vh"
            folder.mkdir()
            (folder / "uv.exe").write_text("")
            wallet = Path(tmp) / ".vibehash" / "wallet.json"  # not the launcher's: must stay
            wallet.parent.mkdir()
            wallet.write_text("{}")
            links = L.shortcut_paths() + [Path(tmp) / "OneDrive" / "Desktop" / "Vibe Hash.lnk"]
            for link in links:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.write_text("")
            (folder / "shortcuts.txt").write_text(f"{links[-1]}\n")  # the one the shell put somewhere else
            L.register(folder / L.EXE, folder, winreg=self.winreg)
            python = L.env_python(folder / "env", "Windows")
            python.parent.mkdir(parents=True)
            python.write_text("")
            ran = []
            with mock.patch.object(L.platform, "system", return_value="Windows"), \
                    mock.patch.object(L.subprocess, "call", side_effect=lambda cmd, **kw: ran.append(cmd) or 0):
                self.assertEqual(L.main(["--uninstall"]), 0)
            self.assertEqual([c[-2:] for c in ran], [["app", "stop"], ["startup", "off"]])  # a running app is stopped first
            self.assertFalse(folder.exists())
            self.assertTrue(wallet.exists())
            self.assertFalse(any(link.exists() for link in links))
            self.assertNotIn(L.UNINSTALL_KEY, self.winreg.keys)
            self.assertIn("wallet", out.getvalue())

    def test_uninstall_with_yes_asks_nothing_and_shows_no_box(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": self.tmp.name + "/vh"}), \
                mock.patch.object(L.screen, "ask") as ask, mock.patch.object(L.screen, "tell") as tell, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            folder = Path(self.tmp.name) / "vh"
            folder.mkdir()
            self.assertEqual(L.main(["--uninstall", "--yes"]), 0)
            self.assertFalse(folder.exists())
            ask.assert_not_called()
            tell.assert_not_called()
            self.assertIn("removed", out.getvalue())

    def test_uninstall_asks_first_in_the_window(self):
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": self.tmp.name + "/vh"}), \
                mock.patch.object(L.screen, "ask", return_value=False), mock.patch("sys.stdout", new_callable=io.StringIO):
            folder = Path(self.tmp.name) / "vh"
            folder.mkdir()
            self.assertEqual(L.main(["--uninstall"]), 0)
            self.assertTrue(folder.exists())  # no

    def test_the_download_installs_itself_then_hands_over(self):
        tmp = Path(self.tmp.name)
        download, folder = tmp / "Downloads" / "VibeHash.exe", tmp / "vh"
        download.parent.mkdir()
        download.write_bytes(b"MZ launcher")
        ran = []
        made = [tmp / "Programs" / "Vibe Hash.lnk", tmp / "OneDrive" / "Desktop" / "Vibe Hash.lnk"]  # where the shell says
        run = lambda cmd, **kw: ran.append(cmd) or mock.Mock(returncode=0, stdout="".join(f"{p}\n" for p in made))  # noqa: E731
        installed = L.install_self(folder, "Windows", exe=download, run=run, winreg=self.winreg)
        self.assertEqual(installed, folder / "VibeHash.exe")
        self.assertEqual(installed.read_bytes(), b"MZ launcher")
        self.assertFalse((folder / "VibeHash.exe.new").exists())  # copied beside, then swapped in
        self.assertEqual(ran[0][:2], ["powershell", "-NoProfile"])  # the shortcuts, through the shell's COM object
        self.assertIn("GetFolderPath('Desktop')", ran[0][-1])  # the real desktop (OneDrive's, say), not a guess
        self.assertIn(str(installed), ran[0][-1])
        self.assertEqual((folder / "shortcuts.txt").read_text().splitlines(), [str(p) for p in made])  # remembered for uninstall
        entry = self.winreg.keys[L.UNINSTALL_KEY]
        self.assertEqual(entry["DisplayName"], "Vibe Hash")
        self.assertEqual(entry["UninstallString"], f'"{installed}" --uninstall')
        self.assertEqual(entry["DisplayVersion"], L.VERSION)
        # the installed copy itself: nothing to copy, no hand-over, the shortcuts not made again (a deleted one stays
        # deleted), the entry refreshed
        self.assertIsNone(L.install_self(folder, "Windows", exe=installed, run=run, winreg=self.winreg))
        self.assertEqual(len(ran), 1)
        self.assertIsNone(L.install_self(folder, "Linux", exe=download, run=run, winreg=self.winreg))  # Windows only
        # put there by the setup wizard (its uninstaller beside it): nothing to copy, no shortcuts, no entry of ours
        wizard = tmp / "Program Files" / "Vibe Hash"
        wizard.mkdir(parents=True)
        (wizard / "VibeHash.exe").write_bytes(b"MZ")
        (wizard / "unins000.exe").write_bytes(b"MZ")
        self.winreg.keys.pop(L.UNINSTALL_KEY)
        self.assertTrue(L.by_setup(wizard / "VibeHash.exe"))
        self.assertIsNone(L.install_self(folder, "Windows", exe=wizard / "VibeHash.exe", run=run, winreg=self.winreg))
        self.assertEqual(len(ran), 1)
        self.assertNotIn(L.UNINSTALL_KEY, self.winreg.keys)
        with mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": str(folder)}), mock.patch.object(L, "install_self", return_value=installed), \
                mock.patch.object(L.subprocess, "Popen") as popen, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(L.main(["--", "--device", "cpu"]), 0)
        self.assertEqual(popen.call_args.args[0], [str(installed), "--", "--device", "cpu"])  # the same arguments, to the installed copy

    def test_the_window_is_hosted_here_with_vibehash_from_the_environment(self):
        folder = Path(self.tmp.name) / "vh"
        sp = L.site_packages(folder / "env", "Linux")
        self.assertEqual(sp, folder / "env" / "lib" / f"python{L.PYTHON}" / "site-packages")
        self.assertEqual(L.site_packages(folder / "env", "Windows"), folder / "env" / "Lib" / "site-packages")
        shown = []
        fake_window = mock.Mock(show=lambda port: shown.append(port) or "window")
        with mock.patch.dict(sys.modules, {"webview": mock.Mock(), "vibehash": mock.Mock(window=fake_window), "vibehash.window": fake_window}), \
                mock.patch.object(L.sys, "path", list(sys.path)):
            self.assertEqual(L.window(folder, 7545, "Linux"), 0)
            self.assertEqual(shown, [7545])
            self.assertEqual(L.sys.path[0], str(sp))  # vibehash came from the environment
        ran = []
        with mock.patch.dict(sys.modules, {"webview": None}):  # a script run, no pywebview here: the environment's python shows it
            self.assertEqual(L.window(folder, 7545, "Linux", run=lambda cmd, **kw: ran.append(cmd) or 1), 1)  # no environment yet
            self.assertEqual(ran, [])
            python = L.env_python(folder / "env", "Linux")
            python.parent.mkdir(parents=True)
            python.write_text("")
            self.assertEqual(L.window(folder, 7545, "Linux", run=lambda cmd, **kw: ran.append(cmd) or 0), 0)
            self.assertEqual(ran[0], [str(python), "-m", "vibehash", "window", "--watch", "7545"])
        with mock.patch.object(L, "window", return_value=0) as w, mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(L.main(["--window", "7000"]), 0)
            self.assertEqual(w.call_args.args[1], 7000)
            self.assertEqual(L.main(["--window"]), 2)
            self.assertIn("--window needs", out.getvalue())

    def test_the_screen_talks_to_the_console_and_the_log(self):
        folder = Path(self.tmp.name) / "vh"
        s = L.Screen(folder, gui=False)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            s.say("hello")
            self.assertFalse(s.show())  # no window without a gui
            self.assertEqual(s.run_while(lambda: 42), 42)  # straight through
        self.assertIn("hello", out.getvalue())
        self.assertIn("hello", (folder / "launcher.log").read_text())

    def test_main_installs_then_runs_the_app_and_passes_arguments_through(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp}), \
                mock.patch.object(L, "install_self", return_value=None), \
                mock.patch.object(L, "ensure_uv", return_value=Path("uv")), \
                mock.patch.object(L, "ensure_env", return_value=Path("py")), \
                mock.patch.object(L, "ensure_vibehash") as vh, mock.patch.object(L, "run_app", return_value=0) as ra, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(L.main(["--update", "--", "--device", "cpu"]), 0)
            self.assertTrue(vh.call_args.kwargs["force"])
            self.assertEqual(ra.call_args.args[1], ["--device", "cpu"])
            self.assertEqual(ra.call_args.kwargs["log"], Path(tmp) / "app.log")
            self.assertEqual(L.main([]), 0)
            self.assertFalse(vh.call_args.kwargs["force"])

    def test_an_install_problem_is_a_line_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VIBEHASH_LAUNCHER_DIR": tmp}), \
                mock.patch.object(L, "install_self", return_value=None), \
                mock.patch.object(L, "ensure_uv", side_effect=SystemExit("no uv for you")), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(L.main([]), 1)
            self.assertIn("no uv for you", out.getvalue())


class AppScriptTests(unittest.TestCase):
    """app.ps1, the one-line route to the same launcher (irm https://vibehash.org/app.ps1 | iex)."""

    def test_it_pins_the_launchers_uv_so_the_launcher_reuses_its_download(self):
        text = (Path(__file__).resolve().parent.parent / "app.ps1").read_text(encoding="utf-8")
        self.assertIn(f"$uvVersion = '{L.UV_VERSION}'", text)

    def test_it_is_ascii(self):  # irm may not decode it as UTF-8
        (Path(__file__).resolve().parent.parent / "app.ps1").read_bytes().decode("ascii")


if __name__ == "__main__":
    unittest.main()
