"""vibehash/update.py: the verified self-update (`vibehash update`, the menu). A fake fetcher stands in for the release
download; nothing goes online and nothing is installed."""
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from vibehash import crypto, settings, update

ROOT = Path(__file__).resolve().parent.parent
RELEASES = settings.RELEASES


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Base(unittest.TestCase):
    def setUp(self):
        self.key = crypto.KeyPair.generate(b"update-test")
        patch = mock.patch.object(settings, "RELEASE_PUBKEYS", [self.key.public.hex()])
        patch.start()
        self.addCleanup(patch.stop)
        self.said = []
        self.reqs = ("numpy==2.0.0 \\\n    --hash=sha256:" + "a" * 64 + "\n").encode()
        self.evo = self.reqs + ("transformers==4.0 \\\n    --hash=sha256:" + "d" * 64 + "\n").encode()

    def manifest(self, **over):
        return {"version": "9.9.9", "wheel": "vibehash-9.9.9-py3-none-any.whl", "wheel_sha256": "b" * 64,
                "requirements": "requirements-9.9.9.txt", "requirements_sha256": hashlib.sha256(self.reqs).hexdigest(),
                "requirements_evo": "requirements-9.9.9-evo.txt", "requirements_evo_sha256": hashlib.sha256(self.evo).hexdigest(),
                "published": 1700000000, "min_launcher": "1.2"} | over

    def files(self, manifest=None, sig=None, reqs=None):
        manifest = self.manifest() if manifest is None else manifest
        sig = self.key.sign(crypto.canonical(manifest)).hex() if sig is None else sig
        return {f"{RELEASES}/release.json": json.dumps(manifest, indent=2).encode(), f"{RELEASES}/release.sig": sig.encode(),
                f"{RELEASES}/requirements-9.9.9.txt": self.reqs if reqs is None else reqs,
                f"{RELEASES}/requirements-9.9.9-evo.txt": self.evo}

    def fetcher(self, files):
        def fetch(url):
            if url not in files:
                raise OSError(f"offline: {url}")
            return files[url]
        return fetch

    def check(self, files, **kw):
        return update.check(fetch=self.fetcher(files), say=self.said.append, **kw)


class CheckTests(Base):
    def test_a_signed_manifest_is_returned_verified(self):
        m = self.check(self.files())
        self.assertEqual(m["version"], "9.9.9")
        self.assertIsInstance(m, update.Verified)
        self.assertEqual(self.said, [])

    def test_the_fields_are_the_ones_the_signer_writes(self):
        self.assertEqual(update.FIELDS, load_script("release_manifest").FIELDS)
        self.assertEqual(len(update.FIELDS), 9)

    def test_extra_fields_are_tolerated_but_missing_ones_are_not(self):
        self.assertIsNotNone(self.check(self.files(manifest=self.manifest(future_field=1))))
        m = self.manifest()
        del m["requirements_evo_sha256"]
        self.assertIsNone(self.check(self.files(manifest=m)))

    def test_a_manifest_signed_by_the_release_script_verifies_here_and_in_the_launcher(self):
        import tempfile
        rm = load_script("release_manifest")
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            (dist / "vibehash-9.9.9-py3-none-any.whl").write_bytes(b"wheel")
            (dist / "requirements-9.9.9.txt").write_text("numpy==2.0.0 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
            (dist / "requirements-9.9.9-evo.txt").write_text("numpy==2.0.0 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
            rm.make_release(dist, "9.9.9", env={"RELEASE_SIGNING_KEY": self.key.secret.hex()})
            files = {f"{RELEASES}/release.json": (dist / "release.json").read_bytes(),
                     f"{RELEASES}/release.sig": (dist / "release.sig").read_bytes()}
            self.assertEqual(self.check(files)["version"], "9.9.9")
            spec = importlib.util.spec_from_file_location("vibehash_launcher_x", ROOT / "launcher" / "vibehash_launcher.py")
            launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launcher)
            manifest, why = launcher.verify_manifest(files[f"{RELEASES}/release.json"], files[f"{RELEASES}/release.sig"].decode(),
                                                     keys=[self.key.public.hex()])
            self.assertEqual((manifest["version"], why), ("9.9.9", ""))

    def test_a_bad_signature_is_refused_with_a_reason(self):
        files = self.files()
        sig = bytearray(bytes.fromhex(files[f"{RELEASES}/release.sig"].decode()))
        sig[5] ^= 1
        files[f"{RELEASES}/release.sig"] = bytes(sig).hex().encode()
        self.assertIsNone(self.check(files))
        self.assertEqual(len(self.said), 1)
        self.assertIn("signature", self.said[0])

    def test_a_manifest_edited_after_signing_is_refused(self):
        files = self.files()
        files[f"{RELEASES}/release.json"] = json.dumps(self.manifest(wheel_sha256="c" * 64)).encode()
        self.assertIsNone(self.check(files))

    def test_no_key_configured_refuses(self):
        with mock.patch.object(settings, "RELEASE_PUBKEYS", []):
            self.assertIsNone(self.check(self.files()))
        self.assertIn("no release key", self.said[0])

    def test_a_key_that_is_not_hex_is_skipped_not_fatal(self):
        with mock.patch.object(settings, "RELEASE_PUBKEYS", ["not hex", self.key.public.hex()]):
            self.assertIsNotNone(self.check(self.files()))

    def test_not_json_and_not_a_dict_and_missing_fields_are_refused(self):
        for body in (b"<html>", b"[1, 2]", json.dumps({"version": "9.9.9"}).encode()):
            files = self.files()
            files[f"{RELEASES}/release.json"] = body
            self.assertIsNone(self.check(files), body)
        partial = {"version": "9.9.9"}  # (validly signed, but not a whole manifest)
        self.assertIsNone(self.check(self.files(manifest=partial)))

    def test_offline_is_a_reason_not_a_traceback(self):
        self.assertIsNone(self.check({}))
        self.assertIn("offline", self.said[0])

    def test_a_requirements_name_cannot_leave_the_release(self):
        self.assertIsNone(self.check(self.files(manifest=self.manifest(requirements="../x.txt"))))


class InstallTests(Base):
    def verified(self, **over):
        return self.check(self.files(manifest=self.manifest(**over)))

    def install(self, manifest, files=None, code=0):
        calls = []
        rc = update.install(manifest, fetch=self.fetcher(self.files() if files is None else files),
                            run=lambda cmd, **kw: calls.append(cmd) or code, say=self.said.append)
        return rc, calls

    def test_it_installs_the_hashed_file_with_hashes_required(self):
        rc, calls = self.install(self.verified())
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][:5], [sys.executable, "-m", "pip", "install", "--require-hashes"])
        self.assertEqual(calls[0][5], "-r")
        self.assertTrue(calls[0][6].endswith("requirements-9.9.9.txt"))

    def test_pips_exit_code_is_returned(self):
        self.assertEqual(self.install(self.verified(), code=2)[0], 2)

    def test_the_file_pip_reads_is_exactly_what_was_hashed(self):
        seen = []

        def run(cmd, **kw):
            seen.append(Path(cmd[6]).read_bytes())
            return 0
        update.install(self.verified(), fetch=self.fetcher(self.files()), run=run, say=self.said.append)
        self.assertEqual(seen, [self.reqs])

    def test_requirements_that_do_not_match_are_never_installed(self):
        rc, calls = self.install(self.verified(), files=self.files(reqs=self.reqs + b"evil==1\n"))
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertIn("hash", self.said[-1])

    def test_the_evo_variant_installs_the_evo_file_and_checks_its_own_hash(self):
        seen = []

        def run(cmd, **kw):
            seen.append((cmd[6], Path(cmd[6]).read_bytes()))
            return 0
        rc = update.install(self.verified(), variant="evo", fetch=self.fetcher(self.files()), run=run, say=self.said.append)
        self.assertEqual(rc, 0)
        self.assertTrue(seen[0][0].endswith("requirements-9.9.9-evo.txt"))
        self.assertEqual(seen[0][1], self.evo)
        files = self.files()
        files[f"{RELEASES}/requirements-9.9.9-evo.txt"] = self.evo + b"evil==1\n"
        calls = []
        rc = update.install(self.verified(), variant="evo", fetch=self.fetcher(files), run=lambda c, **k: calls.append(c) or 0,
                            say=self.said.append)
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertNotEqual(update.install(self.verified(), variant="nope", fetch=self.fetcher(self.files()),
                                           run=lambda c, **k: 0, say=self.said.append), 0)

    def test_a_manifest_that_did_not_come_from_check_is_refused(self):
        rc, calls = self.install(self.manifest())
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])

    def test_offline_while_fetching_the_requirements(self):
        files = self.files()
        del files[f"{RELEASES}/requirements-9.9.9.txt"]
        rc, calls = self.install(self.verified(), files=files)
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])


class UpdateFlowTests(Base):
    def go(self, files, installed="0.5.27", code=0, **kw):
        calls = []
        rc = update.update(fetch=self.fetcher(files), run=lambda cmd, **k: calls.append(cmd) or code, say=self.said.append,
                           installed=installed, **kw)
        return rc, calls

    def test_a_newer_verified_release_is_named_and_installed(self):
        rc, calls = self.go(self.files())
        self.assertEqual(rc, 0)
        self.assertIn("--require-hashes", calls[0])
        self.assertTrue(any("9.9.9" in line and "verified" in line for line in self.said), self.said)

    def test_the_same_version_installs_nothing_unless_asked_to(self):
        rc, calls = self.go(self.files(), installed="9.9.9")
        self.assertEqual((rc, calls), (0, []))
        self.assertTrue(any("up to date" in line for line in self.said))
        rc, calls = self.go(self.files(), installed="9.9.9", reinstall=True)  # (what installing the Evo extras does)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)

    def test_an_older_release_is_never_installed_even_when_asked(self):
        for reinstall in (False, True):
            rc, calls = self.go(self.files(), installed="10.0.0", reinstall=reinstall)
            self.assertNotEqual(rc, 0, reinstall)
            self.assertEqual(calls, [])

    def test_an_installed_version_that_cannot_be_compared_changes_nothing(self):
        rc, calls = self.go(self.files(), installed="0.5.28.dev0")
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])

    def test_evo_variant_reinstalls_even_when_current(self):
        rc, calls = self.go(self.files(), installed="9.9.9", reinstall=True, variant="evo")
        self.assertEqual(rc, 0)
        self.assertTrue(calls[0][6].endswith("-evo.txt"))

    def test_verification_failure_is_nonzero_and_installs_nothing(self):
        with mock.patch.object(settings, "RELEASE_PUBKEYS", []):
            rc, calls = self.go(self.files())
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])


class CliTests(unittest.TestCase):
    def test_the_command_runs_the_same_update_and_exits_with_its_code(self):
        from vibehash import cli
        args = cli.parser().parse_args(["update"])
        self.assertIs(args.fn, cli.cmd_update)
        with mock.patch.object(update, "update", return_value=0) as upd:
            cli.cmd_update(args)
        upd.assert_called_once_with(reinstall=False, variant="base")
        with mock.patch.object(update, "update", return_value=0) as upd:
            cli.cmd_update(cli.parser().parse_args(["update", "--evo"]))
        upd.assert_called_once_with(reinstall=True, variant="evo")
        with mock.patch.object(update, "update", return_value=1), self.assertRaises(SystemExit) as ctx:
            cli.cmd_update(args)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
