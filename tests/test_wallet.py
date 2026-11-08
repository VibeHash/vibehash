"""Wallet key files (wallet.py): the v1 password and plaintext formats, the v2 format whose key lives in the OS
credential store, and the move from plaintext to v2. A fake in-memory keyring stands in for the real store."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import keyring
    import keyring.backend
    import keyring.errors
except ImportError:  # pragma: no cover
    keyring = None

from vibehash import credstore
from vibehash import wallet as wallets
from vibehash.crypto import KeyPair


def memory_backend():
    class Memory(keyring.backend.KeyringBackend):
        priority = 1

        def __init__(self):
            self.data = {}

        def set_password(self, service, username, password):
            self.data[(service, username)] = password

        def get_password(self, service, username):
            return self.data.get((service, username))

        def delete_password(self, service, username):
            if (service, username) not in self.data:
                raise keyring.errors.PasswordDeleteError("absent")
            del self.data[(service, username)]
    return Memory()


def use_memory_keyring(case: unittest.TestCase):
    """Swap the OS credential store for an in-memory one for this test (never the real Credential Manager); skips
    the test when keyring isn't installed. -> the backend, whose `.data` holds what was kept."""
    if keyring is None:
        case.skipTest("keyring is not installed")
    old = keyring.get_keyring()
    backend = memory_backend()
    keyring.set_keyring(backend)
    case.addCleanup(keyring.set_keyring, old)
    return backend


@unittest.skipIf(keyring is None, "keyring is not installed")
class StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "wallet.json"
        old = keyring.get_keyring()
        self.backend = memory_backend()
        keyring.set_keyring(self.backend)
        self.addCleanup(keyring.set_keyring, old)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("VIBEHASH_PASSWORD", None)
        self.keys = KeyPair.generate(b"wallet-test")

    def entries(self):
        return {k: v for k, v in self.backend.data.items() if k[0] == "vibehash"}


class StoreWalletTests(StoreCase):
    def test_v2_has_no_secret_and_round_trips_through_the_store(self):
        wallets.save(self.keys, self.path, None, "m/44'/0'/0'/0/0", store=True)
        doc = json.loads(self.path.read_text())
        self.assertEqual(doc["version"], 2)
        self.assertNotIn("secret", doc)
        self.assertNotIn(self.keys.secret.hex(), self.path.read_text())
        self.assertEqual(doc["crypto"]["kdf"], "keyring")
        self.assertEqual((doc["crypto"]["service"], doc["crypto"]["account"]), ("vibehash", self.keys.address))
        self.assertEqual(wallets.kind(self.path), "store")
        self.assertTrue(wallets.is_encrypted(self.path))
        self.assertEqual(wallets.hd_path(self.path), "m/44'/0'/0'/0/0")
        self.assertEqual(len(self.entries()), 1)
        self.assertEqual(wallets.load(self.path, None).secret, self.keys.secret)

    def test_a_missing_entry_says_the_words_restore_the_wallet(self):
        wallets.save(self.keys, self.path, None, store=True)
        self.backend.data.clear()
        with self.assertRaises(wallets.WalletError) as e:
            wallets.load(self.path, None)
        self.assertIn("24 recovery words", str(e.exception))

    def test_an_unreadable_store_is_its_own_error_and_falls_back_to_nothing(self):
        wallets.save(self.keys, self.path, None, store=True)
        with mock.patch("vibehash.credstore.get", side_effect=credstore.Unavailable("keychain is locked")):
            with self.assertRaises(wallets.WalletError) as e:
                wallets.load(self.path, "any password")
        self.assertIn("keychain is locked", str(e.exception))
        self.assertIn("24 recovery words", str(e.exception))
        self.assertNotIn("another user's", str(e.exception))  # not the "entry missing" message

    def test_without_a_store_save_refuses(self):
        with mock.patch("vibehash.credstore.available", return_value=False):
            with self.assertRaises(wallets.WalletError) as e:
                wallets.save(self.keys, self.path, None, store=True)
        self.assertIn("credential store", str(e.exception))
        self.assertFalse(self.path.exists())

    def test_a_failed_file_write_removes_the_store_entry(self):
        real = os.open

        def refuse(p, *a, **kw):
            if Path(p) == self.path:
                raise OSError("disk full")
            return real(p, *a, **kw)
        with mock.patch("vibehash.wallet.os.open", side_effect=refuse), self.assertRaises(OSError):
            wallets.save(self.keys, self.path, None, store=True)
        self.assertEqual(self.entries(), {})
        self.assertFalse(self.path.exists())

    def test_two_files_for_one_address_share_the_entry_and_both_load(self):
        wallets.save(self.keys, self.path, None, store=True)
        second = self.dir / "second.json"
        wallets.save(self.keys, second, None, store=True)  # a restore of the same words
        self.assertEqual(len(self.entries()), 1)
        self.assertEqual(wallets.load(self.path, None).secret, self.keys.secret)
        self.assertEqual(wallets.load(second, None).secret, self.keys.secret)
        self.assertNotEqual(json.loads(self.path.read_text())["crypto"]["nonce"], json.loads(second.read_text())["crypto"]["nonce"])

    def test_a_failed_second_write_leaves_the_first_file_loadable(self):
        wallets.save(self.keys, self.path, None, store=True)
        second = self.dir / "second.json"
        real = os.open

        def refuse(p, *a, **kw):
            if Path(p) == second:
                raise OSError("disk full")
            return real(p, *a, **kw)
        with mock.patch("vibehash.wallet.os.open", side_effect=refuse), self.assertRaises(OSError):
            wallets.save(self.keys, second, None, store=True)
        self.assertEqual(len(self.entries()), 1)  # the first file's entry stays
        self.assertEqual(wallets.load(self.path, None).secret, self.keys.secret)

    def test_a_leftover_bak_does_not_keep_its_old_permissions(self):
        wallets.save(self.keys, self.path, None)
        bak = Path(str(self.path) + ".bak")
        bak.write_text("old")
        os.chmod(bak, 0o666)
        self.assertTrue(wallets.migrate_plaintext(self.path))
        self.assertIn(self.keys.secret.hex(), bak.read_text())  # replaced, not appended to or truncated in place
        if os.name != "nt":
            self.assertEqual(bak.stat().st_mode & 0o777, 0o600)

    def test_an_existing_file_is_refused_before_any_store_write(self):
        self.path.write_text("{}")
        with self.assertRaises(wallets.WalletError):
            wallets.save(self.keys, self.path, None, store=True)
        self.assertEqual(self.entries(), {})


class V1Tests(StoreCase):
    def test_password_wallets_still_load(self):
        wallets.save(self.keys, self.path, "pw")
        self.assertEqual(wallets.kind(self.path), "password")
        self.assertTrue(wallets.is_encrypted(self.path))
        self.assertEqual(json.loads(self.path.read_text())["version"], 1)
        self.assertEqual(wallets.load(self.path, "pw").secret, self.keys.secret)
        with self.assertRaises(wallets.WalletError):
            wallets.load(self.path, "wrong")

    def test_plaintext_wallets_still_load(self):
        wallets.save(self.keys, self.path, None)
        self.assertEqual(wallets.kind(self.path), "plain")
        self.assertFalse(wallets.is_encrypted(self.path))
        self.assertEqual(wallets.load(self.path, None).secret, self.keys.secret)


class MigrationTests(StoreCase):
    def test_plaintext_becomes_v2_with_a_bak_until_the_next_load(self):
        wallets.save(self.keys, self.path, None, "m/44'/0'/0'/0/0")
        bak = Path(str(self.path) + ".bak")
        self.assertTrue(wallets.migrate_plaintext(self.path))
        self.assertEqual(wallets.kind(self.path), "store")
        self.assertEqual(json.loads(self.path.read_text())["address"], self.keys.address)
        self.assertEqual(wallets.hd_path(self.path), "m/44'/0'/0'/0/0")
        self.assertTrue(bak.exists())
        self.assertIn(self.keys.secret.hex(), bak.read_text())
        self.assertFalse(self.path.with_suffix(".tmp").exists())
        self.assertEqual(wallets.load(self.path, None).secret, self.keys.secret)
        self.assertFalse(bak.exists())

    def test_nothing_else_is_migrated(self):
        wallets.save(self.keys, self.path, None)
        with mock.patch.dict(os.environ, {"VIBEHASH_PASSWORD": "x"}):
            self.assertFalse(wallets.migrate_plaintext(self.path))
        self.assertEqual(wallets.kind(self.path), "plain")
        with mock.patch("vibehash.credstore.available", return_value=False):
            self.assertFalse(wallets.migrate_plaintext(self.path))
        self.assertEqual(wallets.kind(self.path), "plain")
        self.assertTrue(wallets.migrate_plaintext(self.path))
        self.assertFalse(wallets.migrate_plaintext(self.path))  # already v2
        other = self.dir / "pw.json"
        wallets.save(self.keys, other, "pw")
        self.assertFalse(wallets.migrate_plaintext(other))


if __name__ == "__main__":
    unittest.main()
