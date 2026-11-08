"""vibehash/credstore.py (a thin wrapper over the OS credential store) and what the CLI builds on it: `wallet mfa`,
and the opt-in authenticator-code gate on send, stake and unstake."""
import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import cli, credstore, nodeadmin
from vibehash.crypto import KeyPair, checksummed

try:
    import keyring
    import keyring.backend
    import keyring.errors
except ImportError:  # the tests are then skipped
    keyring = None


class MemoryKeyring(keyring.backend.KeyringBackend if keyring else object):
    """An in-memory credential store: what a machine with a keychain looks like to credstore."""
    priority = 1

    def __init__(self):
        self.data = {}

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def get_password(self, service, username):
        return self.data.get((service, username))

    def delete_password(self, service, username):
        if (service, username) not in self.data:
            raise keyring.errors.PasswordDeleteError("no such entry")
        del self.data[(service, username)]


@unittest.skipIf(keyring is None, "keyring is not installed")
class Store(unittest.TestCase):
    """Each test gets a fresh in-memory store as the process's keyring."""

    def setUp(self):
        self.old = keyring.get_keyring()
        self.mem = MemoryKeyring()
        keyring.set_keyring(self.mem)
        self.addCleanup(keyring.set_keyring, self.old)


class CredstoreTests(Store):
    def test_put_get_delete_round_trip(self):
        self.assertTrue(credstore.available())
        self.assertIsNone(credstore.get("svc", "acct"))
        credstore.put("svc", "acct", "secret")
        self.assertEqual(credstore.get("svc", "acct"), "secret")
        credstore.delete("svc", "acct")
        self.assertIsNone(credstore.get("svc", "acct"))
        credstore.delete("svc", "acct")  # deleting what isn't there is not an error
        self.assertEqual(self.mem.data, {})  # the probe left nothing behind

    def test_a_failing_backend_is_not_available(self):
        from keyring.backends import fail
        keyring.set_keyring(fail.Keyring())
        self.assertFalse(credstore.available())

    def test_a_backend_that_cannot_round_trip_is_not_available(self):
        self.mem.get_password = lambda s, u: None  # stores, but never gives back
        self.assertFalse(credstore.available())

    def test_without_the_package_nothing_is_available_and_nothing_is_stored(self):
        with mock.patch.dict("sys.modules", {"keyring": None}):
            self.assertFalse(credstore.available())
            self.assertIsNone(credstore.get("svc", "acct"))


class WalletMfaTests(Store):
    NOW = 1_700_000_000.0

    def setUp(self):
        super().setUp()
        self.keys = KeyPair.generate(b"mfa-test")
        self.now = [self.NOW]
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in ("VIBEHASH_CODE", "VIBEHASH_PASSWORD"):
            os.environ.pop(var, None)
        for patch in (mock.patch.object(cli, "_now", lambda: self.now[0]),
                      mock.patch.object(cli, "load_wallet", return_value=self.keys),
                      mock.patch.object(cli, "wallet_of", return_value="w.json")):
            patch.start()
            self.addCleanup(patch.stop)
        cli._used_codes.clear()

    def code(self, step_offset=0):
        key = nodeadmin.mfa_key(credstore.get("vibehash-totp", self.keys.address))
        return nodeadmin.totp(key, int(self.now[0] // 30) + step_offset)

    def run_cmd(self, fn, **args):
        out, err = io.StringIO(), io.StringIO()
        ns = SimpleNamespace(**({"wallet": None, "node": None, "fee": 0, "code": None} | args))
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                fn(ns)
                status = None
            except SystemExit as e:
                status = e.code
        return status, out.getvalue(), err.getvalue()

    def turn_on(self):
        return self.run_cmd(cli.cmd_wallet_mfa, action="on")

    def test_on_stores_a_key_and_prints_the_uri_for_the_checksummed_address(self):
        status, out, _ = self.turn_on()
        self.assertIsNone(status)
        self.assertIsNotNone(credstore.get("vibehash-totp", self.keys.address))
        self.assertIn("otpauth://totp/", out)
        self.assertIn(checksummed(self.keys.address), out)
        self.assertIn("Vibe%20Hash", out)
        self.assertIn("does not stop malware", out)  # the honest scope is said
        self.assertIn("on", self.run_cmd(cli.cmd_wallet_mfa, action="status")[1])

    def test_status_says_off_before(self):
        self.assertIn("off", self.run_cmd(cli.cmd_wallet_mfa, action="status")[1])

    def test_on_twice_does_not_replace_the_key(self):
        self.turn_on()
        first = credstore.get("vibehash-totp", self.keys.address)
        status, _, _ = self.turn_on()
        self.assertNotEqual(status, None)
        self.assertEqual(credstore.get("vibehash-totp", self.keys.address), first)

    def test_on_is_refused_without_a_credential_store(self):
        with mock.patch.object(credstore, "available", return_value=False):
            status, _, _ = self.turn_on()
        self.assertIn("VIBEHASH_PASSWORD", str(status))
        self.assertIsNone(credstore.get("vibehash-totp", self.keys.address))

    def test_off_needs_a_current_code(self):
        self.turn_on()
        right = self.code()
        status, _, _ = self.run_cmd(cli.cmd_wallet_mfa, action="off", code="000000" if right != "000000" else "111111")
        self.assertEqual(status, 1)
        self.assertIsNotNone(credstore.get("vibehash-totp", self.keys.address))
        status, _, _ = self.run_cmd(cli.cmd_wallet_mfa, action="off", code=right)
        self.assertIsNone(status)
        self.assertIsNone(credstore.get("vibehash-totp", self.keys.address))

    def sends(self, fn=None, **args):
        """Run a send command against a fake client: (its exit status, the client)."""
        client = mock.MagicMock()
        client.min_fee.return_value = 0
        client.send_tx.return_value = "ab" * 32
        with mock.patch.object(cli, "NodeClient", return_value=client), mock.patch.object(cli, "node_of", return_value="http://x"):
            status, _, _ = self.run_cmd(fn or cli.cmd_stake, amount=5, **args)
        return status, client

    def test_with_no_key_a_send_goes_as_before(self):
        status, client = self.sends()
        self.assertIsNone(status)
        client.send_tx.assert_called_once()

    def test_with_the_gate_on_a_send_without_a_code_ends_before_a_transaction_is_built(self):
        self.turn_on()
        other = checksummed(KeyPair.generate(b"other").address)
        for fn, extra in ((cli.cmd_stake, {}), (cli.cmd_unstake, {}), (cli.cmd_send, {"to": other})):
            with mock.patch.object(cli.unattended, "interactive", return_value=False):
                status, client = self.sends(fn, **extra)
            self.assertEqual(status, 1, fn.__name__)
            client.send_tx.assert_not_called()
            client.min_fee.assert_not_called()
            self.assertEqual(client.mock_calls, [], fn.__name__)  # not even the node was asked

    def test_a_blank_prompt_ends_before_a_transaction_is_built(self):
        self.turn_on()
        with mock.patch("getpass.getpass", return_value=""), mock.patch.object(cli.unattended, "interactive", return_value=True):
            status, client = self.sends()
        self.assertEqual(status, 1)
        client.send_tx.assert_not_called()

    def test_a_wrong_code_ends_before_a_transaction_is_built(self):
        self.turn_on()
        right = {self.code(-1), self.code(), self.code(1)}
        wrong = next(c for c in ("123456", "654321", "000000") if c not in right)
        status, client = self.sends(code=wrong)
        self.assertEqual(status, 1)
        client.send_tx.assert_not_called()

    def test_the_right_code_goes_through_once_only(self):
        self.turn_on()
        code = self.code()
        status, client = self.sends(code=code)
        self.assertIsNone(status)
        client.send_tx.assert_called_once()
        status, client = self.sends(code=code)  # the same code again, in the same process
        self.assertEqual(status, 1)
        client.send_tx.assert_not_called()
        self.now[0] += 30  # the next step's code is good
        status, client = self.sends(code=self.code())
        self.assertIsNone(status)
        client.send_tx.assert_called_once()

    def test_a_step_either_side_is_taken(self):
        self.turn_on()
        status, _ = self.sends(code=self.code(-1))
        self.assertIsNone(status)

    def test_the_code_comes_from_the_flag_then_the_environment_then_a_prompt(self):
        self.turn_on()
        with mock.patch.dict(os.environ, {"VIBEHASH_CODE": self.code()}):
            status, _ = self.sends()
        self.assertIsNone(status)
        self.now[0] += 30
        with mock.patch("getpass.getpass", return_value=self.code()), mock.patch.object(cli.unattended, "interactive", return_value=True):
            status, _ = self.sends()
        self.assertIsNone(status)

    def test_help_says_what_the_gate_does_not_stop(self):
        self.assertIn("does not stop malware", cli.MFA_SCOPE)
        for name in ("send", "stake", "unstake"):
            with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                cli.main([name, "--help"])
            self.assertIn("--code", out.getvalue())


if __name__ == "__main__":
    unittest.main()
