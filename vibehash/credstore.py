"""This user's credential store (Windows Credential Manager, macOS Keychain, Secret Service on Linux), through the
`keyring` package: a few secrets kept by (service, account) outside any file of ours. Generic on purpose: the
wallet's authenticator key lives here (`vibehash wallet mfa`), and so can anything else a command wants kept away
from the config folder.

`keyring` is imported inside the functions, so the core package imports without it (it is in the `app` extra); with
it missing, or with no backend on this machine (a headless server), `available()` is False and `get` answers None.

What this is not: a defence against malware that already runs as this user, which can ask the same store."""
from __future__ import annotations

WALLET_TOTP = "vibehash-totp"  # the service a wallet's authenticator key is kept under, by its address (`wallet mfa`)
PROBE_SERVICE = "vibehash-probe"
PROBE_ACCOUNT = "probe"


def _keyring():
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def available() -> bool:
    """Whether a real credential store works here: the package imports, its backend isn't the one that refuses
    everything, and a probe entry can be written, read back and deleted."""
    keyring = _keyring()
    if keyring is None:
        return False
    try:
        from keyring.backends import fail
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return False
        keyring.set_password(PROBE_SERVICE, PROBE_ACCOUNT, "probe")
        try:
            return keyring.get_password(PROBE_SERVICE, PROBE_ACCOUNT) == "probe"
        finally:
            keyring.delete_password(PROBE_SERVICE, PROBE_ACCOUNT)
    except Exception:  # noqa: BLE001  (backends raise their own kinds: a locked keychain, no D-Bus, ...)
        return False


def get(service: str, account: str) -> str | None:
    """What is kept for (service, account), or None: nothing, no store, or one that can't be read now."""
    keyring = _keyring()
    if keyring is None:
        return None
    try:
        return keyring.get_password(service, account)
    except Exception:  # noqa: BLE001
        return None


def put(service: str, account: str, value: str) -> None:
    """Keep `value` for (service, account), replacing what was there. RuntimeError if there is no store."""
    keyring = _keyring()
    if keyring is None:
        raise RuntimeError("the keyring package isn't installed (pip install keyring)")
    try:
        keyring.set_password(service, account, value)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the credential store refused it: {e}") from e


def delete(service: str, account: str) -> None:
    """Forget (service, account); nothing kept is not an error."""
    keyring = _keyring()
    if keyring is None:
        return
    try:
        keyring.delete_password(service, account)
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the credential store refused it: {e}") from e
