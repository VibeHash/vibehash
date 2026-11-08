"""Wallet key files.

With a password, the secret key is encrypted with AES-256-GCM under a key derived by scrypt (N=2^15, r=8, p=1). That
needs the `cryptography` package; we don't hand-roll a cipher. Without one, on a desktop, the key is a random AES-256
key kept in the OS credential store (Windows Credential Manager, macOS Keychain, Secret Service) and the file holds
only ciphertext (version 2: unreadable by releases before 0.5.28). The secret is stored in the clear only with
`--insecure-plaintext` (tests, throwaway keys) or where no store exists.

Keys made from recovery words (mnemonic.py) are stored the same way, plus the derivation path they came from, so the
words can restore them anywhere. The app never writes the words to disk: they are shown once, to be written down.
Headless setups (nohup, Docker) still get a words file, because nobody is watching a screen. Files made before
recovery words, without the path, load as before.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import credstore
from .crypto import KeyPair, public_key
from .mnemonic import SCHEME

SCRYPT = {"n": 2**15, "r": 8, "p": 1}
STORE_SERVICE = "vibehash"  # the credential-store service a version 2 wallet's key is kept under (its address is the account)
RESTORE_HINT = "restore the wallet from its 24 recovery words (vibehash wallet restore)"


class WalletError(Exception):
    pass


def _kdf(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=128 * r * n * 2, dklen=32)


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise WalletError("encrypted wallets need the `cryptography` package: pip install cryptography") from e
    return AESGCM


def _write(doc: dict, path: Path) -> None:
    """The wallet file, new and readable by this user only; an existing one is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, indent=2)


def _head(keys: KeyPair, hd_path: str | None) -> dict:
    doc = {"version": 1, "address": keys.address, "public": keys.public.hex()}
    if hd_path:
        doc["hd"] = {"scheme": SCHEME, "path": hd_path}
    return doc


def _store_doc(keys: KeyPair, hd_path: str | None) -> dict:
    """A version 2 document for `keys`, after putting its random key in the credential store (the caller removes
    that entry again if the file then can't be written). WalletError when there is no store."""
    if not credstore.available():
        raise WalletError("no credential store works on this system, so the key can't be kept out of the file; set "
                          "a password, or pass --insecure-plaintext for a throwaway key")
    key, nonce = os.urandom(32), os.urandom(12)
    try:
        credstore.put(STORE_SERVICE, keys.address, key.hex())
    except credstore.Unavailable as e:
        raise WalletError(f"the credential store refused the wallet's key: {e}") from e
    doc = _head(keys, hd_path)
    doc["version"] = 2
    doc["crypto"] = {"kdf": "keyring", "service": STORE_SERVICE, "account": keys.address, "cipher": "aes-256-gcm",
                     "nonce": nonce.hex(), "ciphertext": _aesgcm()(key).encrypt(nonce, keys.secret, keys.public).hex()}
    return doc


def _forget(address: str) -> None:
    try:
        credstore.delete(STORE_SERVICE, address)
    except credstore.Unavailable:
        pass


def save(keys: KeyPair, path: str | Path, password: str | None, hd_path: str | None = None, *, store: bool = False) -> None:
    """`hd_path`: the derivation path, when the key came from recovery words. With no password, `store=True` keeps
    the key in the OS credential store and only ciphertext in the file (version 2); the store entry is written first
    and removed again if the file can't be. Without a password and without `store` the secret is written in the
    clear, which is for tests and for `--insecure-plaintext`."""
    path = Path(path)
    if path.exists():
        raise WalletError(f"{path} already exists; refusing to overwrite a wallet")
    doc = _head(keys, hd_path)
    if password:
        salt, nonce = os.urandom(16), os.urandom(12)
        key = _kdf(password, salt, **SCRYPT)
        doc["crypto"] = {"kdf": "scrypt", **SCRYPT, "salt": salt.hex(), "cipher": "aes-256-gcm", "nonce": nonce.hex(),
                         "ciphertext": _aesgcm()(key).encrypt(nonce, keys.secret, keys.public).hex()}
    elif store:
        doc = _store_doc(keys, hd_path)
    else:
        doc["secret"] = keys.secret.hex()
    try:
        _write(doc, path)
    except BaseException:
        if doc["version"] == 2:
            _forget(keys.address)
        raise


def load(path: str | Path, password: str | None) -> KeyPair:
    path = Path(path)
    doc = json.loads(path.read_text())
    if "secret" in doc:
        secret = bytes.fromhex(doc["secret"])
    elif doc.get("crypto", {}).get("kdf") == "keyring":
        c = doc["crypto"]
        try:
            held = credstore.get(c["service"], c["account"])
        except credstore.Unavailable as e:
            raise WalletError(f"this wallet's key is in the credential store, which can't be read now ({e}); unlock "
                              f"it and try again, or {RESTORE_HINT}") from e
        if held is None:
            raise WalletError("this wallet's key is in another user's or computer's credential store; " + RESTORE_HINT)
        try:
            secret = _aesgcm()(bytes.fromhex(held)).decrypt(bytes.fromhex(c["nonce"]), bytes.fromhex(c["ciphertext"]),
                                                            bytes.fromhex(doc["public"]))
        except WalletError:
            raise
        except Exception as e:
            raise WalletError("the credential store's key doesn't open this wallet (corrupted?); " + RESTORE_HINT) from e
    else:
        c = doc["crypto"]
        if not password:
            raise WalletError("this wallet is encrypted; a password is required")
        key = _kdf(password, bytes.fromhex(c["salt"]), c["n"], c["r"], c["p"])
        try:
            secret = _aesgcm()(key).decrypt(bytes.fromhex(c["nonce"]), bytes.fromhex(c["ciphertext"]), bytes.fromhex(doc["public"]))
        except Exception as e:
            raise WalletError("wrong password or corrupted wallet") from e
    keys = KeyPair(secret, public_key(secret))
    if keys.public.hex() != doc["public"]:
        raise WalletError("wallet file is inconsistent")
    if doc.get("version") == 2:
        _backup(path).unlink(missing_ok=True)  # a migration's copy of the old plaintext: gone once the new file has worked
    return keys


def _backup(path: Path) -> Path:
    return Path(str(path) + ".bak")


def kind(path: str | Path) -> str:
    """"plain" (the secret is in the file), "password" (scrypt) or "store" (the key is in the credential store)."""
    doc = json.loads(Path(path).read_text())
    if "secret" in doc:
        return "plain"
    return "store" if doc.get("crypto", {}).get("kdf") == "keyring" else "password"


def is_encrypted(path: str | Path) -> bool:
    return kind(path) != "plain"


def migrate_plaintext(path: str | Path) -> bool:
    """Move a plaintext wallet's key into the credential store: the file is replaced, whole, by a version 2 one, and
    a copy of the old file stays at `<path>.bak` until the next `load` of the new one works (so a store that turns
    out not to answer costs nothing). Never when $VIBEHASH_PASSWORD is set (headless setups own their files), when
    the file isn't plaintext, or when there is no store. -> whether it migrated."""
    path = Path(path)
    if "VIBEHASH_PASSWORD" in os.environ or kind(path) != "plain" or not credstore.available():
        return False
    old = path.read_text()
    keys = load(path, None)
    doc = _store_doc(keys, hd_path(path))
    tmp, bak = path.with_suffix(".tmp"), _backup(path)
    try:
        for target, text in ((tmp, json.dumps(doc, indent=2)), (bak, old)):
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        bak.unlink(missing_ok=True)
        _forget(keys.address)
        raise
    return True


def hd_path(path: str | Path) -> str | None:
    """The derivation path of a wallet made from recovery words, or None for one made without them."""
    return (json.loads(Path(path).read_text()).get("hd") or {}).get("path")
