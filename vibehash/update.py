"""`vibehash update`, and the menu's Update: install a newer release only when it is signed.

Each release publishes release.json (the manifest), release.sig (an Ed25519 signature over crypto.canonical of the
manifest, by a key in settings.RELEASE_PUBKEYS) and requirements-<version>.txt (every package the release needs, each
pinned by hash, the release's own wheel included, PyTorch left out so the GPU build `install-gpu` chose stays) on the
latest GitHub release. `check()` fetches and verifies the manifest; `install()` fetches the requirements file the manifest
names, checks its sha256 against the manifest's, and runs `pip install --require-hashes -r` on it, so pip refuses any file
that isn't listed with a hash. Anything that doesn't verify installs nothing and says one line why.

The launcher (launcher/vibehash_launcher.py) does the same in its own stdlib-only code, as a separate process; the two
share nothing at run time but the manifest's field names (FIELDS, the same as scripts/release_manifest.py writes).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

from . import crypto, settings

FIELDS = ("version", "wheel", "wheel_sha256", "requirements", "requirements_sha256", "requirements_evo",
          "requirements_evo_sha256", "published", "min_launcher")  # (all required; extra fields are tolerated)
VARIANTS = {"base": ("requirements", "requirements_sha256"), "evo": ("requirements_evo", "requirements_evo_sha256")}


class Verified(dict):
    """A manifest whose signature `check()` verified. Only check() makes one, and `install()` takes nothing else, so
    a manifest dict from anywhere else can't be installed from."""


def fetch(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vibehash-update"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_version(text) -> tuple[int, ...] | None:
    if isinstance(text, str) and re.fullmatch(r"\d+(\.\d+)*", text):
        return tuple(int(part) for part in text.split("."))
    return None


def installed_version() -> str:
    from . import __version__
    return __version__


def check(fetch=fetch, say=print) -> Verified | None:
    """The latest release's manifest if its signature verifies against settings.RELEASE_PUBKEYS and it is a whole,
    sane manifest; else None, after `say`ing why (one line)."""
    keys = list(settings.RELEASE_PUBKEYS)
    if not keys:
        say("no release key is configured in this copy of Vibe Hash, so nothing can be verified: update skipped")
        return None
    try:
        body = fetch(f"{settings.RELEASES}/release.json")
        sig_text = fetch(f"{settings.RELEASES}/release.sig").decode("ascii", "replace")
    except OSError as e:
        say(f"couldn't fetch the release manifest (offline?): {e}")
        return None
    try:
        manifest = json.loads(body.decode("utf-8"))
        sig = bytes.fromhex(sig_text.strip())
    except ValueError:
        say("the release manifest isn't readable: update skipped")
        return None
    if not isinstance(manifest, dict):
        say("the release manifest isn't readable: update skipped")
        return None
    message = crypto.canonical(manifest)
    for key in keys:
        try:
            if crypto.verify(bytes.fromhex(key), message, sig):
                break
        except ValueError:
            continue  # (a key in the list that isn't hex: the others are still tried)
    else:
        say("the release's signature doesn't match any release key: update skipped")
        return None
    names = [manifest.get(VARIANTS[v][0]) for v in VARIANTS]
    if not set(FIELDS) <= set(manifest) or parse_version(manifest.get("version")) is None \
            or not all(isinstance(n, str) and re.fullmatch(r"requirements-[0-9][0-9.]*(-evo)?\.txt", n) for n in names) \
            or not all(isinstance(manifest.get(VARIANTS[v][1]), str) for v in VARIANTS):
        say("the release manifest is signed but not one this version understands: update skipped")
        return None
    return Verified(manifest)


def is_newer(manifest: dict, installed: str | None = None) -> bool:
    have = parse_version(installed if installed is not None else installed_version())
    return have is None or parse_version(manifest["version"]) > have


def install(manifest: dict, variant: str = "base", fetch=fetch, run=subprocess.call, say=print) -> int:
    """pip's exit code for installing exactly the requirements file `manifest` (from check()) names, with
    --require-hashes: the "base" file (the launcher's set: finetune and app) or the "evo" one (those plus the Evo extras,
    a few hundred MB more); nonzero, with nothing run, when the file isn't the one the manifest hashes."""
    if variant not in VARIANTS:
        say(f"not installing: no such set of packages as {variant!r}")
        return 1
    if not isinstance(manifest, Verified):
        say("not installing: that manifest wasn't verified")
        return 1
    try:
        name_key, hash_key = VARIANTS[variant]
        reqs = fetch(f"{settings.RELEASES}/{manifest[name_key]}")
    except OSError as e:
        say(f"couldn't fetch the requirements file (offline?): {e}")
        return 1
    if hashlib.sha256(reqs).hexdigest() != manifest[hash_key].lower():
        say("the requirements file doesn't match the hash in the signed manifest: nothing installed")
        return 1
    with tempfile.TemporaryDirectory(prefix="vibehash-update-") as tmp:
        path = os.path.join(tmp, manifest[name_key])
        with open(path, "wb") as f:
            f.write(reqs)
        cmd = [sys.executable, "-m", "pip", "install", "--require-hashes", "-r", path]
        say("> " + " ".join(cmd))
        return run(cmd)


def update(reinstall: bool = False, variant: str = "base", fetch=fetch, run=subprocess.call, say=print,
           installed: str | None = None) -> int:
    """The whole thing, for `vibehash update` and the menu: check, say which version was verified, install. Nothing
    newer: nothing is done, unless `reinstall` (put back the current release's packages, e.g. for the Evo extras,
    which the requirements include); an older release is never installed. -> an exit code (0: done or already current)."""
    manifest = check(fetch=fetch, say=say)
    if manifest is None:
        return 1
    have = installed if installed is not None else installed_version()
    say(f"Vibe Hash {manifest['version']} is the latest release, and its signature is verified (installed: {have}).")
    if parse_version(have) is None and manifest["version"] != have:  # (0.5.28.dev0, say): newer or older can't be told
        say(f"can't compare the installed version {have!r} with {manifest['version']}; nothing changed")
        return 1
    if not is_newer(manifest, have):
        older = parse_version(manifest["version"]) < (parse_version(have) or ())
        if older:
            say("that is older than what is installed; nothing changed")
            return 1
        if not reinstall:
            say("You are up to date.")
            return 0
    code = install(manifest, variant=variant, fetch=fetch, run=run, say=say)
    if code == 0:
        say("Updated. Quit and start `vibehash` again to use it.")
    else:
        say(f"pip stopped with exit code {code}.")
    return code
