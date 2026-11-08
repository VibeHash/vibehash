"""Make and sign a release's manifest: what the launcher (launcher/vibehash_launcher.py), `vibehash update`
(vibehash/update.py) and the menu check before they install anything.

Run by the `manifest` job of .github/workflows/publish.yml after the wheel is on PyPI, from a checkout with the built
wheel in dist/ and the compiled requirements (COMPILE below) already written to dist/requirements-<version>.txt:

    python scripts/release_manifest.py --dist dist

It does three things:
1. pins the wheel: the `vibehash==<version>` line of the requirements gets the built wheel's own sha256 (added if the
   compile left it out, which it does: `uv pip compile pyproject.toml` lists the dependencies, not the project);
2. writes dist/release.json, the manifest, a JSON object with exactly these fields:
       version, wheel, wheel_sha256, requirements, requirements_sha256, published, min_launcher
   (the requirements are hashed after step 1; `published` is a unix time; `min_launcher` is the oldest launcher that
   understands this manifest);
3. writes dist/release.sig: the lowercase hex of the Ed25519 signature, by the key in $RELEASE_SIGNING_KEY (64 hex
   characters, the secret from scripts/make_release_key.py), over vibehash.crypto.canonical(manifest): the manifest as
   JSON with sorted keys, separators "," and ":", ASCII only, UTF-8 bytes. The signature covers the dict, not the file's
   layout, so release.json can be pretty-printed.

With no key it signs nothing and writes nothing (exit code 3): a release without a manifest is still a valid PyPI
release, and every installer then skips updating, which is the safe way to fail.

Rotating the key: make a new pair with scripts/make_release_key.py, add its public key to RELEASE_PUBKEYS in
launcher/vibehash_launcher.py and vibehash/settings.py next to the old one, and publish a release signed with the OLD
key (installers then know both). Switch RELEASE_SIGNING_KEY to the new secret for the releases after that, and remove
the old public key in a release after the launchers in the field have had time to update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibehash import crypto  # noqa: E402

FIELDS = ("version", "wheel", "wheel_sha256", "requirements", "requirements_sha256", "published", "min_launcher")
MIN_LAUNCHER = "1.2"  # the first launcher that verifies this manifest
# How the requirements are compiled (the workflow runs this, with -o dist/requirements-<version>.txt added). Every extra
# the launcher and the menu install; --universal so Windows-only and macOS-only packages (pywebview's) are in; torch
# left out, so installing the file never touches the GPU build `vibehash install-gpu` chose.
COMPILE = ("uv pip compile pyproject.toml --extra finetune --extra app --extra evo --generate-hashes --universal "
           "--no-emit-package torch --python-version 3.12").split()


class NoKey(Exception):
    """$RELEASE_SIGNING_KEY is missing or isn't a 32-byte secret in hex."""


def pin_wheel(requirements: str, version: str, wheel_sha256: str) -> str:
    """`requirements` with its vibehash line (and that entry's hash lines) replaced by the built wheel's, or added."""
    out: list[str] = []
    skipping = False
    for line in requirements.splitlines():
        if skipping and line[:1] in (" ", "\t"):  # the entry's --hash and "# via" lines
            continue
        skipping = bool(re.match(r"vibehash(\[[^\]]*\])?\s*(==|@|~=|>=)", line, re.I))
        if not skipping:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    out.append(f"vibehash=={version} \\")
    out.append(f"    --hash=sha256:{wheel_sha256}")
    return "\n".join(out) + "\n"


def secret_from(env) -> bytes:
    text = (env.get("RELEASE_SIGNING_KEY") or "").strip()
    try:
        secret = bytes.fromhex(text)
    except ValueError:
        raise NoKey("RELEASE_SIGNING_KEY isn't hex") from None
    if len(secret) != 32:
        raise NoKey("RELEASE_SIGNING_KEY is missing or isn't a 32-byte secret in hex")
    return secret


def make_release(dist: Path, version: str, env=None, now: int | None = None, min_launcher: str = MIN_LAUNCHER) -> dict:
    """Write dist/release.json and dist/release.sig (and pin the wheel in dist/requirements-<version>.txt). Raises
    NoKey before touching anything when there is no key."""
    secret = secret_from(os.environ if env is None else env)
    wheels = sorted(dist.glob(f"vibehash-{version}-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one vibehash-{version}-*.whl in {dist}, found {len(wheels)}")
    name = f"requirements-{version}.txt"
    wheel_sha = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
    reqs = pin_wheel((dist / name).read_text(encoding="utf-8"), version, wheel_sha).encode("utf-8")
    manifest = {"version": version, "wheel": wheels[0].name, "wheel_sha256": wheel_sha, "requirements": name,
                "requirements_sha256": hashlib.sha256(reqs).hexdigest(),
                "published": int(time.time()) if now is None else int(now), "min_launcher": min_launcher}
    sig = crypto.sign(secret, crypto.canonical(manifest))
    (dist / name).write_bytes(reqs)
    (dist / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dist / "release.sig").write_text(sig.hex() + "\n", encoding="ascii")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pin the wheel in the compiled requirements, then write and sign release.json.")
    ap.add_argument("--dist", default="dist", help="the folder with the built wheel and requirements-<version>.txt")
    ap.add_argument("--pyproject", default="pyproject.toml", help="where the version is read from")
    ap.add_argument("--version", help="the version (default: pyproject.toml's)")
    a = ap.parse_args(argv)
    version = a.version or tomllib.loads(Path(a.pyproject).read_text(encoding="utf-8"))["project"]["version"]
    try:
        manifest = make_release(Path(a.dist), version)
    except NoKey as e:
        print(f"not signing: {e}. This release gets no manifest, so installers will skip updating to it.", file=sys.stderr)
        return 3
    print(f"signed release {manifest['version']}: {manifest['wheel']} ({manifest['wheel_sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
