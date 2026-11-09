"""Make and sign a release's manifest: what the launcher (launcher/vibehash_launcher.py), `vibehash update`
(vibehash/update.py) and the menu check before they install anything.

Two steps, in two jobs of the publish workflow, so the signing key never meets third-party code:

    python scripts/release_manifest.py compile --out reqs     # `build` job, no key: runs uv (COMPILE_BASE, COMPILE_EVO)
    python scripts/release_manifest.py sign --dist dist --reqs reqs   # `manifest` job, after PyPI: stdlib + vibehash.crypto only

`compile` writes reqs/requirements-<version>.txt (the launcher's install set: the finetune and app extras) and
reqs/requirements-<version>-evo.txt (those plus the evo extra, for the menu's Evo Hash and `vibehash update --evo`).
They go in their own artifact, not dist/, which the PyPI upload reads. `sign` does three things:
1. pins the wheel: the `vibehash==<version>` line of both files gets the built wheel's own sha256 (added if the
   compile left it out, which it does: `uv pip compile pyproject.toml` lists the dependencies, not the project);
2. writes reqs/release.json, the manifest, a JSON object with these fields:
       version, wheel, wheel_sha256, requirements, requirements_sha256, requirements_evo, requirements_evo_sha256,
       published, min_launcher
   (the requirements are hashed after step 1; `published` is a unix time; `min_launcher` is the oldest launcher that
   understands this manifest; readers require these fields and tolerate extras);
3. writes reqs/release.sig: the lowercase hex of the Ed25519 signature, by the key in $RELEASE_SIGNING_KEY (64 hex
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
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibehash import crypto  # noqa: E402

FIELDS = ("version", "wheel", "wheel_sha256", "requirements", "requirements_sha256", "requirements_evo",
          "requirements_evo_sha256", "published", "min_launcher")
MIN_LAUNCHER = "1.2"  # the first launcher that verifies this manifest
# How the requirements are compiled (`compile` adds -o). --universal so Windows-only and macOS-only packages
# (pywebview's) are in; torch left out, so installing a file never touches the GPU build `vibehash install-gpu` chose.
# --python-version is the resolution's floor, so it is requires-python's (3.10): a 3.10 or 3.11 install verifies too.
COMPILE_BASE = ("uv pip compile pyproject.toml --extra finetune --extra app --generate-hashes --universal "
                "--no-emit-package torch --python-version 3.10").split()
COMPILE_EVO = ("uv pip compile pyproject.toml --extra finetune --extra app --extra evo --generate-hashes --universal "
               "--no-emit-package torch --python-version 3.10").split()


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


def compile_requirements(out: Path, version: str, run=subprocess.run) -> None:
    """Both requirements files into `out`, by running uv (no key is, or should be, in this process's environment)."""
    out.mkdir(parents=True, exist_ok=True)
    run([*COMPILE_BASE, "-o", str(out / f"requirements-{version}.txt")], check=True)
    run([*COMPILE_EVO, "-o", str(out / f"requirements-{version}-evo.txt")], check=True)


def make_release(dist: Path, version: str, env=None, now: int | None = None, min_launcher: str = MIN_LAUNCHER,
                 reqs: Path | None = None) -> dict:
    """Write release.json and release.sig into `reqs` (default: `dist`), pinning the wheel in the two requirements files
    there. Raises NoKey before touching anything when there is no key."""
    secret = secret_from(os.environ if env is None else env)
    reqs = dist if reqs is None else reqs
    wheels = sorted(dist.glob(f"vibehash-{version}-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one vibehash-{version}-*.whl in {dist}, found {len(wheels)}")
    names = (f"requirements-{version}.txt", f"requirements-{version}-evo.txt")
    wheel_sha = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
    texts = [pin_wheel((reqs / n).read_text(encoding="utf-8"), version, wheel_sha).encode("utf-8") for n in names]
    manifest = {"version": version, "wheel": wheels[0].name, "wheel_sha256": wheel_sha,
                "requirements": names[0], "requirements_sha256": hashlib.sha256(texts[0]).hexdigest(),
                "requirements_evo": names[1], "requirements_evo_sha256": hashlib.sha256(texts[1]).hexdigest(),
                "published": int(time.time()) if now is None else int(now), "min_launcher": min_launcher}
    sig = crypto.sign(secret, crypto.canonical(manifest))
    for name, text in zip(names, texts):
        (reqs / name).write_bytes(text)
    (reqs / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (reqs / "release.sig").write_text(sig.hex() + "\n", encoding="ascii")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compile the hash-pinned requirements, or pin the wheel in them and sign release.json.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("compile", "run uv to write both requirements files (needs no key)"),
                       ("sign", "pin the wheel, write and sign release.json (needs $RELEASE_SIGNING_KEY)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--pyproject", default="pyproject.toml", help="where the version is read from")
        p.add_argument("--version", help="the version (default: pyproject.toml's)")
        if name == "compile":
            p.add_argument("--out", default="reqs", help="the folder for the requirements files")
        else:
            p.add_argument("--dist", default="dist", help="the folder with the built wheel")
            p.add_argument("--reqs", default="reqs", help="the folder with the requirements files; release.json and release.sig go here")
    a = ap.parse_args(argv)
    version = a.version or tomllib.loads(Path(a.pyproject).read_text(encoding="utf-8"))["project"]["version"]
    if a.cmd == "compile":
        compile_requirements(Path(a.out), version)
        return 0
    try:
        manifest = make_release(Path(a.dist), version, reqs=Path(a.reqs))
    except NoKey as e:
        print(f"not signing: {e}. This release gets no manifest, so installers will skip updating to it.", file=sys.stderr)
        return 3
    print(f"signed release {manifest['version']}: {manifest['wheel']} ({manifest['wheel_sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
