"""Make the Ed25519 keypair that signs releases (see scripts/release_manifest.py).

    python scripts/make_release_key.py release.key

Prints the PUBLIC key (hex) to paste into RELEASE_PUBKEYS in launcher/vibehash_launcher.py and vibehash/settings.py, and
writes the SECRET key (64 hex characters) to the file you name, readable by you alone (0o600), never to the screen. Put
that file's contents in the GitHub Actions secret RELEASE_SIGNING_KEY of the `pypi` environment, keep one offline copy,
and delete the file. It won't overwrite a key file that exists.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibehash import crypto  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python scripts/make_release_key.py SECRET_KEY_FILE", file=sys.stderr)
        return 2
    path = Path(argv[0])
    pair = crypto.KeyPair.generate()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # O_EXCL: never over an existing key
    except FileExistsError:
        print(f"{path} exists already; choose another file (this never overwrites a key)", file=sys.stderr)
        return 1
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(pair.secret.hex() + "\n")
    print(f"secret key written to {path} (keep it private; it is the RELEASE_SIGNING_KEY secret)")
    print("public key, to paste into RELEASE_PUBKEYS in launcher/vibehash_launcher.py and vibehash/settings.py:")
    print(f"    {pair.public.hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
