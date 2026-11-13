# Operations

One page on what runs where, who owns it, and the routine jobs. The node runbook itself is
`README.md` "Run a public node". `HANDOFF.md` is the development history and is no longer appended to.

Commits before November 2026 include the Luh Labs LLC hackathon code under MIT; `agents/` remains in the repository as the B2B pipeline.

This page names roles, not people, hosts or addresses. Those live in the password manager and the
server's own configuration, never in the repository.

## What runs where

| Role | What it is | Where it is configured |
|---|---|---|
| Public validator | The main network's node: HTTPS on port 443, the chain database, blob checkpoints, the node wallet | `deploy/setup-node.sh` on one small Linux server |
| Evo validator | A second, named node on the same server for the Evo network (`--name`, its own port, chain id and data folder) | the same script, run again with `--name` and `--es` |
| Website | The static site, built by `scripts/build_site.py` and published on GitHub Pages | the repository's Pages settings |
| PyPI project | The `vibehash` package that users install | PyPI, which trusts the release workflow (below) |
| Release workflow | `.github/workflows/publish.yml`: tests, build, publish on a `v*` tag | the repository's Actions settings and its `pypi` environment |

## Who owns which account

Each account belongs to the VibeHash organisation, held by whoever has the owner role, not to a person's
private login. When an owner changes, hand the role over and rotate the secrets below.

| Role | Account |
|---|---|
| Source and CI | the VibeHash GitHub organisation |
| Packages | the VibeHash organisation's PyPI project (publishing is by trusted publisher, so no API token is stored) |
| Server and snapshots | the organisation's cloud account |
| DNS and domain | the organisation's registrar and DNS account |
| Alert channel | the webhook named by `ALERT_WEBHOOK` on the server |

## Deploy a node

On a fresh Ubuntu 24.04 server, run `deploy/setup-node.sh` (its header lists the flags: `--domain`,
`--chain-id`, `--name`, `--es`, `--stake`, `--enable-tor`). Running it again upgrades the node and
keeps its wallet and chain. Then lock down SSH as `README.md` "Lock down the server" says.

The node's files are in `/var/lib/vibehash`; its settings and secrets are in `/etc/vibehash/env`.
Read the log with `journalctl -u vibehash-node -f`.

## Back up and restore

The script installs `deploy/backup.sh` with a daily timer (`vibehash-backup.timer`) and
`deploy/healthcheck.sh` with a five-minute one (`vibehash-health.timer`). Either failing posts to the
alert webhook through `vibehash-alert@.service`. The backup keeps an online copy of the chain database,
the genesis and the node wallet on the server, and sends the checkpoints and that folder to an rclone
remote when `BACKUP_RCLONE_REMOTE` is set in `/etc/vibehash/env`. Until a remote is set, the blobs
are not backed up. Also keep weekly snapshots of the server on in the cloud account.

To restore on a new server: run `setup-node.sh` first (same `--name` and `--chain-id`), then follow
`/usr/local/lib/vibehash/backup.sh --restore-hint`. The wallet in a backup is encrypted with
`VIBEHASH_PASSWORD`, which is not backed up: keep it in the organisation's password manager.
`README.md` "Backups, alerts and Tor" has the details, and what `last_backup.json` and `health.json` hold.

## Publish a release

Once, before the first tag: set up the release signing key. Installers and `vibehash update` install only
a release whose manifest is signed by a key they ship with.

- Run `python scripts/make_release_key.py release.key`. It prints the public key and writes the secret
  half to `release.key`, readable by you alone.
- Put the secret in the GitHub Actions secret `RELEASE_SIGNING_KEY` of the `pypi` environment. Keep
  one offline copy and delete the file.
- Paste the public key into `RELEASE_PUBKEYS` in both `launcher/vibehash_launcher.py` and
  `vibehash/settings.py`, and merge that. Do it **before the first tag**: `launcher.yml` fails a tag
  build while the list is empty, and a release without `VibeHash-Setup.exe` breaks the site's
  installer link (it 404s).

Each release:

1. Raise the version in `pyproject.toml` and merge it to `main` with the tests green.
2. Tag it and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `publish.yml` checks the tag matches the version, runs the whole test suite, builds the wheel and
   source archive, compiles the hash-pinned `requirements-<version>.txt` and
   `requirements-<version>-evo.txt` (for Python 3.10 and up), and publishes to PyPI. Its `manifest` job
   then signs `release.json` with `RELEASE_SIGNING_KEY` and attaches `release.json`, `release.sig` and
   the two requirements files to the tag's GitHub release, where the launcher and `vibehash update`
   fetch them. Without the secret it warns and attaches nothing: the release is on PyPI, but installers
   don't update to it. Follow it under the repository's Actions tab, and check that the release lists
   those four files and the installer.
4. Upgrade the servers by running `deploy/setup-node.sh` again (restart every named node too, so they
   run the same release). A release that raises the block version (`ledger.VERSION`; 5 came with fine-tune
   check rule 3) must reach every validator before any of them makes a block: older nodes refuse those blocks,
   and workers of an older release stop until they update.

## Rotate secrets

Secrets live in `/etc/vibehash/env` on the server. After changing one, restart the node
(`systemctl restart vibehash-node`, and each named node's unit).

- **Admin token** (`VIBEHASH_ADMIN_TOKEN`): make a new one with `vibehash admin --new-token`, replace the
  line, restart. Anyone holding the old token is locked out.
- **Admin MFA key** (`VIBEHASH_ADMIN_MFA`): make a new one with `vibehash admin --new-mfa`, replace the
  line, restart, and re-enrol the authenticator app.
- **Node wallet password** (`VIBEHASH_PASSWORD`): the wallet file is encrypted with it. There is no
  change-password command: rebuild the wallet from its recovery words with `vibehash wallet restore
  --out FILE`, which asks for the new password (the words stay offline), put the new value in the env
  file, restart, and take a fresh backup so the backup and the password match.

After any rotation, check `journalctl -u vibehash-node -n 50` and the health check's `health.json`.

## What never goes in the repository

Nothing below is committed, pasted into an issue or a pull request, or left in a doc, a test or a comment:

- Secrets, by the name of the environment variable that holds them: `VIBEHASH_PASSWORD`,
  `VIBEHASH_ADMIN_TOKEN`, `VIBEHASH_ADMIN_MFA`, `RELEASE_SIGNING_KEY`, `HF_TOKEN`, `VIBEHASH_GEO_TOKEN`,
  `ALERT_WEBHOOK`, the credentials in `BACKUP_RCLONE_REMOTE`, and for `agents/` `MODEL_API_KEY` and
  `DISPATCH_TOKEN`.
- Wallet files, chain databases and their backups, `.onion` private keys and hostname copies, and
  recovery words.
- Real hostnames, IP addresses and SSH users of the servers; say "the public validator" instead.
- Personal paths (a Windows or macOS user folder), people's handles and email addresses, and wallet
  addresses of real people.
- Operational diaries: who did what on which machine, and when.

Such notes belong in a private operations repository, or in this page in role form. The `gitleaks` job in
`.github/workflows/tests.yml` scans every push and pull request with the rules in `.gitleaks.toml`
(documentation-range and private IPs, user paths, consumer email addresses, `otpauth://` seeds and wallet
addresses in docs, plus gitleaks' own secret rules; `tests/` is exempt from the custom rules above for fixtures, never from gitleaks' own). The scanner is a pinned, checksum-verified binary (no license key): to
bump it, change `GITLEAKS_VERSION` and `GITLEAKS_SHA256` in that job together, taking the checksum from the
release's `checksums.txt`.

## Hand-off items

- The website's Pages repository (`vibehash.org`, built by `scripts/build_site.py`) still sits under the
  founder's personal GitHub account. Transfer it to the VibeHash organisation, then update the URL in
  `scripts/build_site.py` and the Pages and DNS settings.
- Check that PyPI's trusted-publisher entry for `vibehash` names the organisation's repository
  (`VibeHash/vibehash`, workflow `publish.yml`, environment `pypi`).
