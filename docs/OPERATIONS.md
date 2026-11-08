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

1. Raise the version in `pyproject.toml` and merge it to `main` with the tests green.
2. Tag it and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `publish.yml` checks the tag matches the version, runs the whole test suite, builds the wheel and
   source archive, and publishes to PyPI. Follow it under the repository's Actions tab.
4. Upgrade the servers by running `deploy/setup-node.sh` again (restart every named node too, so they
   run the same release).

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
