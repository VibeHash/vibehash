# CLAUDE.md

Rules for any session working in this repository. They are short on purpose. Read the code before you break one.

## What this is

VibeHash is a Python proof-of-useful-work testnet: workers train models, the chain pays for verified work. Read
`README.md` (users and node operators), `DESIGN.md` (the protocol spec), `docs/OPERATIONS.md` (how it runs, by role)
and `docs/CLI.md` (every command).

The business has two halves. This repository is the public one: the open chain, its miners and the open domain
models they train. The private engine (customer jobs, trained on in-house machines from those open models)
lives in a separate private repository. The next section says what belongs where.

## Public repository, private business

This repository is public under MIT. Anything pushed here can be copied and used commercially by anyone, for good.
Treat every commit as published.

Belongs here:
- The chain, the miner, public jobs, Evo, and the open domain models with their published recipes.
- The task scorer and the receipt format. Customers trust a scorer they can read and run on their own test set.

Never here (it goes in the private repository, or nowhere):
- Customer data in any form: records, samples, test fixtures, logs, screenshots. Tests use public or synthetic data.
- Customer names, contacts, task specs, prices, contracts and sales notes.
- Business plans, strategy reviews, market research and pricing.
- Weights of paid or listed models, private training recipes and settings, curated datasets and their generators.
- Container builds and listing configuration for marketplaces.

Rules:
- The dependency runs one way. The private repository installs a pinned `vibehash`; nothing here imports private
  code, names its paths or describes its internals.
- Customer jobs never run on the public chain. A node that holds a customer's model runs `--model-access owner`
  (the default is `public`), and customer models are served from the private deployment only.
- When you can't tell whether something is business IP, leave it out of this repository and ask.

## Never commit

- Secrets. They are named by env var, never by value: `VIBEHASH_PASSWORD`, `VIBEHASH_ADMIN_TOKEN`,
  `VIBEHASH_ADMIN_MFA`, `RELEASE_SIGNING_KEY`, `HF_TOKEN`, `VIBEHASH_GEO_TOKEN`, `ALERT_WEBHOOK`, rclone credentials,
  and the LLM keys of `agents/`.
- IPs, hostnames or SSH users of real machines. Personal paths, handles, emails, other people's wallet addresses.
- `otpauth://` URIs, recovery words, wallet files, chain databases, screenshots of the local app, operational diaries.
- If a doc must say where something runs, name the role ("the public validator"), not the host.
- CI runs gitleaks with `.gitleaks.toml` and fails the PR on a hit. Fix the text; do not add an allowlist entry to pass.

## Where secrets live

- Env vars; `/etc/vibehash*/env` on nodes; GitHub Actions secrets (the `pypi` environment); the OS credential store
  for desktop wallets (`vibehash/credstore.py`).
- Never print or log a secret. A new flag that takes a secret must also accept an env var, because argv shows in `ps`.

## Wallet rules

- The app shows the recovery words once and writes them nowhere. Headless setups get a words file, by design
  (`vibehash/unattended.py`); do not extend that to the app.
- Wallets are v2 in the OS credential store. Plaintext wallets only behind `--insecure-plaintext`.
- Any change that touches wallets needs a test on the in-memory keyring backend. `tests/test_wallet.py` shows how.
- `vibehash wallet mfa` makes the CLI ask for a TOTP code before a send, stake or unstake (`tests/test_wallet_mfa.py`).
- A non-loopback admin bind needs `--admin-mfa`; hold, release and review take an `X-Admin-Code` step-up. Keep both.

## Consensus and wire changes

- Changes to `ledger.py`, `consensus.py`, `economics.py`, `supply.py`, `fees.py`, the block and record formats in
  `wire.py`, or the round logic in `coordinator.py` need a `DESIGN.md` note, a block-version bump (`VERSION` in
  `ledger.py`) when old nodes would disagree, and tests in `tests/test_consensus.py` or `tests/test_rules_v*.py`.
- Node-local policy is not a chain rule and must stay that way: admin holds, review mode, `--model-access owner|public`
  (it only gates `GET /job/<id>/model`, with a signed request header). A fork over policy splits the network.

## No new workload in `coordinator.py`

- New job kinds wait for the plugin boundary. Do not wire one into the coordinator.
- DeepSeek is frozen. The `deepseek` setting defaults to `no`; the pilot prompt is behind `VIBEHASH_DEEPSEEK_PILOT=1`.

## Networking

- Node HTTP goes through `NodeClient` (`vibehash/client.py`), one opener, proxy and Tor aware via `vibehash/proxy.py`.
  Do not add a second `urllib.request.urlopen` path for node traffic: it would skip the proxy and leak the address.
- Proxies are `socks5h://` only (the proxy resolves names). `--tor`, `--proxy` and `--tor-mode` are the flags.
- Two planes: JSON always goes through the proxy; large checkpoints and datasets go direct only under
  `--tor-mode control-only`, and `strict` refuses them. A `.onion` node URL routes over Tor on its own.
- No `http://` endpoints except loopback and `.onion`. The admin geo lookup is opt-in and HTTPS, and off under a proxy.

## Updates and releases

- A release carries a signed manifest (`release.json`, `release.sig`) made by `scripts/release_manifest.py`. Clients
  check it against `RELEASE_PUBKEYS`. `vibehash update [--evo]` (`vibehash/update.py`) installs only a verified release.
- Never bypass `--require-hashes` in the launcher or `vibehash/update.py`. Hashed installs are the supply-chain check.
- A release is a version bump in `pyproject.toml` plus a `v*` tag. Workflow `uses:` stay pinned to commit SHAs.

## Tests and CI

- Before a PR run `python -m unittest discover -s tests -p "test_*.py"`. It needs torch and the extras
  (`pip install -e ".[finetune,evo,app]"`). Without torch, run the modules for what you touched.
- New modules get a test module. Never add a module to a hand-written CI list: `discover` must reach it.
- The `nightly` job runs the whole suite; the `gitleaks` job scans the tree. Both live in `.github/workflows/tests.yml`.
- `tests/protocol.py` and `tests/reference/` hold the old reference protocol, transformer and MoE code that tests
  compare against. They are test fixtures, not part of the `vibehash` package.

## Docs

- `HANDOFF.md` is frozen history. Do not append to it. Operations go in `docs/OPERATIONS.md`.
- No dated "tonight" plans and no machine-specific diaries in the repo.
- No business documents in the repo: plans, pricing, strategy and research reviews go in the private repository.
- Label facts and assumptions apart in any report. Say what you ran, not what you expect.

## `agents/`

- Part of the product: the B2B managed fine-tune pipeline, contributed by Luh Labs LLC, under the root `LICENSE`.
- Tested by `tests/test_agents.py`. Its LLM keys come from `.env.local` only; git ignores it.
- It is public. New customer-specific code, prompts, recipes or data go in the private repository, not here.

## Style

- Docstrings are prose and explain why. Short imperative sentences; plain words; no emoji.
- No TODO or FIXME markers. Open work goes in an issue.
- `from __future__ import annotations` at the top of a module; stdlib imports first.
- Commit subjects are plain sentences saying what changed. End each commit with the `Co-Authored-By` trailer your
  session provides.
