# Operating a node

Running a node day to day: its data folder and disk use, fees, rate limits, logs, audits, upgrades, halts and backups.

## The data folder

`--data-dir` holds everything the node keeps. On a server made with `setup-node.sh` it is `/var/lib/vibehash/data`, beside the node's wallet and genesis in `/var/lib/vibehash`.

| In the folder | What it is |
|---|---|
| `chain.sqlite3` | The blocks, their commit certificates and round records, model weights, and the node's own settings (such as admin holds) |
| `blobs/` | Checkpoints: this round's, plus those of recent rounds kept for re-checking |
| `datasets/` | Training datasets jobs use, checked against each job's hash |
| `node.lock` | Held while a node runs here |

Only one node can run on a folder at a time. A second one is refused, because two nodes on one folder would sign rival blocks with one key.

### Keep-rounds and disk use

`--keep-rounds N` (default 10) is how many recent rounds' opened checkpoints stay downloadable, so anyone can re-run those rounds' checks with `chain verify`. Older checkpoints are deleted as rounds end. More rounds means a longer window for audits and more disk.

Other limits on disk and memory:

- `--max-job-mb MB` (default 64) relays only jobs whose model, one checkpoint at the job's precision, is at most that big. The default suits a 2 GB server; jobs may be up to 1024 MB.
- A dataset no job uses yet counts against its uploader's quota (20 GB an address), and a finished upload no job has used for three days is deleted.

## Fees

Transaction fees go to the validator that proposes the block. A node relays transactions paying at least `--min-fee` (0.0001 VIBE by default; `0` relays free ones), advertised in `/status`. When the mempool is full, higher fees get in first. Worker messages are free.

## Rate limits

Each client address gets its own budget of reads (50 a second, bursts of 500), worker messages, transactions (2 a second, bursts of 20) and uploads, at most 32 connections (256 in all, `--max-connections`), and timeouts that cut off slow clients. Over budget, a node answers 429 and clients wait and retry.

Clients on the node's own computer are never limited, so a local testnet is never throttled. Behind a reverse proxy every client looks local, so limit at the proxy or add `--rate-limit-local`. `--rate-limit off` turns the budgets off, and `--rate-limit tx=1/10,read=100` changes some (requests a second, then burst).

## Logs

- A node under systemd: `journalctl -u vibehash-node -f`.
- `vibehash testnet` keeps its node's log in `node.log` in the testnet's folder.
- `-v` logs every detail rather than just what happens.

`GET /status` shows the height, round, phase, leader, fees, versions and whether the node has halted.

## Audit the chain

```bash
vibehash chain verify --node https://node.example.org --genesis genesis.json
```

This re-checks every block from genesis with its commit certificate and replays every round from its record. For the rounds within the node's keep-rounds it also re-runs the checks that need model weights. See [vibehash chain](../reference/chain.md).

## Upgrading safely

Install the new release (`python -m pip install -U vibehash`, or run `setup-node.sh` again on a server) and restart the node. The chain, wallet and genesis are kept.

Blocks carry a version. A node refuses blocks of a version it doesn't know, and validators vote only for blocks of the current version. So on a network with several validators, every validator must run a release that knows a new block version **before** any block of it appears. Upgrade all of them, not just one.

`--min-client VERSION` (default 0.5.17) is the oldest release whose workers may register; older ones are told to upgrade.

## Halts

If a final block can't be saved to disk, the node stops itself. That happens when the database already has a different block at that height (as when two nodes share a folder) or the disk is full or failing. A halted node:

- no longer leads, proposes, votes, opens rounds, syncs or takes transactions;
- keeps serving the blocks it saved, with `/status` saying it halted.

`vibehash node` exits with the reason. Fix the cause (free disk space, stop the second node) and restart it: it carries on from what's on disk.

{% hint style="info" %}
A halt is the node protecting itself, not the network stopping. The other validators carry on if those still online hold more than two thirds of the stake; see [Validators](validators.md).
{% endhint %}

## Backups

What matters:

- **The node's wallet**, and its recovery words, kept offline. On a server, the wallet password is in `/etc/vibehash/env`.
- **The genesis file**, which every node of the network needs.
- **The data folder.** Other nodes can resync it, but a network run by one node has no other copy of its chain.

On a server made with `setup-node.sh`, back up `/var/lib/vibehash` and `/etc/vibehash`.

{% hint style="danger" %}
Never start a validator from a backup while the original is still running. Two nodes with one validator wallet sign conflicting votes, and that costs all of its stake.
{% endhint %}
