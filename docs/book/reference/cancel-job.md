# vibehash cancel-job

Cancel a job you posted; the budget it hasn't spent comes back.

```
vibehash cancel-job [-h] [--node NODE] [--wallet WALLET] [--fee FEE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [job_id]
```

Arguments:

- `job_id` (optional): default: the last job you trained.

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash cancel-job
vibehash cancel-job my-first-job
```
