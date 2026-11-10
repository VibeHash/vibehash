# vibehash post-job

Post a training job from a job file, paying its budget in (`train` writes the file for you).

Post a job the network trains round by round until its budget is spent. The budget leaves your balance now and is paid out only for work that passes its checks; `cancel-job` returns what is left. `job template` writes a starter file, and the guide explains every field.

```
vibehash post-job [-h] [--node NODE] [--wallet WALLET] --spec FILE --budget VIBE [--dataset FILE] [--fee FEE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--spec FILE` (required): the job file (JSON).
- `--budget VIBE` (required): VIBE to spend on it, in all.
- `--dataset FILE`: an .npz dataset to upload to the node and train on.
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash post-job --spec job.json --budget 150 --dataset examples/demo-dataset.npz
```
