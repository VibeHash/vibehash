# vibehash chain

Audit the chain yourself, without trusting the node.

- [`vibehash chain verify`](#vibehash-chain-verify): download every block, re-check it from genesis, and replay every round

## vibehash chain verify

Download every block, re-check it from genesis, and replay every round.

```
vibehash chain verify [-h] [--node NODE] [--genesis FILE] [--no-replay] [--no-recheck] [--receipts FILE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--genesis FILE`: the genesis file to check the chain against (default: the node's).
- `--no-replay`: skip replaying rounds from their records.
- `--no-recheck`: don't download recent rounds' checkpoints to re-run their weight-based checks.
- `--receipts FILE`: a worker's receipts.jsonl to check (default: every worker's on this computer).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash chain verify
vibehash chain verify --genesis genesis.json
```
