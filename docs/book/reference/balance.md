# vibehash balance

An address's balance, stake, coins unbonding and reputation.

```
vibehash balance [-h] [--node NODE] [--wallet WALLET] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [address]
```

Arguments:

- `address` (optional): the address to look up (default: your wallet's).

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash balance
vibehash balance vibe1...
```
