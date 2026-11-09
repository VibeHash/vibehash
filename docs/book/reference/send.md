# vibehash send

Send VIBE to an address.

Send VIBE to an address. With `wallet mfa on`, a send, stake or unstake from this wallet first asks for the code an authenticator app shows (--code, else $VIBEHASH_CODE, else asked for). That stops casual use of an unlocked machine and mistaken or scripted sends; it does not stop malware that already reads this user's credential store. The worker never needs a code.

```
vibehash send [-h] [--node NODE] [--wallet WALLET] --to ADDRESS --amount VIBE [--fee FEE] [--code CODE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--to ADDRESS` (required): where to send it: an address copied from the wallet that owns it (vibe1...). Its capital letters are a checksum, so a mistyped one is refused.
- `--amount VIBE` (required): how much to send, e.g. 2.5.
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).
- `--code CODE`: the authenticator code, when `wallet mfa on` is set (default $VIBEHASH_CODE, else asked for); each code is good once.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash send --to vibe1... --amount 2.5
```
