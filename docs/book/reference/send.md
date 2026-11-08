# vibehash send

Send VIBE to an address.

```
vibehash send [-h] [--node NODE] [--wallet WALLET] --to ADDRESS --amount VIBE [--fee FEE]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--to ADDRESS` (required): where to send it: an address copied from the wallet that owns it (vibe1...). Its capital letters are a checksum, so a mistyped one is refused.
- `--amount VIBE` (required): how much to send, e.g. 2.5.
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).

Examples:

```bash
vibehash send --to vibe1... --amount 2.5
```
