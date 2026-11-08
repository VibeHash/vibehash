# vibehash stake

Lock VIBE as stake: what a worker's checks and a validator's votes are backed by, and what a cheat loses.

```
vibehash stake [-h] [--node NODE] [--wallet WALLET] --amount VIBE [--fee FEE]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--amount VIBE` (required): how much, e.g. 10.
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).

Examples:

```bash
vibehash stake --amount 10
```
