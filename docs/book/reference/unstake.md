# vibehash unstake

Release staked VIBE; it is spendable after 20 rounds, and can still be slashed until then.

```
vibehash unstake [-h] [--node NODE] [--wallet WALLET] --amount VIBE [--fee FEE]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--amount VIBE` (required): how much, e.g. 10.
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).

Examples:

```bash
vibehash unstake --amount 10
```
