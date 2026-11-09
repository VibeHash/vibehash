# vibehash balance

An address's balance, stake, coins unbonding and reputation.

```
vibehash balance [-h] [--node NODE] [--wallet WALLET] [address]
```

Arguments:

- `address` (optional): the address to look up (default: your wallet's).

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).

Examples:

```bash
vibehash balance
vibehash balance vibe1...
```
