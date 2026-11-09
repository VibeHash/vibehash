# vibehash chain

Audit the chain yourself, without trusting the node.

- [`vibehash chain verify`](#vibehash-chain-verify): download every block, re-check it from genesis, and replay every round

## vibehash chain verify

Download every block, re-check it from genesis, and replay every round.

```
vibehash chain verify [-h] [--node NODE] [--genesis FILE] [--no-replay] [--no-recheck] [--receipts FILE]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--genesis FILE`: the genesis file to check the chain against (default: the node's).
- `--no-replay`: skip replaying rounds from their records.
- `--no-recheck`: don't download recent rounds' checkpoints to re-run their weight-based checks.
- `--receipts FILE`: a worker's receipts.jsonl to check (default: every worker's on this computer).

Examples:

```bash
vibehash chain verify
vibehash chain verify --genesis genesis.json
```
