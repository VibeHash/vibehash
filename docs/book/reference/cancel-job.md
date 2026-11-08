# vibehash cancel-job

Cancel a job you posted; the budget it hasn't spent comes back.

```
vibehash cancel-job [-h] [--node NODE] [--wallet WALLET] [--fee FEE] [job_id]
```

Arguments:

- `job_id` (optional): default: the last job you trained.

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--fee FEE`: transaction fee in VIBE, paid to the block's proposer (default: the node's minimum).

Examples:

```bash
vibehash cancel-job
vibehash cancel-job my-first-job
```
