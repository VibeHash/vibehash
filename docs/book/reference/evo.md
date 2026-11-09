# vibehash evo

Evo Hash's models.

- [`vibehash evo export`](#vibehash-evo-export): save an Evo Hash job's model (the tuned Qwen) as a Hugging Face folder, worked out from the chain

## vibehash evo export

Save an Evo Hash job's model (the tuned Qwen) as a Hugging Face folder, worked out from the chain.

```
vibehash evo export [-h] [--out OUT] [--node NODE] [--cpu] [--float32] [job_id]
```

Arguments:

- `job_id` (optional): default: the newest Evo job that has trained.

Flags:

- `--out OUT`: folder to save (default JOB-model).
- `--node NODE`: the Evo network's node (default http://127.0.0.1:8545, testnet --evo2's).
- `--cpu`: work it out on the CPU even with a GPU.
- `--float32`: save float32 weights (twice the size) instead of bfloat16.
