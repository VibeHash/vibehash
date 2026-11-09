# vibehash evo

Evo Hash's models.

- [`vibehash evo export`](#vibehash-evo-export): save an Evo Hash job's model (the tuned Qwen) as a Hugging Face folder, worked out from the chain

## vibehash evo export

Save an Evo Hash job's model (the tuned Qwen) as a Hugging Face folder, worked out from the chain.

```
vibehash evo export [-h] [--out OUT] [--node NODE] [--cpu] [--float32] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [job_id]
```

Arguments:

- `job_id` (optional): default: the newest Evo job that has trained.

Flags:

- `--out OUT`: folder to save (default JOB-model).
- `--node NODE`: the Evo network's node (default http://127.0.0.1:8545, testnet --evo2's).
- `--cpu`: work it out on the CPU even with a GPU.
- `--float32`: save float32 weights (twice the size) instead of bfloat16.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).
