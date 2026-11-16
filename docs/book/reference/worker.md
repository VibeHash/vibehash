# vibehash worker

Contribute compute and earn VIBE.

Mine: train and check other workers' units, and earn VIBE. One worker trains on one device. For several GPUs in one computer or pod, run one worker per GPU, each with its own wallet (two with one wallet would undo each other's work, so the second refuses to start): `vibehash wallet create --out gpu1.json`, then `vibehash worker --device cuda:1 --wallet gpu1.json`. Each wallet gets its own cache; the model files they download are shared. With no terminal (nohup, Docker, a pod), set $VIBEHASH_PASSWORD to unlock the wallet.

```
vibehash worker [-h] [--node NODE] [--wallet WALLET] [--device DEVICE] [--memory-mb MEMORY_MB] [--flops FLOPS] [--cache-dir CACHE_DIR] [--plain] [--watch [PORT]] [--deepseek {ask,yes,no}] [--job JOB] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--device DEVICE`: auto (first GPU found, else CPU), cpu, cuda, cuda:1, xpu, mps. One device a worker: a second GPU gets its own worker and wallet. Default: `auto`.
- `--memory-mb MEMORY_MB`: override detected memory.
- `--flops FLOPS`: override benchmark.
- `--cache-dir CACHE_DIR`: where checkpoints, datasets and receipts are kept (default ~/.vibehash/cache/WALLET-ADDRESS: one per wallet).
- `--plain`: print log lines instead of the live mining dashboard.
- `--watch PORT`: also serve what the dashboard shows as JSON at http://127.0.0.1:PORT/ (default port 7545), read-only and for this computer only, so a window or a script can follow the worker step by step: its round, what it is doing, its counts, its account, the latest events.
- `--deepseek {ask,yes,no}`: take DeepSeek work (its ~30 GB of model files): yes, or no (the default; DeepSeek is frozen). ask is only for the pilot (VIBEHASH_DEEPSEEK_PILOT=1). Remembered.
- `--job JOB`: mine only this job's rounds and sit the other jobs' out, or several jobs' as JOB,JOB (default auto: every job this worker can do). saved: the jobs the app's Pick your vibe chose, followed as they change. The dashboard's job line changes it while it runs: Tab or 0-9.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash worker
vibehash worker --device cuda:1 --wallet gpu1.json
vibehash worker --job fruit-demo
VIBEHASH_PASSWORD='choose-one' nohup vibehash worker --plain > worker.log 2>&1 &
```
