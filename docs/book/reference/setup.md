# vibehash setup

Get this computer ready to work: node, wallet, GPU support, stake.

```
vibehash setup [-h] [--node NODE] [--wallet WALLET] [--gpu {auto,skip}] [--stake STAKE] [--yes] [--deepseek {yes,no}] [--models-dir MODELS_DIR] [--models-gb MODELS_GB] [--keep-models {yes,no}] [--tls-fingerprint SHA256] [--tls-ca FILE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address, e.g. 192.168.1.20 or http://node.example:8545.
- `--wallet WALLET`: wallet file to use or create (default ~/.vibehash/wallet.json).
- `--gpu {auto,skip}`: find the GPU and install PyTorch for it. Default: `auto`.
- `--stake STAKE`: stake this much VIBE without asking.
- `--yes`: install GPU support without asking. With no terminal (nohup, Docker, a pod) nothing is asked: the wallet's password comes from $VIBEHASH_PASSWORD, its recovery words go to a file only you can read, and model files take the defaults.
- `--deepseek {yes,no}`: whether `worker` takes DeepSeek work (its ~30 GB of model files) without asking; remembered.
- `--models-dir MODELS_DIR`: where big model files go (default: the drive with the most room).
- `--models-gb MODELS_GB`: the most space they may use, in GB (default 40).
- `--keep-models {yes,no}`: keep them after a job ends (default no).
- `--tls-fingerprint SHA256`: trust the https node's certificate with this fingerprint (how workers trust a self-signed testnet node; ask its operator).
- `--tls-ca FILE`: verify the https node's certificate with this CA file.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash setup
vibehash setup --node 192.168.1.20
VIBEHASH_PASSWORD='choose-one' nohup vibehash setup --yes --deepseek no > setup.log 2>&1
```
