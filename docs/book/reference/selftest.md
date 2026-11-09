# vibehash selftest

Check this computer is ready to mine: a real training test, its GPUs, the node, the clock, disk space, and what it can take.

```
vibehash selftest [-h] [--node NODE] [--device DEVICE] [--model NAME] [--yes] [--offline] [--json] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node to check against (default: the one `setup` saved, else the public network's).
- `--device DEVICE`: the device to train on, as for `worker` (default auto).
- `--model NAME`: also train one real unit of a ready model, e.g. qwen3-0.6b (downloads its weights once, asked first).
- `--yes, -y`: don't ask before --model's download.
- `--offline`: skip everything that goes online (node, PyPI, Hugging Face).
- `--json`: machine-readable output: every check, what it can take, next.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash selftest
vibehash selftest --model qwen3-0.6b
```
