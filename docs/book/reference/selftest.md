# vibehash selftest

Check this computer is ready to mine: a real training test, its GPUs, the node, the clock, disk space, and what it can take.

```
vibehash selftest [-h] [--node NODE] [--device DEVICE] [--model NAME] [--yes] [--offline] [--json]
```

Flags:

- `--node NODE`: node to check against (default: the one `setup` saved, else the public network's).
- `--device DEVICE`: the device to train on, as for `worker` (default auto).
- `--model NAME`: also train one real unit of a ready model, e.g. qwen3-0.6b (downloads its weights once, asked first).
- `--yes, -y`: don't ask before --model's download.
- `--offline`: skip everything that goes online (node, PyPI, Hugging Face).
- `--json`: machine-readable output: every check, what it can take, next.

Examples:

```bash
vibehash selftest
vibehash selftest --model qwen3-0.6b
```
