# vibehash window

The app's window: the mining dashboard in a window of its own, beside a running `app` or `worker --watch`.

The window (window.py): the dashboard (the round step by step, the feed, your wallet, the network) in the system's own web view through pywebview (`pip install vibehash[app]`; the downloadable app has it). It waits for the worker's feed, then shows it; closing it doesn't stop mining. Without pywebview, or a web view on this Linux, the dashboard opens in the browser instead.

```
vibehash window [-h] [--watch PORT]
```

Flags:

- `--watch PORT`: the feed's port (default 7545).
