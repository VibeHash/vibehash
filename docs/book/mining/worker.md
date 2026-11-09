# The worker

This page covers `vibehash worker`, the command that mines: the flags you're most likely to need, running several GPUs, picking a job, running with no screen, and the logs it keeps. The full flag list is on [vibehash worker](../reference/worker.md).

## Start mining

After [setup](../getting-started/first-run.md), this is all you need:

```bash
vibehash worker
```

It uses the node and wallet setup saved, trains on the first GPU it finds (or the CPU), and runs until you press Ctrl+C. When you stop it, it prints a summary: rounds, units accepted and VIBE earned. If you'd rather not keep a terminal open, the [desktop app](../getting-started/desktop-app.md) runs the same worker for you.

## The flags that matter

| Flag | What it does |
|---|---|
| `--device` | What to train on: `auto` (the default: first GPU, else CPU), `cpu`, `cuda`, `cuda:1`, `xpu`, `mps`. |
| `--wallet` | A different wallet file from the one setup saved. |
| `--node` | A different network's node, such as the [Evo network](evo-hash.md). |
| `--job` | Mine only this job's rounds and sit the others out. |
| `--plain` | Print plain log lines instead of the live dashboard. |
| `--watch` | Also serve what the dashboard shows as JSON on this computer (port 7545 unless you give another). |
| `--deepseek` | `yes` or `no` to [DeepSeek work](deepseek.md); remembered. |
| `--memory-mb` | Override the memory it detected, if it got it wrong. |

## The live dashboard

In a terminal, the worker shows a live mining dashboard: a drill boring through rock while you train, the round and its phase, your speed, units mined, and VIBE earned as it lands.

The `job:` line under the round picks what to mine: **auto** (every job your computer can do, the default) or one of the jobs with budget left. Tab, `j`/`k` or the arrow keys move along it, `a` or `0` is auto, and `1` to `9` pick a job by its number.

If the dashboard looks garbled, or you want plain lines, add `--plain`. Piped output and services get plain lines automatically, and setting `NO_COLOR` turns colours off.

## Mining one job

Without the dashboard, pick a job by name:

```bash
vibehash worker --job fruit-demo
```

If the node doesn't list that job, the worker stops at once and names the jobs it does have.

## Several GPUs

One worker trains on one GPU. For more, run a worker for each, **each with its own wallet**, so each earns and stakes on its own. Two workers with one wallet would take the same units and undo each other's work, so the second refuses to start.

```bash
vibehash wallet create --out gpu1.json
```
```bash
vibehash worker --device cuda:1 --wallet gpu1.json
```

Each wallet gets its own cache (`~/.vibehash/cache/ADDRESS`). Model weights are shared: when two workers need the same model, one downloads it while the other waits.

## Running with no screen

On a server, a rented GPU, or anything with nobody at the keyboard, the wallet password comes from `VIBEHASH_PASSWORD`:

```bash
VIBEHASH_PASSWORD='choose-one' nohup vibehash worker --plain > worker.log 2>&1 &
```

See [Running with nobody at the keyboard](../getting-started/first-run.md#running-with-nobody-at-the-keyboard) for setup on such machines.

## Following it from elsewhere: --watch

`--watch` serves what the dashboard shows, live, as JSON at `http://127.0.0.1:7545/`: the round and phase, what the worker is doing, its counts this round and this session, and its balance, stake and reputation as the node last said, plus the latest events. It's read-only, for this computer only, and has nothing of your wallet in it. With it running you can open the app's window or the pickaxe beside it:

```bash
vibehash worker --watch
```
```bash
vibehash window
```

## Logs and records

The worker keeps files whether the dashboard is on or not, so a night of mining can be looked at in the morning. They're in `~/.vibehash/logs` (or your `VIBEHASH_HOME`), and the worker prints where when it starts.

- **`worker-DATE.log`**: what the dashboard's feed showed, one timed line each. A file stops at 10 MB and starts a new one, and only the last 14 days are kept.
- **`rounds.jsonl`**: a line for each round you took part in: units given, trained, committed, revealed and accepted (and why not, when the node said), checks done, time spent, what the block paid or took, and your balance and stake after it. The `outcome` field says it in words. Files from several computers can be put together.

Your receipts, signed by the node for every message it accepted, are in `~/.vibehash/cache/ADDRESS/receipts.jsonl`. They prove it if a node ever drops your work; see [Troubleshooting](troubleshooting.md).
