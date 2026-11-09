# vibehash app

Mine without the questions: get ready, join, keep the worker running, start at login; `app status`, `pause`, `resume` and `stop` drive it.

- [`vibehash app start`](#vibehash-app-start): get ready, join the network and mine (what plain `app` does)
- [`vibehash app pause`](#vibehash-app-pause): stop mining for now; the worker leaves its round
- [`vibehash app resume`](#vibehash-app-resume): mine again
- [`vibehash app stop`](#vibehash-app-stop): stop the app and its worker
- [`vibehash app status`](#vibehash-app-status): is it running, paused, and what the worker is doing
- [`vibehash app startup`](#vibehash-app-startup): start the app when you log in (on), or not (off); it then starts without the window, which the avatar or `vibehash window` opens

## vibehash app start

Get ready, join the network and mine (what plain `app` does).

```
vibehash app start [-h] [--node NODE] [--wallet WALLET] [--device DEVICE] [--watch PORT] [--gpu {auto,skip}] [--recheck-gpu] [--job JOB] [--no-password] [--avatar] [--no-window]
```

Flags:

- `--node NODE`: node address (default: the one saved, else the public network's).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--device DEVICE`: what the worker trains on: auto, cpu, cuda, cuda:1, xpu, mps.
- `--watch PORT`: the worker's feed's port (default 7545).
- `--gpu {auto,skip}`: install PyTorch for the GPU (auto) or not.
- `--recheck-gpu`: test the GPU again, even if an earlier run found it ready.
- `--job JOB`: mine only this job, as `worker --job`.
- `--no-password`: make a first wallet without a password, so nothing is asked (what the downloadable app does); the file is then as safe as your user account.
- `--avatar`: also show the floating avatar on the desktop (`avatar`), which the downloadable app does; needs a screen.
- `--no-window`: don't open the window (`window`) as it starts; a click on the avatar or `vibehash window` opens it later. How a start-at-login entry runs.

## vibehash app pause

Stop mining for now; the worker leaves its round.

```
vibehash app pause [-h]
```

## vibehash app resume

Mine again.

```
vibehash app resume [-h]
```

## vibehash app stop

Stop the app and its worker.

```
vibehash app stop [-h]
```

## vibehash app status

Is it running, paused, and what the worker is doing.

```
vibehash app status [-h]
```

## vibehash app startup

Start the app when you log in (on), or not (off); it then starts without the window, which the avatar or `vibehash window` opens.

```
vibehash app startup [-h] [--avatar] {on,off}
```

Arguments:

- `mode`: one of on, off.

Flags:

- `--avatar`: with the floating avatar (on only).
