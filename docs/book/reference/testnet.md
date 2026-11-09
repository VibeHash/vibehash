# vibehash testnet

Run a whole network on this computer: node, workers and a job.

```
vibehash testnet [-h] [--workers 1-8] [--validators 1-7] [--dir DIR] [--host HOST] [--port PORT] [--job {gpu,small,none}] [--budget BUDGET] [--device DEVICE] [--rounds ROUNDS] [--reset] [--plain] [--watch [PORT]] [--evo] [--evo2] [--evo2-rounds N] [--evo-model MODEL] [--register-s SECONDS] [--compute-s SECONDS] [--reveal-s SECONDS] [--verify-s SECONDS] [--min-fee MIN_FEE]
```

Flags:

- `--workers 1-8`: workers mining, each with its own wallet (default 2).
- `--validators 1-7`: validator nodes, on --port and the ports after it; they take turns leading rounds and vote on every block (default 1; fixed when the testnet is first made).
- `--dir DIR`: where its wallets and chain live (default ~/.vibehash/testnet).
- `--host HOST`: address the node listens on (default 127.0.0.1; 0.0.0.0 lets other computers join).
- `--port PORT`: the node's port (default 8545); more validators take the ports after it.
- `--job {gpu,small,none}`: the job template to train (see `job template`), or none: post your own, with `train --node` and the testnet's requester.json wallet. Default: `gpu`.
- `--budget BUDGET`: VIBE per job; a new job is posted when one runs out. Default: `500`.
- `--device DEVICE`: what the workers train on: auto, cpu, cuda, xpu, mps. Default: `auto`.
- `--rounds ROUNDS`: stop after this many rounds (default: run until Ctrl+C).
- `--reset`: delete this testnet's chain and wallets and start over.
- `--plain`: print log lines instead of the live mining dashboard.
- `--watch PORT`: also serve what the dashboard shows, every worker's, as JSON at http://127.0.0.1:PORT/ (default port 7545), for a window or a script on this computer to follow.
- `--evo`: Evo mode: training by evolution, the chain is the model. A network whose genesis takes Evo jobs, training an open model on Countdown puzzles, its blocks holding only seeds and scores (default folder ~/.vibehash/testnet-evo).
- `--evo2`: Evo Hash (Evo v2): the fast recipe on every layer, a short job per task in turn (countdown, arithmetic, arc_easy, then round again), each starting where the last ended (es2.py).
- `--evo2-rounds N`: Evo v2: rounds a job runs, its step size planned over them (default 12: a task's quick early gains, then the next task).
- `--evo-model MODEL`: Evo mode's model: a Hugging Face name, from this computer's cache or downloaded once, or a model's folder (default Qwen/Qwen2.5-0.5B-Instruct, about 1 GB).
- `--register-s SECONDS`: how long a round's register phase lasts (default 3).
- `--compute-s SECONDS`: how long a round's compute phase lasts (default 8).
- `--reveal-s SECONDS`: how long a round's reveal phase lasts (default 3).
- `--verify-s SECONDS`: how long a round's verify phase lasts (default 4).
- `--min-fee MIN_FEE`: smallest fee in VIBE this node relays a transaction for (default 0.0001 VIBE; 0 relays free ones).

Examples:

```bash
vibehash testnet
vibehash testnet --validators 4
vibehash testnet --job none --workers 3 --device cuda
vibehash testnet --evo2
```
