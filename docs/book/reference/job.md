# vibehash job

Training jobs: how one is going, follow it live, download its model, its history, or a starter job file.

- [`vibehash job status`](#vibehash-job-status): how a training job is going: rounds, accuracy, budget left
- [`vibehash job watch`](#vibehash-job-watch): follow a training job live, until its budget is spent: accuracy climbing round by round, budget left, events
- [`vibehash job download`](#vibehash-job-download): save a job's trained model to use with `predict`, a fine-tune's LoRA adapter, or with --merged the whole fine-tuned model
- [`vibehash job history`](#vibehash-job-history): a job's rounds one by one (loss, accuracy, units, VIBE paid, budget left), from the node, for any job; saved in ~/.vibehash/jobs/JOB
- [`vibehash job template`](#vibehash-job-template): write a starter job file, for full control with post-job

## vibehash job status

How a training job is going: rounds, accuracy, budget left.

```
vibehash job status [-h] [--node NODE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [job_id]
```

Arguments:

- `job_id` (optional): default: the last job you trained.

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash job status
vibehash job status fruit-demo
```

## vibehash job watch

Follow a training job live, until its budget is spent: accuracy climbing round by round, budget left, events.

```
vibehash job watch [-h] [--node NODE] [--plain] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [job_id]
```

Arguments:

- `job_id` (optional): default: the last job you trained.

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--plain`: print a line per event instead of the live dashboard.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash job watch
```

## vibehash job download

Save a job's trained model to use with `predict`, a fine-tune's LoRA adapter, or with --merged the whole fine-tuned model.

```
vibehash job download [-h] [--out OUT] [--merged] [--float32] [--node NODE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [job_id]
```

Arguments:

- `job_id` (optional): default: the last job you trained.

Flags:

- `--out OUT`: file to save (default JOB.npz), or a fine-tune's folder (default DATA-lora, or DATA-merged).
- `--merged`: a fine-tune: the base model with the adapter folded into its weights, a model folder that loads as the base does, no PEFT (downloads the base's weights once).
- `--float32`: with --merged (or an Evo job): save float32 weights, which keep the change whole, not the base's bf16 (twice the size).
- `--node NODE`: node address (default: the one `setup` saved).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash job download
vibehash job download fruit-demo --out fruit.npz
vibehash job download my-chats --merged
```

## vibehash job history

A job's rounds one by one (loss, accuracy, units, VIBE paid, budget left), from the node, for any job; saved in ~/.vibehash/jobs/JOB.

```
vibehash job history [-h] [--node NODE] [--csv FILE] [--json] [--tor] [--proxy URL] [--tor-mode {control-only,strict}] [job_id]
```

Arguments:

- `job_id` (optional): default: the last job you trained.

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--csv FILE`: also write the whole history to this CSV file.
- `--json`: print the whole history as JSON, for scripts.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash job history fruit-demo --csv fruit-demo.csv
```

## vibehash job template

Write a starter job file, for full control with post-job.

```
vibehash job template [-h] [--out OUT] [--id ID] [--gpu] [--evo MODEL] [--tasks TASKS] [--rounds ROUNDS] [--max-tokens N] [--loop-stop PERIOD,RUN] [--task-tokens TASK=N,...] [--cross-margin X]
```

Flags:

- `--out OUT`: file to write (default job.json).
- `--id ID`: the job's name (default my-first-job).
- `--gpu`: a larger float32 model sized for GPUs.
- `--evo MODEL`: an Evo job on MODEL that never ends, taking its tasks in turn: a genesis starter job (genesis create --es --starter-job).
- `--tasks TASKS`: with --evo: the tasks, comma-separated, curated (the default: the ones a small model learns from), vibe (the Qwen3.5-0.8B job's, for max_tokens 384-512) or all.
- `--rounds ROUNDS`: with --evo: rounds on each task before the next (default 12).
- `--max-tokens N`: with --evo: the longest answer, in tokens (default 64; 512 with --tasks vibe, whose math and instructions need room to reason).
- `--loop-stop PERIOD,RUN`: with --evo: end an answer once RUN tokens in a row each repeat the token up to PERIOD before (a loop), or off (default 32,48 with --tasks vibe, else off).
- `--task-tokens TASK=N,...`: with --evo: shorter answers for some tasks, such as math=448,gsm8k=384, or off (default math=448,gsm8k=384,ifeval=256 with --tasks vibe, else off).
- `--cross-margin X`: with --evo: the margin, in logits, between machines that do the arithmetic differently, with 0.1 for alike ones and checkers paired by setup (default 0.5), or off for one margin of 0.5 and no pairing.

Examples:

```bash
vibehash job template --out job.json
vibehash job template --gpu
```
