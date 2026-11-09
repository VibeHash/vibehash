# A test network on your computer

`vibehash testnet` runs a whole network on one computer: one or more validator nodes, a few workers and a training job. It needs no wallets, files or flags, and it is the quickest way to see rounds, blocks, payments and consensus working before you set up anything public.

## Start one

```bash
vibehash testnet
```

This makes a new network in `~/.vibehash/testnet` (or `--dir DIR`), starts a node on port 8545, starts two workers, posts a demo job and shows the live mining dashboard. Each round you see what the workers earn. Stop it with Ctrl+C. Running it again carries on where it stopped.

What it creates the first time:

| File | What it is |
|---|---|
| `genesis.json` | The network, with a random chain id such as `vibehash-local-...` |
| `node.json`, `validator-2.json`, ... | The validators' wallets. They vote with 10 VIBE of bootstrap stake each |
| `worker-1.json`, ... | The workers' wallets. They start with nothing and earn their stake |
| `requester.json` | The one wallet handed coins (10,000 VIBE), to pay for the demo jobs |
| `node.log` | The node's log |

{% hint style="warning" %}
A testnet's wallets are unencrypted throwaway keys. Never send real coins to them.
{% endhint %}

While it runs, it prints commands to try in another terminal, such as checking a worker's balance and auditing the chain:

```bash
vibehash balance --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/worker-1.json
```
```bash
vibehash chain verify --node http://127.0.0.1:8545 --genesis ~/.vibehash/testnet/genesis.json
```

## More validators

To watch consensus with several validators, give it more. Each gets its own port (8545, 8546, ...), they take turns leading rounds, and every block needs votes from validators holding more than two thirds of the stake:

```bash
vibehash testnet --reset --validators 4
```

The number of validators is fixed when the testnet is first made, so add `--reset` if you have run one before. `--reset` deletes that testnet's chain and wallets and starts over. With four validators you can see why four is the smallest set that survives one going down; see [Validators](validators.md).

## Workers, devices and jobs

| Flag | What it does |
|---|---|
| `--workers 1-8` | How many workers mine, each with its own wallet (default 2) |
| `--device DEVICE` | What the workers train on: `auto` (default), `cpu`, `cuda`, `xpu` or `mps` |
| `--job {gpu,small,none}` | The job template to train (default `gpu`), or `none` to post your own |
| `--budget VIBE` | VIBE per job (default 500); a new job is posted when one runs out |
| `--rounds N` | Stop after this many rounds |
| `--plain` | Print log lines instead of the live dashboard |
| `--watch [PORT]` | Also serve every worker's dashboard as JSON at `http://127.0.0.1:7545/` |

With `--job none`, no demo job is posted. Post your own with `train --node` and the testnet's `requester.json` wallet:

```bash
vibehash testnet --job none --workers 3 --device cuda
```
```bash
vibehash train my-data.csv --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json --budget 150
```

A testnet's rounds are shorter than a real node's: register 3 s, compute 8 s, reveal 3 s and verify 4 s by default. `--register-s`, `--compute-s`, `--reveal-s` and `--verify-s` change them, and `--min-fee` sets the relay fee.

## Evo Hash

```bash
vibehash testnet --evo2
```

This runs Evo Hash, training by evolution: a node and two miners train Qwen2.5-0.5B-Instruct on many tasks, and each block keeps only the seeds and scores of the round. It needs an NVIDIA GPU with PyTorch (`vibehash install-gpu`) and `pip install vibehash[evo]`, and downloads the model and the tasks' datasets once.

- It lives in its own folder, `~/.vibehash/testnet-evo2`, with a genesis that takes Evo jobs and workers that start staked.
- `--evo2-rounds N` sets each job's length (default 12).
- `--evo-model MODEL` picks another Hugging Face model.
- `--evo` is the first, simpler version (one layer, Countdown only), in `~/.vibehash/testnet-evo`.
- `vibehash evo export`, in another terminal, writes the tuned model out as a Hugging Face folder.

An existing testnet folder made without Evo jobs can't become an Evo one: use `--reset` or another `--dir`.

## Letting other computers join

```bash
vibehash testnet --host 0.0.0.0
```

Other computers on your network can then join with `vibehash setup --node YOUR-LAN-ADDRESS`, and the testnet prints how to send them VIBE from the requester wallet. Because its wallets are unencrypted test keys, use this only for trying things. For anything longer-lived, set up a network by hand ([Genesis and TLS](genesis-and-tls.md)) or on a server ([A public node](public-node.md)).

Only one copy of a testnet can run on a folder at a time; a second is refused. Every flag is in [vibehash testnet](../reference/testnet.md).
