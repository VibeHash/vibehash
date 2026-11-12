# GPU run: a GSM8K fine-tune across two GPU generations

The run sheet for one measured run: a small network of rented GPUs fine-tunes Qwen3-0.6B on GSM8K, then each kind of
GPU replays every checked segment so the paper can report how close honest cross-hardware checks come to their limits.
It produces three things: an open maths adapter with a published recipe, its GSM8K test accuracy against the base
model's, and a table of check margins by the GPU that trained and the GPU that replayed.

Everything here runs on a private test network (`vibehash testnet`), not the public chain: its coins are worthless, and
nothing it does touches the public network.

## What runs where

| Role | GPU | Runs |
|---|---|---|
| Node pod | RTX 4090 (Ada) | the test network's node, one worker, the job's posting, and scoring |
| Worker 2 | RTX 4090 (Ada) | a worker |
| Worker 3 | A100 (Ampere) | a worker |
| Worker 4 | A100 (Ampere) | a worker |

Two of each generation, so every check is replayed both on the GPU type that trained it and on the other. Four workers
are more than the three a round needs before it plants canaries.

RunPod settings for every pod: a PyTorch template, Secure Cloud (its addresses don't move when a pod restarts), and a
volume at `/workspace`. On the node pod only, list `8545` under **Expose HTTP Ports**: RunPod then serves it at
`https://NODE_POD_ID-8545.proxy.runpod.net` with a valid certificate. That address is `NODE` below. The proxy closes a
request after 100 seconds; uploads go in chunks, each well inside that.

## 1. Every pod: install from the repository

```bash
cd /workspace
git clone https://github.com/VibeHash/vibehash.git && cd vibehash
git checkout <the commit this run uses>
pip install -e ".[finetune,evo]" peft
export VIBEHASH_HOME=/workspace/.vibehash VIBEHASH_PASSWORD='<a password for this run only>'
```

The template's PyTorch is kept as it is. Record `git rev-parse HEAD`, `python -c "import torch; print(torch.__version__)"`
and `nvidia-smi --query-gpu=name,driver_version --format=csv` on each pod for the paper.

## 2. Node pod: start the network and post the job

```bash
vibehash testnet --host 0.0.0.0 --workers 1 --device cuda --job none --keep-rounds 1000 --plain
```

`--keep-rounds 1000` keeps every round's opened checkpoints, so the margins can be measured after training ends. In a
second terminal:

```bash
python scripts/gsm8k_job.py prepare --revision 740312add88f781978c0658806c59bc2815b9866 --out gsm8k-train.jsonl
# must print sha256 76ea6dde2bdd0d82e9642fdd3253bb8d440b8c77c3bff50d601ff769a432666b
vibehash train gsm8k-train.jsonl --base qwen3-0.6b --node http://127.0.0.1:8545 \
  --wallet $VIBEHASH_HOME/testnet/requester.json --budget 200 --id gsm8k --yes --plain
```

7,473 chats at 8 sequences a step, 16 steps a unit and 8 units a round is about 7 rounds an epoch; 200 VIBE at 10 a
round buys 20 rounds, close to three epochs.

## 3. Workers 2–4: join

```bash
vibehash setup --node NODE --yes
vibehash worker --node NODE --device cuda --plain
```

Each prints its address when its wallet is made. Collect them, with the node pod's worker, in `pods.json`:

```json
{"<address of the node pod's worker>": "RTX 4090 #1", "<worker 2>": "RTX 4090 #2",
 "<worker 3>": "A100 #1", "<worker 4>": "A100 #2"}
```

## 3b. Stake every worker

A fine-tune trains nothing until at least two staked workers have registered (`FINETUNE_QUORUM`): its work has to be
checked by workers. A test network's workers start with no stake, so give each some of the requester's coins.
`scripts/gpu_run_pod.sh` does all of this by itself: the node pod runs `scripts/gpu_run_faucet.py`, which sends 20 VIBE
to each worker that registers without stake, and each worker's pod stakes 10 once it arrives. By hand, on the node pod,
one block apart (a second transfer sent before the first lands is refused for its nonce):

```bash
vibehash send --node http://127.0.0.1:8545 --wallet $VIBEHASH_HOME/testnet/requester.json --to <worker address> --amount 20
```

then on each worker's pod (the node pod's own worker uses `$VIBEHASH_HOME/testnet/worker-1.json`):

```bash
vibehash stake --node NODE --amount 10
vibehash balance --node NODE        # must show stake: 10 VIBE
```

Until two are staked, the node's log says the job waits for staked workers, and no units are assigned.

## 4. When the budget is spent: the adapter and its score

On the node pod:

```bash
vibehash job download gsm8k --node http://127.0.0.1:8545
python scripts/gsm8k_job.py score --base Qwen/Qwen3-0.6B --base-revision <the job's pinned commit> \
  --revision 740312add88f781978c0658806c59bc2815b9866
python scripts/gsm8k_job.py score --base Qwen/Qwen3-0.6B --base-revision <the job's pinned commit> \
  --revision 740312add88f781978c0658806c59bc2815b9866 --adapter gsm8k-train-lora
```

`job download` names the adapter's folder after the data file (`gsm8k-train-lora`) and prints the base model's pinned
commit in its loading example: that's the `--base-revision`. Keep both score outputs.

## 5. The margins, on each GPU type

Stop the workers first, so the replays have the GPU to themselves. Then, on one 4090 pod and one A100 pod:

```bash
python scripts/finetune_margins.py --node NODE --labels pods.json --out margins-$(hostname).csv
```

Each replays every checked segment of every round on its own GPU and writes one row per segment and part (B, m, v):
the distance, the share of the rule's limit it used (under 1.0 passes), and whether the real check agrees. The
`agrees_with_rule` column must be true on every row; a false one means the script's arithmetic and the check's
differ, and the table can't be used until that's fixed.

## 6. Before stopping the pods

Copy off: the margins CSVs, `pods.json`, both score outputs, the adapter folder, the node's log, and the versions from
step 1. A stopped pod keeps `/workspace`; a terminated one doesn't.

## Not yet in this run

- A cheating worker on real hardware: the cheats (`shortcut`, `freezer`, …) exist in the simulator's workers, not in
  `vibehash worker`.
- A single-GPU control run of the same training, to compare the network's total GPU time against one GPU's.
