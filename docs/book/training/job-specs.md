# Job specs and posting a job

`vibehash train` writes and posts a job for you. This page is for when you want to set every training setting yourself: it explains the job file, how to post it with your own dataset, and why a posted job may wait for a node's operators to confirm it.

## 1. Write a starter file

```bash
vibehash job template
```

This writes `job.json`, a small working job you can edit. `--out FILE` names the file, `--id NAME` names the job, and `--gpu` gives a larger model sized for GPUs.

The starter file looks like this:

```json
{
  "id": "my-first-job",
  "fee_per_round": 1000000000,
  "model": {"n_in": 8, "n_out": 4, "n_hidden": 32},
  "precision": "float64",
  "data": {"kind": "synthetic", "seed": 7},
  "init_seed": 11,
  "lr": 0.1,
  "batch_size": 32,
  "steps_per_unit": 40,
  "checkpoint_every": 5,
  "units_per_round": 12,
  "memory_mb": 1024,
  "tolerance": 1e-06,
  "max_update_ratio": 0.5
}
```

## 2. The fields in plain words

| Field | What it does | Allowed |
|---|---|---|
| `id` | The job's unique name. | 1 to 64 characters: a-z, 0-9, `.`, `_`, `-` |
| `fee_per_round` | What you pay for a round in which every unit passes, in **syn** (1 VIBE = 100,000,000 syn). The example is 10 VIBE. | a whole number, 0 or more |
| `model.n_in`, `model.n_out` | How many input columns, and how many categories to predict. | 1 to 4,096; 2 to 1,000 |
| `model.n_hidden` | The hidden layers' widths: one number, or a list such as `[512, 512]` for two layers. | 1 to 8 layers, each up to 8,192 |
| `precision` | `float32` runs on GPUs. `float64` always runs on CPUs. | `float32` or `float64` |
| `data` | What to train on: generated test data (`synthetic` and a `seed`), or your own file (`npz` and its hash). | `post-job --dataset` fills this in |
| `lr`, `batch_size` | The learning rate, and examples per training step. | over 0 up to 10; 1 to 4,096 |
| `steps_per_unit` | Training steps in one unit of work. | up to 100,000 |
| `checkpoint_every` | Steps per checked segment of a unit. | must divide `steps_per_unit` |
| `units_per_round` | How much work each round hands out. | 1 to 10,000 |
| `memory_mb` | Machines with less memory than this aren't given the job (graphics memory on a GPU, RAM on a CPU). | 1 to 1,000,000 |
| `tolerance` | How closely a re-check must match. Use 0.0001 for `float32` and 0.000001 for `float64`. | up to 0.01 |
| `max_update_ratio` | The largest change one unit may make to the model. | up to 10 |
| `compute_s` | Optional: a longer training phase for this job's rounds, in seconds. Left out, rounds are sized from the model. | 5 to 3,600 |

The network checks every field when the job is posted and rejects anything out of range. A node takes jobs up to 64 MB of model weights (about 16 million `float32` weights) unless its operator raises that.

## 3. Your own dataset

A dataset for `post-job` is an `.npz` file with four arrays: `x_train`, `y_train`, `x_val` and `y_val`. The `_val` rows are held back to score the model. The file is loaded without running any code inside it, and every worker checks its hash before using it.

## 4. Post it

```bash
vibehash post-job --spec job.json --budget 150 --dataset my-data.npz
```

`--budget` is the most you'll spend, in VIBE. It must cover at least one round (`fee_per_round`). `--dataset` uploads the file to the node and fills in the job's `data` for you.

The budget leaves your balance as soon as the job lands in a block, and is held for the job. It's paid out only for work that passes its checks; [cancel-job](../reference/cancel-job.md) returns what's left.

## 5. A job may wait for confirmation

Some nodes are run with **review** on. Their operators confirm or deny jobs in the [admin app](../nodes/admin-app.md), and a node with review on starts every newly posted job held until an operator confirms it. An operator can also hold a job at any time.

While a job is held:

- That node leads no rounds for it. Another node that doesn't hold it may still run it.
- Your budget stays where it is, untouched. Nothing is spent.
- `job watch` shows it waiting rather than training.

If your job waits for a long time, ask the network's operators, or cancel it to get your budget back.

## Next

- [Following a job and getting results](results.md)
- [What training costs](costs.md)
- [vibehash job](../reference/job.md), [vibehash post-job](../reference/post-job.md)
