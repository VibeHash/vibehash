# Following a job and getting results

This page covers what to do after you post a job: checking how it's going, reading its scores, downloading the model, using it, and cancelling it to get unspent budget back.

All the commands below work on the last job you trained unless you give a job name, such as `vibehash job status fruit-3fa2`.

## How it's going

```bash
vibehash job status
```

A short summary: whether it's still training or finished, rounds done, units of training accepted, the latest accuracy and loss (with where it started and its best), and the budget left with about how many more rounds that buys.

```bash
vibehash job watch
```

The live dashboard: accuracy and loss charted round by round, the budget left, and whether it's your job's turn. It runs until the budget is spent. Ctrl+C stops watching, never the training. Add `--plain` for one line per event instead.

```bash
vibehash job history
```

Every round one by one: loss, accuracy, units accepted, VIBE paid and budget left. It comes from the node, so it works for any job, from any computer. A copy is kept in `~/.vibehash/jobs/JOB`. `--csv FILE` saves it as a spreadsheet, and `--json` prints it for scripts.

## Reading the scores

Each round, the model is scored on **held-back data**: part of your data that it never trains on. That's a fair test of how it will do on data it hasn't seen.

| Score | What it means | Better is |
|---|---|---|
| **Accuracy** | For a table model, the share of held-back rows it labels right. For a fine-tune, the share of held-back reply tokens it predicts right. | higher |
| **Loss** | How far its predictions are from the right answers, as one number. It moves more smoothly than accuracy, so it's the best sign that training is still helping. | lower |

For a fine-tune, loss is the main number to watch. Each round's score is of the model that round started from, so the newest round's training shows up in the next round's score.

## Downloading the model

```bash
vibehash job download
```

- **A table model** is saved as one file named after your data, such as `my-data-model.npz`. Beside it go `my-data-model-training-history.csv` and `my-data-model-TRAINING.md`, a short page on how it was trained.
- **A fine-tune** is saved as a folder, such as `chats-lora/`, holding the adapter and the same record. See [Fine-tuning a language model](fine-tuning.md).
- **An Evo job** keeps no weight files: its model is worked out from the chain and saved as a Hugging Face folder.

You can download while the job is still training to get the model so far. `--out` picks the file or folder name.

{% hint style="info" %}
Download from the computer you posted from. That computer kept your column names, category names and input scaling. Downloaded elsewhere, a table model has no names, and the rows you predict on must already be scaled the way its dataset was.
{% endhint %}

## Using a table model

```bash
vibehash predict new-rows.csv
```

`predict` uses the model you downloaded last (or name one: `vibehash predict my-data-model.npz new-rows.csv`). The new rows need the same input columns. It shows the first 20 rows with their predictions; `--out predictions.csv` saves them all.

## Cancelling and getting the budget back

```bash
vibehash cancel-job
```

Only the wallet that posted a job can cancel it. The budget it hasn't spent comes back to your balance once the cancel lands in a block. If a round is being paid in that same block, that round's fee is still paid: you keep paying for work that was done, and get back the rest.

A job you don't cancel simply runs until its budget is spent, and `job status` then says it's finished.

## Next

- [What training costs](costs.md)
- [vibehash job](../reference/job.md), [vibehash predict](../reference/predict.md), [vibehash cancel-job](../reference/cancel-job.md)
