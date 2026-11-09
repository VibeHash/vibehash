# Quick start: train on a CSV

This page takes you from a spreadsheet of examples to a trained model and its predictions, with four commands. It uses the sample data Vibe Hash ships with, so you can try it before using your own.

## 1. Get some data

```bash
vibehash example
```

This writes two files to the current folder:

- `fruit.csv`: 320 measured fruit with their names, to train on.
- `fruit-new.csv`: 12 more without names, to predict.

Your own data works the same way. Save it as a CSV file with:

- a header row naming the columns,
- numeric input columns (measurements, counts, prices),
- the column to predict, last. It holds categories, as names or numbers: 2 to 1,000 different values.

If the column to predict isn't the last one, name it with `--label`, for example `--label species`.

## 2. Train it

```bash
vibehash train fruit.csv --budget 100
```

`train` does the setting up for you:

1. It reads the table and sizes a model to it.
2. It scales the inputs, so columns on very different scales train well together.
3. It holds back a fifth of the rows. The model never trains on them; they're used to score it fairly.
4. It uploads the data, posts the job, and prints the plan: rows to train on, rows held back, the model's size, and the budget.

`--budget 100` means you spend at most 100 VIBE. At the default of 10 VIBE a round, that's up to 10 rounds of training, and you pay only for work that passes its checks. See [What training costs](costs.md).

{% hint style="info" %}
**Trying it on your own computer first?** Start a test network in one terminal with `vibehash testnet`, then pay with its ready-made requester wallet in another:

```bash
vibehash train fruit.csv --budget 100 --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json
```

{% endhint %}

## 3. Watch it learn

In a terminal, `train` follows the job live once it's posted: accuracy and loss charted round by round, the units of work accepted, the budget left, and whether it's your job's turn to train. Jobs take turns a round at a time, so yours starts at the next round that is its turn.

Press Ctrl+C to stop watching. That stops only the watching, not the training. To look again later:

```bash
vibehash job watch
```

## 4. Download the model

```bash
vibehash job download
```

This saves the model as one file named after your data, here `fruit-model.npz`, with a short record of how it was trained beside it. You can download before the budget is spent to get the model so far.

## 5. Predict

```bash
vibehash predict fruit-new.csv
```

`predict` uses the model you downloaded last and shows each row with its prediction. The new file needs the same input columns as the training data. To save every row with its prediction:

```bash
vibehash predict fruit-new.csv --out predictions.csv
```

## Your own data

```bash
vibehash train my-data.csv --budget 100
vibehash job watch
vibehash job download
vibehash predict new-rows.csv
```

Commands like `job watch` and `job download` work on the last job you trained unless you name another. The job's name is printed when you post it (it's made from the file name plus four random characters), or choose one with `--id`.

## Next

- [Following a job and getting results](results.md): status, history, scores and cancelling
- [What training costs](costs.md)
- [vibehash train](../reference/train.md), [vibehash predict](../reference/predict.md), [vibehash example](../reference/example.md): every flag
