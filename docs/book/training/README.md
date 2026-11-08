# Training a model

This section is for you if you want a model trained: you bring data and a VIBE budget, and miners on the network do the training. This page explains what kinds of training jobs exist, what happens to a job from start to finish, and where to go next.

## What you can train

| Kind of job | What you give it | What you get back | Command |
|---|---|---|---|
| A small model trained from scratch | a CSV table of examples | a model file that labels new rows | `vibehash train data.csv` |
| A fine-tune of an open-weight language model | a file of chats or text | a LoRA adapter for the base model | `vibehash train chats.jsonl --base NAME` |
| An Evo job | nothing: the network draws puzzles itself | the tuned model, worked out from the chain | `vibehash train --evo` |

**Tables (CSV).** This is what the public network trains today. Your CSV has a header row, numeric input columns, and one column to predict, such as a fruit's name from its measurements. The model learns to sort rows into those categories (from 2 up to 1,000 of them). `train` sizes the model to your data for you. Start with the [quick start](quickstart.md).

**Fine-tunes.** You give a file of example conversations and pick a base model, such as Qwen3 0.6B. Workers with GPUs train a small add-on (a LoRA adapter) that changes how the model answers. This works for models one gaming GPU can hold, on a network whose node runs a version that supports it: a test network on your own computer today, and the public network once its node is updated. See [Fine-tuning a language model](fine-tuning.md).

**Evo jobs.** These train a model by evolution on puzzles the chain draws, with no data file. `train --evo` only works on networks set up to take them, such as a test network on your computer (`vibehash testnet --evo`). The public Evo network runs its own job, which miners join; see [Evo Hash](../mining/evo-hash.md).

**DeepSeek jobs** are run by the network's owner, not posted by users. They're described under [DeepSeek jobs](../mining/deepseek.md).

## The life of a job

1. **You post it.** `train` checks your data, shows what it will train and what it costs, uploads the data, and posts the job. Your whole budget moves out of your balance into the job's escrow.
2. **It may wait.** Jobs take turns, one round at a time, so a new job starts at the next round that is its turn. Some nodes hold new jobs until an operator confirms them. See [Job specs and posting a job](job-specs.md).
3. **Miners train it.** Each round, workers train small units of your model and other workers re-check them. Only work that passes its checks is paid for, out of your escrow.
4. **You follow it.** A live dashboard shows accuracy and loss round by round, and the budget left. Each round is scored on part of your data the model never trains on.
5. **It finishes** when the budget is spent. You can also cancel it at any time and get back what it hasn't spent.
6. **You download the result** and use it: `vibehash predict` for a table model, or the adapter in your usual tools for a fine-tune.

You don't need to keep your computer on while a job trains. The job lives on the network; `vibehash job watch` picks it up again at any time.

## Before you start

- Vibe Hash installed: [Install Vibe Hash](../getting-started/install.md).
- A wallet with enough VIBE for the budget: [Wallets and recovery words](../wallet/wallet.md).
- On a test network on your own computer, you can pay with its ready-made requester wallet instead; the [quick start](quickstart.md) shows how.

## Where to go next

- [Quick start: train on a CSV](quickstart.md)
- [Fine-tuning a language model](fine-tuning.md)
- [Job specs and posting a job](job-specs.md), for full control
- [Following a job and getting results](results.md)
- [What training costs](costs.md)
