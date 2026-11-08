# vibehash train

Train a model on a CSV file: posts it as a job for the network to train.

```
vibehash train [-h] [--base MODEL] [--rank RANK] [--lr LR] [--max-len MAX_LEN] [--units UNITS] [--yes] [--label LABEL] [--evo] [--budget BUDGET] [--fee FEE] [--hidden HIDDEN] [--id ID] [--node NODE] [--wallet WALLET] [--plain] [data]
```

Arguments:

- `data` (optional): CSV with a header row: numeric input columns, plus the column to predict (on a terminal, leave it out to pick a model from a menu).

Flags:

- `--base MODEL`: fine-tune an open-weight model on a file of chats or text (JSON lines of {"messages": [...]} or {"text": ...}): a name from `models`, org/name or a huggingface.co address. Without a data file, shows what the model would take.
- `--rank RANK`: fine-tunes: LoRA rank (default 16).
- `--lr LR`: fine-tunes: Adam's learning rate (default: by model size).
- `--max-len MAX_LEN`: fine-tunes: longest sequence in tokens; longer chats are cut (default 512, at most 2048; longer sequences make disputes too big to play from home).
- `--units UNITS`: fine-tunes: units of training a round (default 8).
- `--yes, -y`: fine-tunes: don't ask before checking weights or posting.
- `--label LABEL`: the column to predict (default: the last one).
- `--evo`: Evo mode: post an Evo job, training --base (default Qwen/Qwen2.5-0.5B-Instruct) by evolution on puzzles the chain draws, no data file. For local test networks for now (`testnet --evo`).
- `--budget BUDGET`: VIBE to spend, paid only for work that passes its checks (default 100).
- `--fee FEE`: VIBE per round of training (default 10; an Evo job's 1).
- `--hidden HIDDEN`: hidden layer widths (default 128,128).
- `--id ID`: job name (default: from the file name).
- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--plain`: print the plan and return, instead of following the job on the live training dashboard (piped output does this anyway).

Examples:

```bash
vibehash train fruit.csv --budget 100
vibehash train my-data.csv --label price --hidden 256,256
vibehash train chats.jsonl --base qwen3-0.6b
```
