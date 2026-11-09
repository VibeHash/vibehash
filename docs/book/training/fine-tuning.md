# Fine-tuning a language model

This page shows how to fine-tune an open-weight language model on your own example conversations: the file format, the command, which models are supported, and what you get back.

A fine-tune doesn't retrain the whole model. Workers train a small add-on, a **LoRA adapter**, that changes how the base model answers. You keep the adapter and load it on top of the base model.

{% hint style="warning" %}
Fine-tuning works on a network whose node runs a version that supports it: a test network on your own computer today, and the public network once its node is updated. The workers need GPUs with PyTorch; your own computer, posting the job, doesn't.
{% endhint %}

## 1. Prepare your data

Save your examples as **JSON lines**: one chat per line, in a file such as `chats.jsonl`.

```json
{"messages": [{"role": "user", "content": "Do you open on Sundays?"}, {"role": "assistant", "content": "Yes, from 10 to 4."}]}
{"messages": [{"role": "user", "content": "Can I return a gift?"}, {"role": "assistant", "content": "Within 30 days, with the receipt."}]}
```

- Every chat needs at least one `assistant` message. Only the assistant's replies are learned; the rest is context.
- Chats go through the model's own chat template and tokenizer, so you don't format anything yourself.
- Instead of chats, lines can be documents: `{"text": "..."}`. A plain `.txt` or `.md` file is read as one document.
- You need at least 2 examples. A tenth (at most 128) is held back to measure progress.

You also need the tokenizer packages on your computer:

```bash
pip install "vibehash[finetune]"
```

## 2. Pick a base model

```bash
vibehash models
```

This lists popular open-weight models, smallest first, with their size, licence, whether they're gated, and how many 12, 24 or 32 GB graphics cards would hold them. Models marked **ready** can be fine-tuned now: one GPU holds them, and they're from a supported family (Llama, Mistral, Qwen2, Qwen3 and SmolLM3). The rest are coming.

To see what a model would take without posting or spending anything:

```bash
vibehash train --base qwen3-0.6b
```

`--base` takes a name from the list, an `org/name` from Hugging Face, or a huggingface.co address.

## 3. Post the fine-tune

```bash
vibehash train chats.jsonl --base qwen3-0.6b
```

On a test network on your own computer, add `--node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json`.

Before anything is spent, `train`:

1. Pins the model to an exact version, so the run can be reproduced.
2. Asks to check the model's weights. The first time a model is used, they're streamed once from Hugging Face and hashed (about 1.5 GB for Qwen3-0.6B); nothing is kept.
3. Reads your data with the model's tokenizer and shows a plan: how much data, what will be trained, how many rounds make one pass over your data (an epoch), and the cost.
4. Asks **Post it?** Answer no and nothing is posted or spent.

The defaults suit most jobs: LoRA rank 16 on the attention layers, a learning rate chosen by model size, sequences up to 512 tokens (longer chats are cut), 8 units of training a round, a budget of 100 VIBE and 10 VIBE a round. Flags such as `--rank`, `--lr`, `--max-len` (at most 2048) and `--units` change them; `--yes` skips the questions. See [vibehash train](../reference/train.md).

## 4. Get the adapter

```bash
vibehash job download
```

This writes a folder named after your data, such as `chats-lora/`, holding:

- `adapter_model.safetensors` and `adapter_config.json`, a standard PEFT adapter;
- a README naming the base model and its pinned version;
- `training-history.csv` and `TRAINING.md`, how it was trained.

Load it with Hugging Face transformers and PEFT, or serve it with vLLM:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", revision="PINNED-COMMIT")
model = PeftModel.from_pretrained(base, "chats-lora")
```

Use the commit the folder's README names, so the adapter meets the exact weights it was trained on.

## Limits

- Only models one gaming GPU can hold. Bigger models need teams of GPUs, which are still being built.
- Repositories with only `.bin` (pickle) weights are refused, because those files can run code.
- Gated models, which need a Hugging Face licence and token, aren't fully supported yet.

## Next

- [Following a job and getting results](results.md)
- [What training costs](costs.md)
- [vibehash models](../reference/models.md)
