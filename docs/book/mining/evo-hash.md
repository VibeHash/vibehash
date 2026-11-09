# Evo Hash

This page explains Evo Hash, a different way of mining that trains a model by evolution, and how to join the public Evo network or try it on your own computer.

## What it is

In an Evo Hash job, each round, miners answer puzzles with slightly twisted copies of a language model. The twists that score better pull the model their way. A block keeps only the twists' seeds and how they scored, and every node works the model out from those. So the trained model is a function of the chain itself: about 2 KB a round, with no weight files to store.

Checking works in the same spirit as other jobs: other miners check your answers by running them through the same twisted model. An Evo unit is never checked by the miner who answered it. More on how machines with different chips agree is on [GPUs](gpus.md#different-hardware-same-answers), and the full design is in the [Evo Hash white paper](https://github.com/VibeHash/vibehash/blob/main/docs/evo-hash.pdf).

## What you need

- An **NVIDIA GPU** with PyTorch. Run `vibehash install-gpu` if you haven't (see [GPUs](gpus.md)).
- For the public Evo network, a GPU with **8 GB or more**.
- The Evo extra, which brings Hugging Face's transformers:

```bash
python -m pip install "vibehash[evo]"
```

The menu offers to install it for you if it's missing.

## The public Evo network

`https://evo.vibehash.org:8443` (network `vibehash-evo-1`) is a public network for Evo jobs, separate from the main one. Its job tunes **Qwen3.5-0.8B** on 16 kinds of problem (math, following instructions, science, common sense) to beat the official model.

Join it with:

```bash
vibehash worker --node https://evo.vibehash.org:8443
```

Or run `vibehash` and pick **Evo Hash** under **Mine**, then option 1. The menu makes a wallet first if you don't have one.

What to expect:

1. The miner downloads the job's model once (about 1.7 GB) and the tasks' datasets.
2. Before taking Evo work, your card checks that it reproduces its own answers exactly. A card that can't doesn't take Evo work.
3. A new wallet starts with no stake and earns its first 5 VIBE as stake. Until then, its work only counts while a staked miner is online to check it.

With several GPUs, run a worker for each with its own wallet and `--device cuda:N`, as on [The worker](worker.md#several-gpus).

{% hint style="warning" %}
transformers may suggest installing `causal_conv1d` or `flash-linear-attention` for speed. Don't. Every miner has to compute exactly the same way, and those kernels round differently, so your answers would fail other miners' checks.
{% endhint %}

## A test network on your own computer

To try Evo Hash with nothing shared, run a whole Evo network locally: a node and two miners, tuning **Qwen2.5-0.5B-Instruct** on questions from public datasets (science, common sense, word-problem math and more).

```bash
vibehash testnet --evo2
```

It's also in the menu: **Evo Hash** under **Mine**, then option 2. The model (about 1 GB) and the datasets download once. Each job is short (12 rounds unless you set `--evo2-rounds N`), and each starts where the last ended, so the model keeps what every task taught it. The testnet carries on where it stopped the next time you run it.

To take the tuned model out as an ordinary Hugging Face folder, while the testnet runs or after, in another terminal:

```bash
vibehash evo export
```

See [vibehash evo](../reference/evo.md) and [vibehash testnet](../reference/testnet.md) for every flag.
