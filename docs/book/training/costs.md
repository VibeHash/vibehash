# What training costs

This page explains what you pay for a training job, how your budget is spent round by round, and why a job may sit waiting without spending anything.

## Two numbers: budget and fee per round

| | What it is | Default with `train` |
|---|---|---|
| **Budget** (`--budget`) | The most the whole job can spend, in VIBE. | 100 VIBE |
| **Fee per round** (`--fee`) | What one round of training pays, when every unit of work in it passes. | 10 VIBE (an Evo job's: 1 VIBE) |

Roughly, the budget divided by the fee is how many rounds you get: 100 VIBE at 10 VIBE a round is up to 10 rounds. `train` prints this before you post, and a fine-tune's plan also says how many passes over your data (epochs) that buys.

In a job file, the fee per round is `fee_per_round`, written in syn: 1 VIBE = 100,000,000 syn. See [Job specs and posting a job](job-specs.md).

## How the budget is spent

1. **When you post**, the whole budget moves from your balance into the job's escrow. It's held there for the job; nobody can spend it but the job.
2. **Each round your job trains**, it pays its fee per round for the share of units that passed their checks. If 9 of 12 units pass, it pays nine twelfths of the fee. Work that fails its checks costs you nothing.
3. **That payment goes into the round's pool** with the newly minted VIBE, and the pool is split: 80% to the workers who trained, 15% to the workers who checked, and 5% to whoever produced the block. Anything nobody earned is burned, not handed to anyone.
4. **When the escrow runs out**, the job is finished. If you cancel first, what's left comes back to you. See [Following a job and getting results](results.md).

A budget must cover at least one round: it has to be more than 0 and at least the fee per round.

## Transaction fees

Posting a job and cancelling it are transactions, so each also pays a small transaction fee to the block's producer, on top of the budget. Each node sets its minimum; the default is 0.0001 VIBE. See [VIBE: balance, sending and staking](../wallet/vibe.md#transaction-fees).

{% hint style="warning" %}
`--fee` means different things on different commands. On `train` it's the job's fee **per round of training**. On `post-job` and `cancel-job` it's the **transaction fee**; there, the fee per round comes from the job file's `fee_per_round`.
{% endhint %}

## A real example

From a fine-tune run on a test network (Qwen3 0.6B, 2,379 short conversations, three workers sharing one GPU): the plan said 10 VIBE a round, so a 100 VIBE budget bought 10 rounds, about 3.3 epochs. The first scored round accepted no units, because the workers were still fetching the model, so it cost nothing. Over the rounds that followed, the job spent 30 VIBE of its 100. Each round took about 3¼ minutes.

## Why a job may wait

A job that isn't training isn't spending. It may wait because:

- **It's another job's turn.** Jobs with budget left take turns, one round at a time, so a new job starts at the next round that is its turn.
- **A node's operators are reviewing it.** Some nodes hold new jobs until an operator confirms them. Your budget stays untouched meanwhile.
- **No suitable workers are online.** Workers with less memory than the job needs aren't given it. A fine-tune's workers also download the model once before they can train.
- **It's too big for the network yet.** Jobs split across teams of GPUs wait until the node can run teams.

If a job waits for too long, you can cancel it and get the whole unspent budget back.

## Next

- [Economics](../concepts/economics.md): where new VIBE comes from and how pay is split
- [vibehash train](../reference/train.md), [vibehash post-job](../reference/post-job.md)
