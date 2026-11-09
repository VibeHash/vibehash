# VIBE: balance, sending and staking

This page covers what you can do with VIBE once you have some: check your balance, send it, stake it, and get it back. It also explains reputation and transaction fees.

## Your balance

```bash
vibehash balance
```

This shows, for your wallet's address:

| Line | What it means |
|---|---|
| balance | VIBE you can spend now |
| stake | VIBE locked as stake (see below) |
| unbonding | unstaked VIBE on its way back, and the block it becomes spendable at |
| reputation | a number from 0.50 to 1.00 that scales your pay |
| tx fee | what this node charges per transaction |

To look up someone else's address, add it: `vibehash balance vibe1...`.

## Sending VIBE

```bash
vibehash send --to vibe1... --amount 2.5
```

Amounts are in VIBE, with up to 8 decimal places. Copy and paste the address rather than typing it. Some of its letters are capitals, and those capitals are a checksum: a mistyped address is refused before anything is signed or sent. The transfer lands in the next block.

## Staking

Stake is VIBE you lock up as a promise to play fair. Training work can only be spot-checked, so stake is what a cheater stands to lose. It backs a worker's checks of other people's work and a validator's votes on blocks.

```bash
vibehash stake --amount 10
```

What stake gets you:

- **5 VIBE** is the minimum for a worker to be fully trusted: to check other workers' results, and to take full-size batches of work.
- **100 VIBE** is needed to join the validators with `vibehash validator join`.

### Starting with nothing

You don't need to buy VIBE to start mining. On networks where workers earn their stake in, such as the public one, your first 5 VIBE of pay goes into your stake instead of your balance. Until then you get small batches, you get work after staked workers, and you don't check anyone else. If your stake ever falls below 5 VIBE, you go back to earning it in, and your pay refills the stake first.

### Bootstrap stake

A network's first validators start with **bootstrap stake**, written into the network's first block. It counts for voting, but it can never be withdrawn or spent, and it is burned if the validator leaves. If you hold some, `balance` says how much of your stake it is.

### What can be taken

Crashes and dropped connections cost you only that round's pay. Stake is at risk only for things an honest machine never does, such as faking results or approving faked work: then half of it is slashed. [How the network checks work](../concepts/verification.md) has the details.

## Unstaking

```bash
vibehash unstake --amount 10
```

Unstaked VIBE doesn't come back at once. It **unbonds for 20 rounds**, and during that time it can still be slashed. `balance` shows it under "unbonding" with the block it becomes spendable at. Bootstrap stake can't be unstaked.

## Reputation

Reputation scales your share of each round's pay, from 0.5 times to 1.0 times.

- A new wallet starts at **0.80**.
- It gains **0.02** each round you deliver verified work.
- It loses **0.05** when you miss a deadline or don't answer a checking task.
- It drops to **0.50** if you're caught cheating.

## Transaction fees

Every transaction (send, stake, unstake, posting or cancelling a job) pays a small fee to whoever produces the block it lands in. The fee comes out of your balance, on top of the amount.

Each node sets the minimum it accepts; the default is **0.0001 VIBE**. Commands pay the node's minimum unless you choose more with `--fee`. Nodes put higher-paying transactions into blocks first, so a higher fee only matters when blocks are full.

{% hint style="info" %}
`--fee` on `send`, `stake`, `unstake`, `post-job` and `cancel-job` is this transaction fee. On `train`, `--fee` means something else: what a training job pays per round. See [What training costs](../training/costs.md).
{% endhint %}

## Units

Amounts are shown in VIBE. Inside the network, amounts are whole numbers of **syn**: 1 VIBE = 100,000,000 syn. You'll only meet syn if you edit a job file by hand, where `fee_per_round` is written in syn.

## Next

- [Wallets and recovery words](wallet.md)
- [Economics](../concepts/economics.md): where new VIBE comes from and how each round's pay is split
- Command pages: [balance](../reference/balance.md), [send](../reference/send.md), [stake](../reference/stake.md), [unstake](../reference/unstake.md)
