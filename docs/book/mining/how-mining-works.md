# How mining works

This page explains what your computer actually does when it mines Vibe Hash: one round step by step, how the work is checked, what gets paid, and what can cost you. It's written for miners; the protocol details are in [How the network checks work](../concepts/verification.md) and [Economics](../concepts/economics.md).

## The idea

Bitcoin miners spend electricity on hashes nobody uses, and one lucky winner per block gets paid. Vibe Hash replaces the hashes with training steps on a real AI model that someone has paid to have trained. There's no lottery: everyone whose work passes its checks is paid in proportion to how much they did, at the same rate per unit of work, on an old laptop or a gaming rig. You don't need a mining pool.

## A round, step by step

The network runs in rounds. On the public network a round takes about 38 seconds and ends with one new block. Jobs with big models get longer rounds, sized so an ordinary computer on a home connection can still download the model, train and upload in time.

1. **Register (about 3 s).** Your worker checks in with its measured speed and memory. Workers with 5 VIBE staked get work first; newcomers still earning their stake get small batches from what's left.
2. **Compute (about 20 s).** The work is split into small **units**, sized so a slow laptop can finish one in time. Faster machines get more units, not bigger ones. Your worker trains its units and, before the deadline, sends a sealed fingerprint of each result (the **commit**).
3. **Reveal (about 5 s).** A random number that nobody could know earlier picks which parts of each unit get checked. Workers reveal their results and those parts.
4. **Verify (about 10 s).** Other workers re-run the chosen parts and report whether they match. Checking is paid work too, and your worker does it for others.
5. **Payment.** Passing results go into the model, and the payments and penalties go into a block. The network's validators vote on the block, and once it's final your pay lands in your balance.

The live dashboard and the desktop app show these same steps as Registered, Training, Committed, Checking others and Paid.

## How the work is checked

Re-doing all the work would double its cost, so Vibe Hash re-runs small random pieces and makes cheating a losing bet. Each unit of training is split into 8 segments, with a saved snapshot of the model between each one.

- **Sealed answer.** Your fingerprint is sent before anything is revealed, so copying someone else's result can't match.
- **Proven snapshots.** Every snapshot you reveal must match your fingerprint, so nobody can swap them after seeing which parts get checked.
- **Sanity check.** Training must start from this round's model and end where you say, and the change must be a sensible size.
- **Spot-check replay.** A randomly chosen staked worker re-runs 3 random segments and compares. Small rounding differences between chips are allowed; skipped work isn't.
- **Duplicate units.** About 1 in 5 units secretly goes to two workers. If their answers differ, the unit is re-run to see who was wrong.
- **Planted fakes.** Two known-fake results per round are slipped to checkers. A checker who approves one is caught being lazy. A checker who rejects honest work is caught too, because every rejection is re-run.

A GPU's result and a CPU's replay never match bit for bit, so the checks allow for that. Workers turn off GPU shortcuts that trade accuracy for speed, so any honest machine's work checks out on another. More on [GPUs](gpus.md).

## What gets paid

Each round makes a pool of VIBE: new coins from the block subsidy, plus the job poster's fee for the units that passed. Posters pay nothing for work that failed. On the public network the subsidy is 3.125 VIBE a round that pays for work, halving about every 4 years. The pool is split three ways:

| Share | Goes to |
|---|---|
| 80% | Workers, by verified training work, scaled by reputation |
| 15% | Checkers, by the work they re-ran, scaled by reputation |
| 5% | The validator that produced the block, only if work was verified |

Anything nobody earned is burned, so nobody gains by rejecting good work.

**Speed.** Each worker's real speed is measured, and units are handed out in proportion. A new worker starts with a few units (at most 4 until it has delivered once) and can take up to twice as many each round it delivers in full. A GPU 20 times faster ends up with about 20 times the units and the pay.

**Memory.** You're only given jobs that fit: graphics memory for GPU jobs, system RAM otherwise.

## Stake

Your **stake** is VIBE locked as a promise: it's what a cheater stands to lose. On the public network you start with none. Your first 5 VIBE of pay goes into your stake instead of your balance. Until then you get small batches of work, only after staked workers have theirs, and you don't check other workers' results. Once your stake is in, you take full work and checks.

If you unstake, the coins wait 20 blocks before you can spend them, and they can still be slashed in that time. If your stake falls below 5 VIBE, you go back to earning it in. See [VIBE: balance, sending and staking](../wallet/vibe.md).

## Reputation

Reputation scales your share between 0.5 and 1.0. New wallets start at 0.8, gain 0.02 for each round they deliver verified work, and lose 0.05 for missing a deadline. It moves per round, not per unit, so a laptop builds it as fast as a big rig.

## Penalties

Crashes and dropped connections only cost you pay. Your stake is at risk only for things an honest machine never does.

| What happened | Stake | Pay | Reputation |
|---|---|---|---|
| Unit passed its checks | kept | paid for the unit | +0.02 for the round |
| Didn't finish before the compute deadline | kept | that unit unpaid | -0.05 |
| Crashed or went offline after committing | kept | whole round unpaid | -0.05, and back to the new-worker limit |
| Didn't answer a checking task | kept | no checking pay | -0.05 |
| Result failed a check, or didn't match its fingerprint | -50% | whole round unpaid | reset to 0.5 |
| Approved a planted fake while checking | -50% | whole round unpaid | reset to 0.5 |
| Rejected honest work while checking | -50% | whole round unpaid | reset to 0.5 |

Half of any slashed stake goes to whoever caught the problem, and the other half is burned.

## Other kinds of jobs

Most of the above is how classic jobs (small models trained on spreadsheets) work. [Evo Hash](evo-hash.md) jobs and [DeepSeek jobs](deepseek.md) follow the same pattern of train, commit, check and pay, with their own checks. Your worker takes whatever jobs your computer can do, unless you pick one with `--job` (see [The worker](worker.md)).
