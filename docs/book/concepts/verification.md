# How the network checks work

Re-doing all the training would double its cost, so Vibe Hash re-runs small random pieces of it and makes cheating a losing bet. This page explains how.

## Units and segments

Each round, a job's model is split into small **units** of work: "starting from this round's model, run so many training steps with batches drawn from this seed". Units are sized so a slow laptop can finish one before the deadline; faster machines get more units, not bigger ones.

Each unit is split into **8 segments**, with a saved snapshot of the model (a checkpoint) between each one. Those snapshots are what make spot-checking possible.

## Sealed answers first

Before anyone reveals anything, each worker sends a signed **commitment**: a fingerprint of its result, built from a Merkle tree over its checkpoints and bound to its own key. Nobody can copy someone else's answer, since it wouldn't match their own fingerprint.

Only after the deadline does the round get its random **beacon**, made from the previous block, a secret the round's leader committed to in advance, and all the commitments. Nobody can know it before committing, so nobody can predict which segments will be checked. Workers then reveal their results and the checkpoints the beacon picked, with proofs that they match the commitment.

## The checks, cheapest first

| Check | What it catches |
|---|---|
| Sealed answer | Copying another worker's result |
| Proven snapshots | Swapping in different checkpoints after seeing which get checked |
| Sanity check | Training that doesn't start from this round's model, doesn't end where the worker says, or changes the model by a suspicious amount |
| Spot-check replay | Skipped or made-up training |
| Duplicate units | About 1 in 5 units secretly goes to two workers. If their answers differ, the whole unit is re-run to see who was wrong |
| Planted fakes | Lazy checkers. Known-fake results that look like real work are slipped to checkers; one who approves a fake is caught |

## Replays and tolerance

A randomly chosen staked worker, the **checker**, re-runs 3 random segments of a unit from the worker's own checkpoints and compares its result with the next checkpoint. Checking is paid work.

Different chips round numbers very slightly differently, so a GPU's result and a CPU's replay never match bit for bit. A result passes if it is within the job's **tolerance** and within 1% of what that segment of training actually changed. That absorbs rounding differences between chips, but not skipped work. Workers also turn off GPU modes that trade accuracy for speed, so replays on another device still match.

## The odds of catching a cheat

Suppose a worker fakes `m` of a unit's `k` segments, and the checker replays `q` different segments picked at random. The chance it gets away with it is `C(k − m, q) / C(k, q)`: the number of ways to pick only honest segments, over all the ways to pick.

| Segments in a unit | Faked | Checked segments | Gets away with it |
|---|---|---|---|
| 8 | 4 (half) | 3 | 7.1% |
| 8 | 1 | 3 | 62.5% |
| 8 | 4 (half) | 1 | 50% |

At the default of 8 segments and 3 checked, faking half a unit is caught 92.9% of the time, and duplicate units give a second, independent chance. The chance of being caught is never below the share faked. Getting caught costs half the worker's stake and the whole round's pay. Faking less saves less work but still risks the stake.

## Disputes and the referee

A rejection is only a claim. When a checker rejects a unit, the node re-runs the disputed segment itself, and a checker who rejected honest work is slashed for a false accusation. A checker who doesn't answer loses reputation and the node checks the work itself.

Fine-tunes of large language models are too big for the node to replay. There, two checkers check each claim. When a worker and a checker disagree, they play a **dispute game**: taking turns, they narrow their disagreement down through the unit, segment, step, layer and operation to one small tile of numbers. The node then acts as **referee**, recomputing that one tile on its CPU with bounds on how far an honest calculation can land. The side that's out of bounds loses. If the referee can't decide (for example, the model's weights couldn't be fetched), the claim ends unchecked and nobody is slashed. Every move is signed and kept in the round's record, so any validator can replay the game and reach the same verdict.

## Checking the node

Every choice the node makes follows from the round's signed messages, which its block commits to, so anyone can replay the round with `vibehash chain verify`. Signed receipts let a worker prove a message the node accepted and then dropped.

## Slashing

Your stake is at risk only for things an honest machine never does: a result that fails a check, approving a planted fake, or rejecting honest work. Each costs **50% of stake**, including coins still unbonding, plus the whole round's pay, and resets reputation to its lowest. Half the slashed amount goes to whoever caught it and half is burned.

Crashes and dropped connections cost only that round's pay and some reputation, never stake. See [Economics](economics.md) for the numbers, and [How mining works](../mining/how-mining-works.md) for a worker's side of a round.
