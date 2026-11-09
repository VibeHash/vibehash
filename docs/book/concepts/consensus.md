# Consensus and blocks

A network's validators have to agree on every block, even if some are offline, slow or dishonest. This page explains blocks, voting, finality, halts and syncing.

## Blocks

The network runs in rounds, and each round ends with one block. On a node with default settings a round takes about 38 seconds: register, compute, reveal and verify. Jobs with big models get longer rounds.

A block holds:

- the transactions it includes (transfers, stakes, job posts and so on);
- a **round summary**: who did how much verified work, and who cheated;
- the hash of the **round record**, every signed message the round was decided from;
- the hash of the block before it, and the proposer's signature.

Payments and penalties aren't trusted from the proposer. Every node works them out again from the summary and the chain's state, so a block whose payments differ is invalid. A proposer can't pay itself extra, skip a slash or overcharge a job poster.

## Who proposes

For each height, validators take turns leading, in a stake-weighted order that depends only on the height. The order is known in advance, and whoever made the previous block can't steer it. The leader runs the round with the workers, then proposes its block together with the round record.

## Voting: prevote and precommit

Voting follows Tendermint's algorithm (Buchman, Kwon and Milosevic, 2018). Each validator first checks the proposal: the block must apply to the chain, and its round must replay exactly from the record. Then it votes in two steps:

1. **Prevote.** A validator prevotes for a block it has checked. Once it sees more than two thirds of the stake prevote for a block, it **locks** on it, and won't vote for any other block at that height unless a later round shows the network moved on.
2. **Precommit.** It then precommits. A block is **final** once validators holding more than two thirds of the stake precommit it in the same round.

If the leader is offline, slow, or proposes something that fails the checks, the others time out and the next validator in the order leads. A block that more than two thirds already prevoted for is proposed again rather than trained again.

## Commit certificates and finality

The signed precommits that made a block final are its **commit certificate**. It's stored and served with the block, and anyone can check it against the validator set without trusting the node that served it.

Finality is immediate: once a block has its certificate, it won't be replaced. Two different blocks can never both be final at one height unless more than a third of the stake signs both ways. A validator that does that leaves two signed, conflicting votes behind, and the proof costs it all of its stake and its place ([Validators](../nodes/validators.md)).

Requiring more than two thirds has a price. If the validators still online hold two thirds of the stake or less, no block can be final and the chain waits. That's why four validators with equal stake is the smallest set that keeps going with one offline.

## Idle chains

On a chain run by validators, nobody waits for a proposal until there's something to put in a block: a funded job or a waiting transaction. A network with nothing to do doesn't churn out empty rounds.

## When a node halts

A node saves each final block, with its certificate and record, before it builds on it or serves it. If it can't (the database already has a different block at that height, or the disk is full or failing), that node **halts**: it stops leading, voting, proposing and syncing, keeps serving only the blocks it saved, and says so in `/status`. A restart carries on from what's on disk.

A halt is one node protecting itself from serving blocks it doesn't have. The network carries on if the validators still running hold more than two thirds of the stake.

## Followers and syncing

A **follower** is a node with no vote. It syncs the chain from its peers, serves it, passes transactions on, and points workers at the current leader.

A node that's behind, because it was offline or a vote names a later height, fetches the missing blocks from a peer with their certificates and records. It trusts none of them: each must apply, replay from its record and carry a valid certificate. On start-up, a node re-checks every stored block, and refuses a data folder where a certificate is missing or forged.

Anyone can run the same checks with `vibehash chain verify` ([vibehash chain](../reference/chain.md)).

## A chain without validators

A network whose genesis names no validators is run by one node, which seals every block itself. It's simpler, but that one node can halt or censor the chain. See [Genesis and TLS](../nodes/genesis-and-tls.md).
