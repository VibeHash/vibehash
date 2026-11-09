# Validators

Validators are the nodes that vote on every block and take turns leading rounds. This page covers joining and leaving the validator set, the stake it takes, how many validators a network needs to survive an outage, and the one mistake that costs a validator everything.

## Joining

Joining takes at least **100 VIBE staked** in the wallet that will validate. Stake enough to reach that first (a wallet that already has some staked needs only the difference), then announce the address where other nodes can reach your node:

```bash
vibehash stake --amount 100 --wallet bob.json
```
```bash
vibehash validator join --url https://bob.example.org:8545 --wallet bob.json
```

Then run your node with the same wallet, announcing that address, and point it at a node of the network:

```bash
vibehash node --genesis genesis.json --wallet bob.json --data-dir bob-data --host 0.0.0.0 --announce https://bob.example.org:8545 --peer https://node.example.org
```

The join takes effect **10 blocks** after the block that holds it. Until then the current validators keep voting, so the set can't be swapped in one block, everyone sees the change coming, and the validators have time to reach the newcomer.

Some rules worth knowing:

- Only a chain whose genesis names validators can add more. On a chain run by a single node, `validator join` says so and stops.
- While a wallet is in the set, or waiting to join, its own `stake` and `unstake` are refused. Its stake stays bonded until it leaves.
- A validator already in the set can announce a new address with `validator join --url` again; that applies at once, since it doesn't change who votes.
- Voting itself needs only the 5 VIBE minimum, so validators that joined before the 100 VIBE rule (or that vote with bootstrap stake from the genesis) keep their vote.

See [VIBE: balance, sending and staking](../wallet/vibe.md) for staking.

## Leaving

```bash
vibehash validator leave --wallet bob.json
```

A leave also takes effect 10 blocks later. The wallet can unstake from the height its leave takes effect, and unstaked coins then wait 20 rounds before they can be spent. A leave sent while a join is still waiting calls the join off.

The last validator with a vote can't leave, since nobody could then propose a block. Bootstrap stake, which a genesis gives a first validator, is burned when that validator leaves.

## Listing

```bash
vibehash validator list --node https://node.example.org
```

This shows each validator's address, stake, share of the votes and URL, then the block being voted on and who is leading it. A validator whose stake has fallen below the minimum is listed with no vote.

## How many validators you need

A block is final once validators holding **more than two thirds** of the stake have signed it. Two such groups always share an honest validator, which is what makes two final blocks at one height impossible. The price is that the chain stops whenever the validators still online hold two thirds or less.

With equal stakes:

| Validators | One offline: what's left | Keeps going? |
|---|---|---|
| 1 | nothing | No |
| 2 | 1 of 2 (50%) | No |
| 3 | 2 of 3 (exactly two thirds, not more) | No |
| 4 | 3 of 4 (75%) | Yes |

So two or three validators are no more fault-tolerant than one: any single one going down halts the chain until it's back. **Four is the smallest set that survives one going down.** In general, surviving `f` offline validators takes at least `3f + 1` with equal stake, and no single validator should hold a third or more. `vibehash genesis check` flags a genesis that falls short ([Genesis and TLS](genesis-and-tls.md)).

A validator that was offline catches up when it's back, fetching the blocks it missed from its peers and checking each one.

## Double-signing

{% hint style="danger" %}
Never run one validator wallet on two machines, or two nodes on one data folder. The two copies would sign different blocks at the same height with the same key, and that is double-signing.
{% endhint %}

A validator that signs two conflicting consensus messages (two proposals, prevotes or precommits for different blocks at the same height and round) leaves signed proof behind. Any validator that sees both turns them into an `evidence` transaction, and anyone may send one. When it lands, the offender:

- loses **all** of its stake and its coins still unbonding;
- leaves the validator set at once and is **banned** for good: it can never join again.

Whoever reported it gets a tenth of the withdrawable part; the rest is burned, bootstrap stake included. Evidence is accepted for 20 blocks after the offence, which is the unbonding period, so a validator can't pull its stake out before it's caught.

A node refuses to start on a data folder another node is already using. Nothing stops the same wallet running on two different machines, so that part is up to you. To move a validator to a new server, stop the old node completely before starting the new one.

For how voting works, see [Consensus and blocks](../concepts/consensus.md). Every flag is in [vibehash validator](../reference/validator.md).
