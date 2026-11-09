# Running a node

This section is for people who run the network itself rather than mine on it: what a node does, the difference between a validator and a follower, and when it is worth running one.

## What a node does

A node is the program `vibehash node`. It keeps a copy of the chain on disk and serves it to anyone as JSON over HTTP or HTTPS. Workers talk to a node to register for rounds, fetch their work, and send back commitments, reveals and checks. Job posters send their jobs and datasets to it, and anyone can read blocks, accounts, jobs and the supply from it.

When a node starts, it re-checks every block it has stored from the genesis onwards, so an edited or corrupted database is refused rather than trusted.

## Validators and followers

A network is run by **validators**: nodes whose wallet has stake and is listed as a validator on chain.

- Validators take turns leading rounds, in a stake-weighted order fixed by the block height.
- The leader runs the round with the workers and proposes a block, together with a record of the round's signed messages.
- Every validator replays the round from that record and votes. A block is final once validators holding more than two thirds of the stake have signed it.

A **follower** is a node whose wallet is not a validator, started with `--peer` pointing at a node of the network. It has no vote. It syncs the chain, checks every block it receives (each must apply, replay from its record and carry a valid commit certificate), serves the chain, passes transactions on, and points workers at whichever validator is leading.

A network whose genesis names no validators runs the old way: one node seals every block by itself.

{% hint style="info" %}
The role a node plays comes from its wallet. Start it with a validator's wallet and it votes and leads; start it with any other wallet and it follows.
{% endhint %}

## When you'd run one

| You want to... | Run |
|---|---|
| Try the whole system on one computer | [A test network on your computer](local-testnet.md) |
| Start a network on a server, for other people to mine on | [A public node](public-node.md) |
| Check the chain yourself, or serve it closer to your own workers | A follower: `vibehash node ... --peer URL` |
| Help run an existing network and earn the proposer's share | A validator: [Validators](validators.md) |

You don't need a node to mine. Workers connect to an existing network's node; see [How mining works](../mining/how-mining-works.md).

## A minimal node

Every node needs the network's genesis file, a wallet and a folder for its data:

```bash
vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0
```

A follower of an existing network adds the address of one of its nodes:

```bash
vibehash node --genesis genesis.json --wallet me.json --data-dir me-data --peer https://node.example.org
```

By default a node listens on 127.0.0.1, port 8545. `--host 0.0.0.0` lets other computers reach it. Every flag is in [vibehash node](../reference/node.md).

## In this section

- [A test network on your computer](local-testnet.md): `vibehash testnet`, with validators, workers and Evo jobs, all on one machine.
- [A public node](public-node.md): `deploy/setup-node.sh` on a fresh Ubuntu server, locking the server down, and upgrading.
- [Genesis and TLS](genesis-and-tls.md): making the file a network starts from, and serving HTTPS.
- [Validators](validators.md): joining, leaving, the stake it takes, and why four validators survive an outage when two or three don't.
- [The admin app](admin-app.md): `vibehash admin`, the operators' window.
- [Operating a node](operations.md): disk, fees, rate limits, logs, audits, upgrades and backups.

For the ideas underneath, see [Consensus and blocks](../concepts/consensus.md) and [How the network checks work](../concepts/verification.md).
