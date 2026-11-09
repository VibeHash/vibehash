# vibehash genesis

The first block of a new network: who starts with coins, stake and a vote.

- [`vibehash genesis create`](#vibehash-genesis-create): write a genesis file: chain id, starting balances, stakes, validators and the issuance schedule
- [`vibehash genesis check`](#vibehash-genesis-check): hold a genesis file against what a network meant to carry value needs

## vibehash genesis create

Write a genesis file: chain id, starting balances, stakes, validators and the issuance schedule.

Write the genesis file every node of a new network starts from. Addresses come from `wallet create`. A validator named here needs stake of at least the minimum, from --stake or --bootstrap; a worker needs none when --no-earn-in isn't given, since its first pay becomes its stake. The file's hash identifies the network, so give every node the same file.

```
vibehash genesis create [-h] --chain-id CHAIN_ID [--alloc ADDR=VIBE] [--stake ADDR=VIBE] [--validator ADDR=URL] --out OUT [--economics {mainnet,testnet}] [--bootstrap ADDR=VIBE] [--no-earn-in] [--es] [--starter-job SPEC.json]
```

Flags:

- `--chain-id CHAIN_ID` (required): the network's name, written into every block.
- `--alloc ADDR=VIBE`: an address and the VIBE it starts with (repeatable).
- `--stake ADDR=VIBE`: an address and the VIBE it starts with staked (repeatable); staked coins can be withdrawn later.
- `--validator ADDR=URL`: a validator and the address other nodes reach it at (it needs a --stake too). Without any, one node runs the chain alone.
- `--out OUT` (required): genesis file to write.
- `--economics {mainnet,testnet}`: issuance schedule: testnet (fast halvings, for trying things) or mainnet (Bitcoin-like, ~4-year halvings on a busy network); either way only rounds that pay for work mint, and halvings count them. Default: `testnet`.
- `--bootstrap ADDR=VIBE`: stake for a first validator that votes but can never be withdrawn or spent; burned if it leaves.
- `--no-earn-in`: require workers to be given stake; by default they can start with none and earn it.
- `--es`: let Evo jobs run on this network ("es": true); it can't be turned on later.
- `--starter-job SPEC.json`: a job on chain from block one, owned by nobody, paying only new coins (fee 0), so the network can mine with no starting coins; a spec as `vibehash job template` writes.

Examples:

```bash
vibehash genesis create --chain-id my-testnet-1 --out genesis.json --bootstrap NODE_ADDR=10 --validator NODE_ADDR=http://192.168.1.20:8545 --alloc REQUESTER_ADDR=500
```

## vibehash genesis check

Hold a genesis file against what a network meant to carry value needs.

```
vibehash genesis check [-h] [--json] file
```

Arguments:

- `file`: the genesis file to check.

Flags:

- `--json`: print the findings as JSON, for scripts.
