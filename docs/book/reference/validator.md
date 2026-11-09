# vibehash validator

Validators vote on blocks and take turns leading rounds.

- [`vibehash validator join`](#vibehash-validator-join): announce your wallet as a validator (it needs 100 VIBE staked)
- [`vibehash validator leave`](#vibehash-validator-leave): stop validating
- [`vibehash validator list`](#vibehash-validator-list): the validators, their stake and voting share

## vibehash validator join

Announce your wallet as a validator (it needs 100 VIBE staked).

```
vibehash validator join [-h] --url URL [--node NODE] [--wallet WALLET]
```

Flags:

- `--url URL` (required): the address other nodes reach your node at, e.g. http://203.0.113.7:8545.
- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).

Examples:

```bash
vibehash validator join --url http://203.0.113.7:8545 --wallet bob.json
```

## vibehash validator leave

Stop validating.

```
vibehash validator leave [-h] [--node NODE] [--wallet WALLET]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).

Examples:

```bash
vibehash validator leave
```

## vibehash validator list

The validators, their stake and voting share.

```
vibehash validator list [-h] [--node NODE]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).

Examples:

```bash
vibehash validator list
```
