# vibehash validator

Validators vote on blocks and take turns leading rounds.

- [`vibehash validator join`](#vibehash-validator-join): announce your wallet as a validator (it needs 100 VIBE staked)
- [`vibehash validator leave`](#vibehash-validator-leave): stop validating
- [`vibehash validator list`](#vibehash-validator-list): the validators, their stake and voting share

## vibehash validator join

Announce your wallet as a validator (it needs 100 VIBE staked).

```
vibehash validator join [-h] --url URL [--node NODE] [--wallet WALLET] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--url URL` (required): the address other nodes reach your node at, e.g. http://203.0.113.7:8545.
- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash validator join --url http://203.0.113.7:8545 --wallet bob.json
```

## vibehash validator leave

Stop validating.

```
vibehash validator leave [-h] [--node NODE] [--wallet WALLET] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash validator leave
```

## vibehash validator list

The validators, their stake and voting share.

```
vibehash validator list [-h] [--node NODE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--node NODE`: node address (default: the one `setup` saved).
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash validator list
```
