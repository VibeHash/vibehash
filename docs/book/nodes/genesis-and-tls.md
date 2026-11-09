# Genesis and TLS

Every network starts from a genesis file, and every node of it must be given the same one. This page covers making and checking a genesis with `vibehash genesis`, and serving a node over HTTPS with `vibehash tls`, including how workers trust a self-signed node.

`deploy/setup-node.sh` does both for you on a server ([A public node](public-node.md)). Read on to set a network up by hand.

## Make the wallets

Make one wallet for the node, plus any you want to hand coins or stake to. Each prints its 24 recovery words once:

```bash
vibehash wallet create --out node.json
```

## Write the genesis

```bash
vibehash genesis create --chain-id my-testnet-1 --out genesis.json --bootstrap NODE_ADDR=10 --validator NODE_ADDR=https://node.example.org --alloc REQUESTER_ADDR=500
```

| Flag | What it sets |
|---|---|
| `--chain-id ID` | Required. The network's name, written into every block and every signature |
| `--out FILE` | Required. The file to write |
| `--alloc ADDR=VIBE` | Spendable coins an address starts with (repeatable) |
| `--stake ADDR=VIBE` | Coins an address starts with staked; they can be withdrawn later (repeatable) |
| `--bootstrap ADDR=VIBE` | Stake for a first validator that votes but can never be withdrawn or spent, and is burned if it leaves |
| `--validator ADDR=URL` | A validator and the address other nodes reach it at. Without any, one node runs the chain alone |
| `--economics testnet\|mainnet` | Issuance schedule (default `testnet`); see [Economics](../concepts/economics.md) |
| `--no-earn-in` | Workers must be given stake. By default they start with none and their first 5 VIBE of pay becomes their stake |
| `--es` | Let Evo jobs run. It can't be turned on later |
| `--starter-job SPEC.json` | A job on chain from block one, owned by nobody and paying only new coins, so a network with no starting coins can still mine |

A validator needs at least 5 VIBE of stake in the genesis, from `--stake` or `--bootstrap`. For several validators, give each its own wallet, stake and `--validator ADDR=URL`; they find each other from the addresses in the genesis.

The genesis hash commits to everything in the file, so it identifies the network. Nodes refuse a data folder from a different genesis, and workers stop if their node switches networks. There is no way to change a genesis later.

## Check it

```bash
vibehash genesis check genesis.json
```

This holds the file against what a network meant to carry value needs, and needs no node or wallet, so anyone can run it before agreeing to validate. Each finding is `ok`, `warn` (a choice to make on purpose, such as a premine or Evo off), `fail` or `info`. Among the failures: a test network's chain id, no supply cap, no validators, or fewer than 4 validators or one holding a third or more of the stake. The command exits with status 1 if anything fails, and `--json` prints the findings for scripts.

## Start the node

```bash
vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0
```

## HTTPS

A node can serve HTTPS itself with `--tls-cert FILE --tls-key FILE`. You have three ways to get a certificate:

| Option | How workers trust it |
|---|---|
| A real CA, such as Let's Encrypt (what `setup-node.sh` does) | Nothing to do: `setup --node https://...` works |
| A private CA | `setup --tls-ca FILE` |
| A self-signed certificate from `vibehash tls cert` | `setup --tls-fingerprint FINGERPRINT` |

### A self-signed certificate

Make one for the names and addresses workers will use. `--host` is repeatable; by default it covers localhost and this computer's name and address. It is valid for 365 days unless you give `--days`:

```bash
vibehash tls cert --host node.example.org
```
```bash
vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0 --tls-cert node-cert.pem --tls-key node-key.pem
```

Both print the certificate's SHA-256 fingerprint, and you can print it again at any time:

```bash
vibehash tls fingerprint node-cert.pem
```

Without the `cryptography` package, `tls cert` prints the openssl command to run instead.

### Workers trusting a self-signed node

Give workers the fingerprint. They pin it in setup, and every later command trusts that exact certificate from that node and nothing else:

```bash
vibehash setup --node https://node.example.org --tls-fingerprint FINGERPRINT
```

The choice is saved against that node's address. `$VIBEHASH_TLS_FINGERPRINT` or `$VIBEHASH_TLS_CA` set the trust without setup.

{% hint style="info" %}
A self-signed certificate is its own CA, so the same `node-cert.pem` also works as a `--tls-ca` file.
{% endhint %}

See [vibehash genesis](../reference/genesis.md), [vibehash tls](../reference/tls.md) and [vibehash node](../reference/node.md) for every flag.
