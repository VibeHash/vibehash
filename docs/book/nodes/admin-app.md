# The admin app

`vibehash admin` is a window for the people who run a network's nodes: a globe of where devices connect from, the jobs, the ledger, the devices and the nodes. This page covers setting a node up for it, signing in, and each tab.

## How it fits together

The app is a small server on your own computer (port 7546 by default, `--port`) with its page in a window of its own (pywebview, from the `app` extra) or the browser (`--browser`). It asks each node's admin routes with that node's token, so no token ever reaches the page. A node serves admin routes only if started with a token; one without still shows what its public routes say.

## Give a node a token

Make a token (at least 24 characters; this prints a good one):

```bash
vibehash admin --new-token
```

Start the node with it, using `--admin-token TOKEN` or the environment variable:

```bash
VIBEHASH_ADMIN_TOKEN=SECRET vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0
```

On a server made with `setup-node.sh`, put `VIBEHASH_ADMIN_TOKEN=...` in `/etc/vibehash/env` and run `systemctl restart vibehash-node` ([A public node](public-node.md)).

## Authenticator codes

For a second factor, give the node an authenticator key as well:

```bash
vibehash admin --new-mfa
```

This prints a new key and how to add it to an authenticator app (Google or Microsoft Authenticator, 1Password, Authy and others) as a time-based setup key. Everyone who runs the node adds the same key to their own app. Start the node with `--admin-mfa KEY` or `VIBEHASH_ADMIN_MFA=KEY`.

The app then asks for the 6-digit code when a node wants one, and tries that one code on every node that asked. A sign-in lasts **12 hours** and is kept in memory only, so restarting the node or the app signs you out.

{% hint style="warning" %}
Wrong guesses are limited. An address that sends 10 wrong tokens or codes within 15 minutes is turned away for the rest of that window. Behind a reverse proxy, that only works per client if the proxy sets `X-Forwarded-For`; otherwise everyone looks like the proxy, and anyone's wrong guesses lock everyone out.
{% endhint %}

The app only sends a token over HTTPS, or to this computer or a private network's address.

## Opening it

```bash
vibehash admin --node https://node.example.org --token SECRET --node http://10.0.0.5:8545
```

Repeat `--node` for each node. The first is the one the jobs and ledger come from. With no flags it opens the nodes saved in `config.json` (under `"admin"`), or the node setup chose.

## The tabs

**Globe.** Devices, nodes and validators, with arcs from each device to its node, beside counts of devices online (registered in the last 10 minutes), jobs, height, the round, validators, the mempool and VIBE in circulation.

**Jobs.** Each job, with its actions:

- **Deny** puts a hold on the job on every node the app knows, with an optional reason only admins see. A held job gets no rounds from that node, and its budget stays in escrow untouched.
- **Confirm** releases a held job, so it takes its turns again.
- **Review new jobs** is a switch: while it's on, every job posted from then on starts held until an admin confirms it.
- **Cancel** and **Post a job** are transactions signed with the admin wallet, unlocked in the page with its password and kept only in the app's memory. Cancel returns a job's unspent budget; Post takes a spec as `post-job --spec` does, and a budget.

{% hint style="info" %}
A hold is a node's own policy, not a rule of the chain. A validator that doesn't hold the job still leads rounds for it, which is why the app sends holds to every node with a token. Holds and the review setting survive a node restart.
{% endhint %}

**Ledger.** Blocks as they land (optionally only rounds that trained), the mempool, block time, miners and units today, and the subsidy.

**Devices.** Each device that registered lately: hardware, speed, memory, last seen, where it connected from and its node. Nodes give connection addresses only to admins, never publicly.

**Nodes.** Each node: up, unreachable or halted; admin, code needed or public only; its version, height, round, peers and review setting. Below, the validators on chain with their stake and votes.

**Settings.** The nodes and tokens (an empty token keeps the saved one), the admin wallet file, and place lookups.

## Place lookups

Places come from devices' connection addresses, looked up at ip-api.com's free service (or `$VIBEHASH_GEO_URL`). Only the addresses are sent, never private or loopback ones, and answers are cached for a month in `~/.vibehash/admin/geo.json`. With lookups off in Settings nothing is sent, and the globe shows only the nodes.

## A shortcut

```bash
vibehash admin --install-shortcut --node https://node.example.org --token SECRET
```

This puts "Vibe Hash Admin" in the Start menu on Windows, the applications menu on Linux, or `~/Applications` on a Mac, running the Python you ran it from, with any `--node`, `--token` and `--wallet` saved for it. `--remove-shortcut` takes it away.

Every flag is in [vibehash admin](../reference/admin.md).
