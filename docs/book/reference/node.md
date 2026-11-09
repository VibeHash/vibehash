# vibehash node

Run a node: serve the chain to workers, and validate if its wallet is a validator.

```
vibehash node [-h] --genesis GENESIS --wallet WALLET --data-dir DATA_DIR [--host HOST] [--port PORT] [--peer URL] [--announce URL] [--onion ADDR.onion] [--model-access {public,owner}] [--dataset DATASET] [--device DEVICE] [--keep-rounds KEEP_ROUNDS] [--max-job-mb MB] [--register-s SECONDS] [--compute-s SECONDS] [--reveal-s SECONDS] [--verify-s SECONDS] [--min-fee MIN_FEE] [--rate-limit SPEC] [--rate-limit-local] [--max-connections MAX_CONNECTIONS] [--min-client VERSION] [--admin-token TOKEN] [--admin-mfa KEY] [--admin-allow-no-mfa] [--tls-cert FILE] [--tls-key FILE] [--tor] [--proxy URL] [--tor-mode {control-only,strict}]
```

Flags:

- `--genesis GENESIS` (required): the network's genesis file (see `genesis create`).
- `--wallet WALLET` (required): this node's wallet; a validator's if it is to vote and lead rounds.
- `--data-dir DATA_DIR` (required): folder for the chain, its records and uploaded checkpoints.
- `--host HOST`: address to listen on (default 127.0.0.1; 0.0.0.0 for other computers).
- `--port PORT`: port to listen on (default 8545).
- `--peer URL`: another node to sync from and gossip to (validators find each other from the chain anyway).
- `--announce URL`: the address other nodes reach this one at, if not --host:--port.
- `--onion ADDR.onion`: this node's Tor hidden-service address, shown in /status.
- `--model-access {public,owner}`: who may download a job's weights from this node: public (anyone; the default), or owner (the job's owner and the validators, by a signed request; starter jobs and jobs whose spec says "public": true stay open). While public, the log says which requests owner would refuse.
- `--dataset DATASET`: add an .npz dataset to this node's store.
- `--device DEVICE`: where the node checks work its checkers didn't and scores each round: cpu (default), auto, cuda, xpu or mps. Fine-tunes want a GPU here (and PyTorch).
- `--keep-rounds KEEP_ROUNDS`: rounds whose opened checkpoints stay downloadable, so anyone can re-run their checks (default 10).
- `--max-job-mb MB`: relay only jobs whose model, one checkpoint at the job's precision, is at most this big (default 64, which a 2 GB server can run; jobs may be up to 1024).
- `--register-s SECONDS`: how long a round's register phase lasts (default 3).
- `--compute-s SECONDS`: how long a round's compute phase lasts (default 20).
- `--reveal-s SECONDS`: how long a round's reveal phase lasts (default 5).
- `--verify-s SECONDS`: how long a round's verify phase lasts (default 10).
- `--min-fee MIN_FEE`: smallest fee in VIBE this node relays a transaction for (default 0.0001 VIBE; 0 relays free ones).
- `--rate-limit SPEC`: per-client request limits: default, off, or overrides like tx=1/10,read=100 (requests a second/burst; buckets: read, worker, tx, upload, other). Clients on this computer are never limited.
- `--rate-limit-local`: limit clients on this computer (127.0.0.1) too, e.g. behind a reverse proxy, where everyone looks local.
- `--max-connections MAX_CONNECTIONS`: connections served at once (default 256).
- `--min-client VERSION`: the oldest Vibe Hash release whose workers may register (default 0.5.17); older ones are told to upgrade, as are releases before 0.5.17, which name no version and can't follow version 4 blocks.
- `--admin-token TOKEN`: serve the operators' routes (/admin/*: hold and release jobs, the devices' addresses) to the admin app (`vibehash admin`) that sends this token, at least 24 characters (`vibehash admin --new-token` makes one); default $VIBEHASH_ADMIN_TOKEN; without one those routes don't exist.
- `--admin-mfa KEY`: an authenticator app's key (`vibehash admin --new-mfa` makes one and says how to add it): the token then also needs the app's 6-digit code, for a sign-in that lasts 12 hours; default $VIBEHASH_ADMIN_MFA.
- `--admin-allow-no-mfa`: UNSAFE: serve the admin routes on a --host other than this computer's with the token alone. Without --admin-mfa such a node refuses to start; this is for test rigs only.
- `--tls-cert FILE`: serve HTTPS with this PEM certificate (see `tls cert`).
- `--tls-key FILE`: the certificate's private key.
- `--tor`: send this command's node traffic through Tor (a local Tor on 9050, or Tor Browser's on 9150), resolving the node's name inside Tor.
- `--proxy URL`: send it through this SOCKS proxy instead: socks5h://HOST:PORT (socks5:// would resolve the node's name on this computer, so it is refused).
- `--tor-mode {control-only,strict}`: control-only: JSON through the proxy, transfers of over 50 MB (checkpoints) direct; strict: everything through it, big transfers refused (default: strict for wallet, send, stake, unstake and job posting, else control-only).

Examples:

```bash
vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0
vibehash node --genesis genesis.json --wallet bob.json --data-dir bob-data --host 0.0.0.0 --announce http://203.0.113.7:8545 --peer http://192.168.1.20:8545
VIBEHASH_ADMIN_TOKEN=SECRET vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0
```
