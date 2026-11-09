# Command reference

Every `vibehash` command and flag, written from the command line's own parser by `scripts/cli_docs.py`, so it always matches the release it was built from. `vibehash COMMAND --help` says the same on your computer.

`vibehash` and `python -m vibehash` are the same program. With no arguments at a terminal it opens a menu. `--node` and `--wallet` default to what `setup` saved. Wallet passwords are read from the terminal, or from `$VIBEHASH_PASSWORD` on machines nobody is watching.

## Global flags

- `-v, --verbose`: log every detail (DEBUG), not just what happens (INFO)
- `--version`: show program's version number and exit

## Commands

- [`vibehash setup`](setup.md): get this computer ready to work: node, wallet, GPU support, stake
- [`vibehash update`](update.md): install the latest Vibe Hash release, only if its signature checks out
- [`vibehash install-gpu`](install-gpu.md): find your graphics card and install the PyTorch build for it
- [`vibehash testnet`](testnet.md): run a whole network on this computer: node, workers and a job
- [`vibehash wallet`](wallet.md): wallets: make one, rebuild one from its recovery words, or show one's address
- [`vibehash genesis`](genesis.md): the first block of a new network: who starts with coins, stake and a vote
- [`vibehash node`](node.md): run a node: serve the chain to workers, and validate if its wallet is a validator
- [`vibehash tls`](tls.md): HTTPS certificates for nodes
- [`vibehash worker`](worker.md): contribute compute and earn VIBE
- [`vibehash app`](app.md): mine without the questions: get ready, join, keep the worker running, start at login; `app status`, `pause`, `resume` and `stop` drive it
- [`vibehash avatar`](avatar.md): a small always-on-top pickaxe on the desktop that shows what the worker is doing; click it for the window, right-click for pause, resume and more
- [`vibehash window`](window.md): the app's window: the mining dashboard in a window of its own, beside a running `app` or `worker --watch`
- [`vibehash admin`](admin.md): the operators' window: a globe of where devices connect from, the jobs to confirm, deny, cancel or post, the ledger live, the devices and the nodes
- [`vibehash cache`](cache.md): big model files kept for jobs: where, how big and why; `cache clear` removes them
- [`vibehash devices`](devices.md): list the CPUs and GPUs this computer can train on
- [`vibehash selftest`](selftest.md): check this computer is ready to mine: a real training test, its GPUs, the node, the clock, disk space, and what it can take
- [`vibehash balance`](balance.md): an address's balance, stake, coins unbonding and reputation
- [`vibehash send`](send.md): send VIBE to an address
- [`vibehash stake`](stake.md): lock VIBE as stake: what a worker's checks and a validator's votes are backed by, and what a cheat loses
- [`vibehash unstake`](unstake.md): release staked VIBE; it is spendable after 20 rounds, and can still be slashed until then
- [`vibehash post-job`](post-job.md): post a training job from a job file, paying its budget in (`train` writes the file for you)
- [`vibehash cancel-job`](cancel-job.md): cancel a job you posted; the budget it hasn't spent comes back
- [`vibehash validator`](validator.md): validators vote on blocks and take turns leading rounds
- [`vibehash example`](example.md): write sample data (fruit.csv) to try training with
- [`vibehash train`](train.md): train a model on a CSV file: posts it as a job for the network to train
- [`vibehash models`](models.md): popular open-weight models to fine-tune (coming), with their size, licence and the GPUs a team would need
- [`vibehash predict`](predict.md): use a downloaded model on new rows of a CSV file
- [`vibehash job`](job.md): training jobs: how one is going, follow it live, download its model, its history, or a starter job file
- [`vibehash evo`](evo.md): Evo Hash's models
- [`vibehash chain`](chain.md): audit the chain yourself, without trusting the node
