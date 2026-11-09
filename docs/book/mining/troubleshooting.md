# Troubleshooting

This page lists the problems people run into most, the messages you're likely to see, and what to do about each. If something isn't here, start with the self test and the logs.

## First, check the computer

```bash
vibehash selftest
```

In a minute or so it checks your software, GPUs, a real training unit, the node, your clock, your wallet and disk space, and tells you what's wrong in plain words and what to run next. In the desktop app it's the **System check** tab, which also has **Copy report for support**: the report leaves out your recovery words and anything like a key or password, and shortens your address.

The worker's logs are in `~/.vibehash/logs` (see [Logs and records](worker.md#logs-and-records)). The desktop app writes what it prints to `app.log` in its own folder (`%LOCALAPPDATA%\VibeHash` on Windows).

## Installing

**`'python' is not recognized` / `command not found`**
Python isn't installed, or isn't on your PATH. On Windows, run the one-line installer again, which offers to install Python, or run the python.org installer and tick "Add python.exe to PATH"; then open a new PowerShell window. On macOS and Linux, type `python3`.

**`No matching distribution found for vibehash`**
Usually your Python is older than 3.10: `python --version` shows which you have. Otherwise update pip with `python -m pip install --upgrade pip`, check you're online, and try again.

**`No module named vibehash`**
Vibe Hash was installed for a different copy of Python. Run the install command again with the same `python` or `python3` you use for everything else. If the installer said Vibe Hash has its own environment, type `vibehash` instead of `python3 -m vibehash`; if it said Python is called `py`, type `py -m vibehash`.

**Windows says the download is from an unknown publisher**
VibeHash-Setup.exe isn't code-signed yet. Choose **More info**, then **Run anyway**.

## Wallet and setup

**`no wallet: run python -m vibehash setup, or add --wallet FILE`**
Run setup once, or add `--wallet` (and `--node`) to the command. To switch networks, run `vibehash setup --node NEW-ADDRESS`.

**`this wallet is encrypted; a password is required`**
Run the command in a normal terminal window so it can ask for your password. To run unattended, set `VIBEHASH_PASSWORD` first.

**I lost my wallet file, or its password**
Rebuild it from your 24 recovery words with `vibehash wallet restore`. Wallets made before version 0.4 have no words, so for those the backed-up file is the only copy. See [Wallets and recovery words](../wallet/wallet.md).

**`another worker ... already mines with wallet ...`**
Two workers with one wallet would undo each other's work. Give a second GPU its own wallet; see [Several GPUs](worker.md#several-gpus).

## Connecting

**`can't reach the node at ...`**
Check you're online and that the address is right. The node may also be offline for a moment; the worker keeps retrying on its own, so you can leave it running. The message says more:

- *doesn't resolve*: check the spelling. On a campus or dorm network that filters names, try the public node or a phone's hotspot.
- *nothing is listening*: the node isn't running there, or uses another port. Check with whoever runs it.
- *certificate didn't check out*: if this Wi-Fi has a sign-in page, sign in first, and check your clock.
- *no answer*: a firewall is in the way. Dorm and campus Wi-Fi often block unusual ports; use the public node or a hotspot.

**`message timestamp too far from node clock`**
Your clock is more than 2 minutes off. Turn on automatic time: on Windows, **Settings > Time & language > Date & time > Sync now**; on Linux, `sudo timedatectl set-ntp true`.

**`node switched networks (genesis changed)`**
The address now runs a different network: it was reset, or points somewhere else. Your wallet is fine. Check the address, then start the worker again.

**This node runs newer rules than this copy of Vibe Hash**
Update with `python -m pip install -U vibehash`, or run the one-line installer again. See [Updating](../getting-started/install.md#updating).

## Mining

**I'm registered but never get any units**
Check three things. Your memory must be at least what the job needs: graphics memory for GPU jobs, system RAM otherwise (add `--memory-mb` if it's detected wrong). New workers get at most 4 units until they've delivered once, so the first round is small. And while you're still earning your stake, you only get units staked workers didn't need, so on a busy network the first rounds can pass with none. That changes once your first 5 VIBE is in.

**`job ... waits: ... it needs 2 staked workers ...`**
Fine-tunes and Evo jobs need at least two staked workers with enough GPU memory, one to train and one to check. If you're still earning your stake, there's nothing to do but keep mining: you earn the rest on other jobs, then take this one too. `vibehash balance` shows the stake filling up.

**`stake at least 500000000 syn to take work`**
This network requires stake instead of letting workers earn it in, and yours is under 5 VIBE. Ask whoever runs it for some VIBE, then stake it with `vibehash stake --amount 5`.

**The GPU ran out of memory**
Close what else uses it (games, browsers, video, another worker on the same card); the next round tries again. Or pick another card with `--device`, or the CPU with `--device cpu` (CSV jobs only).

**The disk is full**
Free some space, or move Vibe Hash to a bigger drive: set `VIBEHASH_HOME` to a folder there, and `vibehash setup --models-dir FOLDER` for big model files. `vibehash cache` shows what model files take up. See [DeepSeek jobs](deepseek.md#vibehash-cache).

**`censored: round N: the node signed a receipt for your commit but sealed the round without it`**
The node accepted your message, promised in a signed receipt to record it, then left it out, which can cost that round's pay and reputation. Your receipts are proof, in `~/.vibehash/cache/YOUR-ADDRESS/receipts.jsonl`. Anyone can check them with `vibehash chain verify --receipts receipts.jsonl`, so share the file with whoever runs the network.

**The dashboard looks garbled**
Add `--plain` for plain log lines. Setting `NO_COLOR` turns colours off.

## GPUs

**`vibehash devices` only lists the CPU**
GPU support isn't installed, or was installed without support for your card (a plain `pip install torch` on Windows does that). Run `vibehash install-gpu`. If it says your driver is too old, update it from your card maker's website and run it again. On Windows, AMD cards mine on the CPU for now; see [GPUs](gpus.md#which-cards-work).

**`GPU training needs PyTorch` / `device ... not found`**
You asked for a GPU with `--device`, but PyTorch can't see one. Run `vibehash devices` to see what's there, or leave out `--device` to fall back automatically.

**A Mac mines on the CPU**
The newest PyTorch needs macOS 15 to use the GPU. Run `vibehash install-gpu`, which installs a PyTorch that works on macOS 14, or update macOS.

## Desktop app

**The app won't change settings in my browser**
The browser page at `http://127.0.0.1:7545/dashboard` is for looking only. Settings change in the app's own window.

**I closed the window and it's still mining**
That's on purpose. Use **Stop** in the window, right-click the pickaxe, or `vibehash app stop`. See [The desktop app](../getting-started/desktop-app.md).

**Mining keeps pausing by itself**
Check **Settings > When to mine**: pausing while gaming, on battery, or only when you're away. Mining carries on about a minute after the reason has gone.
