# Your first run

This page explains what `vibehash setup` asks and does, how to look after your recovery words, how to run with nobody at the keyboard, and where Vibe Hash keeps its files. If you installed the Windows desktop app, it does all of this for you without questions; see [The desktop app](desktop-app.md).

## What setup does

The one-line installers start setup for you. You can also run it any time:

```bash
vibehash setup
```

It goes through five steps. Re-running it later keeps whatever is already done.

1. **Node.** Press Enter to join the public network, `node.vibehash.org`, or type the address of another network's node (such as `192.168.1.20`). You only need another address if someone who runs a different network gave it to you.
2. **Wallet.** Choose a wallet password and type it again. Setup creates your wallet, prints its address (it starts with `vibe1`) and shows its 24 recovery words, once.
3. **Graphics card.** If it finds a card PyTorch can use, it asks before a download of up to 3 GB. Say no, or have no card, and training uses the CPU. You can do this step later with `vibehash install-gpu` (see [GPUs](../mining/gpus.md)).
4. **Space for big models.** Some jobs need big model files, such as DeepSeek's (about 30 GB). They're only downloaded when you take that work. Setup asks where they should go, how much space they may use (40 GB unless you say otherwise), and whether to keep them after a job ends (no, by default).
5. **Stake.** On the public network there's nothing to do: your first 5 VIBE of pay becomes your stake. If you already hold VIBE, it offers to stake it.

Then it asks **Start mining now?** Press Enter and the worker starts; Ctrl+C stops it. To start again later:

```bash
vibehash worker
```

Every command after setup uses the node and wallet it saved, so you rarely need `--node` or `--wallet`. All of setup's flags are on [vibehash setup](../reference/setup.md).

## Your recovery words

{% hint style="danger" %}
Write the 24 words down, in order, and keep them offline. They bring your wallet back if its file or password is lost, and **anyone who sees them can take your coins**. Never type them into a website or share them.
{% endhint %}

To rebuild a wallet from its words:

```bash
vibehash wallet restore
```

It asks for the words (the first four letters of each are enough) and a new password. More on [Wallets and recovery words](../wallet/wallet.md).

## Running with nobody at the keyboard

On a server, a rented GPU or anything started with `nohup`, tmux or Docker, nothing can wait for an answer. Then:

- The wallet password comes from the `VIBEHASH_PASSWORD` environment variable, for setup and for every `worker` start. Without it, setup and the worker stop at once and say so.
- A new wallet's recovery words are **not** printed into the log, where other people might read them. They go to `recovery-words-ADDRESS.txt` in your Vibe Hash folder, readable only by you, and the log says where. Copy them somewhere offline and delete the file.
- `--yes` installs GPU support without asking, and `--deepseek yes` or `--deepseek no` answers the DeepSeek question in advance.

```bash
VIBEHASH_PASSWORD='choose-one' nohup vibehash setup --yes --deepseek no > setup.log 2>&1
```
```bash
VIBEHASH_PASSWORD='choose-one' nohup vibehash worker --plain > worker.log 2>&1 &
```

In PowerShell, set the variable with `$env:VIBEHASH_PASSWORD = "choose-one"` instead.

## Where your files live

Everything is in one folder: `~/.vibehash` in your home folder (`C:\Users\YOUR-NAME\.vibehash` on Windows). On a RunPod pod it's `/workspace/.vibehash`. To put it somewhere else, set `VIBEHASH_HOME` to another folder.

| What | Where in that folder |
|---|---|
| Your wallet, encrypted with your password | `wallet.json` |
| Saved node and settings | `config.json` |
| Each wallet's checkpoints, datasets and receipts | `cache/ADDRESS/` |
| Worker logs and a line per round | `logs/` |
| Recovery words, when they were saved to a file | `recovery-words-ADDRESS.txt` |

Keep a copy of `wallet.json` somewhere safe. Big model files go wherever you chose in step 4; `vibehash cache` shows where.

## The menu

Run `vibehash` on its own, with no command, and it opens a menu of what people do most:

```bash
vibehash
```

Pick with the arrow keys or a number. The top shows your wallet and balance. The sections are **Mine** (start mining, Evo Hash, your balance, your devices, a system check, the DeepSeek job), **Train** (train or fine-tune a model, follow your job, download it) and **Settings** (set up again, your wallet, model files, updating, the guide, and more commands). Each item shows the command it runs, and you're back in the menu when it's done.
