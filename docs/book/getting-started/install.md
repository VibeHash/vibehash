# Install Vibe Hash

This page covers every way to get Vibe Hash onto your computer: the Windows setup wizard, the one-line installers, pip, and rented GPUs. It ends with how to update and uninstall.

{% hint style="info" %}
Vibe Hash is a test network. VIBE has no cash value, and the software hasn't been audited.
{% endhint %}

## Windows: the setup wizard

The easiest way on Windows is **VibeHash-Setup.exe**, from [vibehash.org](https://vibehash.org) or the [latest release](https://github.com/VibeHash/vibehash/releases/latest/download/VibeHash-Setup.exe). It works like any other app's installer.

1. Run VibeHash-Setup.exe. Windows warns that it's from an unknown publisher, because it isn't code-signed yet. Choose **More info**, then **Run anyway**.
2. Pick your language, whether to install for everyone or just you, and where.
3. Choose whether you want a desktop icon and whether Vibe Hash should start when you log in (off unless you tick it).
4. Finish, and let it start.

The first run then installs everything else into `%LOCALAPPDATA%\VibeHash`: a Python of its own (your computer's Python, if you have one, is never touched), Vibe Hash itself, and the GPU build of PyTorch for your graphics card, a one-time download of up to 3 GB. It makes a wallet without a password, shows its recovery words once in the window, keeps them in a file only you can read, and starts mining. Every run after that just mines, and checks for a newer Vibe Hash once a day.

What happens next is on [The desktop app](desktop-app.md).

## The one-line installers

If you'd rather use a terminal, one line installs Vibe Hash and starts the guided setup, which asks you the rest (network, wallet password, GPU support).

On Windows, in PowerShell. It finds Python 3.10 or newer, and offers to install it for you with winget if you don't have it:

```powershell
irm https://vibehash.org/install.ps1 | iex
```

On macOS or Linux, in Terminal:

```bash
curl -fsSL https://vibehash.org/install.sh | sh
```

On macOS and Linux you need Python 3.10 or newer first: from python.org or `brew install python` on a Mac, or your package manager on Linux (on Ubuntu or Debian, `sudo apt install python3 python3-venv`). If your Python doesn't take packages directly, as Homebrew's, Debian's and Ubuntu's don't, the script gives Vibe Hash its own environment in `~/.vibehash/venv` and tells you to type `vibehash` instead of `python3 -m vibehash`.

Setup is explained step by step on [Your first run](first-run.md).

## With pip

You can also install it yourself. Vibe Hash is on PyPI and needs Python 3.10 or newer:

```bash
python -m pip install --upgrade vibehash
```

That adds a `vibehash` command (also available as `python -m vibehash`). Then run `vibehash setup`. For the app window, install the extra that brings `pywebview`:

```bash
python -m pip install "vibehash[app]"
```

## On a rented GPU (RunPod)

1. Start a pod from a template that has PyTorch.
2. Open its terminal (SSH or Jupyter).
3. Run the macOS/Linux line above.

The PyTorch that's already on the pod is kept exactly as it is: the script never reinstalls or downgrades it, and setup sees it works and installs nothing.

A pod keeps only its `/workspace` volume when it restarts, so on a pod Vibe Hash keeps its settings and wallet in `/workspace/.vibehash` by itself. After a restart, run the installer again: Vibe Hash itself was wiped, but your wallet is still there. To run a worker with nobody watching, see [unattended runs](first-run.md#running-with-nobody-at-the-keyboard).

## Updating

- **Desktop app:** it checks once a day and updates itself. When a new version is out, the window says so, and it installs the next time the app starts.
- **One-line installer:** run the same line again any time. Setup keeps the wallet and settings it already made.
- **pip:** `python -m pip install --upgrade vibehash`.
- **The menu:** run `vibehash` and choose **Settings > Update Vibe Hash**.

## Uninstalling

- **Windows setup wizard:** open **Settings > Apps**, find Vibe Hash and choose **Uninstall**. It stops Vibe Hash, removes everything it put on the machine, and then asks whether to delete your wallet, recovery words and settings too. They're kept unless you say yes.
- **pip installs:** turn off start at login first if you turned it on (`vibehash app startup off`), then `python -m pip uninstall vibehash`. Your wallet and settings stay in `~/.vibehash`.

{% hint style="danger" %}
Before you delete a wallet, make sure you have its 24 recovery words written down. Without them, the coins in that wallet are gone for good.
{% endhint %}
