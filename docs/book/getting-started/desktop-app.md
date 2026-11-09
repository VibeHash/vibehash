# The desktop app

This page covers `vibehash app`: mining you start once and then forget, with a window that shows what your computer is doing, a small pickaxe on the desktop, and automatic pauses for when you're gaming or away. The Windows download runs it for you.

## Starting it

If you installed with VibeHash-Setup.exe, just open **Vibe Hash** from the Start menu or the desktop. From a terminal:

```bash
vibehash app
```

It gets your computer ready without asking questions: it saves the node (the public network unless you chose another), makes a wallet if there isn't one, installs GPU support, then starts the worker and keeps it running, starting it again if it ever stops. The first time, the window walks you through this with a progress bar, shows your new wallet's address and recovery words once, and gives a short tour.

A wallet made by `vibehash app` asks for a password once at a terminal, or reads it from `VIBEHASH_PASSWORD`. The Windows download makes its first wallet **without** a password, so nothing is asked; the file is then as safe as your Windows account. Either way, the recovery words go to a file only you can read.

{% hint style="danger" %}
Write the recovery words down and keep them offline. Anyone who sees them can take your coins. The Wallet tab lets you check your copy and then delete the words file.
{% endhint %}

## The window

The window is the system's own web view (Edge's on Windows, Safari's on a Mac), through the small `pywebview` package. The download has it; with pip, add it with `pip install "vibehash[app]"`. Without it, the same page opens in your browser at `http://127.0.0.1:7545/dashboard`, where you can look but not change settings.

At the top it says what the worker is doing, with **Pause mining** and **Stop** on the right. The tabs:

| Tab | What it shows |
|---|---|
| **Mining** | The round, your speed and device, what you've earned this session and your reputation. The round step by step: Registered, Training, Committed, Checking others, Paid. Below, a feed of what's happening, which you can filter to Pay or Problems, or search. |
| **Wallet** | Your spendable balance, stake, coins unbonding, and earnings this session. Recent rounds (units, checks, pay), your address with a copy button, and your recovery words: show and confirm them, then delete the words file. |
| **Network** | Miners online and today, validators, block time, what kinds of machines are online, who did the most work today (shortened addresses only), and the jobs with their budget left. |
| **System check** | The same checks as `vibehash selftest`, including one real training unit, with a button to copy a report for support. It's only in the app's own window. |
| **Settings** | When to mine, start at login, the floating pickaxe, reduce motion, model files (how much space they use, a cap, and a Clear button), the node you're connected to, and your Vibe Hash folder. |

## Pause, resume and stop

- **Pause mining** stops mining for now; the worker leaves its round. The button then reads **Resume mining**.
- **Stop** asks first, then the worker leaves its round and the app closes. Start Vibe Hash again to carry on.

From another terminal, the same orders:

```bash
vibehash app pause
vibehash app resume
vibehash app stop
vibehash app status
```

**Closing the window doesn't stop mining.** The app keeps going in the background. Open the window again from the pickaxe, the Start menu, or `vibehash window`.

## The floating pickaxe

The download also puts a small pickaxe on your desktop, always on top. It rests, swings while your computer trains, glows while it checks other miners' work, and throws sparks when a round pays. Click it to open the window; right-click it for pause, resume and more; drag it anywhere. Turn it off and on under **Settings > The floating pickaxe**. From a terminal, `vibehash app --avatar` starts it with the app.

## Start at login

Turn on **Settings > Start when I log in**, or tick it in the setup wizard, or run:

```bash
vibehash app startup on
```

At login, mining starts in the background without the window. Open it from the pickaxe or the Start menu. `vibehash app startup off` turns it off.

## Automatic pauses

Under **Settings > When to mine**:

- **Pause while I'm gaming:** holds mining off while a game or any app is full screen.
- **Pause on battery:** mines only while the laptop is plugged in.
- **Only when I'm away:** mines after a number of minutes with no keyboard or mouse (5 unless you pick another), and stops when you're back.

Once the reason has gone, mining carries on about a minute later, so a game flicking out of full screen doesn't restart work each time. Windows supports all three. On Linux only the battery setting works, and elsewhere the window greys out what your computer can't tell and says why.

Every flag is on [vibehash app](../reference/app.md), [vibehash avatar](../reference/avatar.md) and [vibehash window](../reference/window.md).
