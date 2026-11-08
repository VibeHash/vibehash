# Wallets and recovery words

This page covers your Vibe Hash wallet: how to make one, how to bring it back from its 24 recovery words, where the file lives, and how to keep it safe.

A wallet is a small file holding the key that controls your VIBE. Your address (it starts with `vibe1`) is public; the key is not.

## Making a wallet

Most people never run a wallet command: `vibehash setup` makes a wallet for you, and so does the desktop app. To make one by hand:

```bash
vibehash wallet create
```

1. Choose a password. It encrypts the file, and you type it twice.
2. Your new address is printed.
3. Your **24 recovery words** are shown, numbered. Write them down in order.

The words are shown once. They are never saved inside the wallet file and won't be shown again.

{% hint style="danger" %}
**Your recovery words are your coins.** Anyone who sees them can take everything in the wallet, from any computer, without your password. Write them on paper (or keep them in a password manager), keep them offline, and never paste them into a website, chat or email. Without them, a lost wallet file or a forgotten password can't be undone.
{% endhint %}

If there is no screen to show the words on (for example the app setting itself up, or a command run in the background with its output going to a log), they are not printed. They go to a file only your user account can read, and a loud note says where. Copy the words somewhere offline, then delete that file.

`--passphrase` adds an optional extra secret on top of the words. A different passphrase gives a different wallet, so if you use one, you need both to restore it.

## Restoring from your words

If you lose the wallet file or its password, rebuild it from the words:

```bash
vibehash wallet restore
```

Type the words separated by spaces (they aren't shown as you type). The first four letters of each word are enough. Then choose a new password. Add `--passphrase` if you gave the words one.

`wallet restore` refuses to overwrite a wallet file that already exists. To restore next to an existing one, give it a new file with `--out`, such as `vibehash wallet restore --out restored.json`.

## Checking a wallet

```bash
vibehash wallet show
```

This prints the wallet's address, whether the file is encrypted, and whether it was made from recovery words. A wallet made before recovery words existed says so: back up that file itself.

## Where the file lives

By default the wallet is `wallet.json` in the `.vibehash` folder in your home folder (`~/.vibehash/wallet.json`). If you set `VIBEHASH_HOME`, it lives in that folder instead. Every command uses the wallet `setup` saved unless you pass `--wallet FILE`.

## Passwords and unattended machines

Commands that spend or sign ask for the wallet password at the terminal. On a machine nobody is watching (a mining rig, a server), set it in the environment instead:

```bash
export VIBEHASH_PASSWORD=your-password
```

In PowerShell: `$env:VIBEHASH_PASSWORD = "your-password"`.

To restore without typing, `VIBEHASH_MNEMONIC` can hold the recovery words (and `VIBEHASH_MNEMONIC_PASSPHRASE` the passphrase, if any). Only do this on a machine you trust, and clear it afterwards.

`--no-password` stores the key unencrypted. Anyone who can read the file can then spend from it, so keep that for throwaway test wallets or machines where your user account is the only protection you need.

## Several wallets

You can keep as many wallets as you like. Name a file when you create one, and point commands at it with `--wallet`:

```bash
vibehash wallet create --out gpu1.json
vibehash balance --wallet gpu1.json
```

A common reason is mining with several graphics cards: each worker runs with its own wallet. Each wallet has its own recovery words, so write each set down and label it.

## Next

- [VIBE: balance, sending and staking](vibe.md)
- [vibehash wallet](../reference/wallet.md), every flag
