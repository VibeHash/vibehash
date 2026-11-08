# vibehash wallet

Wallets: make one, rebuild one from its recovery words, or show one's address.

- [`vibehash wallet create`](#vibehash-wallet-create): make a wallet and show its 24 recovery words once
- [`vibehash wallet restore`](#vibehash-wallet-restore): rebuild a wallet from its recovery words (read from the terminal or $VIBEHASH_MNEMONIC)
- [`vibehash wallet show`](#vibehash-wallet-show): a wallet's address, whether the file is encrypted, and whether it has recovery words

## vibehash wallet create

Make a wallet and show its 24 recovery words once.

```
vibehash wallet create [-h] [--out OUT] [--no-password] [--passphrase]
```

Flags:

- `--out OUT`: wallet file to write (default ~/.vibehash/wallet.json).
- `--no-password`: don't encrypt the file (for unattended machines; anyone who can read it can spend from it).
- `--passphrase`: add a BIP39 passphrase to the words (asked for; optional).

Examples:

```bash
vibehash wallet create
vibehash wallet create --out gpu1.json
```

## vibehash wallet restore

Rebuild a wallet from its recovery words (read from the terminal or $VIBEHASH_MNEMONIC).

```
vibehash wallet restore [-h] [--out OUT] [--no-password] [--passphrase] [--path PATH]
```

Flags:

- `--out OUT`: wallet file to write (default ~/.vibehash/wallet.json).
- `--no-password`: don't encrypt the file (for unattended machines; anyone who can read it can spend from it).
- `--passphrase`: the words had a BIP39 passphrase (asked for).
- `--path PATH`: derivation path (default m/44'/1'/0'/0'/0').

Examples:

```bash
vibehash wallet restore
```

## vibehash wallet show

A wallet's address, whether the file is encrypted, and whether it has recovery words.

```
vibehash wallet show [-h] [--wallet WALLET]
```

Flags:

- `--wallet WALLET`: wallet file (default: the one `setup` saved).

Examples:

```bash
vibehash wallet show
vibehash wallet show --wallet gpu1.json
```
