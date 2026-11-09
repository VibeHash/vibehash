# vibehash wallet

Wallets: make one, rebuild one from its recovery words, or show one's address.

- [`vibehash wallet create`](#vibehash-wallet-create): make a wallet and show its 24 recovery words once
- [`vibehash wallet restore`](#vibehash-wallet-restore): rebuild a wallet from its recovery words (read from the terminal or $VIBEHASH_MNEMONIC)
- [`vibehash wallet mfa`](#vibehash-wallet-mfa): ask for an authenticator code before this wallet sends, stakes or unstakes (opt-in)
- [`vibehash wallet show`](#vibehash-wallet-show): a wallet's address, whether the file is encrypted, and whether it has recovery words

## vibehash wallet create

Make a wallet and show its 24 recovery words once.

```
vibehash wallet create [-h] [--out OUT] [--no-password] [--insecure-plaintext] [--passphrase]
```

Flags:

- `--out OUT`: wallet file to write (default ~/.vibehash/wallet.json).
- `--no-password`: no password: the key is kept in this account's credential store (Windows Credential Manager, macOS Keychain, Secret Service) and the file holds only ciphertext.
- `--insecure-plaintext`: write the key into the file in the clear, for tests and throwaway keys only: anyone who can read the file can spend from it.
- `--passphrase`: add a BIP39 passphrase to the words (asked for; optional).

Examples:

```bash
vibehash wallet create
vibehash wallet create --out gpu1.json
```

## vibehash wallet restore

Rebuild a wallet from its recovery words (read from the terminal or $VIBEHASH_MNEMONIC).

```
vibehash wallet restore [-h] [--out OUT] [--no-password] [--insecure-plaintext] [--passphrase] [--path PATH]
```

Flags:

- `--out OUT`: wallet file to write (default ~/.vibehash/wallet.json).
- `--no-password`: no password: the key is kept in this account's credential store (Windows Credential Manager, macOS Keychain, Secret Service) and the file holds only ciphertext.
- `--insecure-plaintext`: write the key into the file in the clear, for tests and throwaway keys only: anyone who can read the file can spend from it.
- `--passphrase`: the words had a BIP39 passphrase (asked for).
- `--path PATH`: derivation path (default m/44'/1'/0'/0'/0').

Examples:

```bash
vibehash wallet restore
```

## vibehash wallet mfa

Ask for an authenticator code before this wallet sends, stakes or unstakes (opt-in).

With `wallet mfa on`, a send, stake or unstake from this wallet first asks for the code an authenticator app shows (--code, else $VIBEHASH_CODE, else asked for). That stops casual use of an unlocked machine and mistaken or scripted sends; it does not stop malware that already reads this user's credential store. The worker never needs a code. The key is kept in this computer's credential store, not in a file.

```
vibehash wallet mfa [-h] [--wallet WALLET] [--code CODE] {on,off,status}
```

Arguments:

- `action`: one of
  - `on`: make a key and show how to add it to an authenticator app
  - `off`: forget it (asks for a current code)
  - `status`: whether it is on

Flags:

- `--wallet WALLET`: wallet file (default: the one `setup` saved).
- `--code CODE`: the authenticator code, when `wallet mfa on` is set (default $VIBEHASH_CODE, else asked for); each code is good once.

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
