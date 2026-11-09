# vibehash admin

The operators' window: a globe of where devices connect from, the jobs to confirm, deny, cancel or post, the ledger live, the devices and the nodes.

The admin app (admin.py): a small server on this computer only, with the nodes behind it, and its page in a window of its own (pywebview, the `app` extra) or the browser. It asks each node's /admin/overview with the token that node was started with (`node --admin-token`; a node without one shows only what its public routes say). Confirm and deny are holds on every node it knows; cancel and post are transactions signed with the admin wallet, unlocked in the page. The nodes, their tokens, the wallet and whether to look up places are kept in config.json under "admin", from the page's Settings or these flags.

```
vibehash admin [-h] [--node URL] [--token TOKEN] [--wallet WALLET] [--port PORT] [--browser] [--install-shortcut] [--remove-shortcut] [--new-token] [--new-mfa]
```

Flags:

- `--node URL`: a node to watch (repeat for more; the first gives the jobs and the ledger); default: the ones saved, else the node setup chose.
- `--token TOKEN`: the admin token for the --node(s) given (default: the one saved for that node, else $VIBEHASH_ADMIN_TOKEN; a saved node without a token of its own gets none).
- `--wallet WALLET`: the admin wallet, for cancelling and posting jobs (default: as other commands).
- `--port PORT`: the local server's port (default 7546).
- `--browser`: open the page in the browser instead of a window.
- `--install-shortcut`: put "Vibe Hash Admin" in the Start menu (the apps menu on Linux, ~/Applications on a Mac), running this Python; --node, --token and --wallet given with it are saved for it to open with.
- `--remove-shortcut`: take that shortcut away again.
- `--new-token`: print a new random admin token for a node's --admin-token.
- `--new-mfa`: print a new authenticator key for a node's --admin-mfa, with how to add it to an authenticator app.

Examples:

```bash
vibehash admin
vibehash admin --node https://node.vibehash.org --token SECRET --node http://10.0.0.5:8545
vibehash admin --browser
vibehash admin --install-shortcut --node https://node.vibehash.org --token SECRET
vibehash admin --new-token
vibehash admin --new-mfa
```
