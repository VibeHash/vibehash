# A public node

`deploy/setup-node.sh` turns a fresh Ubuntu 24.04 server into a network's public node, serving HTTPS on its own domain. This page covers running it, a second network on the same server, locking the server down, upgrading and Cloudflare.

## What the script does

In one run, as root, it:

1. Checks the domain points at this server (or at Cloudflare's proxy).
2. Installs `python3-venv`, `certbot` and `ufw`, then Vibe Hash from PyPI into `/opt/vibehash`, run by a system user `vibehash`.
3. Writes a random wallet password to `/etc/vibehash/env` and creates the node's wallet, `/var/lib/vibehash/node-wallet.json`, showing its recovery words **once**.
4. Writes the genesis, `/var/lib/vibehash/genesis.json`: this node as the first validator with 10 VIBE of bootstrap stake, the mainnet issuance schedule, and workers earning their stake in.
5. Opens the firewall for SSH (first run only), HTTP (for certificate renewals) and HTTPS.
6. Gets a Let's Encrypt certificate, with a renewal hook that copies it for the node and restarts it.
7. Runs the node as the systemd service `vibehash-node`, which restarts after a crash or reboot, then checks it answers on `/status`.

## Run it

First point the domain's **A** record at the server's IPv4 address. The node listens on IPv4 only, so give it no AAAA record. Then copy the script over and run it:

```bash
scp deploy/setup-node.sh root@SERVER-IP:
```
```bash
ssh root@SERVER-IP bash setup-node.sh --domain node.example.org --fund YOUR-WALLET-ADDRESS=10000
```

| Flag | Meaning |
|---|---|
| `--domain NAME` | Required. The node's domain |
| `--fund ADDR=VIBE` | Give one wallet VIBE to pay for the first training jobs, since with earn-in nobody else starts with any |
| `--email ADDR` | Optional. Let's Encrypt uses it only for account notices |
| `--chain-id ID` | The network's name (default `vibehash-public-1`) |
| `--economics mainnet\|testnet` | Issuance schedule (default `mainnet`) |
| `--name`, `--port`, `--es`, `--stake` | A second network on the same server (below) |

`--fund`, `--es` and `--stake` count only the first time, when the genesis is made. Getting the certificate accepts Let's Encrypt's terms.

{% hint style="danger" %}
Write down the recovery words the script prints. They are shown only once, and they are the validator's wallet.
{% endhint %}

When it finishes it prints the node's address and validator address. Follow the log with:

```bash
journalctl -u vibehash-node -f
```

The node's files are in `/var/lib/vibehash`, and its wallet password is in `/etc/vibehash/env`. Include both in backups.

For a network that takes Evo jobs, it also installs `datasets`, fetches the Evo tasks' questions (the node re-scores miners' answers with them) and gives rounds a 180-second verify phase.

To turn on the [admin app](admin-app.md)'s routes, add `VIBEHASH_ADMIN_TOKEN=...` and `VIBEHASH_ADMIN_MFA=...` lines to `/etc/vibehash/env` (made with `vibehash admin --new-token` and `--new-mfa`) and run `systemctl restart vibehash-node`. Running the script again keeps them.

## A second network on the same server

One server can run another network's node beside the first, such as an Evo Hash testnet. Give it a name, its own domain (an A record to the same server), a port other than 443 and its own chain id:

```bash
ssh -t YOU@SERVER sudo bash setup-node.sh --name evo --domain evo.example.org --port 8443 --chain-id my-evo-1 --es --stake YOUR-GPU-WALLET=10 --fund YOUR-WALLET-ADDRESS=10000
```

- `--name evo` keeps it apart: `/var/lib/vibehash-evo`, `/etc/vibehash-evo`, the service `vibehash-node-evo` and its own certificate. The first node is left as it is.
- A named node must have its own `--chain-id`; the script refuses the default.
- `--es` makes it take Evo Hash jobs. `--stake ADDR=VIBE` starts a wallet staked: an Evo job's work is checked only by staked workers, so someone must be staked before newcomers can earn theirs.

Both nodes share the server's Vibe Hash install, so after upgrading one, restart the others too:

```bash
sudo systemctl restart 'vibehash-node*'
```

## Lock down the server

Anyone with a shell on the server has the validator's wallet, so close SSH to everyone but you right after the first run. Keep your session open throughout; your host's web console gets you back in if a step goes wrong.

1. Make a user of your own with sudo and your SSH key:
   ```bash
   adduser YOU && usermod -aG sudo YOU && rsync --archive --chown=YOU:YOU ~/.ssh /home/YOU
   ```
   Check `ssh YOU@SERVER` and `sudo -v` work from a second terminal before going on.
2. Turn off root login and passwords. The file is named `00-` because sshd keeps the first value it reads, and Ubuntu's cloud images may ship `50-cloud-init.conf` with `PasswordAuthentication yes`:
   ```bash
   printf 'PermitRootLogin no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n' | sudo tee /etc/ssh/sshd_config.d/00-hardening.conf
   sudo sshd -t && sudo systemctl reload ssh
   ```
3. Allow SSH only over [Tailscale](https://tailscale.com), so port 22 isn't open to the internet (or `ufw allow from YOUR-IP to any port 22` instead, if your address doesn't change):
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
   sudo ufw allow in on tailscale0 to any port 22 && sudo ufw delete allow OpenSSH
   ```

From then on `SERVER` is the server's Tailscale name. The script opens SSH only while the firewall is still off, so later runs don't reopen port 22.

## Upgrading

Running the script again upgrades Vibe Hash from PyPI and restarts the node. The wallet, genesis and chain are kept. Upgrade as your own user:

```bash
scp deploy/setup-node.sh YOU@SERVER:
```
```bash
ssh -t YOU@SERVER sudo bash setup-node.sh --domain node.example.org
```

See [Operating a node](operations.md) for upgrading a network with several validators safely.

## Cloudflare in front

The domain may point at Cloudflare's proxy instead of straight at the server. The script notices (Cloudflare answers `/cdn-cgi/trace` for the domain) and carries on. The record must still lead to this server: Cloudflare passes port 80 through for the certificate, and HTTPS on 443 or 8443 on to the node.

Behind the proxy, connections come from Cloudflare's addresses, so the node's per-client rate limits count those rather than your users'.
