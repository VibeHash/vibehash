# vibehash tls

HTTPS certificates for nodes.

- [`vibehash tls cert`](#vibehash-tls-cert): make a self-signed certificate for a testnet node
- [`vibehash tls fingerprint`](#vibehash-tls-fingerprint): print a certificate's SHA-256 fingerprint, for setup --tls-fingerprint

## vibehash tls cert

Make a self-signed certificate for a testnet node.

```
vibehash tls cert [-h] [--cert CERT] [--key KEY] [--host HOST] [--days DAYS]
```

Flags:

- `--cert CERT`: certificate file to write (default node-cert.pem).
- `--key KEY`: private key file to write (default node-key.pem).
- `--host HOST`: a name or IP address workers reach the node at (repeat for several; default: localhost and this computer's name and address).
- `--days DAYS`: days the certificate is valid (default 365).

Examples:

```bash
vibehash tls cert --host 192.168.1.20
```

## vibehash tls fingerprint

Print a certificate's SHA-256 fingerprint, for setup --tls-fingerprint.

```
vibehash tls fingerprint [-h] cert
```

Arguments:

- `cert`: the PEM certificate file.

Examples:

```bash
vibehash tls fingerprint node-cert.pem
```
