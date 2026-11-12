"""Send stake money to every worker that registers on the GPU run's private test network (docs/GPU-RUN.md).

    python scripts/gpu_run_faucet.py http://127.0.0.1:8545 $VIBEHASH_HOME/testnet/requester.json

A fine-tune assigns nothing until two staked workers register (coordinator.FINETUNE_QUORUM), and a test network's
workers start with no stake. Rented pods can't be reached to run a command on, so the node pod runs this beside the
network: each round it reads who registered, and sends any worker without stake or the money for it 20 VIBE from the
requester's wallet, one transfer a block (a second sent before the first lands is refused for its nonce). Each worker's
own `gpu_run_pod.sh` stakes 10 of it once it arrives. The coins are the test network's own, worth nothing elsewhere.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibehash.client import NodeClient, NodeError  # noqa: E402
from vibehash.crypto import address_of  # noqa: E402
from vibehash.tasks import SYN_PER_VIBE  # noqa: E402

AMOUNT = 20  # VIBE a worker is sent: it stakes 10 and keeps the rest for fees
NEEDS = 15 * SYN_PER_VIBE  # a worker with less than this, and no stake, gets sent AMOUNT


def registrants(node: NodeClient, height: int) -> list[str]:
    """The addresses that registered for the round sealed at `height`, from the signed registrations in its record."""
    try:
        record = node.get(f"/round/{height}/record")
    except NodeError:
        return []
    out = []
    for entry in record.get("registrations", []):
        pub = entry.get("pubkey") if isinstance(entry, dict) else None
        out.append(address_of(bytes.fromhex(pub)) if pub else entry.get("address"))
    return [a for a in out if a]


def wants_money(node: NodeClient, address: str) -> bool:
    account = node.account(address)
    return account.get("stake", 0) == 0 and account.get("balance", 0) < NEEDS


def main(once: bool = False) -> None:
    url, wallet = sys.argv[1], sys.argv[2]
    node, sent, height = NodeClient(url), set(), 0
    print(f"faucet: watching {url} for workers without stake", flush=True)
    while True:
        try:
            tip = node.status()["height"]
            due = [a for h in range(max(height + 1, tip - 3), tip + 1) for a in registrants(node, h)]
            height = tip
            for address in dict.fromkeys(due):
                if address in sent or not wants_money(node, address):
                    continue
                # This Python's own vibehash, wherever PATH points.
                done = subprocess.run([sys.executable, "-m", "vibehash", "send", "--node", url, "--wallet", wallet,
                                       "--to", address, "--amount", str(AMOUNT)], stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True)
                if done.returncode == 0:
                    sent.add(address)
                    print(f"faucet: sent {AMOUNT} VIBE to {address}", flush=True)
                    break  # one transfer a block: the rest wait for the next pass
                print(f"faucet: sending to {address} failed, retrying next block: "
                      f"{(done.stderr or done.stdout).strip()[-300:]}", flush=True)
                break
        except Exception as e:  # a node that's restarting or busy: the next pass tries again
            print(f"faucet: {type(e).__name__}: {e}", flush=True)
        if once:
            return
        time.sleep(15)


if __name__ == "__main__":
    main(once="--once" in sys.argv)
