#!/usr/bin/env bash
# One pod of the GPU run (docs/GPU-RUN.md), on a RunPod PyTorch pod, so each pod takes one pasted command.
#
#   bash gpu_run_pod.sh node                   the node pod: the private network, its own worker, the GSM8K job
#   bash gpu_run_pod.sh worker https://ID-8545.proxy.runpod.net
#                                              every other pod: join, wait for stake money, stake, mine
#   bash gpu_run_pod.sh fund ADDRESS...        on the node pod, by hand: what its faucet does for each worker
#
# A pod can run it at boot, as its start command, and keep RunPod's own services (SSH, the web terminal):
#   bash -c 'curl -fsSLO https://raw.githubusercontent.com/VibeHash/vibehash/gpu-run/scripts/gpu_run_pod.sh &&
#            (bash gpu_run_pod.sh node &) ; exec /start.sh'
#
# Set VIBEHASH_PASSWORD first (a password for this run only: it locks the pod's wallet). Logs go to
# /workspace/run-logs, which outlives a stopped pod.
set -euo pipefail

BRANCH=gpu-run
DATA_REV=740312add88f781978c0658806c59bc2815b9866
DATA_SHA=76ea6dde2bdd0d82e9642fdd3253bb8d440b8c77c3bff50d601ff769a432666b
LOCAL=http://127.0.0.1:8545  # GSM8K_BUDGET (VIBE, 10 a round) sets the job size; default 200
export VIBEHASH_HOME=/workspace/.vibehash
: "${VIBEHASH_PASSWORD:?set VIBEHASH_PASSWORD first: export VIBEHASH_PASSWORD='a password for this run only'}"
LOGS=/workspace/run-logs
mkdir -p "$LOGS"

install() {
    # The branch, installed over the template's own PyTorch (pip keeps a PyTorch that's already there).
    if [ ! -d /workspace/vibehash ]; then
        git clone --branch "$BRANCH" https://github.com/VibeHash/vibehash.git /workspace/vibehash
    fi
    cd /workspace/vibehash
    pip install -q -e ".[finetune,evo]" peft
    {
        echo "commit $(git rev-parse HEAD)"
        python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    } | tee "$LOGS/versions-$(hostname).txt"
}

height() {
    python -c "from vibehash.client import NodeClient; print(NodeClient('$1').status()['height'])"
}

next_block() {
    # Returns once the chain at $1 is a block past where it was: one transfer per block from the same wallet.
    local start
    start=$(height "$1")
    until [ "$(height "$1")" -gt "$start" ]; do sleep 5; done
}

stake_from() {
    # Stakes 10 VIBE from wallet $2 on node $1, once the money is there.
    until python - "$1" "$2" <<'EOF'
import json, sys
from vibehash.client import NodeClient
from vibehash.tasks import SYN_PER_VIBE
node, wallet = sys.argv[1], sys.argv[2]
address = json.load(open(wallet))["address"]  # a wallet file says its address in the clear
sys.exit(0 if NodeClient(node).account(address).get("balance", 0) >= 15 * SYN_PER_VIBE else 1)
EOF
    do sleep 10; done
    vibehash stake --node "$1" --wallet "$2" --amount 10 < /dev/null
}

case "${1:-}" in
node)
    install
    nohup vibehash testnet --host 0.0.0.0 --workers 1 --device cuda --job none --keep-rounds 1000 --plain \
        > "$LOGS/testnet.log" 2>&1 &
    until grep -q "node up" "$LOGS/testnet.log" 2>/dev/null; do sleep 2; done
    T=$VIBEHASH_HOME/testnet
    python scripts/gsm8k_job.py prepare --revision "$DATA_REV" --out /workspace/gsm8k-train.jsonl
    echo "$DATA_SHA  /workspace/gsm8k-train.jsonl" | sha256sum -c -
    vibehash train /workspace/gsm8k-train.jsonl --base qwen3-0.6b --node "$LOCAL" --wallet "$T/requester.json" \
        --budget "${GSM8K_BUDGET:-200}" --id gsm8k --yes --plain < /dev/null | tee "$LOGS/job.txt"
    # Every worker that registers without stake money is sent some (its own pod stakes it), this pod's own included.
    nohup python scripts/gpu_run_faucet.py "$LOCAL" "$T/requester.json" > "$LOGS/faucet.log" 2>&1 &
    stake_from "$LOCAL" "$T/worker-1.json"
    echo "Node pod ready: its worker is staked, and the faucet funds each worker that joins (log: $LOGS/faucet.log)."
    echo "Other pods join with:  bash gpu_run_pod.sh worker https://<this pod's ID>-8545.proxy.runpod.net"
    ;;
worker)
    NODE=${2:?give the node pod address: https://ID-8545.proxy.runpod.net}
    install
    vibehash setup --node "$NODE" --yes < /dev/null | tee "$LOGS/setup.txt"
    WALLET=$(python -c "from vibehash import settings; print(settings.default_wallet())")
    ADDR=$(python -c "import json; print(json.load(open('$WALLET'))['address'])")
    echo
    echo "This worker's address: $ADDR   (give it to: bash gpu_run_pod.sh fund ... on the node pod)"
    echo "Waiting for its stake money…"
    stake_from "$NODE" "$WALLET"
    nohup vibehash worker --node "$NODE" --device cuda --plain > "$LOGS/worker.log" 2>&1 &
    echo "Mining. Log: $LOGS/worker.log"
    ;;
fund)
    shift
    T=$VIBEHASH_HOME/testnet
    cd /workspace/vibehash
    for addr in "$@"; do
        next_block "$LOCAL"
        vibehash send --node "$LOCAL" --wallet "$T/requester.json" --to "$addr" --amount 20 < /dev/null
    done
    ;;
*)
    sed -n '2,13p' "$0"
    exit 2
    ;;
esac
