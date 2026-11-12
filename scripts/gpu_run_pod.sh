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
DATA_SHA=99911ce45fe50b8e1e70b4288e425591dc9f483e6c94403169f4f23c7405efc7
LOCAL=http://127.0.0.1:8545  # GSM8K_BUDGET (VIBE, 10 a round) sets the job size; default 200
export VIBEHASH_HOME=/workspace/.vibehash
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring  # see install()
: "${VIBEHASH_PASSWORD:?set VIBEHASH_PASSWORD first: export VIBEHASH_PASSWORD='a password for this run only'}"
LOGS=/workspace/run-logs
mkdir -p "$LOGS"

install() {
    # The branch, installed over the template's own PyTorch (pip keeps a PyTorch that's already there).
    if [ ! -d /workspace/vibehash ]; then
        git clone --branch "$BRANCH" https://github.com/VibeHash/vibehash.git /workspace/vibehash
    else  # a pod restarted with its disk kept: catch up with the branch
        git -C /workspace/vibehash pull -q --ff-only
    fi
    cd /workspace/vibehash
    # Ubuntu's own cryptography has no pip record, so pip can't replace it; install the newer one beside it first.
    pip install -q --ignore-installed cryptography
    # A credential store: sending or staking first checks the wallet's opt-in authenticator gate there, and refuses
    # when no store can be read. A pod has no desktop keychain, so keyrings.alt's file store stands in (test coins).
    pip install -q -e ".[finetune,evo]" peft keyring keyrings.alt
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

wait_done() {
    # Returns once the GSM8K job on node $1 has spent its budget: less than one round's 10 VIBE left in escrow.
    until python - "$1" <<'EOF'
import sys
from vibehash.client import NodeClient
from vibehash.tasks import SYN_PER_VIBE
try:
    left = NodeClient(sys.argv[1]).get("/job/gsm8k")["escrow"]
except Exception:  # the node busy or restarting: ask again
    sys.exit(1)
sys.exit(0 if left < 10 * SYN_PER_VIBE else 1)
EOF
    do sleep 30; done
}

margins() {
    # Replays every checked segment on this pod's GPU and prints the table into the pod's log, where it can be read
    # back with RunPod's tools (they can't copy files off a pod).
    python /workspace/vibehash/scripts/finetune_margins.py --node "$1" --device cuda \
        --scratch /workspace/margins-scratch --out /workspace/margins.csv
    echo "=== MARGINS CSV $(hostname) ==="
    cat /workspace/margins.csv
    echo "=== END MARGINS ==="
}

case "${1:-}" in
node)
    install
    # Its log also goes to the pod's own log, where RunPod's tools can read it.
    nohup vibehash testnet --host 0.0.0.0 --workers 1 --device cuda --job none --keep-rounds 1000 --plain \
        2>&1 | tee "$LOGS/testnet.log" &
    until grep -q "node up" "$LOGS/testnet.log" 2>/dev/null; do sleep 2; done
    T=$VIBEHASH_HOME/testnet
    python scripts/gsm8k_job.py prepare --revision "$DATA_REV" --out /workspace/gsm8k-train.jsonl
    echo "$DATA_SHA  /workspace/gsm8k-train.jsonl" | sha256sum -c -
    vibehash train /workspace/gsm8k-train.jsonl --base qwen3-0.6b --node "$LOCAL" --wallet "$T/requester.json" \
        --budget "${GSM8K_BUDGET:-200}" --id gsm8k --yes --plain < /dev/null | tee "$LOGS/job.txt"
    # Every worker that registers without stake money is sent some (its own pod stakes it), this pod's own included.
    nohup python -u scripts/gpu_run_faucet.py "$LOCAL" "$T/requester.json" 2>&1 | tee "$LOGS/faucet.log" &
    stake_from "$LOCAL" "$T/worker-1.json"
    echo "Node pod ready: its worker is staked, and the faucet funds each worker that joins (log: $LOGS/faucet.log)."
    echo "Other pods join with:  bash gpu_run_pod.sh worker https://<this pod's ID>-8545.proxy.runpod.net"

    # When the budget is spent: the adapter, its GSM8K score against the base model's, and this GPU's margins.
    wait_done "$LOCAL"
    echo "=== JOB SPENT: finishing ==="
    cd /workspace
    vibehash job download gsm8k --node "$LOCAL" < /dev/null
    BASE_REV=$(python -c "import json, re; from vibehash.client import NodeClient; \
print(re.search(r'@([0-9a-f]{40})', json.dumps(NodeClient('$LOCAL').get('/job/gsm8k')['spec'])).group(1))")
    S=/workspace/vibehash/scripts/gsm8k_job.py
    python "$S" score --base Qwen/Qwen3-0.6B --base-revision "$BASE_REV" --revision "$DATA_REV" --batch 32
    python "$S" score --base Qwen/Qwen3-0.6B --base-revision "$BASE_REV" --revision "$DATA_REV" --batch 32 \
        --adapter /workspace/gsm8k-train-lora
    margins "$LOCAL"
    echo "=== RUN COMPLETE ==="
    ;;
worker)
    NODE=${2:?give the node pod address: https://ID-8545.proxy.runpod.net}
    install
    # --gpu skip: keep the template's PyTorch, so every pod in the run computes with the same build.
    vibehash setup --node "$NODE" --gpu skip --yes < /dev/null | tee "$LOGS/setup.txt"
    WALLET=$(python -c "from vibehash import settings; print(settings.default_wallet())")
    ADDR=$(python -c "import json; print(json.load(open('$WALLET'))['address'])")
    # Mine first: the node pod's faucet pays workers once they register, and a worker registers by mining.
    nohup vibehash worker --node "$NODE" --device cuda --plain 2>&1 | tee "$LOGS/worker.log" &
    echo "This worker's address: $ADDR. Mining; it stakes once the faucet's money arrives."
    stake_from "$NODE" "$WALLET"
    echo "Staked 10 VIBE."

    # When the budget is spent: stop mining, then replay every checked segment on this GPU.
    wait_done "$NODE"
    pkill -f "vibehash worker" || true
    echo "=== JOB SPENT: measuring margins ==="
    margins "$NODE"
    echo "=== RUN COMPLETE ==="
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
