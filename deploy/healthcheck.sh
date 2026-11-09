#!/usr/bin/env bash
# Check that a Vibe Hash node answers and its chain moves. setup-node.sh runs it every five minutes from a systemd timer;
# exit 2 (so the unit's OnFailure= alert fires) if the node doesn't answer, says it halted itself, or has been mid-round
# or holding transactions for 15 minutes without a new block.
#
#   URL=https://node.example.org healthcheck.sh [--name NAME]
#
# /status has the tip's height and hash but not its time, so tip_time below is when this check first saw the tip's
# height. Every run rewrites $DATA/health.json (atomically), which the admin console reads:
#   {"checked_at": <unix>, "height": <int or null>, "tip_time": <unix or null>, "ok": <bool>, "reason": <string or null>}
# (height and tip_time are null when the node didn't answer; the previous run's height and tip_time are what the
# next run compares against.) A chain with no job seals no blocks, so a node that is idle with nothing
# waiting is healthy at any height. --name NAME (or $SUFFIX=-NAME) picks a named node's folder, as in setup-node.sh.
set -euo pipefail

NAME="${SUFFIX:+${SUFFIX#-}}"
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    *) echo "health: unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -z "$NAME" || "$NAME" =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || { echo "health: --name takes lowercase letters, digits and dashes" >&2; exit 1; }
SUFFIX=${NAME:+-$NAME}
DATA=${DATA:-/var/lib/vibehash$SUFFIX}
[ -n "${URL:-}" ] || { echo "health: set URL to the node's address" >&2; exit 1; }

err=$(mktemp)
if status=$(curl -fsS --max-time 10 "$URL/status" 2>"$err"); then UP=1; else UP=""; fi
CURL_ERR=$(cat "$err"); rm -f "$err"
UP="$UP" STATUS="${status:-}" CURL_ERR="$CURL_ERR" URL="$URL" python3 -c '
import json, os, sys, time
state_file, now, limit = sys.argv[1], int(time.time()), 15 * 60
try:
    with open(state_file) as f:
        prev = json.load(f)
except (OSError, ValueError):
    prev = {}
height = tip_time = None
ok, reason = True, None
if not os.environ["UP"]:
    ok, reason = False, "%s did not answer: %s" % (os.environ["URL"], os.environ["CURL_ERR"].strip())
else:
    s = json.loads(os.environ["STATUS"])
    height = s["height"]
    # The height the last run saw keeps its tip_time; a new height is dated now.
    tip_time = prev["tip_time"] if prev.get("height") == height and prev.get("tip_time") else now
    waiting = s.get("phase") != "idle" or s.get("mempool", 0) > 0
    if s.get("halted"):
        ok, reason = False, "the node halted itself: %s" % s["halted"]
    elif waiting and now - tip_time >= limit:
        ok, reason = False, "block %d has been the tip for %d minutes (phase %s, %d transactions waiting)" % (
            height, (now - tip_time) // 60, s.get("phase"), s.get("mempool", 0))
with open(state_file + ".tmp", "w") as f:
    json.dump({"checked_at": now, "height": height, "tip_time": tip_time, "ok": ok, "reason": reason}, f)
os.replace(state_file + ".tmp", state_file)
if not ok:
    sys.exit("health: " + reason)
' "$DATA/health.json" || exit 2
