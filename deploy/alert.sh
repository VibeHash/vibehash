#!/usr/bin/env bash
# Tell the operator a Vibe Hash unit failed: POST the failure and the unit's last 20 log lines to $ALERT_WEBHOOK
# (set in $ETC/env). Discord reads the JSON's "content", Slack and ntfy its "text": both are sent. With no webhook it
# only logs, to the journal, and succeeds. The unit vibehash-alert@.service runs it, from another unit's OnFailure=.
#
#   alert.sh [--name NAME] UNIT
set -euo pipefail

NAME="${SUFFIX:+${SUFFIX#-}}"
while [ $# -gt 1 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    *) echo "alert: unknown option: $1" >&2; exit 1 ;;
  esac
done
[ $# -eq 1 ] || { echo "usage: alert.sh [--name NAME] UNIT" >&2; exit 1; }
UNIT="$1"
[[ -z "$NAME" || "$NAME" =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || { echo "alert: --name takes lowercase letters, digits and dashes" >&2; exit 1; }
ETC=${ETC:-/etc/vibehash${NAME:+-$NAME}}

ALERT_WEBHOOK=""
if [ -f "$ETC/env" ]; then set -a; . "$ETC/env"; set +a; fi
if [ -z "${ALERT_WEBHOOK:-}" ]; then
  echo "alert: $UNIT failed, and ALERT_WEBHOOK isn't set in $ETC/env, so nobody was told"
  exit 0
fi
journalctl -u "$UNIT" -n 20 --no-pager 2>&1 | HOST=$(hostname) UNIT="$UNIT" python3 -c '
import json, os, sys
text = "%s: unit %s failed\n%s" % (os.environ["HOST"], os.environ["UNIT"], sys.stdin.read())
print(json.dumps({"text": text, "content": text[:1900]}))  # (Discord refuses more than 2000 characters)
' | curl -fsS --max-time 20 -H 'Content-Type: application/json' --data @- "$ALERT_WEBHOOK" >/dev/null
