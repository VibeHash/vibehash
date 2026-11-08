#!/usr/bin/env bash
# Back up a Vibe Hash node: an online copy of its chain database, its genesis and wallet, and (if a remote is set)
# its checkpoint blobs, sent to an rclone remote. setup-node.sh installs this and a daily systemd timer for it.
#
#   backup.sh [--name NAME] [--restore-hint]
#
# --name NAME is the named node setup-node.sh made with --name (folders /var/lib/vibehash-NAME and /etc/vibehash-NAME);
# without it, the main node. $SUFFIX (-NAME) in the environment does the same. $ETC/env may set:
#   BACKUP_RCLONE_REMOTE  an rclone remote path, such as spaces:vibehash-backups/public-1
# Without it the chain is backed up on this server only, and the blobs not at all (a loud warning says so).
#
# The server is small and a checkpoint in blobs/ reaches 1 GiB, so a blob is never copied to local disk: rclone reads
# blobs/ and sends it straight to the remote. Only small files (the chain database, gzipped, and the genesis and wallet)
# are staged in $DATA/backup/ first. A failing step ends with a one-line reason and a non-zero status, so the unit's
# OnFailure= alert fires. Exit 3: not enough free disk for the staged copy.
#
# Asking for a backup now: the node (an unprivileged user) writes {"requested_at": <unix>} to $DATA/backup/request,
# atomically; vibehash-backup.path sees it and starts the backup, and a run deletes the file when it starts.
# What happened: every run, failed or not, ends by writing $DATA/backup/last_backup.json (atomically):
#   {"timestamp": <unix>, "chain_bytes": <size of the chain archive written, 0 if none>,
#    "remote_ok": <true only if BACKUP_RCLONE_REMOTE is set and both rclone steps succeeded>, "error": <string or null>}
# Without a remote, remote_ok is false and error is "no BACKUP_RCLONE_REMOTE: blobs not backed up".
#
# To restore (on a new server: run setup-node.sh first, with the same --name and --chain-id, so the user, units and
# folders exist; the wallet's password, VIBEHASH_PASSWORD in $ETC/env, is not backed up: keep it somewhere safe):
#   1. systemctl stop vibehash-node$SUFFIX
#   2. get the chain and the keys back: rclone copy "$REMOTE/chain/" "$DATA/backup/"   (or use the local $DATA/backup/)
#   3. gunzip -c "$DATA/backup/chain-<time>.sqlite3.gz" > "$DATA/data/chain.sqlite3", and delete chain.sqlite3-wal
#      and chain.sqlite3-shm beside it (to bring the wallet and genesis back too: copy node-wallet.json and
#      genesis.json from $DATA/backup/ to $DATA/)
#   4. rclone sync "$REMOTE/blobs/" "$DATA/data/blobs/"
#   5. chown -R vibehash:vibehash "$DATA" && systemctl start vibehash-node$SUFFIX
# `backup.sh --restore-hint` prints this with the paths filled in.
set -euo pipefail

NAME="${SUFFIX:+${SUFFIX#-}}" HINT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --restore-hint) HINT=1; shift ;;
    *) echo "backup: unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -z "$NAME" || "$NAME" =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || { echo "backup: --name takes lowercase letters, digits and dashes" >&2; exit 1; }
SUFFIX=${NAME:+-$NAME}
DATA=${DATA:-/var/lib/vibehash$SUFFIX} ETC=${ETC:-/etc/vibehash$SUFFIX} KEEP=14  # (DATA and ETC in the environment are for a dry run)
CHAIN_BYTES=0 REMOTE_OK=false ERROR="" RAW="" STAGE="$DATA/backup"
fail() { echo "backup: $*" >&2; ERROR="$*"; exit 1; }
# However the run ends, say how it went in last_backup.json (once the stage exists) and drop any half-written copy.
finish() {
  local rc=$?
  [ -z "$RAW" ] || rm -f "$RAW" "$RAW.gz.part"
  if [ "$rc" -ne 0 ] && [ -z "$ERROR" ]; then ERROR="backup stopped (exit $rc)"; fi
  [ -d "$STAGE" ] || return 0
  CHAIN_BYTES="$CHAIN_BYTES" REMOTE_OK="$REMOTE_OK" ERROR="$ERROR" python3 -c '
import json, os, sys, time
out = {"timestamp": int(time.time()), "chain_bytes": int(os.environ["CHAIN_BYTES"]),
       "remote_ok": os.environ["REMOTE_OK"] == "true", "error": os.environ["ERROR"] or None}
with open(sys.argv[1] + ".tmp", "w") as f:
    json.dump(out, f)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
' "$STAGE/last_backup.json" || true
}
trap finish EXIT

if [ -n "$HINT" ]; then
  cat <<TEXT
To restore $DATA from a backup (the wallet's password, VIBEHASH_PASSWORD in $ETC/env, is not backed up):
  systemctl stop vibehash-node$SUFFIX
  rclone copy "\$BACKUP_RCLONE_REMOTE/chain/" $DATA/backup/        # or use the archives already in $DATA/backup/
  gunzip -c $DATA/backup/chain-<time>.sqlite3.gz > $DATA/data/chain.sqlite3
  rm -f $DATA/data/chain.sqlite3-wal $DATA/data/chain.sqlite3-shm
  rclone sync "\$BACKUP_RCLONE_REMOTE/blobs/" $DATA/data/blobs/
  chown -R vibehash:vibehash $DATA && systemctl start vibehash-node$SUFFIX
(on a new server, run setup-node.sh first with the same --name and --chain-id; to bring the wallet and genesis back
too: cp $DATA/backup/node-wallet.json $DATA/backup/genesis.json $DATA/)
TEXT
  exit 0
fi

rm -f "$STAGE/request"  # (a request for this run, if the node made one)
[ -f "$ETC/env" ] || fail "no $ETC/env: is this the right --name?"
set -a; . "$ETC/env"; set +a
BACKUP_RCLONE_REMOTE=${BACKUP_RCLONE_REMOTE:-}
BACKUP_RCLONE_REMOTE=${BACKUP_RCLONE_REMOTE%/}  # (a trailing slash would double up in the paths below)
CHAIN="$DATA/data/chain.sqlite3"
[ -f "$CHAIN" ] || fail "no chain database at $CHAIN"
command -v sqlite3 >/dev/null || fail "sqlite3 isn't installed (apt-get install sqlite3)"

umask 077  # the stage holds the wallet
install -d -m 700 "$STAGE"
# The copy is the database's size, and gzip then makes a smaller one beside it: 2x is enough, and the node's own disk
# must not fill up under it.
need=$(( 2 * $(stat -c %s "$CHAIN") ))
free=$(df --output=avail -B1 "$STAGE" | tail -n 1 | tr -d ' ')
if [ "$free" -lt "$need" ]; then
  ERROR="not enough free disk in $STAGE: $free bytes free, $need needed (twice the chain database)"
  echo "backup: $ERROR" >&2
  exit 3
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
RAW="$STAGE/chain-$stamp.sqlite3"
# .backup is sqlite's online backup: a consistent copy while the node runs.
sqlite3 "$CHAIN" ".backup '$RAW'" || fail "sqlite3 couldn't copy the chain database"
gzip -c "$RAW" > "$RAW.gz.part" || fail "gzip failed"
mv "$RAW.gz.part" "$RAW.gz"
rm -f "$RAW"
CHAIN_BYTES=$(stat -c %s "$RAW.gz")
for f in genesis.json node-wallet.json; do
  [ -f "$DATA/$f" ] || fail "no $DATA/$f"
  install -m 600 "$DATA/$f" "$STAGE/$f"
done

# The newest $KEEP chain archives stay (their names sort by time).
find "$STAGE" -maxdepth 1 -name 'chain-*.sqlite3.gz' | sort -r | tail -n +"$((KEEP + 1))" | while IFS= read -r old; do
  rm -f "$old"
done
echo "backup: chain saved as $STAGE/chain-$stamp.sqlite3.gz"

if [ -z "$BACKUP_RCLONE_REMOTE" ]; then
  ERROR="no BACKUP_RCLONE_REMOTE: blobs not backed up"
  cat >&2 <<TEXT
backup: WARNING: the BLOBS (checkpoints) ARE NOT BACKED UP, and the chain copy above is on this server only.
backup: Set BACKUP_RCLONE_REMOTE=remote:bucket/path in $ETC/env (make the remote with
backup: \`rclone config --config $ETC/rclone.conf\`), and the next run sends both off this server.
TEXT
  exit 0
fi

command -v rclone >/dev/null || fail "rclone isn't installed (apt-get install rclone)"
# Blobs go from their folder straight to the remote, never to a local copy.
if [ -d "$DATA/data/blobs" ]; then
  rclone sync "$DATA/data/blobs/" "$BACKUP_RCLONE_REMOTE/blobs/" --transfers 2 || fail "rclone couldn't sync the blobs to $BACKUP_RCLONE_REMOTE"
fi
rclone copy "$STAGE/" "$BACKUP_RCLONE_REMOTE/chain/" || fail "rclone couldn't copy the chain backup to $BACKUP_RCLONE_REMOTE"
REMOTE_OK=true
echo "backup: sent to $BACKUP_RCLONE_REMOTE"
