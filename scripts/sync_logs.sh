#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 user@host[:/remote/Log-TIR/logs] [local_remote_logs_dir]" >&2
  exit 1
fi

REMOTE="$1"
LOCAL_DIR="${2:-remote-logs}"
INTERVAL="${SYNC_INTERVAL_SECONDS:-60}"

case "$REMOTE" in
  *:*) REMOTE_LOGS="$REMOTE" ;;
  *) REMOTE_LOGS="$REMOTE:~/Log-TIR/logs/" ;;
esac

mkdir -p "$LOCAL_DIR"

while true; do
  rsync -az --partial "$REMOTE_LOGS" "$LOCAL_DIR/"
  sleep "$INTERVAL"
done

