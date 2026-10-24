#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/myenv/bin/activate"

POOL="${CELERY_POOL:-}"
if [[ -z "$POOL" ]]; then
  case "$(uname -s)" in
    Darwin) POOL=solo ;;
    *) POOL=prefork ;;
  esac
fi

if [[ "$POOL" == "solo" ]]; then
  exec celery -A SAMS worker --loglevel=info --pool=solo
fi
exec celery -A SAMS worker --loglevel=info --pool=prefork --concurrency="${CELERY_CONCURRENCY:-1}"
