#!/usr/bin/env bash
# Arranque de un worker: aplica (opcional) netem y lanza rpc-server en CPU (4 hilos = i5-4590).
set -euo pipefail
if [ -n "${NETEM:-}" ]; then /codigo/tc.sh eth0 "$NETEM"; fi
echo "[start_worker] $(hostname) $(date -u +%FT%T.%NZ) NETEM=${NETEM:-none}"
exec rpc-server -H 0.0.0.0 -p 50052 -t 4
