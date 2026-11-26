#!/usr/bin/env bash
# Arranque del master: aplica (opcional) netem antes de cargar y lanza llama-server con RPC.
set -euo pipefail
if [ -n "${NETEM:-}" ]; then /codigo/tc.sh eth0 "$NETEM"; fi
echo "[start_master] $(date -u +%FT%T.%NZ) TS=$TS CTX=$CTX KVT=$KVT PARALLEL=${PARALLEL:-1} RPC=$RPC NETEM=${NETEM:-none}"
echo "[start_master] llama.cpp commit: $(cat /opt/llama/LLAMA_COMMIT)"
exec llama-server -m /models/qwen2.5-7b-instruct-q8_0.gguf \
  --rpc "$RPC" -ngl 99 --tensor-split "$TS" -c "$CTX" -n 300 \
  -ctk "$KVT" -ctv "$KVT" -fa on --parallel "${PARALLEL:-1}" -lm none -sm layer --fit off \
  --host 0.0.0.0 --port 8080 --metrics ${EXTRA_ARGS:-}
