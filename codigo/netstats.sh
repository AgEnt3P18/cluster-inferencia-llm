#!/usr/bin/env bash
# Imprime contadores de red de la interfaz en JSON (para bytes antes/después de cada corrida).
IFACE="${1:-eth0}"
S=/sys/class/net/$IFACE/statistics
printf '{"iface":"%s","rx_bytes":%s,"tx_bytes":%s,"rx_packets":%s,"tx_packets":%s,"rx_dropped":%s,"tx_dropped":%s}\n' \
  "$IFACE" "$(cat $S/rx_bytes)" "$(cat $S/tx_bytes)" "$(cat $S/rx_packets)" "$(cat $S/tx_packets)" \
  "$(cat $S/rx_dropped)" "$(cat $S/tx_dropped)"
