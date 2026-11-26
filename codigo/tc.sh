#!/usr/bin/env bash
# Emulación de red con tc netem (egress de la interfaz).
# Uso: tc.sh [eth0] [A|B|clear]
set -euo pipefail
IFACE="${1:-eth0}"
SCENARIO="${2:-A}"

case "$SCENARIO" in
  A) DELAY="0.5ms"; LOSS="0%";   RATE="1000mbit" ;;
  B) DELAY="2ms";   LOSS="0.1%"; RATE="100mbit"  ;;
  clear) tc qdisc del dev "$IFACE" root 2>/dev/null || true; tc qdisc show dev "$IFACE"; exit 0 ;;
  *) echo "Uso: $0 [eth0] [A|B|clear]"; exit 1 ;;
esac

tc qdisc del dev "$IFACE" root 2>/dev/null || true
tc qdisc add dev "$IFACE" root netem delay "$DELAY" loss "$LOSS" rate "$RATE"
tc qdisc show dev "$IFACE"
