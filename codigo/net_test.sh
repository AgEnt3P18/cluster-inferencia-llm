#!/usr/bin/env bash
# Parte 2: aplica netem A y B en todos los nodos y verifica con ping, iperf3 y tc -s.
# Requiere el clúster arriba (docker compose up -d). Salida en ../logs/red/
set -uo pipefail
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
OUT=../logs/red; mkdir -p "$OUT"
NODES="master worker-1 worker-2 worker-3"
docker exec -d worker-1 iperf3 -s
sleep 1
for S in base A B; do
  for n in $NODES; do
    if [ "$S" = base ]; then docker exec $n /codigo/tc.sh eth0 clear; else docker exec $n /codigo/tc.sh eth0 $S; fi
  done > "$OUT/tc_qdisc_show_$S.txt" 2>&1
  docker exec master ping -c 20 worker-1 > "$OUT/ping_$S.txt" 2>&1
  docker exec master iperf3 -c worker-1 -t 10 -J > "$OUT/iperf3_$S.json" 2>&1
  docker exec master iperf3 -c worker-1 -t 10 -R -J > "$OUT/iperf3_${S}_reverse.json" 2>&1
  for n in $NODES; do echo "--- $n"; docker exec $n tc -s qdisc show dev eth0; done > "$OUT/tc_s_$S.txt" 2>&1
  echo "== $S"; tail -2 "$OUT/ping_$S.txt"
  python -c "import json;d=json.load(open('$OUT/iperf3_$S.json'));e=d['end'];print('iperf3 Mbps:',round(e['sum_received']['bits_per_second']/1e6,1),'retrans:',e['sum_sent'].get('retransmits'))"
done
for n in $NODES; do docker exec $n /codigo/tc.sh eth0 clear >/dev/null; done
docker exec worker-1 pkill iperf3
