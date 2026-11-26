#!/usr/bin/env bash
# Orquestador: para cada configuración y escenario reinicia el clúster en frío,
# mide el cold start, aplica netem, corre 1 calentamiento + 3 repeticiones y guarda evidencias.
# Uso: ./run_experiments.sh [CONFIG ...]    (por defecto todas las de config.json)
set -uo pipefail
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
LOGS=../logs
mkdir -p "$LOGS/runs" "$LOGS/series"
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=($(python -c "import json;print(' '.join(json.load(open('config.json'))['configs']))"))
SCENARIOS=${SCENARIOS:-"A B"}
REPS=${REPS:-3}
NODES="master worker-1 worker-2 worker-3"

cfg() { python -c "import json,sys;c=json.load(open('config.json'))['configs']['$1'];print(c['$2'])"; }

for C in "${CONFIGS[@]}"; do
  export TS=$(cfg $C tensor_split) CTX=$(cfg $C ctx) KVT=$(cfg $C kv_type) RPC=$(cfg $C rpc)
  EXTRA=$(python -c "import json;print(json.dumps(json.load(open('config.json'))['configs']['$C']))")
  for S in $SCENARIOS; do
    R="$LOGS/runs/$C-$S"; mkdir -p "$R"
    echo "=== $C escenario $S  TS=$TS CTX=$CTX KVT=$KVT ==="
    docker compose down -t 5 >/dev/null 2>&1
    # limpiar la caché de página (evita warm start entre A y B)
    docker run --rm --privileged alpine sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" >/dev/null
    export NETEM=${LOAD_NETEM:-}
    t_up=$(python -c "import time;print(time.time())")
    docker compose up -d >/dev/null 2>&1
    until curl -sf http://localhost:18080/health >/dev/null 2>&1; do
      sleep 0.5
      if [ "$(docker inspect -f '{{.State.Running}}' master 2>/dev/null)" != "true" ]; then
        echo "master se detuvo"; docker logs master > "$R/llama-server.log" 2>&1; break; fi
    done
    t_ready=$(python -c "import time;print(time.time())")
    echo "{\"config\":\"$C\",\"scenario\":\"$S\",\"load_netem\":\"${LOAD_NETEM:-none}\",\"t_up\":$t_up,\"t_ready\":$t_ready,\"load_s\":$(python -c "print($t_ready-$t_up)")}" > "$R/coldstart.json"
    cat "$R/coldstart.json"
    # aplicar escenario de red en egress de todos los nodos
    for n in $NODES; do echo "--- $n" >> "$R/tc_show.txt"; docker exec $n /codigo/tc.sh eth0 $S >> "$R/tc_show.txt"; done
    # muestreadores (0.5 s)
    for n in $NODES; do docker exec -d $n bash -c "/codigo/sampler.sh /logs/runs/$C-$S/mem_$n.csv 0.5"; done
    sleep 2
    docker exec master python3 /codigo/collect.py --label "$C-$S-warmup" --config $C --scenario $S --rep 0 --warmup \
      --out /logs/metricas.jsonl --series-dir /logs/series --samplers-dir /logs/runs/$C-$S --extra "$EXTRA"
    for r in $(seq 1 $REPS); do
      docker exec master python3 /codigo/collect.py --label "$C-$S-r$r" --config $C --scenario $S --rep $r \
        --out /logs/metricas.jsonl --series-dir /logs/series --samplers-dir /logs/runs/$C-$S --extra "$EXTRA"
    done
    # evidencias
    docker compose ps > "$R/compose_ps.txt" 2>&1
    docker stats --no-stream > "$R/docker_stats.txt" 2>&1
    for n in $NODES; do docker inspect $n > "$R/inspect_$n.json"; docker exec $n tc -s qdisc show dev eth0 > "$R/tc_s_$n.txt"; done
    docker exec master nvidia-smi > "$R/nvidia-smi_master.txt" 2>&1
    for n in worker-1 worker-2 worker-3; do docker exec $n bash -c "nvidia-smi || echo 'sin GPU (worker CPU)'" > "$R/nvidia-smi_$n.txt" 2>&1; done
    for n in $NODES; do docker exec $n pkill -f sampler.sh; done
    docker logs master > "$R/llama-server.log" 2>&1
    for n in worker-1 worker-2 worker-3; do docker logs $n > "$R/rpc-server_$n.log" 2>&1; done
  done
done
docker compose down -t 5 >/dev/null 2>&1
echo "fin"
