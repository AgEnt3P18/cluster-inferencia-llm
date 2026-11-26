#!/usr/bin/env bash
# Mediciones complementarias (se ejecutan con el clúster abajo):
#   1) OOM: modelo completo en un worker (4 GiB)      -> logs/oom/
#   2) Ancho de banda de memoria GPU y CPU (roofline)  -> logs/bw/
#   3) Servidor GPU única (opción B medida en el host) -> logs/metricas.jsonl (config GPU1) + logs/baseline_gpu_unica/
#   4) Concurrencia (--parallel 1/2/4) en la ganadora  -> logs/metricas.jsonl (config <BEST>-par<k>) + logs/concurrencia/
#   5) Cold start con la red B activa durante la carga -> logs/runs/<BEST>-B-coldload/
# Uso: ./extras.sh BEST_CONFIG    (p. ej. ./extras.sh C5)
set -uo pipefail
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
BEST=${1:-C5}
L=../logs
IMG=cluster-llm:latest
MODEL=/models/qwen2.5-7b-instruct-q8_0.gguf
cfg() { python -c "import json;c=json.load(open('config.json'))['configs']['$1'];print(c['$2'])"; }
docker compose down -t 5 >/dev/null 2>&1

# ---------- 1) OOM en un worker
O=$L/oom; mkdir -p $O; docker rm -f oom-worker >/dev/null 2>&1
CMD="llama-cli -m $MODEL --n-gpu-layers 99 -lm none -c 4096 -p Hola -n 16"
echo "$(date -u +%FT%TZ) docker run --name oom-worker --memory 4g --memory-swap 4g --cpus 4 $IMG $CMD" > $O/oom_worker_cmd.txt
docker run --name oom-worker --memory 4g --memory-swap 4g --cpus 4 -v llm_models:/models:ro $IMG $CMD > $O/oom_worker_log.txt 2>&1 </dev/null
echo "exit code: $?" > $O/oom_worker_exitcode.txt
docker inspect oom-worker > $O/oom_worker_inspect.json
docker inspect -f 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} Memory={{.HostConfig.Memory}}' oom-worker >> $O/oom_worker_exitcode.txt
docker run --rm --privileged alpine dmesg | grep -iE "oom|killed process" | tail -6 > $O/oom_worker_dmesg.txt
docker rm -f oom-worker >/dev/null; cat $O/oom_worker_exitcode.txt

# ---------- 2) Ancho de banda de memoria
B=$L/bw; mkdir -p $B
docker run --rm --gpus all $IMG gpu_bw > $B/gpu_bw.json 2>/dev/null; tail -1 $B/gpu_bw.json
docker run --rm --cpus 4 $IMG sysbench memory --memory-block-size=1M --memory-total-size=64G --threads=4 --memory-oper=read run > $B/cpu_bw_sysbench_4hilos.txt 2>&1
grep -E "transferred" $B/cpu_bw_sysbench_4hilos.txt
docker run --rm --gpus all $IMG nvidia-smi --query-gpu=name,memory.total,memory.used,clocks.max.mem,clocks.max.sm --format=csv > $B/nvidia_smi_gpu.csv

# ---------- 3) GPU única (modelo completo en la RTX 3060 12 GB del host)
G=$L/baseline_gpu_unica; mkdir -p $G
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > $G/vram_idle_mib.txt
docker run -d --name gpu-unica --gpus all -p 18081:8080 -v llm_models:/models:ro -v "$(cd $L && pwd):/logs" -v "$(pwd):/codigo:ro" $IMG \
  llama-server -m $MODEL -ngl 99 -c 4096 -n 300 -ctk f16 -ctv f16 -fa on --parallel 1 -lm none --fit off --host 0.0.0.0 --port 8080 >/dev/null
t_up=$(python -c "import time;print(time.time())")
until curl -sf localhost:18081/health >/dev/null; do sleep 0.5; done
t_ready=$(python -c "import time;print(time.time())")
echo "{\"t_up\":$t_up,\"t_ready\":$t_ready,\"load_s\":$(python -c "print($t_ready-$t_up)")}" > $G/coldstart.json
docker exec -d gpu-unica bash -c "/codigo/sampler.sh /logs/baseline_gpu_unica/mem_master.csv 0.5"
sleep 2
docker exec gpu-unica python3 /codigo/collect.py --label GPU1-warmup --config GPU1 --scenario local --warmup --out /logs/metricas.jsonl --series-dir /logs/series --samplers-dir /logs/baseline_gpu_unica --extra '{"nota":"RTX 3060 12 GB, modelo completo, sin RPC"}'
for r in 1 2 3; do
  docker exec gpu-unica python3 /codigo/collect.py --label GPU1-r$r --config GPU1 --scenario local --rep $r --out /logs/metricas.jsonl --series-dir /logs/series --samplers-dir /logs/baseline_gpu_unica --extra '{"nota":"RTX 3060 12 GB, modelo completo, sin RPC"}'
done
docker exec gpu-unica nvidia-smi > $G/nvidia-smi.txt; docker logs gpu-unica > $G/llama-server.log 2>&1
docker exec gpu-unica pkill -f sampler.sh; docker rm -f gpu-unica >/dev/null

# ---------- 4) Concurrencia en la ganadora
export TS=$(cfg $BEST tensor_split) KVT=$(cfg $BEST kv_type) RPC=$(cfg $BEST rpc) NETEM=
K=$L/concurrencia; mkdir -p $K
for P in 1 2 4; do
  docker compose down -t 5 >/dev/null 2>&1
  export PARALLEL=$P CTX=$((4096 * P))
  docker compose up -d >/dev/null 2>&1
  until curl -sf localhost:18080/health >/dev/null; do sleep 0.5; done
  for n in master worker-1 worker-2 worker-3; do docker exec $n /codigo/tc.sh eth0 A >/dev/null; done
  docker exec master python3 /codigo/collect.py --label $BEST-par$P-warmup --config $BEST-par$P --scenario A --warmup --out /logs/concurrencia/warmup.jsonl --series-dir /logs/series >/dev/null
  pids=""
  for u in $(seq 1 $P); do
    docker exec master python3 /codigo/collect.py --label $BEST-par$P-u$u --config $BEST-par$P --scenario A --rep $u \
      --out /logs/concurrencia/par$P-u$u.jsonl --series-dir /logs/series --extra "{\"parallel\":$P,\"usuario\":$u}" & pids="$pids $!"
  done
  wait $pids
  cat $K/par$P-u*.jsonl >> $K/concurrencia.jsonl
  docker logs master > $K/llama-server_par$P.log 2>&1
done
docker compose down -t 5 >/dev/null 2>&1
unset PARALLEL

# ---------- 5) Cold start con la red B aplicada antes de cargar (t_distribución_RPC a 100 Mbps)
R=$L/runs/$BEST-B-coldload; mkdir -p $R
export CTX=$(cfg $BEST ctx) NETEM=B
docker run --rm --privileged alpine sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" >/dev/null
t_up=$(python -c "import time;print(time.time())")
docker compose up -d >/dev/null 2>&1
until curl -sf localhost:18080/health >/dev/null; do sleep 0.5; done
t_ready=$(python -c "import time;print(time.time())")
echo "{\"config\":\"$BEST\",\"scenario\":\"B\",\"load_netem\":\"B\",\"t_up\":$t_up,\"t_ready\":$t_ready,\"load_s\":$(python -c "print($t_ready-$t_up)")}" | tee $R/coldstart.json
for n in master worker-1 worker-2 worker-3; do docker exec $n tc -s qdisc show dev eth0 > $R/tc_s_$n.txt; done
docker logs master > $R/llama-server.log 2>&1
docker compose down -t 5 >/dev/null 2>&1
# ---------- 6) Tamaño de buffers por dispositivo (log verboso solo durante la carga) para cada configuración
V=$L/buffers; mkdir -p $V
for C in $(python -c "import json;print(' '.join(json.load(open('config.json'))['configs']))"); do
  export TS=$(cfg $C tensor_split) CTX=$(cfg $C ctx) KVT=$(cfg $C kv_type) RPC=$(cfg $C rpc) NETEM= EXTRA_ARGS=-v
  docker compose up -d >/dev/null 2>&1
  until curl -sf localhost:18080/health >/dev/null; do sleep 0.5; done
  docker logs master 2>&1 | grep -iE "buffer size|offload|assigned to device|KV self size|kv_cache|CUDA0|RPC" > $V/buffers_$C.txt
  for n in worker-1 worker-2 worker-3; do echo "$n $(docker exec $n cat /sys/fs/cgroup/memory.current)"; done >> $V/buffers_$C.txt
  docker compose down -t 5 >/dev/null 2>&1
done
unset EXTRA_ARGS
echo fin-extras
