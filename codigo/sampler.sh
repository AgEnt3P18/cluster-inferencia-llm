#!/usr/bin/env bash
# Muestrea cada 0.5 s la memoria del cgroup del contenedor (lo mismo que reporta docker stats)
# y, si hay GPU, la VRAM usada. Salida CSV: epoch,mem_current_bytes,anon_bytes,vram_used_mib,gpu_util,temp_c,sm_clock_mhz,power_w
OUT="$1"; PERIOD="${2:-0.5}"
echo "epoch,mem_current_bytes,anon_bytes,vram_used_mib,gpu_util,temp_c,sm_clock_mhz,power_w" > "$OUT"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
while true; do
  t=$(date +%s.%N)
  cur=$(cat /sys/fs/cgroup/memory.current)
  anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
  if [ "$HAS_GPU" = 1 ]; then
    g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ' ')
  else g=",,,,"; fi
  echo "$t,$cur,$anon,$g" >> "$OUT"
  sleep "$PERIOD"
done
