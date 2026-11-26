#!/usr/bin/env python3
"""Renderiza como imagen (estilo terminal) extractos LITERALES de los archivos de logs/ para el informe.
No se modifica ningún valor: cada panel indica el archivo de origen y solo se seleccionan líneas (grep)."""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(ROOT, "..", "logs"); FIG = os.path.join(ROOT, "..", "figuras")
NODES = ("master", "worker-1", "worker-2", "worker-3", "NAME", "CONTAINER")


def read(p, pat=None, keep=None, head=None, tail=None):
    lines = open(os.path.join(L, p), encoding="utf-8", errors="ignore").read().splitlines()
    lines = [l.rstrip() for l in lines if l.strip()]
    if pat: lines = [l for l in lines if re.search(pat, l)]
    if keep: lines = [l for l in lines if l.split()[1:2] and (l.split()[0] in keep or l.split()[1] in keep)]
    if head: lines = lines[:head]
    if tail: lines = lines[-tail:]
    return lines


def panel(blocks, fname, width=13):
    """blocks: lista de (titulo/comando, [líneas])"""
    n = sum(len(b[1]) + 2 for b in blocks)
    fig = plt.figure(figsize=(width, 0.165 * n + 0.3)); fig.patch.set_facecolor("#0b1220")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, n + 1)
    y = n + 0.4
    for title, lines in blocks:
        ax.text(0.008, y, title, family="monospace", fontsize=7.6, color="#4ade80", va="top", weight="bold"); y -= 1
        for l in lines:
            ax.text(0.008, y, l[:175], family="monospace", fontsize=7.1, color="#e2e8f0", va="top"); y -= 1
        y -= 1
    fig.savefig(os.path.join(FIG, fname), dpi=170, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    panel([("$ docker compose ps        # logs/runs/C4-A/compose_ps.txt", read("runs/C4-A/compose_ps.txt")),
           ("$ docker stats --no-stream  # logs/runs/C4-A/docker_stats.txt (solo los 4 nodos)", read("runs/C4-A/docker_stats.txt", keep=NODES)),
           ("$ docker inspect -f '{{.Name}} {{.NetworkSettings...IPAddress}} {{.HostConfig.Memory}}'   # extraído de logs/runs/C4-A/inspect_*.json",
            [f"{n:9s} " + " ".join(re.findall(r'"IPAddress": "(10\.0\.50\.\d+)"', open(os.path.join(L, f"runs/C4-A/inspect_{n}.json")).read())[:1]) +
             "  Memory=" + re.findall(r'"Memory": (\d+)', open(os.path.join(L, f"runs/C4-A/inspect_{n}.json")).read())[0] +
             "  OOMKilled=" + re.findall(r'"OOMKilled": (\w+)', open(os.path.join(L, f"runs/C4-A/inspect_{n}.json")).read())[0]
             for n in ("master", "worker-1", "worker-2", "worker-3")]),
           ("$ nvidia-smi (master)       # logs/runs/C4-A/nvidia-smi_master.txt", read("runs/C4-A/nvidia-smi_master.txt", pat=r"NVIDIA-SMI|GPU  Name|RTX 3060|MiB /")),
           ("$ nvidia-smi (worker-1)     # logs/runs/C4-A/nvidia-smi_worker-1.txt", read("runs/C4-A/nvidia-smi_worker-1.txt")),
           ("$ docker logs worker-1      # logs/runs/C4-A/rpc-server_worker-1.log", read("runs/C4-A/rpc-server_worker-1.log", pat=r"start_worker|load_backend|Starting RPC|endpoint|CPU:|transport|Accepted", head=8))],
          "ev_topologia.png")
    panel([("$ llama-server -v ... (C4)   # logs/buffers/buffers_C4.txt", read("buffers/buffers_C4.txt", pat=r"using device|model buffer size|KV buffer size")),
           ("$ llama-server -v ... (C6)   # logs/buffers/buffers_C6.txt", read("buffers/buffers_C6.txt", pat=r"using device|assigned to device|model buffer size|KV buffer size", head=40)[:4] +
            read("buffers/buffers_C6.txt", pat=r"model buffer size|KV buffer size"))],
          "ev_particion.png")
    panel([("$ cat logs/oom/oom_worker_cmd.txt", read("oom/oom_worker_cmd.txt")),
           ("$ cat logs/oom/oom_worker_exitcode.txt", read("oom/oom_worker_exitcode.txt")),
           ("$ dmesg | grep -i oom          # logs/oom/oom_worker_dmesg.txt", [l[:170] for l in read("oom/oom_worker_dmesg.txt", pat=r"invoked oom-killer|Memory cgroup out of memory|oom-kill:")])],
          "ev_oom.png")
    blocks = []
    for s in ("A", "B"):
        blocks.append((f"$ tc qdisc show dev eth0 (escenario {s}, 4 nodos)   # logs/red/tc_qdisc_show_{s}.txt", read(f"red/tc_qdisc_show_{s}.txt", pat="qdisc")))
        blocks.append((f"$ ping -c 20 worker-1 (escenario {s})   # logs/red/ping_{s}.txt", read(f"red/ping_{s}.txt", tail=3)))
        blocks.append((f"$ tc -s qdisc show dev eth0 (master, tras iperf3 en {s})   # logs/red/tc_s_{s}.txt", read(f"red/tc_s_{s}.txt", head=4)[1:4]))
    panel(blocks, "ev_red.png")


if __name__ == "__main__":
    main()
