#!/usr/bin/env python3
"""Modelo de capacidad (pre-lab y Parte 4): pesos, KV, memoria total, max_ctx, tiempos de carga.
Imprime las tablas en JSON (logs/analisis/capacidad.json)."""
import json, os

GiB, MiB, GB = 1024**3, 1024**2, 1e9
N = 7.62e9                    # parámetros
FILE = 8_098_525_184          # bytes del GGUF fusionado (medido)
BPP = 34 / 32                 # Q8_0: bloque de 32 pesos int8 + escala fp16 = 34 B
L, KVH, HD = 28, 4, 128

def kv_tok(bytes_elem):       # 2 (K y V) x capas x cabezas KV x head_dim x bytes
    return 2 * L * KVH * HD * bytes_elem

KV_F16 = kv_tok(2)            # 57 344 B
KV_Q8 = kv_tok(34 / 32)       # 30 464 B (q8_0: 34 B por bloque de 32)
RUNTIME_GPU = 0.45 * GiB      # contexto CUDA + cuBLAS (medido aprox. en nvidia-smi)
RUNTIME_CPU = 0.10 * GiB      # proceso rpc-server en reposo (medido en docker stats)
ACT = 0.30 * GiB              # buffer de cómputo (activaciones) con ubatch 512


def total_mem(batch, ctx, kv=KV_F16, runtime=RUNTIME_GPU):
    return N * BPP + batch * ctx * kv + runtime + ACT


def max_ctx(limit, shard, kv=KV_F16, runtime=RUNTIME_CPU, act=ACT, frac_layers=1.0):
    """max_ctx = (RAM_límite - runtime - shard - activaciones) / KV_por_token (de las capas del nodo)."""
    return (limit - runtime - shard - act) / (kv * frac_layers)


def main():
    out = {}
    out["pesos"] = {"N": N, "bytes_por_param": BPP, "bytes_teoricos": N * BPP, "bytes_archivo": FILE,
                    "GiB_archivo": FILE / GiB}
    out["kv_por_token"] = {"f16_B": KV_F16, "q8_0_B": KV_Q8}
    out["kv_sesion_MiB"] = {str(c): {"f16": c * KV_F16 / MiB, "q8_0": c * KV_Q8 / MiB} for c in (2048, 4096, 8192, 32768)}
    out["memoria_total_GiB"] = {f"batch{b}_ctx{c}": total_mem(b, c) / GiB for b in (1, 8) for c in (2048, 4096)}
    out["descarga_s"] = {"100Mbps": FILE * 8 / 100e6, "1Gbps": FILE * 8 / 1e9}
    # max_ctx por worker para cada reparto (28 capas repartidas según tensor-split)
    # orden de --tensor-split en llama.cpp: worker-1, worker-2, worker-3, master
    splits = {"C1/C2 1,1,1,1": [1, 1, 1, 1], "C3 4,4,8": [4, 4, 8], "C4 4,4,4,8": [4, 4, 4, 8],
              "C5 1,1,1,7": [1, 1, 1, 7], "C6 1,5": [1, 5]}
    mc = {}
    for name, s in splits.items():
        frac = s[0] / sum(s)                     # fracción del modelo en el worker-1
        shard = frac * FILE
        lim = 4 * GiB
        mc[name] = {"frac_worker": frac, "shard_GiB": shard / GiB,
                    "max_ctx_f16": max_ctx(lim, shard, KV_F16, frac_layers=frac),
                    "max_ctx_q8": max_ctx(lim, shard, KV_Q8, frac_layers=frac)}
    out["max_ctx_worker_4GiB"] = mc
    # tiempos de carga teóricos del modelo completo y de un shard de 1/4
    out["t_carga_s"] = {t: {"modelo": FILE / (bw * 1e6), "shard_1_4": FILE / 4 / (bw * 1e6)}
                        for t, bw in (("NVMe", 3500), ("SATA", 550), ("HDD", 160))}
    os.makedirs("../logs/analisis", exist_ok=True)
    json.dump(out, open("../logs/analisis/capacidad.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
