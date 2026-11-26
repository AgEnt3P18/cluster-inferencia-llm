#!/usr/bin/env python3
"""Análisis de resultados: lee logs/metricas.jsonl y logs/runs/*, genera tablas (CSV/JSON/LaTeX)
y las figuras obligatorias en figuras/. Uso: python analyze.py"""
import csv, glob, json, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "..", "logs")
FIG = os.path.join(ROOT, "..", "figuras")
OUT = os.path.join(LOGS, "analisis")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
GiB, MiB = 1024**3, 1024**2
CFG = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))["configs"]
SLO_TTFT, SLO_TPOT = 6.0, 150.0
FILE = 8_098_525_184
N_LAYER = 28
LAYER_B = 233_570_000            # bytes aprox. de un bloque transformer Q8_0 (ver capa_bytes())
OUT_B = 152064 * 3584 * 34 / 32  # lm_head Q8_0
EMB_B = 152064 * 3584 * 34 / 32  # token_embd (queda en la CPU del master)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})
C_A, C_B = "#2563eb", "#dc2626"
COL = {"C1": "#64748b", "C2": "#0891b2", "C3": "#7c3aed", "C4": "#ea580c", "C5": "#16a34a", "C6": "#ca8a04"}


def load_records():
    recs = [json.loads(l) for l in open(os.path.join(LOGS, "metricas.jsonl"), encoding="utf-8") if l.strip()]
    return recs


def q(v, p):
    return float(np.percentile(v, p))


def layer_split(ts):
    """Reproduce el reparto de capas de llama.cpp (split-mode layer): n_layer+1 capas (incluye salida)."""
    w = np.array(ts, float); cum = np.cumsum(w / w.sum())
    n = N_LAYER + 1
    dev = []
    for il in range(n):
        dev.append(int(np.searchsorted(cum, il / n, side="right")))
    counts = [dev[:N_LAYER].count(i) for i in range(len(ts))]
    out_dev = dev[N_LAYER]
    return counts, out_dev


def capa_bytes():
    return (FILE - EMB_B - OUT_B) / N_LAYER


def node_weights(cfg):
    """Devuelve (capas, bytes, out_dev) en orden de NODO: [master, worker-1, worker-2, ...].
    llama.cpp ordena los dispositivos como RPC0..RPCk-1, CUDA0, así que el tensor-split se lee en ese orden."""
    ts = [float(x) for x in CFG[cfg]["tensor_split"].split(",")]
    counts_d, out_d = layer_split(ts)
    lb = capa_bytes()
    w_d = [c * lb for c in counts_d]
    w_d[out_d] += OUT_B
    order = [len(ts) - 1] + list(range(len(ts) - 1))      # master primero, luego workers
    counts = [counts_d[i] for i in order]; w = [w_d[i] for i in order]
    out_dev = order.index(out_d)
    return counts, w, out_dev


def layer_ranges(cfg):
    """Rango de capas por nodo según el orden del pipeline (w1 -> w2 -> w3 -> master)."""
    ts = [float(x) for x in CFG[cfg]["tensor_split"].split(",")]
    counts_d, _ = layer_split(ts)
    rng, acc = [], 0
    for c in counts_d:
        rng.append((acc, acc + c - 1)); acc += c
    order = [len(ts) - 1] + list(range(len(ts) - 1))
    return [rng[i] for i in order]


def summarize(recs):
    rows = {}
    for r in recs:
        if r["warmup"] or r["config"] not in CFG:
            continue
        if r.get("extra", {}).get("tensor_split") != CFG[r["config"]]["tensor_split"]:
            continue   # corrida con un reparto distinto al declarado (p. ej. C3 con el orden de dispositivos invertido)
        rows.setdefault((r["config"], r["scenario"]), []).append(r)
    table = []
    for (c, s), rs in sorted(rows.items()):
        m = lambda k: [x["metrics"][k] for x in rs if x["metrics"][k] is not None]
        ttft, tps, tpot, pp = m("ttft_s"), m("gen_tps"), m("tpot_ms"), m("prompt_eval_tps")
        net = [x["metrics"]["net_tx_bytes"] + x["metrics"]["net_rx_bytes"] for x in rs]
        drops = m("tc_dropped")
        ram_m = max((x["nodes"].get("master", {}).get("anon_peak_bytes") or 0) for x in rs)
        ram_w = max(max((x["nodes"].get(n, {}).get("anon_peak_bytes") or 0) for n in ("worker-1", "worker-2", "worker-3")) for x in rs)
        vram = max((x["nodes"].get("master", {}).get("vram_peak_mib") or 0) for x in rs)
        row = {"config": c, "scenario": s, "n": len(rs),
               "ttft_med": st.median(ttft), "ttft_iqr": q(ttft, 75) - q(ttft, 25),
               "tps_med": st.median(tps), "tps_iqr": q(tps, 75) - q(tps, 25),
               "tpot_med": st.median(tpot), "tpot_iqr": q(tpot, 75) - q(tpot, 25),
               "pp_med": st.median(pp), "net_MB_med": st.median(net) / 1e6,
               "drops_med": st.median(drops) if drops else 0,
               "ram_master_GiB": ram_m / GiB, "ram_worker_max_GiB": ram_w / GiB, "vram_master_MiB": vram,
               "total_s_med": st.median(m("total_s")),
               "prompt_ms_med": st.median(m("prompt_ms")), "pred_ms_med": st.median(m("predicted_ms"))}
        row["slo"] = row["ttft_med"] <= SLO_TTFT and row["tpot_med"] <= SLO_TPOT
        table.append(row)
    return table


def write_tables(table):
    keys = list(table[0].keys())
    with open(os.path.join(OUT, "tabla_resumen.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(table)
    json.dump(table, open(os.path.join(OUT, "tabla_resumen.json"), "w"), indent=2)


def fig_ab(table):
    cfgs = sorted({r["config"] for r in table})
    get = lambda c, s, k: next((r[k] for r in table if r["config"] == c and r["scenario"] == s), np.nan)
    x = np.arange(len(cfgs)); wd = .38
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, (k, e, lab, lim) in enumerate([("ttft_med", "ttft_iqr", "TTFT (s)", SLO_TTFT),
                                          ("tps_med", "tps_iqr", "Generación (tok/s)", 1000 / SLO_TPOT),
                                          ("tpot_med", "tpot_iqr", "TPOT (ms/token)", SLO_TPOT)]):
        a = ax[i]
        for j, (s, col) in enumerate((("A", C_A), ("B", C_B))):
            vals = [get(c, s, k) for c in cfgs]; err = [get(c, s, e) / 2 for c in cfgs]
            b = a.bar(x + (j - .5) * wd, vals, wd, yerr=err, color=col, label=f"Escenario {s}", capsize=3)
            for xi, v in zip(x + (j - .5) * wd, vals):
                a.text(xi, v, f"{v:.1f}" if k != "tpot_med" else f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        a.axhline(lim, color="k", ls="--", lw=1)
        a.text(len(cfgs) - .5, lim, " SLO", va="bottom", ha="right", fontsize=8)
        a.set_xticks(x, cfgs); a.set_title(lab)
    ax[0].legend(fontsize=8)
    fig.suptitle("A (1 Gbps, 0.5 ms) vs. B (100 Mbps, 2 ms, 0.1 % pérdida): mediana de 3 repeticiones, barras = IQR/2", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ab_barras.png")); plt.close(fig)


def fig_series(best):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for s, col in (("A", C_A), ("B", C_B)):
        for c, ls in ((best, "-"), ("C1", ":")):
            p = os.path.join(LOGS, "series", f"{c}-{s}-r1.json")
            if not os.path.exists(p):
                continue
            t = np.array(json.load(open(p, encoding="utf-8"))["chunk_t"])
            dt = np.diff(t)
            inst = 1 / np.convolve(dt, np.ones(10) / 10, mode="valid")
            ax[0].plot(t[10:], inst, ls, color=col, lw=1.2, label=f"{c}-{s}")
            ax[1].plot(t, np.arange(1, len(t) + 1), ls, color=col, lw=1.2, label=f"{c}-{s}")
    ax[0].axhline(1000 / SLO_TPOT, color="k", ls="--", lw=1); ax[0].set_xlabel("tiempo desde el POST (s)")
    ax[0].set_ylabel("tok/s (media móvil 10 tokens)"); ax[0].legend(fontsize=8); ax[0].set_title("Velocidad instantánea durante la generación")
    ax[1].set_xlabel("tiempo desde el POST (s)"); ax[1].set_ylabel("tokens acumulados"); ax[1].legend(fontsize=8)
    ax[1].set_title("Tokens acumulados (el corte con el eje x es el TTFT)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_serie_temporal.png")); plt.close(fig)


def parse_buffers(cfg):
    """Lee logs/buffers/buffers_<cfg>.txt (log verboso de llama-server) -> MiB por dispositivo y tipo de buffer."""
    import re
    d, mem = {}, {}
    pat = re.compile(r"(CUDA0|CUDA_Host|CPU|RPC0\[10\.0\.50\.(\d+):50052\])\s+(model|KV|compute|rotation|output) buffer size =\s+([\d.]+) MiB")
    for line in open(os.path.join(LOGS, "buffers", f"buffers_{cfg}.txt"), encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            g = m.group(1)
            dev = "master_gpu" if g == "CUDA0" else "master_cpu" if g in ("CUDA_Host", "CPU") else f"worker-{int(m.group(2)) - 11}"
            d.setdefault(dev, {}).setdefault(m.group(3), 0.0)
            d[dev][m.group(3)] += float(m.group(4))
        m2 = re.match(r"(worker-\d) (\d+)", line)
        if m2:
            mem[m2.group(1)] = int(m2.group(2))
    return d, mem


def fig_memoria(best, table):
    """Descomposición de memoria por nodo (pesos, KV, activaciones, runtime, libre) con los buffers reales de llama.cpp."""
    bp = os.path.join(LOGS, "baseline_gpu_unica", "vram_idle_mib.txt")   # VRAM del escritorio de Windows sin el clúster
    base = float(open(bp).read().split()[0]) if os.path.exists(bp) else 0
    allrows = {}
    for cfg in CFG:
        d, mem = parse_buffers(cfg)
        g = d.get("master_gpu", {})
        vram = next(r["vram_master_MiB"] for r in table if r["config"] == cfg and r["scenario"] == "A")
        used = vram - base
        rows = [{"nodo": "master (VRAM)", "pesos": g.get("model", 0), "kv": g.get("KV", 0),
                 "act": g.get("compute", 0) + g.get("rotation", 0), "runtime": max(used - sum(g.values()), 0),
                 "limite": 8192, "medido": used}]
        for w in sorted(k for k in d if k.startswith("worker")):
            x = d[w]; cur = mem.get(w, 0) / MiB
            rows.append({"nodo": w + " (RAM)", "pesos": x.get("model", 0), "kv": x.get("KV", 0),
                         "act": x.get("compute", 0) + x.get("rotation", 0), "runtime": max(cur - sum(x.values()), 0),
                         "limite": 4096, "medido": cur})
        for r in rows:
            r["libre"] = r["limite"] - (r["pesos"] + r["kv"] + r["act"] + r["runtime"])
        allrows[cfg] = {"nodos": rows, "master_cpu": d.get("master_cpu", {})}
    json.dump(allrows, open(os.path.join(OUT, "memoria_nodos.json"), "w"), indent=2)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, cfg in zip(axs, ("C1", best)):
        rows = allrows[cfg]["nodos"]
        lbl = [r["nodo"].replace(" (", "\n(") for r in rows]
        bottom = np.zeros(len(rows))
        for k, col, name in (("pesos", "#1e40af", "Pesos (shard)"), ("kv", "#f59e0b", "Caché KV"), ("act", "#a855f7", "Activaciones (compute)"),
                             ("runtime", "#94a3b8", "Runtime (CUDA / proceso)"), ("libre", "#d1fae5", "Libre")):
            v = np.array([max(r[k], 0) / 1024 for r in rows])
            ax.bar(lbl, v, bottom=bottom, color=col, label=name, edgecolor="white")
            for xi, (b, vi) in enumerate(zip(bottom, v)):
                if vi > .18:
                    ax.text(xi, b + vi / 2, f"{vi:.2f}", ha="center", va="center", fontsize=7.5, color="white" if k in ("pesos", "act") else "k")
            bottom += v
        ax.set_ylabel("GiB")
        ax.set_title(f"{cfg}: --tensor-split {CFG[cfg]['tensor_split']}, ctx {CFG[cfg]['ctx']}, KV {CFG[cfg]['kv_type']}", fontsize=9)
    axs[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Memoria por nodo (buffers reales de llama.cpp). Límites: 8 GB de VRAM de referencia en el master, 4 GiB en cada worker", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_memoria_nodos.png")); plt.close(fig)
    return allrows


def roofline(table):
    """Techo de decode batch 1 de un pipeline: t_step = sum_i (W_i + KV_i) / BW_i (cada etapa lee su shard una vez por token)."""
    bw_gpu = json.loads(open(os.path.join(LOGS, "bw", "gpu_bw.json")).read().strip().splitlines()[-1])["GBps"] * 1e9
    bw_cpu = json.loads(open(os.path.join(LOGS, "bw", "cpu_bw_stream.json")).read().strip().splitlines()[0])["read_GBps"] * 1e9
    out = {"bw_gpu_Bps": bw_gpu, "bw_cpu_4hilos_Bps": bw_cpu, "t_ctx_medio": 391 + 150, "configs": {}}
    t_ctx = out["t_ctx_medio"]   # contexto medio durante la generación (prompt + mitad de la respuesta)
    for cfg in CFG:
        d, _ = parse_buffers(cfg)
        t, parts = 0.0, {}
        for dev, x in d.items():
            if dev == "master_cpu":
                continue
            kv_tok = x.get("KV", 0) * MiB / CFG[cfg]["ctx"]
            b = x.get("model", 0) * MiB + kv_tok * t_ctx
            ti = b / (bw_gpu if dev == "master_gpu" else bw_cpu)
            parts[dev] = ti * 1e3; t += ti
        techo = 1 / t
        med = {s: next(r["tps_med"] for r in table if r["config"] == cfg and r["scenario"] == s) for s in ("A", "B")}
        out["configs"][cfg] = {"t_step_min_ms": t * 1e3, "etapas_ms": parts, "techo_tok_s": techo, "medido_A": med["A"],
                               "medido_B": med["B"], "eta_A": med["A"] / techo, "eta_B": med["B"] / techo}
    recs = [json.loads(l) for l in open(os.path.join(LOGS, "metricas.jsonl"), encoding="utf-8") if l.strip()]
    g = [r for r in recs if r["config"] == "GPU1" and not r["warmup"]]
    if g:
        tps = st.median([r["metrics"]["gen_tps"] for r in g]); techo = bw_gpu / FILE
        out["GPU1"] = {"techo_tok_s": techo, "medido": tps, "eta": tps / techo}
        json.dump({"tps_med": tps, "ttft_med": st.median([r["metrics"]["ttft_s"] for r in g]),
                   "tpot_med": st.median([r["metrics"]["tpot_ms"] for r in g]),
                   "pp_med": st.median([r["metrics"]["prompt_eval_tps"] for r in g]),
                   "vram_peak_mib": max((r["nodes"].get("master", {}).get("vram_peak_mib") or 0) for r in g),
                   "bw_gpu_gbs": bw_gpu / 1e9, "n": len(g)}, open(os.path.join(OUT, "baseline_gpu_unica.json"), "w"), indent=2)
    json.dump(out, open(os.path.join(OUT, "roofline.json"), "w"), indent=2)
    return out


def concurrencia():
    p = os.path.join(LOGS, "concurrencia", "concurrencia.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    res = {}
    for r in recs:
        res.setdefault(r["extra"]["parallel"], []).append(r)
    out = {}
    for P, rs in sorted(res.items()):
        t0 = min(r["ts"]["t0"] for r in rs); t1 = max(r["ts"]["t_last"] for r in rs)
        tok = sum(r["metrics"]["gen_tokens"] for r in rs)
        out[P] = {"usuarios": len(rs), "tok_s_agregado": tok / (t1 - t0),
                  "tok_s_por_usuario_med": st.median([r["metrics"]["gen_tps"] for r in rs]),
                  "ttft_max_s": max(r["metrics"]["ttft_s"] for r in rs), "ttft_med_s": st.median([r["metrics"]["ttft_s"] for r in rs]),
                  "tpot_max_ms": max(r["metrics"]["tpot_ms"] for r in rs), "wall_s": t1 - t0}
    json.dump(out, open(os.path.join(OUT, "concurrencia.json"), "w"), indent=2)
    return out


def fig_coldwarm(table):
    rows = []
    recs = [json.loads(l) for l in open(os.path.join(LOGS, "metricas.jsonl"), encoding="utf-8") if l.strip()]
    for p in sorted(glob.glob(os.path.join(LOGS, "runs", "*", "coldstart.json"))):
        cs = json.load(open(p)); c, s = cs["config"], cs["scenario"]
        tag = os.path.basename(os.path.dirname(p))
        if c not in CFG or tag not in (f"{c}-{s}", f"{c}-{s}-coldload"):
            continue
        warm_ttft = next((r["ttft_med"] for r in table if r["config"] == c and r["scenario"] == s), None)
        first = next((r for r in recs if r["label"] == f"{c}-{s}-warmup" and r.get("extra", {}).get("tensor_split") == CFG[c]["tensor_split"]), None)
        rows.append({"config": c, "scenario": s, "run": tag, "load_s": cs["load_s"], "load_netem": cs.get("load_netem"),
                     "ttft_primera_s": first["metrics"]["ttft_s"] if first else None, "ttft_warm_s": warm_ttft})
    g = os.path.join(LOGS, "baseline_gpu_unica", "coldstart.json")
    if os.path.exists(g):
        cs = json.load(open(g)); first = next((r for r in recs if r["label"] == "GPU1-warmup"), None)
        w = [r["metrics"]["ttft_s"] for r in recs if r["config"] == "GPU1" and not r["warmup"]]
        rows.append({"config": "GPU1", "scenario": "A", "run": "GPU1", "load_s": cs["load_s"], "load_netem": "none",
                     "ttft_primera_s": first["metrics"]["ttft_s"], "ttft_warm_s": st.median(w)})
    json.dump(rows, open(os.path.join(OUT, "cold_warm.json"), "w"), indent=2)
    rows_a = [r for r in rows if r["run"] in [f"{c}-A" for c in CFG] + ["GPU1"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(rows_a))
    ax.bar(x - .2, [r["load_s"] for r in rows_a], .4, color="#475569", label="Carga del modelo + distribución RPC")
    ax.bar(x - .2, [r["ttft_primera_s"] for r in rows_a], .4, bottom=[r["load_s"] for r in rows_a], color="#f97316", label="TTFT 1.ª petición")
    ax.bar(x + .2, [r["ttft_warm_s"] for r in rows_a], .4, color="#22c55e", label="TTFT warm (mediana)")
    for xi, r in zip(x, rows_a):
        ax.text(xi - .2, r["load_s"] + r["ttft_primera_s"], f"{r['load_s'] + r['ttft_primera_s']:.0f} s", ha="center", va="bottom", fontsize=8)
        ax.text(xi + .2, r["ttft_warm_s"], f"{r['ttft_warm_s']:.1f} s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [r["config"] for r in rows_a]); ax.set_ylabel("s")
    ax.set_title("Cold start (caché de página vaciada, contenedores reiniciados) vs. warm start — escenario A")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cold_warm.png")); plt.close(fig)
    return rows


def main():
    recs = load_records()
    table = summarize(recs)
    write_tables(table)
    for r in table:
        print(f"{r['config']}-{r['scenario']}: TTFT {r['ttft_med']:.2f}±{r['ttft_iqr']:.2f}s  tg {r['tps_med']:.2f} tok/s  "
              f"TPOT {r['tpot_med']:.0f} ms  pp {r['pp_med']:.1f}  net {r['net_MB_med']:.1f} MB  drops {r['drops_med']}  "
              f"RAMw {r['ram_worker_max_GiB']:.2f}  VRAM {r['vram_master_MiB']:.0f}  SLO {r['slo']}")
    ok = [r for r in table if r["scenario"] == "A" and r["slo"]]
    best = max(ok, key=lambda r: r["tps_med"])["config"] if ok else max(table, key=lambda r: r["tps_med"])["config"]
    json.dump({"best": best}, open(os.path.join(OUT, "ganadora.json"), "w"))
    print("ganadora:", best)
    fig_ab(table); fig_series(best); fig_memoria(best, table); fig_coldwarm(table)
    # Los diagramas de topología se dibujan en TikZ: codigo/diagramas/ (fig_topologia*.png)
    rf = roofline(table)
    for c, v in rf["configs"].items():
        print(f"roofline {c}: techo {v['techo_tok_s']:.2f} tok/s  eta_A {v['eta_A']:.2f}  eta_B {v['eta_B']:.2f}  etapas {v['etapas_ms']}")
    print("GPU1", rf.get("GPU1"))
    print("concurrencia", json.dumps(concurrencia(), indent=1))


if __name__ == "__main__":
    main()
