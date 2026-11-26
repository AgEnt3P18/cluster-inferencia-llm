#!/usr/bin/env python3
"""Cliente de medición para llama-server (se ejecuta dentro del contenedor master).

1. Lee el prompt estandarizado (prompt.txt) y aplica la plantilla de chat de Qwen2.5.
2. Hace un POST streaming a http://localhost:8080/completion.
3. Registra t0, t_first, t_last, número de chunks y la marca de tiempo de cada chunk.
4. Parsea el bloque `timings` final (prompt_per_second, predicted_per_second, ...).
5. Mide bytes de red de eth0 y descartes de tc antes/después.
6. Añade los picos de RAM/VRAM de los muestreadores (sampler.sh) en la ventana [t0, t_last].
7. Escribe una línea JSON en metricas.jsonl con el hash SHA-256 de la línea anterior.

Uso:
  python3 collect.py --label C1-A-r1 --config C1 --scenario A --rep 1 \
      --out /logs/metricas.jsonl --series-dir /logs/series --samplers-dir /logs/runs/C1-A
"""
import argparse, csv, glob, hashlib, http.client, json, os, re, subprocess, time

SYSTEM = "Eres un asistente experto en análisis financiero. Responde en español."


def netstats(iface="eth0"):
    base = f"/sys/class/net/{iface}/statistics/"
    out = {}
    for k in ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets"):
        with open(base + k) as f:
            out[k] = int(f.read())
    return out


def tc_stats(iface="eth0"):
    try:
        txt = subprocess.run(["tc", "-s", "qdisc", "show", "dev", iface],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {"raw": "", "dropped": None, "sent_bytes": None}
    m = re.search(r"Sent (\d+) bytes (\d+) pkt \(dropped (\d+), overlimits (\d+) requeues (\d+)\)", txt)
    return {"raw": txt.strip(),
            "sent_bytes": int(m.group(1)) if m else None,
            "dropped": int(m.group(3)) if m else None}


def sampler_peaks(samplers_dir, t0, t1):
    """Pico de memoria (cgroup) y VRAM por nodo en la ventana [t0, t1]."""
    peaks = {}
    if not samplers_dir:
        return peaks
    for path in glob.glob(os.path.join(samplers_dir, "mem_*.csv")):
        node = os.path.basename(path)[4:-4]
        mem = anon = vram = None
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    t = float(row["epoch"])
                except (ValueError, TypeError):
                    continue
                if t < t0 - 0.5 or t > t1 + 0.5:
                    continue
                mem = max(mem or 0, int(row["mem_current_bytes"]))
                anon = max(anon or 0, int(row["anon_bytes"] or 0))
                if row.get("vram_used_mib"):
                    vram = max(vram or 0, float(row["vram_used_mib"]))
        peaks[node] = {"mem_peak_bytes": mem, "anon_peak_bytes": anon, "vram_peak_mib": vram}
    return peaks


def last_hash(path):
    if not os.path.exists(path):
        return "0" * 64
    last = b""
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                last = line.rstrip(b"\n")
    return hashlib.sha256(last).hexdigest() if last else "0" * 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--prompt", default="/codigo/prompt.txt")
    ap.add_argument("--n-predict", type=int, default=300)
    ap.add_argument("--out", default="/logs/metricas.jsonl")
    ap.add_argument("--series-dir", default="/logs/series")
    ap.add_argument("--samplers-dir", default="")
    ap.add_argument("--extra", default="{}", help="JSON con metadatos de la configuración")
    a = ap.parse_args()

    user = open(a.prompt, encoding="utf-8").read().strip()
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    body = json.dumps({
        "prompt": prompt, "n_predict": a.n_predict, "temperature": 0, "seed": 42,
        "stream": True, "cache_prompt": False, "ignore_eos": True,
    })

    net0, tc0 = netstats(), tc_stats()
    conn = http.client.HTTPConnection(a.host, a.port, timeout=900)
    t0 = time.time()
    conn.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    t_first = None
    chunk_ts, n_chunks, text, timings, final = [], 0, [], {}, {}
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        now = time.time()
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        d = json.loads(payload)
        content = d.get("content", "")
        if content:
            if t_first is None:
                t_first = now
            chunk_ts.append(round(now - t0, 6))
            n_chunks += 1
            text.append(content)
        if d.get("stop"):
            timings = d.get("timings", {})
            final = {k: d.get(k) for k in ("tokens_predicted", "tokens_evaluated", "stop_type", "truncated")}
            break
    t_last = time.time()
    conn.close()
    net1, tc1 = netstats(), tc_stats()

    gen_tps = timings.get("predicted_per_second")
    rec = {
        "label": a.label, "config": a.config, "scenario": a.scenario, "rep": a.rep, "warmup": a.warmup,
        "ts": {"t0": t0, "t_first": t_first, "t_last": t_last},
        "metrics": {
            "ttft_s": (t_first - t0) if t_first else None,
            "prompt_eval_tps": timings.get("prompt_per_second"),
            "gen_tps": gen_tps,
            "tpot_ms": (1000.0 / gen_tps) if gen_tps else None,
            "total_s": t_last - t0,
            "prompt_tokens": timings.get("prompt_n"),
            "gen_tokens": timings.get("predicted_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "n_chunks": n_chunks,
            "net_rx_bytes": net1["rx_bytes"] - net0["rx_bytes"],
            "net_tx_bytes": net1["tx_bytes"] - net0["tx_bytes"],
            "net_rx_packets": net1["rx_packets"] - net0["rx_packets"],
            "net_tx_packets": net1["tx_packets"] - net0["tx_packets"],
            "tc_dropped": (tc1["dropped"] - tc0["dropped"]) if tc1["dropped"] is not None and tc0["dropped"] is not None else None,
        },
        "server_final": final,
        "timings": timings,
        "nodes": sampler_peaks(a.samplers_dir, t0, t_last),
        "extra": json.loads(a.extra),
        "prev_hash": last_hash(a.out),
    }
    os.makedirs(a.series_dir, exist_ok=True)
    with open(os.path.join(a.series_dir, a.label + ".json"), "w", encoding="utf-8") as f:
        json.dump({"label": a.label, "t0": t0, "chunk_t": chunk_ts, "text": "".join(text)}, f, ensure_ascii=False)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    m = rec["metrics"]
    print(f"{a.label}: TTFT={m['ttft_s']:.3f}s pp={m['prompt_eval_tps']:.1f} tok/s "
          f"tg={m['gen_tps']:.2f} tok/s TPOT={m['tpot_ms']:.1f} ms net_tx={m['net_tx_bytes']} rx={m['net_rx_bytes']}")


if __name__ == "__main__":
    main()
