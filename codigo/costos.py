#!/usr/bin/env python3
"""Costos en COP: CAPEX, OPEX, TCO a 3 años, COP/1M tokens, comparación A/B/C, break-even y sensibilidad.
Entradas: costos_params.json + logs/analisis/tabla_resumen.json (+ baseline GPU única).
Salidas: ../costos.xlsx (con fórmulas), logs/analisis/costos.json, figuras Pareto / break-even / tornado."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "..", "logs"); OUT = os.path.join(LOGS, "analisis"); FIG = os.path.join(ROOT, "..", "figuras")
P = json.load(open(os.path.join(ROOT, "costos_params.json"), encoding="utf-8"))
CFG = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))["configs"]
TAB = json.load(open(os.path.join(OUT, "tabla_resumen.json")))
BASE = json.load(open(os.path.join(OUT, "baseline_gpu_unica.json"))) if os.path.exists(os.path.join(OUT, "baseline_gpu_unica.json")) else None
BEST = json.load(open(os.path.join(OUT, "ganadora.json")))["best"]
H, D, KWH = P["horas_dia"], P["dias_anio"], P["energia_cop_kwh"]
INTERNET = P["internet_cop_mes"] * 12
SEG_ANIO = 3600 * H * D
DEM = P["demanda"]; DEMANDA_TOK = DEM["req_dia"] * DEM["tokens_salida_req"] * D   # 37.5 M
TRM = P["trm_cop_usd"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})


def capex_items_cluster(n_workers):
    items = []
    for it in P["capex_cluster"]:
        cant = n_workers if it["item"].startswith("Worker") else it["cant"]
        items.append({**it, "cant": cant})
    return items


def opex(capex, watts, kwh=KWH):
    energia = watts * H * D * kwh / 1000
    mant = P["mantenimiento_pct_capex"] * capex
    dep = capex / P["vida_util_anios"]
    return {"energia": energia, "internet": INTERNET, "mantenimiento": mant, "depreciacion": dep,
            "total_sin_dep": energia + INTERNET + mant}


def tco(capex, op):
    return capex + P["anios_tco"] * op["total_sin_dep"]


def cop_1m(tco3, tps, util):
    tok_anio = tps * SEG_ANIO * util
    return tco3 / (P["anios_tco"] * tok_anio / 1e6)


def api_cop_1m(api, scale=1.0):
    usd_req = (DEM["tokens_prompt_req"] * api["usd_in"] + DEM["tokens_salida_req"] * api["usd_out"]) / 1e6 * scale
    return usd_req / DEM["tokens_salida_req"] * 1e6 * TRM


def med(c, s, k):
    return next(r[k] for r in TAB if r["config"] == c and r["scenario"] == s)


def main():
    res = {"configs": {}, "opciones": {}, "apis": []}
    # --- configuraciones del clúster
    for c, cf in CFG.items():
        if not any(r["config"] == c for r in TAB):
            continue
        items = capex_items_cluster(cf["workers"])
        capex = sum(i["cant"] * i["precio"] for i in items)
        watts = sum(i["cant"] * i["potencia_w"] for i in items)
        op = opex(capex, watts); t3 = tco(capex, op)
        tps = med(c, "A", "tps_med")
        res["configs"][c] = {"workers": cf["workers"], "capex": capex, "watts": watts, "opex": op, "tco3": t3,
                             "tps_A": tps, "tps_B": med(c, "B", "tps_med"), "tpot_A": med(c, "A", "tpot_med"),
                             "slo_A": med(c, "A", "slo"), "slo_B": med(c, "B", "slo"),
                             "cop1m": {u: cop_1m(t3, tps, u) for u in (0.25, 0.5, 1.0)},
                             "capacidad_tok_anio": tps * SEG_ANIO,
                             "cop1m_demanda": t3 / (P["anios_tco"] * DEMANDA_TOK / 1e6) if tps * SEG_ANIO >= DEMANDA_TOK else None}
    best = res["configs"][BEST]
    # --- opción B: servidor GPU único
    capex_b = sum(i["cant"] * i["precio"] for i in P["opcion_b_rtx4090"])
    op_b = opex(capex_b, P["opcion_b_potencia_w"]); t3b = tco(capex_b, op_b)
    tps_3060 = BASE["tps_med"] if BASE else None
    # misma eficiencia que la medida en la 3060 (η ≈ 0.96 del techo) y misma relación BW efectivo/nominal: escala con el BW nominal
    tps_4090 = tps_3060 * P["opcion_b_rtx4090_bw_gbs"] / P["rtx3060_bw_nominal_gbs"] if BASE else None
    res["opciones"]["B_rtx4090"] = {"capex": capex_b, "opex": op_b, "tco3": t3b, "tps": tps_4090, "metodo": "estimado por roofline: tok/s medido en RTX 3060 12 GB × 1008/360 GB/s (BW nominal 4090/3060)",
                                    "cop1m": {u: cop_1m(t3b, tps_4090, u) for u in (0.25, 0.5, 1.0)}}
    up = P["upgrades"]["gpu_3060_12gb"]["precio"]
    capex_b1 = P["capex_cluster"][0]["precio"] + up + 324000 + 300000
    op_b1 = opex(capex_b1, 300); t3b1 = tco(capex_b1, op_b1)
    res["opciones"]["B_rtx3060_12gb"] = {"capex": capex_b1, "opex": op_b1, "tco3": t3b1, "tps": tps_3060, "metodo": "medido en el host (RTX 3060 12 GB, modelo completo en VRAM)",
                                         "cop1m": {u: cop_1m(t3b1, tps_3060, u) for u in (0.25, 0.5, 1.0)}}
    for a in P["apis"]:
        res["apis"].append({**a, "cop_1m": api_cop_1m(a), "cop_anio_demanda": api_cop_1m(a) * DEMANDA_TOK / 1e6})
    api_ref = min(res["apis"], key=lambda a: a["cop_1m"])
    api_mid = sorted(res["apis"], key=lambda a: a["cop_1m"])[1]
    # --- break-even
    be = {}
    for name, t3, cap in (("cluster_" + BEST, best["tco3"], best["capacidad_tok_anio"]),
                          ("B_rtx4090", t3b, tps_4090 * SEG_ANIO), ("B_rtx3060_12gb", t3b1, tps_3060 * SEG_ANIO)):
        be[name] = {a["proveedor"]: {"tok_anio_breakeven": (t3 / P["anios_tco"]) / (a["cop_1m"] / 1e6),
                                     "capacidad_tok_anio": cap} for a in res["apis"]}
    res["breakeven"] = be
    # --- sensibilidad (ganadora, base utilización 50 %)
    base_u = 0.5
    def c1m(kwh=KWH, u=base_u, tf=1.0):
        op = opex(best["capex"], best["watts"], kwh); return cop_1m(tco(best["capex"], op), best["tps_A"] * tf, u)
    base_val = c1m()
    sens = {"base": base_val,
            "energia": (c1m(kwh=600), c1m(kwh=1000)),
            "utilizacion": (c1m(u=1.0), c1m(u=0.25)),
            "throughput": (c1m(tf=1.3), c1m(tf=0.7))}
    be_api = {s: (best["tco3"] / 3) / (api_mid["cop_1m"] * s / 1e6) for s in (0.5, 1.0, 1.5)}
    res["sensibilidad"] = sens; res["sens_breakeven_api"] = {"proveedor": api_mid["proveedor"], **{str(k): v for k, v in be_api.items()}}
    # --- upgrades: impacto en la ganadora con el modelo de etapas medido (roofline.json)
    rf = json.load(open(os.path.join(OUT, "roofline.json")))["configs"][BEST]
    et = rf["etapas_ms"]; tpot = best["tpot_A"]; over = tpot - rf["t_step_min_ms"]        # sobrecosto RPC+red medido
    ups = {}
    def add(key, new_tps, capex_new, watts_new, nota):
        op = opex(capex_new, watts_new); t3 = tco(capex_new, op)
        c_old = best["cop1m"][1.0]; c_new = cop_1m(t3, new_tps, 1.0)
        ups[key] = {"desc": P["upgrades"][key]["desc"] if key in P["upgrades"] else key, "inversion": P["upgrades"].get(key, {}).get("precio", 0),
                    "tps_antes": best["tps_A"], "tps_despues": new_tps, "mejora_tps_pct": 100 * (new_tps / best["tps_A"] - 1),
                    "tco3_despues": t3, "cop1m_antes": c_old, "cop1m_despues": c_new, "nota": nota}
        inv = ups[key]["inversion"]
        ups[key]["cop_invertido_por_cop1m_reducido"] = inv / (c_old - c_new) if c_old > c_new else None
        ups[key]["cop_invertido_por_tok_s"] = inv / (new_tps - best["tps_A"]) if new_tps > best["tps_A"] else None
    # GPU 12 GB: el modelo cabe entero en el master (medido: GPU1); se libera el worker
    add("gpu_3060_12gb", tps_3060, capex_b1, 300, "medido en el host (GPU1); el worker deja de ser necesario")
    # switch 10 GbE: reduce el RTT de ~1.12 ms a ~0.2 ms; ~2 RTT por token y worker (pendiente medida A->B)
    k_rtt = (best["tpot_A"] and (med(BEST, "B", "tpot_med") - best["tpot_A"]) / (4.131 - 1.119))
    new_tpot = tpot - k_rtt * (1.119 - 0.2)
    add("switch_10g", 1000 / new_tpot, best["capex"] + P["upgrades"]["switch_10g"]["precio"], best["watts"] + 20,
        f"modelo: ΔTPOT = {k_rtt:.2f} ms por ms de RTT (medido entre A y B); RTT 10 GbE ≈ 0.2 ms")
    # CPU/RAM de los workers: etapa CPU con el doble de ancho de banda (DDR4 doble canal moderno)
    new_tpot = et["master_gpu"] + et["worker-1"] / 2 + over
    add("cpu_worker", 1000 / new_tpot, best["capex"] + 2500000, best["watts"], "worker nuevo (~2.5 M COP) con el doble de BW de memoria: la etapa CPU baja a la mitad (Amdahl)")
    P["upgrades"]["cpu_worker"] = {"desc": "Reemplazar el worker por un equipo con el doble de ancho de banda de memoria", "precio": 2500000, "fuente": "estimado"}
    ups["cpu_worker"]["desc"] = P["upgrades"]["cpu_worker"]["desc"]; ups["cpu_worker"]["inversion"] = 2500000
    ups["cpu_worker"]["cop_invertido_por_cop1m_reducido"] = 2500000 / (ups["cpu_worker"]["cop1m_antes"] - ups["cpu_worker"]["cop1m_despues"])
    ups["cpu_worker"]["cop_invertido_por_tok_s"] = 2500000 / (ups["cpu_worker"]["tps_despues"] - best["tps_A"])
    add("ram_master", best["tps_A"], best["capex"] + P["upgrades"]["ram_master"]["precio"], best["watts"], "la RAM del master no está en la ruta crítica del decode (usa < 2 GB)")
    add("nvme_workers", best["tps_A"], best["capex"] + P["upgrades"]["nvme_workers"]["precio"], best["watts"], "solo acorta el cold start (carga ≈ 11 s); no cambia tok/s")
    res["upgrades"] = ups
    json.dump(res, open(os.path.join(OUT, "costos.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ================= figuras =================
    # Pareto COP/1M (100 % utilización) vs tok/s
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pts = []
    for c, r in res["configs"].items():
        for s, mk in (("A", "o"), ("B", "s")):
            tps = r["tps_" + s]; y = cop_1m(r["tco3"], tps, 1.0); ok = r["slo_" + s]
            pts.append((tps, y, f"{c}-{s}", ok))
            ax.scatter(tps, y / 1e3, marker=mk, s=60, color="#16a34a" if ok else "#dc2626", edgecolor="k", zorder=3)
            ax.annotate(f"{c}-{s}", (tps, y / 1e3), textcoords="offset points", xytext=(5, 4), fontsize=8)
    # frontera de Pareto (maximizar tok/s, minimizar costo)
    srt = sorted(pts, key=lambda p: -p[0]); front = []; mn = 1e18
    for p in srt:
        if p[1] < mn: front.append(p); mn = p[1]
    ax.plot([p[0] for p in front], [p[1] / 1e3 for p in front], "--", color="#334155", lw=1, label="Frontera de Pareto")
    ax.axvline(1000 / 150, color="k", ls=":", lw=1); ax.text(1000 / 150, ax.get_ylim()[1] * .95, " TPOT 150 ms", fontsize=8)
    ax.scatter([], [], color="#16a34a", label="cumple SLO"); ax.scatter([], [], color="#dc2626", label="no cumple SLO")
    ax.set_xlabel("tokens/s generación (mediana)"); ax.set_ylabel("miles de COP por 1M tokens (100 % jornada)")
    ax.set_title("Pareto costo vs. rendimiento (círculo = escenario A, cuadrado = B)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_pareto.png")); plt.close(fig)

    # break-even: COP/1M vs tokens/año
    T = np.logspace(6.5, 10, 300)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, t3, cap, col in ((f"Clúster {BEST} (medido)", best["tco3"], best["capacidad_tok_anio"], "#2563eb"),
                               ("B: RTX 4090 (estimado)", t3b, tps_4090 * SEG_ANIO, "#7c3aed"),
                               ("B': master con RTX 3060 12 GB (medido)", t3b1, tps_3060 * SEG_ANIO, "#0891b2")):
        y = t3 / 3 / (T / 1e6); m = T <= cap
        ax.plot(T[m], y[m], color=col, lw=2, label=name); ax.plot(T[~m], y[~m], color=col, lw=1, ls=":")
        ax.scatter([cap], [t3 / 3 / (cap / 1e6)], color=col, zorder=3)
    for a, ls in zip(res["apis"], ("-", "--", "-.")):
        ax.axhline(a["cop_1m"], color="#ea580c", ls=ls, lw=1.2, label=f"API {a['proveedor']}: {a['cop_1m']:,.0f} COP/1M")
    ax.axvline(DEMANDA_TOK, color="k", ls=":", lw=1); ax.text(DEMANDA_TOK * 1.05, ax.get_ylim()[0] if False else 3e3, "demanda real\n37.5 M tok/año", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tokens generados por año"); ax.set_ylabel("COP por 1M tokens")
    ax.set_title("Break-even: COP por 1M tokens vs. volumen anual (punteado = por encima de la capacidad medida)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_breakeven.png")); plt.close(fig)

    # tornado
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    labels = [("utilizacion", "Utilización 100 % / 25 %"), ("throughput", "Throughput +30 % / −30 %"), ("energia", "Energía 600 / 1 000 COP/kWh")]
    for i, (k, lab) in enumerate(labels):
        lo, hi = sens[k]
        ax[0].barh(i, (lo - base_val) / 1e3, left=base_val / 1e3, color="#16a34a"); ax[0].barh(i, (hi - base_val) / 1e3, left=base_val / 1e3, color="#dc2626")
        ax[0].text(lo / 1e3, i, f"{lo/1e3:,.0f} ", ha="right", va="center", fontsize=8); ax[0].text(hi / 1e3, i, f" {hi/1e3:,.0f}", ha="left", va="center", fontsize=8)
    ax[0].axvline(base_val / 1e3, color="k", lw=1); ax[0].set_yticks(range(3), [l for _, l in labels])
    ax[0].set_xlabel("miles de COP por 1M tokens"); ax[0].set_title(f"Tornado {BEST} (base: 50 % de la jornada, 800 COP/kWh)")
    xs = ["−50 %", "base", "+50 %"]; vals = [be_api[0.5], be_api[1.0], be_api[1.5]]
    ax[1].bar(xs, [v / 1e9 for v in vals], color="#ea580c")
    for xi, v in enumerate(vals): ax[1].text(xi, v / 1e9, f"{v/1e9:.1f} G", ha="center", va="bottom", fontsize=8)
    ax[1].axhline(best["capacidad_tok_anio"] / 1e9, color="#2563eb", ls="--"); ax[1].text(2.4, best["capacidad_tok_anio"] / 1e9, "capacidad", color="#2563eb", fontsize=8, ha="right", va="bottom")
    ax[1].set_ylabel("tokens/año para igualar\nla API (miles de millones)", fontsize=9); ax[1].set_title(f"Break-even vs. precio API ({api_mid['proveedor'].split(' (')[0]})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tornado.png")); plt.close(fig)

    write_xlsx(res)
    print(json.dumps({c: {"capex": r["capex"], "tco3": r["tco3"], "tps": r["tps_A"], "cop1m_100": r["cop1m"][1.0], "cop1m_dem": r["cop1m_demanda"]} for c, r in res["configs"].items()}, indent=1))
    print("B4090", res["opciones"]["B_rtx4090"]["tco3"], tps_4090, "B3060", t3b1, tps_3060)
    print("APIs", [(a["proveedor"], round(a["cop_1m"])) for a in res["apis"]])
    print("sens", sens, be_api)


# ============ hoja de cálculo con fórmulas ============
HDR = Font(bold=True, color="FFFFFF"); FILL = PatternFill("solid", fgColor="1E3A8A"); BOLD = Font(bold=True)
THIN = Border(*(Side(style="thin", color="CBD5E1"),) * 4); MONEY = '#,##0'


def header(ws, row, cols):
    for j, c in enumerate(cols, 1):
        x = ws.cell(row=row, column=j, value=c); x.font = HDR; x.fill = FILL; x.alignment = Alignment(wrap_text=True, vertical="center")


def autow(ws, widths):
    for i, w in enumerate(widths, 1): ws.column_dimensions[get_column_letter(i)].width = w


def write_xlsx(res):
    wb = Workbook()
    # ---- Parámetros
    ws = wb.active; ws.title = "Parametros"
    params = [("Tarifa energía (COP/kWh)", KWH), ("Internet (COP/mes)", P["internet_cop_mes"]), ("Horas/día", H), ("Días/año", D),
              ("Mantenimiento (% CAPEX)", P["mantenimiento_pct_capex"]), ("Vida útil (años)", P["vida_util_anios"]), ("Años TCO", P["anios_tco"]),
              ("TRM COP/USD (26-nov-2026)", TRM), ("Solicitudes/día", DEM["req_dia"]), ("Tokens de salida/solicitud", DEM["tokens_salida_req"]),
              ("Tokens de prompt/solicitud (medido)", DEM["tokens_prompt_req"]), ("Demanda tokens/año", "=B10*B11*B5")]
    header(ws, 1, ["Parámetro", "Valor", "Fuente"])
    for i, (k, v) in enumerate(params, 2):
        ws.cell(row=i, column=1, value=k); ws.cell(row=i, column=2, value=v)
    ws["C2"] = "Enunciado del examen"; ws["C9"] = P["trm_fuente"]; ws["C12"] = "logs/metricas.jsonl (prompt_n)"
    ws["B6"].number_format = "0%"; autow(ws, [36, 16, 80])
    # ---- CAPEX
    wc = wb.create_sheet("CAPEX"); header(wc, 1, ["Ítem", "Cantidad", "Precio unitario COP", "Total COP", "Potencia W (c/u)", "Fuente (tienda, URL, fecha)"])
    items = capex_items_cluster(3)
    for i, it in enumerate(items, 2):
        wc.cell(row=i, column=1, value=it["item"]); wc.cell(row=i, column=2, value=it["cant"]); wc.cell(row=i, column=3, value=it["precio"]).number_format = MONEY
        wc.cell(row=i, column=4, value=f"=B{i}*C{i}").number_format = MONEY; wc.cell(row=i, column=5, value=it["potencia_w"]); wc.cell(row=i, column=6, value=it["fuente"])
    n = len(items) + 2
    wc.cell(row=n, column=1, value="Total CAPEX (clúster 4 nodos)").font = BOLD
    wc.cell(row=n, column=4, value=f"=SUM(D2:D{n-1})").number_format = MONEY; wc.cell(row=n, column=5, value=f"=SUMPRODUCT(B2:B{n-1},E2:E{n-1})")
    wc.cell(row=n, column=6, value="Potencia total bajo carga (W) en columna E")
    autow(wc, [52, 10, 18, 16, 14, 70])
    # ---- OPEX
    wo = wb.create_sheet("OPEX"); header(wo, 1, ["Concepto", "Cálculo", "COP/año"])
    rows = [("Energía", "W × h/día × días/año × COP/kWh / 1000", f"=CAPEX!E{n}*Parametros!B4*Parametros!B5*Parametros!B2/1000"),
            ("Internet", "COP/mes × 12", "=Parametros!B3*12"),
            ("Mantenimiento", "% × CAPEX", f"=Parametros!B6*CAPEX!D{n}"),
            ("Total OPEX (sin depreciación)", "energía + internet + mantenimiento", "=SUM(C2:C4)"),
            ("Depreciación (informativa)", "CAPEX / vida útil — no se suma al TCO para no contar dos veces el CAPEX", f"=CAPEX!D{n}/Parametros!B7")]
    for i, (a, b, c) in enumerate(rows, 2):
        wo.cell(row=i, column=1, value=a); wo.cell(row=i, column=2, value=b); wo.cell(row=i, column=3, value=c).number_format = MONEY
    wo["A5"].font = BOLD; autow(wo, [32, 70, 16])
    # ---- TCO por configuración
    wt = wb.create_sheet("TCO_configs")
    header(wt, 1, ["Config", "Workers", "CAPEX", "Potencia W", "Energía/año", "Internet/año", "Mant./año", "OPEX/año", "TCO 3 años",
                   "tok/s A (medido)", "tok/s B (medido)", "SLO A", "Capacidad tok/año (100 %)",
                   "COP/1M 25 %", "COP/1M 50 %", "COP/1M 100 %", "COP/1M demanda real (37.5 M)"])
    for i, (c, r) in enumerate(res["configs"].items(), 2):
        k = r["workers"]
        vals = [c, k, f"=CAPEX!C2+B{i}*CAPEX!C3+CAPEX!D4+CAPEX!D5+CAPEX!D6+CAPEX!D7", f"=CAPEX!E2+B{i}*CAPEX!E3",
                f"=D{i}*Parametros!B4*Parametros!B5*Parametros!B2/1000", "=Parametros!B3*12", f"=Parametros!B6*C{i}", f"=E{i}+F{i}+G{i}",
                f"=C{i}+Parametros!B8*H{i}", round(r["tps_A"], 3), round(r["tps_B"], 3), "sí" if r["slo_A"] else "no",
                f"=J{i}*3600*Parametros!B4*Parametros!B5",
                f"=I{i}/(Parametros!B8*M{i}*0.25/1e6)", f"=I{i}/(Parametros!B8*M{i}*0.5/1e6)", f"=I{i}/(Parametros!B8*M{i}/1e6)",
                f'=IF(M{i}>=Parametros!B13,I{i}/(Parametros!B8*Parametros!B13/1e6),"no alcanza la demanda")']
        for j, v in enumerate(vals, 1):
            x = wt.cell(row=i, column=j, value=v)
            if j in (3, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17): x.number_format = MONEY
    wt.cell(row=len(res["configs"]) + 3, column=1, value="tok/s = mediana de 3 repeticiones (logs/metricas.jsonl). Fórmulas: TCO = CAPEX + 3×OPEX; tokens/año = tok/s × 3600 × h × días × utilización; COP/1M = TCO/(3 × tokens/año / 1e6)")
    autow(wt, [8, 8, 13, 10, 12, 12, 11, 12, 13, 10, 10, 7, 16, 13, 13, 13, 22])
    # ---- Comparación
    wm = wb.create_sheet("Comparacion")
    header(wm, 1, ["Opción", "CAPEX", "OPEX/año", "TCO 3 años", "tok/s", "COP/1M (100 %)", "COP/1M demanda real", "Latencia (TPOT)", "Privacidad", "Riesgo", "Método / fuente"])
    b = res["opciones"]["B_rtx4090"]; b1 = res["opciones"]["B_rtx3060_12gb"]; bc = res["configs"][BEST]
    comp = [(f"A. Clúster 4 nodos — {BEST} (medido)", bc["capex"], bc["opex"]["total_sin_dep"], bc["tco3"], bc["tps_A"], bc["cop1m"][1.0],
             bc["tco3"] / (3 * DEMANDA_TOK / 1e6), f"{bc['tpot_A']:.0f} ms", "Sí", "Alto: 4 equipos, red, RPC experimental", "logs/metricas.jsonl"),
            ("B. Servidor GPU único RTX 4090 24 GB", b["capex"], b["opex"]["total_sin_dep"], b["tco3"], b["tps"], b["cop1m"][1.0],
             b["tco3"] / (3 * DEMANDA_TOK / 1e6), f"{1000 / b['tps']:.0f} ms", "Sí", "Bajo: un equipo; punto único de falla", b["metodo"]),
            ("B'. Master con RTX 3060 12 GB (sin workers)", b1["capex"], b1["opex"]["total_sin_dep"], b1["tco3"], b1["tps"], b1["cop1m"][1.0],
             b1["tco3"] / (3 * DEMANDA_TOK / 1e6), f"{1000 / b1['tps']:.0f} ms", "Sí", "Bajo; sin margen para modelos > 12 GB", b1["metodo"])]
    for a in res["apis"]:
        comp.append((f"C. API {a['proveedor']}", 0, a["cop_anio_demanda"], 3 * a["cop_anio_demanda"], None, a["cop_1m"], a["cop_1m"], "depende de internet",
                     "No (viola la restricción)", "Datos de clientes salen del edificio", a["fuente"]))
    for i, row in enumerate(comp, 2):
        for j, v in enumerate(row, 1):
            x = wm.cell(row=i, column=j, value=round(v, 2) if isinstance(v, float) else v)
            if j in (2, 3, 4, 6, 7): x.number_format = MONEY
    wm.cell(row=len(comp) + 3, column=1, value="API: COP/1M = (391 × USD_in + 300 × USD_out)/1e6 / 300 × 1e6 × TRM (incluye el costo del prompt por cada token generado). OPEX API = costo de la demanda real (37.5 M tokens/año).")
    autow(wm, [44, 13, 13, 13, 8, 14, 16, 14, 22, 36, 60])
    # ---- Break-even
    wb_ = wb.create_sheet("BreakEven"); header(wb_, 1, ["Opción", "API", "COP/1M API", "Costo anual opción (TCO/3)", "Tokens/año para igualar la API", "Capacidad medida tok/año", "¿Alcanzable?"])
    i = 2
    for name, d in res["breakeven"].items():
        t3 = {"cluster_" + BEST: bc["tco3"], "B_rtx4090": b["tco3"], "B_rtx3060_12gb": b1["tco3"]}[name]
        for a in res["apis"]:
            wb_.cell(row=i, column=1, value=name); wb_.cell(row=i, column=2, value=a["proveedor"]); wb_.cell(row=i, column=3, value=round(a["cop_1m"], 1))
            wb_.cell(row=i, column=4, value=round(t3 / 3)).number_format = MONEY
            wb_.cell(row=i, column=5, value=f"=D{i}/(C{i}/1e6)").number_format = MONEY
            wb_.cell(row=i, column=6, value=round(d[a["proveedor"]]["capacidad_tok_anio"])).number_format = MONEY
            wb_.cell(row=i, column=7, value=f'=IF(F{i}>=E{i},"sí","no")'); i += 1
    autow(wb_, [20, 40, 12, 22, 26, 22, 12])
    # ---- Sensibilidad
    wsn = wb.create_sheet("Sensibilidad"); header(wsn, 1, ["Variable", "Valor bajo", "COP/1M bajo", "Valor alto", "COP/1M alto", "Base COP/1M"])
    s = res["sensibilidad"]
    for i, (k, lo, hi) in enumerate((("energia", "600 COP/kWh", "1 000 COP/kWh"), ("utilizacion", "100 %", "25 %"), ("throughput", "+30 %", "−30 %")), 2):
        wsn.cell(row=i, column=1, value=k); wsn.cell(row=i, column=2, value=lo); wsn.cell(row=i, column=3, value=round(s[k][0])).number_format = MONEY
        wsn.cell(row=i, column=4, value=hi); wsn.cell(row=i, column=5, value=round(s[k][1])).number_format = MONEY; wsn.cell(row=i, column=6, value=round(s["base"])).number_format = MONEY
    sb = res["sens_breakeven_api"]
    wsn.cell(row=6, column=1, value=f"Break-even vs. precio API ({sb['proveedor']})").font = BOLD
    for j, k in enumerate(("0.5", "1.0", "1.5")):
        wsn.cell(row=7 + j, column=1, value=f"precio API × {k}"); wsn.cell(row=7 + j, column=2, value=round(sb[k])).number_format = MONEY
    wsn.cell(row=11, column=1, value=f"Base: configuración {BEST}, 50 % de la jornada, 800 COP/kWh. Valores calculados por costos.py con las mismas fórmulas de la hoja TCO_configs.")
    autow(wsn, [44, 16, 14, 16, 14, 14])
    # ---- Upgrades
    wu = wb.create_sheet("Upgrades"); header(wu, 1, ["Upgrade", "Inversión COP", "tok/s antes", "tok/s después", "Mejora tok/s %", "TCO 3 años después", "COP/1M antes (100 %)", "COP/1M después", "COP invertidos / COP/1M reducido", "COP invertidos / (tok/s)", "Nota"])
    for i, (k, u) in enumerate(res["upgrades"].items(), 2):
        vals = [u["desc"], u["inversion"], round(u["tps_antes"], 2), round(u["tps_despues"], 2), round(u["mejora_tps_pct"], 1), round(u["tco3_despues"]), round(u["cop1m_antes"]), round(u["cop1m_despues"]),
                round(u["cop_invertido_por_cop1m_reducido"], 1) if u["cop_invertido_por_cop1m_reducido"] else "sin mejora", round(u["cop_invertido_por_tok_s"]) if u["cop_invertido_por_tok_s"] else "sin mejora", u["nota"]]
        for j, v in enumerate(vals, 1):
            x = wu.cell(row=i, column=j, value=v)
            if j in (2, 6, 7, 8, 10): x.number_format = MONEY
    autow(wu, [60, 13, 10, 12, 12, 16, 16, 14, 18, 16, 80])
    # ---- Fuentes
    wf = wb.create_sheet("Fuentes"); header(wf, 1, ["Ítem", "Fuente", "Fecha"])
    srcs = [(it["item"], it["fuente"]) for it in P["capex_cluster"] + P["opcion_b_rtx4090"]] + [(v["desc"], v["fuente"]) for v in P["upgrades"].values()] + [(a["proveedor"], a["fuente"]) for a in P["apis"]] + [("TRM", P["trm_fuente"])]
    for i, (a, b2) in enumerate(srcs, 2):
        wf.cell(row=i, column=1, value=a); wf.cell(row=i, column=2, value=b2); wf.cell(row=i, column=3, value=P["fecha_cotizacion"])
    autow(wf, [60, 100, 12])
    wb.save(os.path.join(ROOT, "..", "costos.xlsx"))


if __name__ == "__main__":
    main()
