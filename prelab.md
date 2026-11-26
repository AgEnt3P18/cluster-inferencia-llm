# Pre-lab — Modelo, memoria y costos

**Estudiante:** Santiago Payares Gonzalez (T00068704) · Arquitectura del Computador, NRC 1214 · UTB

## 1. Perfil elegido y por qué

**Perfil A (referencia), adaptado a un solo host.** El host disponible tiene Ryzen 5 5600 (6C/12T), 32 GB DDR4,
**una** RTX 3060 de 12 GB y NVMe, con Windows 11 + Docker Desktop (WSL2). El modelo de 7B Q8_0 es el del perfil A y la
red Docker 10.0.50.0/24 con `tc netem` funciona (verificado), así que no hace falta bajar al perfil B. Como hay una sola GPU,
la partición es **lógica**, como permite el enunciado:

| Nodo | IP | Límite | Dispositivo del shard | Equivale en el perfil A a |
|---|---|---|---|---|
| master | 10.0.50.11 | `mem_limit 16g` | RTX 3060 (CUDA0) | master con RTX 3060 |
| worker-1..3 | .12 / .13 / .14 | `mem_limit 4g`, `cpus 4` | CPU (`rpc-server -t 4`) | GTX 1050 Ti de **4 GB** + i5-4590 (4 núcleos) |

El límite de 4 GiB de cada worker emula la VRAM de 4 GB de la GTX 1050 Ti, que es donde vive el shard en el perfil A.

## 2. Modelo descargado y verificado

| Campo | Valor |
|---|---|
| Modelo | Qwen2.5-7B-Instruct, GGUF **Q8_0** (repositorio oficial `Qwen/Qwen2.5-7B-Instruct-GGUF`) |
| Archivos | 3 partes: 3 980 069 248 + 3 942 935 680 + 175 520 480 B, fusionadas con `llama-gguf-split --merge` |
| Tamaño exacto | **8 098 525 184 B = 8.10 GB = 7.54 GiB** |
| SHA-256 (fusionado) | `52c5bf1406d31fd335a9d18154071c8d29d74680e13af45bb76543b4ddd5dbf1` |
| SHA-256 partes | coinciden con los `lfs.oid` de Hugging Face (`logs/prelab/sha256_partes.txt`, `hf_tree_*.json`) |
| Licencia | Apache 2.0 |
| Descarga medida | 8.10 GB en 577.8 s (≈ 14 MB/s ≈ 112 Mbps efectivos de internet; `logs/prelab/descarga_modelo.txt`) |
| Descarga calculada | 100 Mbps: 8.10e9 × 8 / 1e8 = **648 s (10.8 min)**; 1 Gbps: **64.8 s** |

## 3. Memoria

**Pesos** = N × bytes/parámetro = 7.62e9 × (34/32) B = **8.10 GB** (Q8_0 guarda bloques de 32 pesos int8 + una escala FP16 = 34 B; coincide con el archivo).

**KV por token** = 2 × capas × cabezas_KV × head_dim × bytes = 2 × 28 × 4 × 128 × 2 B = **57 344 B = 56 KiB** (FP16). Con KV `q8_0`: × 34/64 = 30 464 B.

| Contexto | KV FP16 | KV q8_0 |
|---|---|---|
| 2 048 | 112 MiB | 59.5 MiB |
| 4 096 | 224 MiB | 119 MiB |
| 8 192 | 448 MiB | 238 MiB |

**Memoria total** = pesos + KV + runtime (≈ 0.45 GiB: contexto CUDA + cuBLAS) + activaciones (≈ 0.30 GiB, buffer de cómputo con ubatch 512):

| | ctx 2 048 | ctx 4 096 |
|---|---|---|
| batch 1 | 8.40 GiB (9.02 GB) | **8.51 GiB (9.14 GB)** |
| batch 8 | 9.17 GiB | 10.04 GiB (10.8 GB) |

**¿Cabe en el nodo más grande?** No. El master del perfil A tiene 8 GB de VRAM y solo los pesos ya ocupan 8.10 GB;
con KV, runtime y activaciones hacen falta ≈ 9.1 GB (batch 1, 4K). Un worker (4 GB) tiene menos de la mitad. La regla
de selección se cumple: 8.10 GB > 8 GB (VRAM del nodo más grande) y 8.10 GB < 8 + 3 × 4 = 20 GB (suma). Hay que particionar.

## 4. Contexto máximo por worker

`max_ctx = (RAM_límite − runtime − shard − activaciones) / KV_por_token_del_nodo`, con RAM_límite = 4 GiB,
runtime = 0.10 GiB, activaciones = 0.30 GiB y KV del nodo = 56 KiB × (capas del nodo / 28):

| Reparto (`--tensor-split` w1,w2,w3,master) | Shard worker-1 | max_ctx (FP16) | max_ctx (q8_0) |
|---|---|---|---|
| 1,1,1,1 (C1/C2) | 1.89 GiB | ≈ 128 000 | ≈ 242 000 |
| 4,4,4,8 (C4) | 1.51 GiB | ≈ 196 000 | ≈ 369 000 |
| 1,1,1,7 (C5) | 0.75 GiB | ≈ 533 000 | ≈ 1 000 000 |

En todos los casos el límite real es el contexto nativo del modelo (32 768) y no la RAM del worker: 4K cabe con holgura.

## 5. Costos (resumen; detalle en `costos.xlsx`)

| CAPEX | Cant. | COP |
|---|---|---|
| Master (Ryzen 7, 32 GB, 1 TB NVMe, RTX 3060 8 GB) | 1 | 4 500 000 |
| Workers (i5-4590, 16 GB, 512 GB SSD, GTX 1050 Ti) | 3 | 4 800 000 |
| Switch Gigabit 8 puertos | 1 | 150 000 |
| Cableado Cat6 + patch panel | 1 | 120 000 |
| UPS APC Easy UPS BV1000 (capitalcolombia.com, 26-nov-2026) | 1 | 324 000 |
| Mano de obra | 1 | 800 000 |
| **Total** | | **10 694 000** |

Los equipos ya existen en la oficina; no se encontraron cotizaciones de equipos usados equivalentes, así que se usan los
precios de referencia del Anexo C (se declara). La UPS sí se cotizó.

| OPEX/año | Cálculo | COP |
|---|---|---|
| Energía | 750 W × 8 h × 250 d × 800 / 1000 | 1 200 000 |
| Internet | 120 000 × 12 | 1 440 000 |
| Mantenimiento | 5 % × CAPEX | 534 700 |
| **Total** | | **3 174 700** |
| Depreciación (informativa) | CAPEX / 5 años | 2 138 800 |

**TCO(3 años)** = 10 694 000 + 3 × 3 174 700 = **COP 20 218 100** (la depreciación no se suma: es el mismo CAPEX
repartido en el tiempo y sumarla lo contaría dos veces).
