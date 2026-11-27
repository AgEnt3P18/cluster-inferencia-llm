# Clúster distribuido de inferencia LLM — Examen parcial integrador

Arquitectura del Computador (NRC 1214) · Universidad Tecnológica de Bolívar
Santiago Payares Gonzalez — T00068704

Clúster de 4 nodos (1 master + 3 workers) que ejecuta **Qwen2.5-7B-Instruct Q8_0** (8.10 GB) particionado por capas
con `llama.cpp` en modo RPC, sobre una red Docker `10.0.50.0/24` con emulación de red `tc netem` (escenarios A y B).

## Contenido

| Ruta | Qué es |
|---|---|
| `informe.pdf` | Informe completo |
| `prelab.md` | Pre-lab: modelo, memoria y CAPEX |
| `costos.xlsx` | CAPEX, OPEX, TCO, comparación, break-even y sensibilidad (con fórmulas) |
| `codigo/` | `Dockerfile`, `docker-compose.yml`, `tc.sh`, `collect.py`, `config.json` y scripts de orquestación y análisis |
| `logs/` | `metricas.jsonl` crudo y evidencias de cada corrida (llama-server, rpc-server, nvidia-smi, docker stats/inspect, tc, ping, iperf3, OOM) |
| `video.mp4` | Defensa oral en video |
| `figuras/` | Figuras del informe (gráficas de `codigo/analyze.py` y `codigo/costos.py`; diagramas de topología en TikZ, fuente en `codigo/diagramas/`) |

## Hardware y versiones

| Componente | Valor |
|---|---|
| CPU | AMD Ryzen 5 5600 (6 núcleos / 12 hilos) |
| RAM | 32 GB DDR4 (28 GB asignados a la VM de WSL2 en `.wslconfig`) |
| GPU | NVIDIA GeForce RTX 3060 12 GB (driver 610.88, CUDA 13.3) |
| SO | Windows 11 Pro + WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2) |
| Docker | Docker Desktop 28.3.3, Compose v2, NVIDIA Container Toolkit |
| llama.cpp | commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (build 12359), `-DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_BACKEND_DL=ON` |
| Imagen base | `nvidia/cuda:12.4.1-devel/runtime-ubuntu22.04` |

Como el host tiene una sola GPU, la partición es **lógica**: el master usa la RTX 3060 y los workers son contenedores
`rpc-server` en CPU (4 hilos y `mem_limit 4g` cada uno, que emulan la GTX 1050 Ti de 4 GB del perfil A).

## Cómo reproducir

Requisitos: Docker con soporte de GPU (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` debe funcionar),
~20 GB de disco y ~16 GB de RAM libres para la VM de Docker. Comandos desde `codigo/` (en Windows, Git Bash con `MSYS_NO_PATHCONV=1`).

```bash
# 1. Imagen (compila llama.cpp con CUDA + RPC; ~20 min)
docker build -t cluster-llm:latest .

# 2. Modelo en un volumen Docker (evita el bind-mount lento de Windows)
docker volume create llm_models
for i in 1 2 3; do f=qwen2.5-7b-instruct-q8_0-0000$i-of-00003.gguf
  curl -L -o $f https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/$f; done
docker run --rm -v llm_models:/models -v "$PWD:/src:ro" alpine cp /src/*.gguf /models/
docker run --rm -v llm_models:/models cluster-llm:latest llama-gguf-split --merge \
  /models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf /models/qwen2.5-7b-instruct-q8_0.gguf
# sha256 esperado: 52c5bf1406d31fd335a9d18154071c8d29d74680e13af45bb76543b4ddd5dbf1

# 3. Levantar una configuración (ej. C5) y medir una corrida
TS=1,1,1,7 CTX=4096 KVT=q8_0 docker compose up -d
until curl -sf localhost:18080/health; do sleep 1; done
for n in master worker-1 worker-2 worker-3; do docker exec $n /codigo/tc.sh eth0 A; done
docker exec master python3 /codigo/collect.py --label C5-A-manual --config C5 --scenario A --out /logs/manual.jsonl

# 4. Campaña completa (C1..C6 × A/B × 1 calentamiento + 3 repeticiones), red y OOM
./run_experiments.sh          # escribe logs/metricas.jsonl y logs/runs/<config>-<escenario>/
./net_test.sh                 # ping / iperf3 / tc -s en base, A y B -> logs/red/
./extras.sh                   # OOM del worker, baseline de GPU única, concurrencia, ancho de banda

# 5. Tablas, figuras y hoja de costos
python capacidad.py && python analyze.py && python costos.py && python evidencias.py
```

El puerto publicado en el host es `18080` (el `8080` del host estaba ocupado por otro servicio); dentro de la red del clúster
`llama-server` escucha en `8080`.

## Configuraciones evaluadas

| # | Workers | Contexto | `--tensor-split` (w1, w2, w3, master) | KV | Idea |
|---|---|---|---|---|---|
| C1 | 3 | 2 048 | 1,1,1,1 | f16 | Línea base |
| C2 | 3 | 4 096 | 1,1,1,1 | f16 | Más contexto |
| C3 | 2 | 2 048 | 4,4,8 | f16 | Menos nodos, proporcional a VRAM |
| C4 | 3 | 4 096 | 4,4,4,8 | q8_0 | Proporcional a VRAM + KV 8 bits |
| C5 | 3 | 4 096 | 1,1,1,7 | q8_0 | Proporcional al ancho de banda de memoria |
| C6 | 1 | 4 096 | 1,5 | q8_0 | Mínimo de nodos |

## Resultados (mediana de 3 repeticiones, `logs/analisis/tabla_resumen.csv`)

| Config | TTFT A (s) | tok/s A | TPOT A (ms) | TTFT B (s) | tok/s B | TPOT B (ms) | SLO A/B |
|---|---|---|---|---|---|---|---|
| C1 | 13.61 | 5.46 | 183 | 17.54 | 4.48 | 223 | no / no |
| C2 | 14.15 | 5.04 | 198 | 17.16 | 4.51 | 222 | no / no |
| C3 | 9.82 | 6.95 | 144 | 11.68 | 6.41 | 156 | no / no |
| C4 | 12.33 | 5.91 | 169 | 15.47 | 5.23 | 191 | no / no |
| C5 | 6.46 | 9.74 | 103 | 9.41 | 8.29 | 121 | no / no |
| **C6** | **3.83** | **16.06** | **62** | **4.62** | **14.78** | **68** | **sí / sí** |

Ganadora: **C6** (η roofline = 0.86; TCO 3 años COP 15 098 100; COP 43 525 por 1M tokens a plena jornada).
Referencia de GPU única (modelo completo en la RTX 3060 12 GB del host): 37.05 tok/s, TTFT 0.19 s.

## Qué falló y cómo se resolvió

| Problema | Solución |
|---|---|
| El target `rpc-server` no existe en esta versión de llama.cpp (ahora es `ggml-rpc-server`) | Compilar todo y crear el enlace `rpc-server -> ggml-rpc-server` en el `Dockerfile` |
| Workers sin GPU no arrancaban con CUDA enlazado estáticamente | `-DGGML_BACKEND_DL=ON`: el backend CUDA se carga dinámicamente y los workers caen al backend CPU |
| El puerto 8080 del host estaba ocupado | Publicar el master en `18080:8080` |
| Copiar el modelo por bind-mount de Windows es lento | Volumen Docker `llm_models` (ext4 dentro de la VM) |
| El intento de OOM en el master con un lastre de VRAM no falla en Windows: WDDM virtualiza la VRAM y desaloja el lastre | Se documenta en `logs/oom/oom_master_nota.txt`; la imposibilidad del master de 8 GB se demuestra por cálculo y el OOM exigido (worker) sí se reproduce |
| `iperf3` reportó 477.6 Mbps en el resumen del receptor en A | El receptor contó 20.08 s (incluye la espera del servidor); los intervalos de 1 s y la prueba inversa (`-R`) dan 954–956 Mbps, que es el valor usado |
| llama.cpp ordena los dispositivos como `RPC0, RPC1, RPC2, CUDA0` (no como los lista `--list-devices`) | La primera corrida de C3 (`8,4,4`) puso el 50 % en worker-1; se detectó con `docker stats`, se reordenaron los `--tensor-split` y se repitió C3. Esas líneas siguen en `metricas.jsonl` (crudo) y `analyze.py` las excluye porque su reparto no coincide con `config.json`; sus evidencias están en `logs/runs/C3-*_orden_split_invertido/` |
| `--fit` de llama.cpp puede reajustar parámetros | Se fija `--fit off` para que el `--tensor-split` declarado sea el que se mide |

## Declaración de uso de IA

Usé Claude Code (Anthropic) como asistente de programación, a modo de copiloto: me ayudó a escribir y depurar los
scripts de orquestación, medición y análisis (`run_experiments.sh`, `collect.py`, `analyze.py`, `costos.py`) y a revisar
la redacción. El montaje del clúster, la ejecución de las mediciones, la interpretación de los resultados y las decisiones
de diseño y de costos son míos. Cada número del informe sale de `logs/metricas.jsonl` y de los archivos de `logs/`.
