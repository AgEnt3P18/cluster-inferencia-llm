# Imagen única para master y workers: llama.cpp compilado con CUDA + RPC
# Commit fijado para reproducibilidad (tag v1.0.0, build b12359).
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build
ARG LLAMA_COMMIT=b1b770d90d94cad9e9ea5b993bdbc383a522fb28
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp && git checkout ${LLAMA_COMMIT}
WORKDIR /src/llama.cpp
RUN cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build -j 12 \
    && git rev-parse HEAD > /src/LLAMA_COMMIT
# Micro-benchmarks: ancho de banda de VRAM (roofline) y lastre de VRAM (emular GPU de 8 GB en la prueba OOM)
COPY gpu_bw.cu vram_ballast.cu /src/
RUN nvcc -O3 -arch=sm_86 -o /src/gpu_bw /src/gpu_bw.cu && nvcc -O3 -arch=sm_86 -o /src/vram_ballast /src/vram_ballast.cu

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 iputils-ping iperf3 curl jq python3 procps libgomp1 ca-certificates sysbench \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/llama.cpp/build/bin/ /opt/llama/bin/
COPY --from=build /src/LLAMA_COMMIT /opt/llama/LLAMA_COMMIT
COPY --from=build /src/gpu_bw /src/vram_ballast /opt/llama/bin/
COPY tc.sh netstats.sh /opt/cluster/
RUN chmod +x /opt/cluster/*.sh && ln -s /opt/llama/bin/ggml-rpc-server /opt/llama/bin/rpc-server
ENV PATH=/opt/llama/bin:/opt/cluster:$PATH \
    LD_LIBRARY_PATH=/opt/llama/bin:$LD_LIBRARY_PATH
WORKDIR /work
