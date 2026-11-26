// Reserva N MiB de VRAM y la mantiene ocupada hasta recibir SIGTERM.
// Se usa SOLO en la prueba OOM del master para emular la RTX 3060 de 8 GB del perfil A
// sobre la RTX 3060 de 12 GB del host (12 288 MiB - 4 096 MiB = 8 192 MiB visibles).
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
  size_t mib = argc > 1 ? atol(argv[1]) : 4096;
  void* p;
  cudaError_t e = cudaMalloc(&p, mib << 20);
  if (e != cudaSuccess) { printf("cudaMalloc %zu MiB failed: %s\n", mib, cudaGetErrorString(e)); return 1; }
  cudaMemset(p, 0, mib << 20);
  size_t fr, tot; cudaMemGetInfo(&fr, &tot);
  printf("ballast: reserved %zu MiB; free now %zu MiB of %zu MiB\n", mib, fr >> 20, tot >> 20);
  fflush(stdout);
  while (true) pause();
}
