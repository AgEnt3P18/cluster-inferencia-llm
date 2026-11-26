// Ancho de banda efectivo de VRAM (copia device-to-device), usado como BW_efectivo en el roofline.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  const size_t n = 1ull << 30;              // 1 GiB
  void *a, *b; cudaMalloc(&a, n); cudaMalloc(&b, n);
  cudaMemset(a, 1, n);
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice);
  const int it = 20;
  cudaEventRecord(s);
  for (int i = 0; i < it; i++) cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  // una copia lee n bytes y escribe n bytes
  double gbs = 2.0 * n * it / (ms / 1e3) / 1e9;
  printf("{\"test\":\"gpu_d2d_copy\",\"bytes\":%zu,\"iters\":%d,\"ms\":%.3f,\"GBps\":%.2f}\n", n, it, ms, gbs);
  return 0;
}
