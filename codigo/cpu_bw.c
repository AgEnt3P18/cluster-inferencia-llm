// Ancho de banda de DRAM tipo STREAM (lectura con suma y triad) con OpenMP.
// Arreglos de 512 MiB (>> caché L3 de 32 MiB) para medir memoria principal, no caché.
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N (64UL * 1024 * 1024)   /* 64 Mi doubles = 512 MiB por arreglo */
int main(void) {
  double *a = malloc(N * sizeof(double)), *b = malloc(N * sizeof(double)), *c = malloc(N * sizeof(double));
  #pragma omp parallel for
  for (size_t i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
  double best_read = 0, best_triad = 0, sink = 0;
  for (int it = 0; it < 6; it++) {
    double t0 = omp_get_wtime(), s = 0;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < N; i++) s += a[i];
    double t1 = omp_get_wtime(); sink += s;
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) c[i] = a[i] + 3.0 * b[i];
    double t2 = omp_get_wtime();
    double r = N * 8.0 / (t1 - t0) / 1e9, tr = 3.0 * N * 8.0 / (t2 - t1) / 1e9;
    if (r > best_read) best_read = r;
    if (tr > best_triad) best_triad = tr;
  }
  printf("{\"test\":\"cpu_stream\",\"threads\":%d,\"array_MiB\":%lu,\"read_GBps\":%.2f,\"triad_GBps\":%.2f,\"sink\":%.0f}\n",
         omp_get_max_threads(), N * 8 / 1048576, best_read, best_triad, sink + c[N / 2]);
  return 0;
}
