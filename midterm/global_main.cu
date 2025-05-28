#include "kernels.cuh"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <numeric>
#include <stdio.h>

#define BLOCK_SIZE 256

unsigned int nextPowerOfTwo(int x) {
  x = (unsigned)x;
  if (x == 0)
    return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++; // 加1得到结果
  return x;
}

int main() {
  printf("--- global memory version ---\n");
  // host memory
  int N = 1e8;
  int res_global_mem = 0;

  int *h_data; // original data
  h_data = (int *)std::malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    h_data[i] = 1;
  }

  int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int *device_sum;
  device_sum = (int *)malloc(sizeof(int) * grid_size);

  // device memory
  int *d_data; // original data
  cudaMalloc((void **)&d_data, sizeof(int) * N);
  cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);

  int *d_device_sum;
  cudaMalloc((void **)&d_device_sum, sizeof(int) * grid_size);

  // launch kernel
  // global memory version
  ReducSumGlobalMem<<<grid_size, BLOCK_SIZE>>>(d_data, d_device_sum, N);
  cudaDeviceSynchronize();
  cudaMemcpy(device_sum, d_device_sum, sizeof(int) * grid_size,
             cudaMemcpyDeviceToHost);
  res_global_mem = std::accumulate(d_device_sum, d_device_sum + grid_size, 0);
  printf("\n");
  printf("final sum on global memory : %d\n", res_global_mem);
  return 0;
}
