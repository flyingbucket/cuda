#include "kernels.cuh"
#include <numeric>
#include <stdio.h>

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

#define BLOCK_SIZE 256

int main() {
  printf("--- shared memory version ---\n");
  printf("block size: %d\n", BLOCK_SIZE);
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
  // shared memory version
  // start recording time
  cudaEvent_t start, stop;
  float elapsd_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  ReducSumSharedMem<<<grid_size, BLOCK_SIZE>>>(d_data, d_device_sum, N);
  cudaDeviceSynchronize();
  cudaMemcpy(device_sum, d_device_sum, sizeof(int) * grid_size,
             cudaMemcpyDeviceToHost);

  res_global_mem = std::accumulate(device_sum, device_sum + grid_size, 0);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsd_time, start, stop);

  printf("elapsed time : %f\n", elapsd_time);
  printf("final sum on global memory : %d\n", res_global_mem);
  return 0;
}
