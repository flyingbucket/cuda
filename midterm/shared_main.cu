#include "kernels.cuh"
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

int main() {
  printf("--- shared memory version ---");
  // host memory
  int N = 1e8;
  int res_shared_mem = 0;

  int *h_data; // original data
  h_data = (int *)std::malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    h_data[i] = 1;
  }

  // prepare paramters
  int grid_size_1 = (N + 1023) / 1024;

  int full_grid_size_1 = nextPowerOfTwo(grid_size_1);
  int grid_size_2 = (full_grid_size_1 + 1023) / 1024;

  int full_grid_size_2 = nextPowerOfTwo(grid_size_2);
  printf("grid_size_1: %d\nfull_grid_size_1: %d\n", grid_size_1,
         full_grid_size_1);
  printf("grid_size_2: %d\nfull_grid_size_2: %d\n", grid_size_2,
         full_grid_size_2);

  // device memory
  int *d_data; // original data
  cudaMalloc((void **)&d_data, sizeof(int) * N);
  cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);

  int *d_block_res_1;
  int *d_block_res_2; // to store sum of ecah block on device
  cudaMalloc((void **)&d_block_res_1, sizeof(int) * full_grid_size_1);
  cudaMemset(d_block_res_1, 0, sizeof(int) * full_grid_size_1);
  cudaMalloc((void **)&d_block_res_2, sizeof(int) * full_grid_size_2);
  cudaMemset(d_block_res_2, 0, sizeof(int) * full_grid_size_2);

  int *d_final_res_global_mem;
  cudaMalloc((void **)&d_final_res_global_mem, sizeof(int));

  int *d_final_res_shared_mem;
  cudaMalloc((void **)&d_final_res_shared_mem, sizeof(int));

  // launch kernel
  // shared memory version
  ReducSumSharedMem<<<grid_size_1, 1024>>>(d_data, d_block_res_1, N);
  ReducSumSharedMem<<<grid_size_2, 1024>>>(d_block_res_1, d_block_res_2,
                                           full_grid_size_1);
  ReducSumSharedMem<<<1, grid_size_2>>>(d_block_res_2, d_final_res_shared_mem,
                                        full_grid_size_2);
  cudaDeviceSynchronize();
  cudaMemcpy(&res_shared_mem, d_final_res_shared_mem, sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("\n");
  printf("final sum on shared memory : %d\n", res_shared_mem);
  return 0;
}
