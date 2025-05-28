#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void ReducSumGlobalMem(int *src, int *res, int N) {
  // basic implementation of reduction sum
  // this kernel will change the src array
  int start = blockDim.x * blockIdx.x;
  int end = min(start + blockDim.x, N);
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset && idx + offset < end) {
      src[idx] += src[idx + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    res[blockIdx.x] = src[idx];
  }
}

unsigned int nextPowerOfTwo(int x) {
  x = (unsigned)x;
  if (x == 0)
    return 1;  // 处理0的特殊情况
  x--;         // 处理x本身为2的幂的情况
  x |= x >> 1; // 把高位影响低位，逐步扩散
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++; // 加1得到结果
  return x;
}

int main() {
  // host memory
  int N = 1e8;
  int res = 0;
  int *h_data; // original data
  h_data = (int *)std::malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    h_data[i] = 1;
  }
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

  int *d_final_res;
  cudaMalloc((void **)&d_final_res, sizeof(int));
  ReducSumGlobalMem<<<grid_size_1, 1024>>>(d_data, d_block_res_1, N);
  ReducSumGlobalMem<<<grid_size_2, 1024>>>(d_block_res_1, d_block_res_2,
                                           full_grid_size_1);
  ReducSumGlobalMem<<<1, grid_size_2>>>(d_block_res_2, d_final_res,
                                        full_grid_size_2);
  cudaDeviceSynchronize();
  cudaMemcpy(&res, d_final_res, sizeof(int), cudaMemcpyDeviceToHost);
  printf("\n");
  printf("final sum : %d\n", res);
  return 0;
}
