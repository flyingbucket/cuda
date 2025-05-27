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

__global__ void FinalSum(int *block_sum, int *res, int grid_size) {
  int tid = threadIdx.x;
  for (int offset = grid_size / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      block_sum[tid] += block_sum[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *res = block_sum[0];
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
  int *h_block_res; // to store sum of each block on host memory
  int grid_size = (N + 1023) / 1024;
  int final_thread_len = nextPowerOfTwo(grid_size);
  printf("grid_size: %d\nfinal_thread_len: %d", grid_size, final_thread_len);
  h_block_res = (int *)std::malloc(sizeof(int) * grid_size);

  // device memory
  int *d_data; // original data
  cudaMalloc((void **)&d_data, sizeof(int) * N);
  cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);
  int *d_block_res; // to store sum of ecah block on device
  cudaMalloc((void **)&d_block_res, sizeof(int) * final_thread_len);
  cudaMemset(d_block_res, 0, sizeof(int) * final_thread_len);
  int *d_final_res;
  cudaMalloc((void **)&d_final_res, sizeof(int));
  ReducSumGlobalMem<<<grid_size, 1024>>>(d_data, d_block_res, N);
  ReducSumGlobalMem<<<1, final_thread_len>>>(d_block_res, d_final_res,
                                             final_thread_len);
  // FinalSum<<<1, final_thread_len>>>(d_block_res, d_final_res,
  // final_thread_len);
  cudaDeviceSynchronize();

  cudaMemcpy(h_block_res, d_block_res, sizeof(int) * grid_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&res, d_final_res, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_block_res, d_block_res, sizeof(int) * grid_size,
             cudaMemcpyDeviceToHost);
  printf("\n");
  printf("final sum : %d\n", res);
  return 0;
}
