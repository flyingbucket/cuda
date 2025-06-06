#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
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

__global__ void ReducSumSharedMem(int *src, int *res, int N) {
  __shared__ int sdata[1024];

  int tid = threadIdx.x;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x; // Global index

  if (gidx < N) {
    sdata[tid] = src[gidx];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    res[blockIdx.x] = sdata[0];
  }
}
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
  // host memory
  int N = 1e8;
  int res_global_mem = 0;
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
  // global memory version
  ReducSumGlobalMem<<<grid_size_1, 1024>>>(d_data, d_block_res_1, N);
  ReducSumGlobalMem<<<grid_size_2, 1024>>>(d_block_res_1, d_block_res_2,
                                           full_grid_size_1);
  ReducSumGlobalMem<<<1, grid_size_2>>>(d_block_res_2, d_final_res_global_mem,
                                        full_grid_size_2);
  cudaDeviceSynchronize();
  cudaMemcpy(&res_global_mem, d_final_res_global_mem, sizeof(int),
             cudaMemcpyDeviceToHost);

  // shared memory version
  // rewrite original data on device (d_data) because the first run will change
  // original data
  cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);
  ReducSumSharedMem<<<grid_size_1, 1024>>>(d_data, d_block_res_1, N);
  ReducSumSharedMem<<<grid_size_2, 1024>>>(d_block_res_1, d_block_res_2,
                                           full_grid_size_1);
  ReducSumSharedMem<<<1, grid_size_2>>>(d_block_res_2, d_final_res_shared_mem,
                                        full_grid_size_2);
  cudaDeviceSynchronize();
  cudaMemcpy(&res_shared_mem, d_final_res_shared_mem, sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("\n");
  printf("final sum on global memory : %d\n", res_global_mem);
  printf("final sum on shared memory : %d\n", res_shared_mem);
  return 0;
}
