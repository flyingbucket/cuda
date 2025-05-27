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
  int end = min(start + blockIdx.x, N);
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
    printf("%d ", res[blockIdx.x]);
  }
}

__global__ void FinalSum(int *block_sum, int *res, int grid_size) {
  int tid = threadIdx.x;
  for (int offset = grid_size / 2; offset > 0; offset /= 2) {
    if (tid + offset < grid_size) {
      block_sum[tid] += block_sum[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *res = block_sum[0];
  }
}

int main() {
  // host memory
  int N = 1e4;
  int res = 0;
  int *h_data; // original data
  h_data = (int *)std::malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    h_data[i] = i % 10;
  }
  int *h_block_res; // to store sum of each block on host memory
  int grid_size = N / 1024 + 1;
  h_block_res = (int *)std::malloc(sizeof(int) * grid_size);

  // device memory
  int *d_data; // original data
  cudaMalloc((void **)&d_data, sizeof(int) * N);
  cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);
  int *d_block_res; // to store sum of ecah block on device
  cudaMalloc((void **)&d_block_res, sizeof(int) * grid_size);
  int *d_final_res;
  cudaMalloc((void **)&d_final_res, sizeof(int));
  ReducSumGlobalMem<<<grid_size, 1024>>>(d_data, d_block_res, N);
  FinalSum<<<1, grid_size>>>(d_block_res, d_final_res, grid_size);
  cudaDeviceSynchronize();
  printf("\n");

  cudaMemcpy(h_block_res, d_block_res, sizeof(int) * grid_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&res, d_final_res, sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < grid_size && i < 10; i++) {
    printf("%d", h_block_res[i]);
    printf(" ");
  }
  printf("\n");
  printf("final sum : %d\n", res);
  return 0;
}
