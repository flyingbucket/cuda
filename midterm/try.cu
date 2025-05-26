#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void ReducSumGlobalMem(int *src, int *res, int N) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset && idx + offset < N) {
      src[idx] += src[idx + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    res[blockIdx.x] = src[blockIdx.x * blockDim.x];
    printf("%d ", res[blockIdx.x]);
  }
}

__global__ void FinalSum(int *block_sum, int *res, int N) {
  int tid = threadIdx.x;

  for (int offset = N / 2; offset > 0; offset >>= 1) {
    if (tid < offset && tid + offset < N) {
      block_sum[tid] += block_sum[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *res = block_sum[0];
  }
}

int main() {
  int N = 1e4;
  int *h_data = (int *)std::malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    h_data[i] = i % 10;
  }

  int grid_size = N / 1024 + 1;
  int *h_block_res = (int *)std::malloc(sizeof(int) * grid_size);
  int h_final_res = 0;

  // device memory
  int *d_data;
  int *d_block_res;
  int *d_final_res;

  cudaMalloc((void **)&d_data, sizeof(int) * N);
  cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_block_res, sizeof(int) * grid_size);
  cudaMalloc((void **)&d_final_res, sizeof(int));

  ReducSumGlobalMem<<<grid_size, 1024>>>(d_data, d_block_res, N);
  cudaDeviceSynchronize();

  FinalSum<<<1, grid_size>>>(d_block_res, d_final_res, grid_size);
  cudaDeviceSynchronize();

  cudaMemcpy(h_block_res, d_block_res, sizeof(int) * grid_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_final_res, d_final_res, sizeof(int), cudaMemcpyDeviceToHost);

  printf("\nBlock partial sums:\n");
  for (int i = 0; i < grid_size && i < 10; i++) {
    printf("%d ", h_block_res[i]);
  }
  printf("\nFinal sum: %d\n", h_final_res);

  // clean up
  free(h_data);
  free(h_block_res);
  cudaFree(d_data);
  cudaFree(d_block_res);
  cudaFree(d_final_res);

  return 0;
}
