#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void square(int *ori, int *target, int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < size) {
    target[idx] = ori[idx] * ori[idx];
  }
}

int main() {
  // ptr on host
  int size = 1e5;
  int *A = (int *)malloc(size * sizeof(int));
  int *res = (int *)malloc(size * sizeof(int));
  // ptr on device
  int *d_A;
  int *d_B;
  cudaMalloc((int **)&d_A, size * sizeof(int));
  cudaMalloc((int **)&d_B, size * sizeof(int));

  // initialize arr A on host
  for (int i = 0; i < size; i++) {
    A[i] = i % 10 + 1;
  }
  // copy data to device
  cudaMemcpy(d_A, A, size * sizeof(int), cudaMemcpyHostToDevice);

  // cuda events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // start kernel
  int block_size = 1024;
  int grid_size = (size + block_size - 1) / block_size;
  square<<<grid_size, block_size>>>(d_A, d_B, size);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float time_cost;
  cudaEventElapsedTime(&time_cost, start, stop);

  // copy back to host
  cudaMemcpy(res, d_B, size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // print the first 15 numbers of result
  for (int i = 0; i < 15; i++) {
    printf("%d,", res[i]);
  }
  printf("\n");
  printf("Time cost:%f\n", time_cost);
}
