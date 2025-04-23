#include <cuda_runtime.h>
#include <stdio.h>

__global__ void twice(float *devicePtr, int N) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    devicePtr[tid] *= 2.0f;
  }
}

int main() {

  // get total global memory of device 0
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned long global_mem = prop.totalGlobalMem;
  printf("total global mem :%lu\n", global_mem);

  float *hostPtr;
  float *devicePtr;
  float *data;
  // destribute host memory
  int N = 2 * global_mem / sizeof(float);
  cudaMallocManaged((void **)&data, N * sizeof(float));
  // cudaHostAlloc((void **)&hostPtr, N * sizeof(float), cudaHostAllocDefault);
  for (int i = 0; i < 2 * global_mem / sizeof(float); i += 1) {
    hostPtr[i] = i;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
  }

  for (int i = 0; i < 10; i += 1) {
    printf("%f ", hostPtr[i]);
  }
  printf("\n");

  // get map hostPtr to devicePtr
  // cudaHostGetDevicePointer((void **)&devicePtr, (void **)&hostPtr, 0);

  // call kernel "twice"
  int grid_size = (N + 1024 - 1) / 1024;
  // twice<<<grid_size, 1024>>>(devicePtr);
  twice<<<grid_size, 1024>>>(data);
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  for (int i = 0; i < 10; i += 1) {
    printf("%f ", hostPtr[i]);
  }
  printf("\n");

  return 0;
}
