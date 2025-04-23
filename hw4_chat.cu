#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void twice(float *arr, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    arr[idx] = arr[idx] * 2.0f;
  }
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t total_global_mem = prop.totalGlobalMem;
  printf("Total global memory: %lu bytes\n", total_global_mem);

  size_t num_floats = 2 * total_global_mem / sizeof(float);
  float *data;

  // Unified Memory allocation
  cudaError_t err = cudaMallocManaged(&data, num_floats * sizeof(float));
  if (err != cudaSuccess) {
    printf("cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
    return 1;
  }

  // Initialize array
  for (size_t i = 0; i < num_floats; ++i) {
    data[i] = (float)i;
  }
  cudaDeviceSynchronize();
  // Launch kernel
  int block_size = 1024;
  int grid_size = (num_floats + block_size - 1) / block_size;

  twice<<<grid_size, block_size>>>(data, num_floats);
  cudaDeviceSynchronize();

  // Check results (only first 10 to keep it fast)
  for (int i = 0; i < 10; ++i) {
    if (data[i] != i * 2.0f) {
      printf("Mismatch at %d: got %f, expected %f\n", i, data[i], i * 2.0f);
      return 1;
    }
  }
  printf("Check passed!\n");

  cudaFree(data);
  return 0;
}
