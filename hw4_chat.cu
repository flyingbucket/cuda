#include <cuda_runtime.h>
#include <stdio.h>

__global__ void twice(float *arr, size_t offset, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    arr[offset + idx] *= 2.0f;
  }
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t total_global_mem = prop.totalGlobalMem;
  printf("Total global memory: %lu bytes\n", total_global_mem);

  size_t total_floats = 2 * total_global_mem / sizeof(float);
  printf("Total float count: %lu\n", total_floats);

  float *data;
  cudaMallocManaged(&data, total_floats * sizeof(float));

  for (size_t i = 0; i < total_floats; ++i) {
    data[i] = (float)i;
  }

  const size_t chunk_size = 100 * 1024 * 1024;
  int block_size = 1024;
  for (size_t offset = 0; offset < total_floats; offset += chunk_size) {
    size_t current_chunk = (offset + chunk_size < total_floats)
                               ? chunk_size
                               : total_floats - offset;
    int grid_size = (current_chunk + block_size - 1) / block_size;
    printf("Launching kernel for offset=%lu, size=%lu, grid=%d\n", offset,
           current_chunk, grid_size);
    twice<<<grid_size, block_size>>>(data, offset, current_chunk);
    cudaDeviceSynchronize();
  }

  // 验证
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
