#include <cuda_runtime.h>
#include <stdio.h>

__global__ void printInfo() {
  int threadId = threadIdx.x;
  const char *student_id = "2311067";

  printf("Student ID: %s, Thread ID: %d\n", student_id, threadId);
}
int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  printf("%d devices fount ", device_count);

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
      printf("cudaGetDeviceProperties failed for device %d\n", i);
      continue;
    }

    printf("Device %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %lu MB\n",
           prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared Memory per Block: %.2f KB\n",
           prop.sharedMemPerBlock / 1024.0);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max Threads per Multiprocessor: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("------------------------------------\n");
  }
  printInfo<<<1, 16>>>();
  cudaDeviceSynchronize();
  return 0;
}
