#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ReducSumBasic(int *src, int *res, int N) {
  // basic implementation of reduction sum
  // this kernel will change the src array
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int offset = N / 2; offset > 0; offset /= 2) {
    if (tid < N and tid + offset < N) {
      src[tid] += src[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *res = src[0];
  }
}
