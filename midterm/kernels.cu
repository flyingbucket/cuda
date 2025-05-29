#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
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

__global__ void ReducSumWarpShfl(int *src, int *res, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int lane = threadIdx.x % 32; // 当前线程在 warp 中的位置

  // 每个线程加载自己的数据（防止越界）
  int val = (idx < N) ? src[idx] : 0;

  // warp 内规约：32 个线程规约为 1 个值
  // 利用 __shfl_down_sync 在 warp 内通信（CUDA 9.0+）
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  extern __shared__ int warp_sum[];

  // 每个 warp 的第一个线程写入结果
  if (lane == 0) {
    int warpId = (threadIdx.x) / 32;
    warp_sum[warpId] = val;
  }
  __syncthreads();
  // 对warp_sum做进一步规约
  if (threadIdx.x < 32) {
    int warpSum = (threadIdx.x < blockDim.x) ? warp_sum[threadIdx.x] : 0;
    for (int offset = 16; offset > 0; offset /= 2) {
      warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
    }
    if (lane == 0) {
      res[blockIdx.x] = warpSum;
    }
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
