#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void ReducSumGlobalMem(int *src, int *res, int N);
__global__ void ReducSumSharedMem(int *src, int *res, int N);
__global__ void ReducSumWarpShfl(int *src, int *res, int N);

#endif
