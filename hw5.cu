#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void make_arr(int *arr) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  arr[tid] = tid;
}

__global__ void get_next(int *val) {
  int lane_id = threadIdx.x % 32;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int current_val = -1;
  if (tid < 64)
    current_val = val[tid];
  if (current_val == -1)
    printf("overflow!!!\n");
  int target = (lane_id + 1) % 32;
  int next_val = __shfl_sync(0xffffffff, current_val, target);
  printf("threadIdx.x:%d,next_val:%d\n", threadIdx.x, next_val);
}

__global__ void add_pre(int *val) {
  int sum = 0;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int current_val = -1;
  if (tid < 64)
    current_val = val[tid];
  if (current_val == -1)
    printf("overflow!!!\n");
  if (threadIdx.x == 0)
    sum = val[0];
  else
    sum = current_val + __shfl_up_sync(0xffffffff, current_val, 1);
  printf("threadIdx.x:%d,val:%d,sum:%d\n", threadIdx.x, current_val, sum);
}

int main() {
  int N = 64;
  int *val;
  cudaMalloc((void **)&val, sizeof(int) * N);
  make_arr<<<2, 32>>>(val);
  int *host_val = (int *)malloc(sizeof(int) * N);
  cudaMemcpy(host_val, val, sizeof(int) * N, cudaMemcpyDeviceToHost);
  printf("val:\n");
  for (int i = 0; i < N; i++) {
    printf("%d,", host_val[i]);
  }
  printf("\n");

  // question1:get val of next thread
  printf("get val of next thread\n");
  get_next<<<2, 32>>>(val);
  cudaDeviceSynchronize();

  // question2:add val of previous thread
  printf("\n");
  printf("add val of previous thread\n");
  add_pre<<<2, 32>>>(val);
  cudaDeviceSynchronize();
  return 0;
}
