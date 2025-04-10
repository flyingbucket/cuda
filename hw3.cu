#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sumBlockKernel(float *input, float *output, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int n = tid + blockDim.x * bid;
  __shared__ float blockArr[256];

  if (n < N) {
    blockArr[tid] = input[n];
  }
  __syncthreads();

  // reduction sum
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      blockArr[tid] += blockArr[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = blockArr[0];
    printf("Block:%d sum:%f\n", blockIdx.x, blockArr[0]);
  }
}

int main() {
  int N = 1e6;
  int blockNum = (N + 256 - 1) / 256;

  float *h_arr = (float *)malloc(N * sizeof(float));
  float *h_output = (float *)malloc(blockNum * sizeof(float));

  for (int i = 0; i < N; i++) {
    h_arr[i] = 1;
  }

  float *d_input, *d_output;
  cudaMalloc((void **)&d_input, N * sizeof(float));
  cudaMalloc((void **)&d_output, blockNum * sizeof(float));
  cudaMemcpy(d_input, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

  sumBlockKernel<<<blockNum, 256, 256>>>(d_input, d_output, N);
  printf("\n");
  cudaMemcpy(h_output, d_output, blockNum * sizeof(float),
             cudaMemcpyDeviceToHost);

  printf("print some results from Host\n");
  for (int i = 0; i < 50; i++) {
    printf("%f ", h_output[i]);
  }
  printf("\n");
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_arr);
  free(h_output);

  return 0;
}
