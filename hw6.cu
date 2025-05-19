#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#define SIZE (1 << 26) // 64M floats, ~256MB

void check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    printf("ERROR %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

int main() {
  int dev0 = 0, dev1 = 1;
  size_t bytes = SIZE * sizeof(float);

  // 查询P2P支持
  int canAccessPeer01 = 0, canAccessPeer10 = 0;
  check(cudaDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1),
        "cudaDeviceCanAccessPeer 0->1");
  check(cudaDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0),
        "cudaDeviceCanAccessPeer 1->0");
  printf("P2P capability: GPU0->GPU1 %d, GPU1->GPU0 %d\n", canAccessPeer01,
         canAccessPeer10);

  // 1. 准备GPU0内存，初始化数据
  check(cudaSetDevice(dev0), "SetDevice 0");
  float *d_data0 = nullptr;
  check(cudaMalloc(&d_data0, bytes), "Malloc GPU0");
  float *ori_data = nullptr;
  ori_data = (float *)malloc(SIZE * sizeof(float));
  for (int i = 0; i < SIZE; i++) {
    ori_data[i] = 1.0f;
  }
  check(cudaMemcpy(d_data0, ori_data, SIZE * sizeof(float),
                   cudaMemcpyHostToDevice),
        "from host copy ori_data to dev0");
  free(ori_data);
  printf("gpu0 data prepared\n");
  // 2. 准备GPU1内存
  check(cudaSetDevice(dev1), "SetDevice 1");
  float *d_data1 = nullptr;
  check(cudaMalloc(&d_data1, bytes), "Malloc GPU1");
  printf("gpu1 memory prepared\n");

  // 计时辅助
  cudaEvent_t start, stop;
  float elapsed_ms;

  // 确保在创建事件前设置成同一个 GPU
  check(cudaSetDevice(dev0), "SetDevice 0 (for event creation)");
  check(cudaEventCreate(&start), "CreateEvent start");
  check(cudaEventCreate(&stop), "CreateEvent stop");
  // 计时辅助

  // --- 1. GPU0->CPU->GPU1 拷贝 ---

  // 准备Host缓冲
  float *h_data = nullptr;
  h_data = (float *)malloc(bytes);

  check(cudaSetDevice(dev0), "SetDevice 0");
  check(cudaEventRecord(start), "EventRecord start");
  // 从GPU0拷贝到Host
  check(cudaMemcpy(h_data, d_data0, bytes, cudaMemcpyDeviceToHost),
        "Memcpy D0->H");
  // 切换到GPU1设备
  check(cudaSetDevice(dev1), "SetDevice 1");
  // 从Host拷贝到GPU1
  check(cudaMemcpy(d_data1, h_data, bytes, cudaMemcpyHostToDevice),
        "Memcpy H->D1");
  check(cudaEventRecord(stop), "EventRecord stop");
  check(cudaEventSynchronize(stop), "EventSynchronize stop");
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "EventElapsedTime");

  printf("GPU0->CPU->GPU1 memcpy time: %.3f ms\n", elapsed_ms);

  free(h_data);

  // --- 2. P2P 拷贝 ---
  // 开启P2P访问
  check(cudaSetDevice(dev0), "SetDevice 0");
  if (canAccessPeer01) {
    cudaDeviceEnablePeerAccess(dev1, 0);
  }
  check(cudaSetDevice(dev1), "SetDevice 1");
  if (canAccessPeer10) {
    cudaDeviceEnablePeerAccess(dev0, 0);
  }

  check(cudaSetDevice(dev0), "SetDevice 0");
  check(cudaEventRecord(start), "EventRecord start");
  // 使用cudaMemcpyPeer从GPU0到GPU1
  check(cudaMemcpyPeer(d_data1, dev1, d_data0, dev0, bytes),
        "MemcpyPeer D0->D1");
  check(cudaEventRecord(stop), "EventRecord stop");
  check(cudaEventSynchronize(stop), "EventSynchronize stop");
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "EventElapsedTime");

  printf("P2P memcpyPeer time: %.3f ms\n", elapsed_ms);

  // 关闭P2P访问
  check(cudaSetDevice(dev0), "SetDevice 0");
  if (canAccessPeer01) {
    cudaDeviceDisablePeerAccess(dev1);
  }
  check(cudaSetDevice(dev1), "SetDevice 1");
  if (canAccessPeer10) {
    cudaDeviceDisablePeerAccess(dev0);
  }

  return 0;
}
