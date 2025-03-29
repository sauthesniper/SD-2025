#include <cuda_runtime.h>

__global__ void BitonicSort(int* v, int size, int pass, int stage);
__global__ void BitonicSortSharedMemory(int* v, int size, int pass, int stage);

void LaunchCudaSortKernel(int* cpuData, int size);