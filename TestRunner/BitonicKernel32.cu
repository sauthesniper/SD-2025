// bitonic_kernel.cu

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

__global__ void BitonicSort(int* v, int size, int stage, int pass) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	size /= 2;

	if(i >= size) return;

	int group = i / pass;
	int dir = i / (stage / 2) % 2;

	int li = i + group * pass;
	int ri = li + pass;

	int aux = li;
	// Math for if(dir == 0) swap(li, ri)
	aux = ri * dir + li * (1 - dir);
	ri = li * dir + ri * (1 - dir);
	li = aux;

	int minval = min(v[li], v[ri]);
	int maxval = max(v[li], v[ri]);
	v[li] = minval;
	v[ri] = maxval;
}

__global__ void BitonicSortSharedMemory(int* v, int size, int stage, int pass) {
	// Thread group memory; Avoids reading from global memory
	extern __shared__ int sharedMem[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	size /= 2;

	if(i >= size) return;

	// Load data into shared memory
	sharedMem[tid] = v[i];
	__syncthreads();

	int group = i / pass;
	int dir = i / (stage / 2) % 2;

	int li = i + group * pass;
	int ri = li + pass;

	bool swap = (dir == 0) ? (sharedMem[li] > sharedMem[ri]) : (sharedMem[li] < sharedMem[ri]);
	if (swap) {
		int aux = sharedMem[li];
		sharedMem[li] = sharedMem[ri];
		sharedMem[ri] = aux;
	}

	__syncthreads();

	v[i] = sharedMem[tid];
}

void LaunchCudaSortKernel(int* cpuData, int size) {
	int* gpuData;

	// Warm up CUDA drivers
	cudaMalloc(0, 0);

	auto start = high_resolution_clock::now();
	// Allocate on the gpu
	cudaMalloc((void**)&gpuData, size * sizeof(int));
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	cout << "GPU memory allocation took " << duration.count() << "ms\n";

	start = high_resolution_clock::now();
	cudaMemcpy(gpuData, cpuData, size * sizeof(int), cudaMemcpyHostToDevice);
	end = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(end - start);
	cout << "Memory transfer to GPU took " << duration.count() << "ms\n";

	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;


	start = high_resolution_clock::now();
	for(int s = 2; s <= size; s *= 2) {
		for(int p = s / 2; p > 0; p /= 2) {
			BitonicSort<<<numBlocks, blockSize>>>(gpuData, size, s, p);

			// Wait for the gpu to finish processing
			cudaDeviceSynchronize();
		}
	}
	end = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(end - start);
	cout << "Sorting on GPU took " << duration.count() << "ms\n";

	start = high_resolution_clock::now();
	// Copy data from gpu
	cudaMemcpy(cpuData, gpuData, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpuData);
	end = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(end - start);
	cout << "Memory transfer to CPU took " << duration.count() << "ms\n";
}


int* ReadNumbersFromFile(const char* filePath, int* size){
	FILE* file = fopen(filePath, "r");
	if(file == NULL){
		perror("Errror opening file\n");
		fclose(file);
		return NULL;
	}

	if(fscanf(file, "%d", size)!=1){
		perror("Error reading size from file\n");
		fclose(file);
		return NULL;
	}

	int* arr = (int*)malloc((*size) * sizeof(int));
	if(arr == NULL){
		perror("Memory allocation failed\n");
		fclose(file);
		return NULL;
	}

	for(int i = 0; i < *size; i++){
		if(fscanf(file, "%d", &arr[i]) != 1){
			perror("Error reading numbers from file\n");
			free(arr);
			fclose(file);
			return NULL;
		}
	}

	fclose(file);
	return arr;
}
int main(int argc, char* argv[]){
	if(argc < 2){
		fprintf(stderr, "Usage: %s <filename0> [filename1...]\n", argv[0]);
		return 0;
	}
	for(int i = 1; i < argc; i++) {
		const char* filePath = argv[i];
		int size;
		int* arr = ReadNumbersFromFile(filePath, &size);

		if(arr == NULL) continue;

		printf("Running test file %s...\n", filePath);

		LaunchCudaSortKernel(arr, size);

		/*for(int j = 0; j < size; j++){
			printf("%i ", arr[j]);
		}
		printf("\n\n");*/

		free(arr);
	}
	return 0;
}

