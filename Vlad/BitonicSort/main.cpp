#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "Implementations/CudaCompute.h"
#include "Implementations/GLSLCompute.h"

using namespace std;
using namespace std::chrono;

int* ReadArr(const char* path, int&size) {
	ifstream fin(path);
	fin >> size;

	int* arr = new int[size];
	for (int i = 0; i < size; i++) {
		fin >> arr[i];
	}

	fin.close();

	return arr;
}

int main() {
	cout << "Reading test file...\n";
	int size;
	int* arr = ReadArr("C:/Dalv/School/University/Classes/Semestrul2/SD/BitonicSort/Tests/shuffle2pow10.in", size);
	int* arrCopy = ReadArr("C:/Dalv/School/University/Classes/Semestrul2/SD/BitonicSort/Tests/shuffle2pow10.in", size);
	cout << "Read " << size << " numbers\n\n";

	GLSLCompute* glslCompute = new GLSLCompute(1920, 1080);
	glslCompute->AnimSort(arr, size);

	delete[] arr;
	delete[] arrCopy;
	return 0;

	cout << "Running STL sort...";
	auto start = high_resolution_clock::now();
	sort(arrCopy, arrCopy + size);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	cout << " STL Sort finished in " << duration.count() << " milliseconds\n\n";


	cout << "Running GPU sort...\n";
	// Launch CUDA kernel
	LaunchCudaSortKernel(arr, size);

	cout << "GPU sort finished\n";
	for (int i = 0; i < size; i++) {
		if(arr[i] != arrCopy[i]) {
			cerr << "\nElement " << i << " was not sorted properly\n";
			break;
		}
		// cout << arr[i] << " ";
	}

	delete[] arr;
	delete[] arrCopy;
	return 0;
}
