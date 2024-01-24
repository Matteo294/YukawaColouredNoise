#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "params.h"

//template <typename T>
struct KernelLaunchInfo {
	dim3 dimGrid, dimBlock;
	int numBlocks, numThreads, numSms, numBlocksPerSm;
	void * const func;

	template <typename T,
          typename = typename std::enable_if_t<std::is_pointer<T>::value>
			  >
	KernelLaunchInfo(T func)
	: func{reinterpret_cast<void*>(func)}
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);

		if (!deviceProp.managedMemory) {
			// This sample requires being run on a device that supports Unified Memory
			fprintf(stderr, "Unified Memory not supported on this device\n");
			exit(-1);
		}

		// This sample requires being run on a device that supports Cooperative Kernel
		// Launch
		if (!deviceProp.cooperativeLaunch) {
			printf(
					"\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
					"Waiving the run\n",
					0);
			exit(-1);
		}

		int constexpr N = SIZE;
		numThreads = 0;
		numBlocks = 0;
		auto const maxBlocks = deviceProp.multiProcessorCount *
			(deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock);

		cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, func);
		numThreads = std::min(numThreads, N * nVectorComponents);
		numBlocks = std::min(numBlocks, maxBlocks);
		numBlocks = std::min(N / numThreads, numBlocks);
		numBlocks = std::max(numBlocks, 1);

		numBlocksPerSm = 0;

		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, func, numThreads,
				numThreads * sizeof(myType));

		numSms = deviceProp.multiProcessorCount;
		if (numBlocks > numBlocksPerSm * numSms) numBlocks = numBlocksPerSm * numSms;
		std::cout << "#numSms = " << numSms << '\n';
		std::cout << "#blocks per SM = " << numBlocksPerSm << '\n';
		std::cout << "#theads = " << numThreads << '\n';
		std::cout << "#blocks = " << numBlocks << '\n';

		dimGrid  = dim3(numBlocks, 1, 1);
		dimBlock = dim3(numThreads, 1, 1);
	}

	inline void Run(void** args, int sMemInBytes = 0) const {
		//TODO: check if func is NULL/nullptr before calling
		checkCuda(cudaLaunchCooperativeKernel(func, dimGrid, dimBlock, args, sMemInBytes, NULL));
	}
};
