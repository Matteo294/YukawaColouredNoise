#pragma once

#include <complex>
#include <cufft.h>
#include <iostream>
#include <curand_kernel.h>
#include <cooperative_groups.h>

#ifdef USE_vkFFT
#define VKFFT_BACKEND 1
#include "vkFFT.h"
#endif

#include "params.h"
#include "device_info.h"
#include "managedVector.h"

#ifdef USE_vkFFT
typedef struct {
	CUdevice device;
	CUcontext context;
	uint64_t device_id;//an id of a device, reported by Vulkan device list
} VkGPU;//an example structure containing Vulkan primitives
#endif

class ColouredNoise {
public:
	ColouredNoise(ManagedVector<myType>& noise, int const& N, double const cutFraction,
			int const& seed, KernelLaunchInfo const& kliRun);

	~ColouredNoise();

	void operator()();

	inline ManagedVector<curandState>& GetState() { return devStates; }
	inline ManagedVector<curandState> const& GetState() const { return devStates; }

private:
	KernelLaunchInfo const RNG;
	KernelLaunchInfo const Mask;
	ManagedVector<myType>& noise;
	ManagedVector<curandState> devStates;
	ManagedVector<myType> momentumMask;
	ManagedVector<std::complex<myType>> tmpNoise;
	std::array<void*, 3> kRNG, kMask;
	cufftHandle planF, planR;
#ifdef USE_vkFFT
	VkFFTApplication app;
#endif

};
