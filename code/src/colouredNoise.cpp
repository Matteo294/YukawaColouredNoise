#include "colouredNoise.h"

namespace cg = cooperative_groups;

dimArray IndexToGlobalCoords(int idx) {
	int i = nDim-1;
	std::array<int, nDim> out;
	while (i >= 0) {
		out[i] = idx % Sizes[i];
		idx /= Sizes[i];
		i--;
	}
	return out;
}

double GlobalCoordinateToMomentumSquared(dimArray const coo) {
	double momentumSquared = 0.0;
	double sgn = 1.0;
	if (coo[nDim-1] > std::floor(Sizes[nDim-1] / 2))
		sgn = -1.0;
	for (int n = 0; n < nDim; ++n) {
		int const fullSize = Sizes[n];
		auto const tmp = (coo[n] <= std::floor(fullSize / 2) ? coo[n]
				: coo[n] - fullSize) * 2.0 * M_PI / fullSize;
		momentumSquared += tmp*tmp;
	}
	// make the result *negative* to mark it as a redundant mode that r2c and c2r FFTs don't need
	return momentumSquared * sgn;
}

__global__ void setup_kernel(int const seed, curandState *state) {
	cg::grid_group grid = cg::this_grid();
// TODO: which one is better/safer?
	curand_init(seed, grid.thread_rank(), 0, &state[grid.thread_rank()]);
	//curand_init(1234 + grid.thread_rank(), grid.thread_rank(), 0, &state[grid.thread_rank()]);
}

template <typename T>
struct FakeComplex { T real, imag; };
__global__ void ApplyMask(FakeComplex<myType> *noise, myType *mask, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	auto const tr = grid.thread_rank();

	for (int n = 0; n < nVectorComponents; ++n)
		for (int i = tr; i < nFreq; i += grid.size()) {
			noise[i + n * nFreq].real *= mask[i];
			noise[i + n * nFreq].imag *= mask[i];
		}
}

__global__ void DoRNG(myType* vec, int size, curandState* state) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	auto const tr = grid.thread_rank();

	// apparently it's more efficient to have a copy of the state in local memory
	auto myState = state[tr];
	for (int i = tr; i < SIZE * nVectorComponents; i += grid.size()) {
		//		vec[i] = fact * curand_normal(&myState);
		vec[i] = curand_normal_double(&myState);
	}
	// and then update the device one back
	state[tr] = myState;
}

ColouredNoise::ColouredNoise(ManagedVector<myType>& noise, int const& N, double const cutFraction,
			int const& seed, KernelLaunchInfo const& kliRun)
	: RNG{DoRNG}
	, Mask{ApplyMask}
	, noise{noise}
	, devStates{ManagedVector<curandState>{kliRun.numBlocks * kliRun.numThreads}}
	, momentumMask{ManagedVector<myType>{nFreq}}
	, tmpNoise{ManagedVector<std::complex<myType> >{nFreq * nVectorComponents}}
	, kRNG{{(void*)&(noise.data()), (void*)&N, (void*)&(devStates.data())}}
	, kMask{{(void*)&(tmpNoise.data()), (void*)&(momentumMask.data()), (void*)&N}}
#ifdef USE_vkFFT
	, app{}
#endif
{
	void *kSetup[] = {(void*)&seed, (void*)&(devStates.data())};
	cudaLaunchCooperativeKernel((void *)setup_kernel, kliRun.dimGrid, kliRun.dimBlock,
			kSetup, 0, NULL);
	cudaDeviceSynchronize();

#if defined(USE_cuFFT)
	dimArray geez = Sizes;
	cufftPlanMany(&planF, nDim, geez.data(), NULL, 1, SIZE, NULL, 1, SIZE,
			CUFFT_D2Z, nVectorComponents);
	cufftPlanMany(&planR, nDim, geez.data(), NULL, 1, SIZE, NULL, 1, SIZE,
			CUFFT_Z2D, nVectorComponents);
	std::cout << "#Done initialising cuFFT\n";
#elif defined(USE_vkFFT)
	VkGPU vkGPU = {};

	cudaSetDevice(vkGPU.device_id);
	cuDeviceGet(&vkGPU.device, vkGPU.device_id);
	cuCtxCreate(&vkGPU.context, 0, vkGPU.device);

	uint64_t bufferSize = (uint64_t)sizeof(std::complex<double>) * nFreq * nVectorComponents;
	auto inputBufferSize = sizeof(double) * SIZE * nVectorComponents;

	//zero-initialize configuration + FFT application
	auto configuration = VkFFTConfiguration{};
	//Device management + code submission
	configuration.device = &vkGPU.device;

	configuration.FFTdim = nDim; //FFT dimension, 1D, 2D or 3D
	std::copy(Sizes.crbegin(), Sizes.crend(), configuration.size);
	configuration.performR2C = 1;
	configuration.inverseReturnToInputBuffer = 1;
	// unrealistic code already includes normalisation on the mask, so we can turn this to *false*
	//	configuration.normalize = true;
	configuration.doublePrecision = true;
	configuration.numberBatches = nVectorComponents;

	configuration.isInputFormatted = true;
	configuration.inputBufferStride[0] = (configuration.size[0]);
	configuration.inputBufferStride[1] = (configuration.size[0]) * (configuration.size[1]);
	configuration.inputBufferStride[2] = (configuration.size[0]) * (configuration.size[1]) * (configuration.size[2]);
	// because the coordinates are fed into configuration.size in *reverse* order, the *first*
	// coordinate is the one that has the division by 2 and +1
	configuration.bufferStride[0] = (configuration.size[0] / 2 + 1);
	configuration.bufferStride[1] = (configuration.size[0] / 2 + 1) * (configuration.size[1]);
	configuration.bufferStride[2] = (configuration.size[0] / 2 + 1) * (configuration.size[1]) * (configuration.size[2]);

	configuration.inputBuffer = noise.blob();
	configuration.inputBufferSize = &inputBufferSize;
	configuration.buffer = tmpNoise.blob();
	configuration.bufferSize = &bufferSize;

	initializeVkFFT(&app, configuration);
	std::cout << "#Done initialising vkFFT\n";
#endif

	double const momentumCut = std::sqrt(nDim) * M_PI * cutFraction;
	auto const cutSq = momentumCut * momentumCut;
	int arr = 0;
	for (int pos = 0; pos < SIZE; ++pos) {
		auto const momSq = GlobalCoordinateToMomentumSquared(IndexToGlobalCoords(pos));
		if (momSq >= 0)
			momentumMask[arr++] = (momSq <= cutSq ? 1.0 / N : 0.0);
	}
}

ColouredNoise::~ColouredNoise() {
#if defined(USE_cuFFT)
	cufftDestroy(planF);
	cufftDestroy(planR);
#endif
}

void ColouredNoise::operator()() {
	RNG.Run(kRNG.data());
	cudaDeviceSynchronize();
	if constexpr(nDim <= 3) {
		// FFT kernel
#if defined(USE_cuFFT)
		cufftExecD2Z(planF, noise.data(),reinterpret_cast<cufftDoubleComplex*>(tmpNoise.data()));
#elif defined(USE_vkFFT)
		VkFFTLaunchParams launchParams = {};
		VkFFTAppend(&app, -1, &launchParams);
#endif
		cudaDeviceSynchronize();
		// filter kernel
		Mask.Run(kMask.data());
		cudaDeviceSynchronize();
		// iFFT kernel
#if defined(USE_cuFFT)
		cufftExecZ2D(planR, reinterpret_cast<cufftDoubleComplex*>(tmpNoise.data()),noise.data());
#elif defined(USE_vkFFT)
		VkFFTAppend(&app, +1, &launchParams);
#endif
		cudaDeviceSynchronize();
	}
}
