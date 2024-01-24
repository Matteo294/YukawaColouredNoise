#include "reductions.cuh"

extern __constant__ myType epsBar;
extern __constant__ myType m2;
extern __constant__ myType lambda;

namespace cg = cooperative_groups;

// TODO: maybe try again to replace these with STL's ones?
__device__ inline double myMax(double a, double b) { return (a > b ? a : b); }
__device__ inline double myAbs(double a) { return (a > 0 ? a : -a); }

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(myMax(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void gpuMaxAbsReduce(myType *vecA, myType *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	extern __shared__ myType tmp[];

	myType temp_max = 0.0;
	// since this is supposed to be a global maximum over all field components and spacetime
	// indices, we can simply iterate the entire field array
	for (int i = grid.thread_rank(); i < size * nVectorComponents; i += grid.size()) {
		temp_max = myMax(myAbs(vecA[i]), temp_max);
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	temp_max = cg::reduce(tile32, temp_max, cg::greater<myType>());

	if (tile32.thread_rank() == 0) {
		tmp[tile32.meta_group_rank()] = temp_max;
	}

	// for some reason, if I synchronise with cta the cluster gives me wrong results for
	// comp > 1 when using "large" lattices (> 64x64x4)
	cg::sync(grid);

	if (tile32.meta_group_rank() == 0) {
		temp_max = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
		temp_max = cg::reduce(tile32, temp_max, cg::greater<myType>());

		if (tile32.thread_rank() == 0) {
			atomicMax(result, temp_max);
		}
	}
	//if (abs(*result) < 1e-4) printf("res %f\n", *result);
}

__global__ void gpuTimeSlices(myType *vecA, myType *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	extern __shared__ myType tmp[];

	for (int comp = 0; comp < nVectorComponents; ++comp) {
		for (int tt = 0; tt < nTimeSlices; ++tt) {
			// TODO: is this bad? It seems to work. I can also set
			// it to zero outside the kernel, but that looks ugly
			result[comp * nTimeSlices + tt] = 0.0;

			myType temp_sum = 0.0;
			for (int i = grid.thread_rank(); i < SpatialVolume; i += grid.size()) {
				temp_sum += vecA[i + comp * size + tt * SpatialVolume];
			}

			cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

			temp_sum = cg::reduce(tile32, temp_sum, cg::plus<myType>());

			if (tile32.thread_rank() == 0) {
				tmp[tile32.meta_group_rank()] = temp_sum;
			}

			// for some reason, if I synchronise with cta the cluster gives me wrong results for
			// comp > 1 when using "large" lattices (> 64x64x4)
			cg::sync(grid);

			if (tile32.meta_group_rank() == 0) {
				temp_sum = tile32.thread_rank() < tile32.meta_group_size()
					? tmp[tile32.thread_rank()] : 0.0;
				temp_sum = cg::reduce(tile32, temp_sum, cg::plus<myType>());

				if (tile32.thread_rank() == 0) {
					atomicAdd(result+(comp*nTimeSlices + tt), temp_sum);
				}
			}
		}
	}
}

__global__ void gpuMagnetisation(myType *vecA, myType *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	extern __shared__ myType tmp[];

	for (int comp = 0; comp < nVectorComponents; ++comp) {
		result[comp] = 0.0;		// TODO: is this bad? It seems to work. I can also set
								// it to zero outside the kernel, but that looks ugly
		myType temp_sum = 0.0;
		for (int i = grid.thread_rank(); i < size; i += grid.size()) {
			temp_sum += vecA[i + comp * size];
		}

		cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

		temp_sum = cg::reduce(tile32, temp_sum, cg::plus<myType>());

		if (tile32.thread_rank() == 0) {
			tmp[tile32.meta_group_rank()] = temp_sum;
		}

		// for some reason, if I synchronise with cta the cluster gives me wrong results for
		// comp > 1 when using "large" lattices (> 64x64x4)
		cg::sync(grid);

		if (tile32.meta_group_rank() == 0) {
			temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
			temp_sum = cg::reduce(tile32, temp_sum, cg::plus<myType>());

			if (tile32.thread_rank() == 0) {
				atomicAdd(result+comp, temp_sum);
			}
		}
	}
}


// -------------------------------------------------------------------------------------------

__device__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	extern __shared__ thrust::complex<double> tmp2[];

	thrust::complex<double> temp_sum = 0.0;
	for (int i = grid.thread_rank(); i < size; i += grid.size()) {
		temp_sum += conj(vecA[i]) * vecB[i];
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

	if (tile32.thread_rank() == 0) {
		tmp2[tile32.meta_group_rank()] = temp_sum;
	}

	cg::sync(cta);

	if (tile32.meta_group_rank() == 0) {
		temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp2[tile32.thread_rank()] : 0.0;
		temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

		if (tile32.thread_rank() == 0) {
		atomicAdd(reinterpret_cast<double*>(result), temp_sum.real());
		atomicAdd(reinterpret_cast<double*>(result)+1, temp_sum.imag());
		}
	}
}

__global__ void gpuTimeSlices_spinors(cp<double> *vecA, double *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	
	for (int i = grid.thread_rank(); i < Sizes[0]; i += grid.size()) {
		
		result[i] = 0.0;

		for (int j = 0; j < Sizes[1]; j++) {
			for (int k=0; k<4; k++){
				result[i] += vecA[4*(i*Sizes[1] + j) + k].real();
			}
		}

	}
}



__global__ void setZero_kernel(cp<double> *v, int const vol){
	cg::grid_group grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()) v[i] = 0.0;
}

__global__ void gpuTraces(double *vec, double *result, int const vol) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	extern __shared__ double tmpTr[];

    //result[j] = 0.0;		// TODO: is this bad? It seems to work. I can also set
                            // it to zero outside the kernel, but that looks ugly
    double temp_sum = 0.0;
    for (int i = grid.thread_rank(); i < vol; i += grid.size()) {
        temp_sum += vec[i];
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
        tmpTr[tile32.meta_group_rank()] = temp_sum;
    }

    // for some reason, if I synchronise with cta the cluster gives me wrong results for
    // comp > 1 when using "large" lattices (> 64x64x4)
    cg::sync(grid);

    if (tile32.meta_group_rank() == 0) {
        temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmpTr[tile32.thread_rank()] : 0.0;
        temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

        if (tile32.thread_rank() == 0) {
            atomicAdd(result, temp_sum);
        }
    }

}
