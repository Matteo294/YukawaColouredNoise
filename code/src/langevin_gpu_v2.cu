#include "langevin_gpu_v2.cuh"

extern __constant__ myType epsBar;
extern __constant__ myType m2;
extern __constant__ myType lambda;

namespace cg = cooperative_groups;

__device__ void PotentialAndMass(myType *ivec, myType *ovec, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	auto const myLambda = lambda / static_cast<myType>(6.0);

	for (int i = grid.thread_rank(); i < size; i += grid.size()) {
		myType interaction = 0.0;
		for (int j = 0; j < nVectorComponents; ++j) {
			auto const tmp = ivec[j * size + i];
			interaction += tmp * tmp;
		}
		for (int j = 0; j < nVectorComponents; ++j) {
			auto const myVal = ivec[j * size + i];
			ovec[j * size + i] += myVal * (m2 + myLambda * interaction);
            //if (j == 0) ovec[j * size + i] -= 0.1; // magnetization sigma
		}
	}
}

__device__ void Evolve(myType *ivec, myType* ovec, myType* noise, int size, myType const eps) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	auto const fact = std::sqrt(2.0 * eps);
	for (int i = grid.thread_rank(); i < size * nVectorComponents; i += grid.size()) {
		ovec[i] -= eps * ivec[i] + fact * noise[i];
	}
}

// TODO: no idea how to do this shared memory business here, or how to make that speed up the computation
// this version does not use I, but it is still here so the function signature is the same
__device__ void gpuSpMV(int *I, int *J, myType *val, int num_rows,
		myType *inputVecX, myType *outputVecY) {

	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {
		int row_elem = i * nElements;//I[i];
		int num_elems_this_row = nElements;//next_row_elem - row_elem;

		// maybe this can be made more efficient?
		for (int comp = 0; comp < nVectorComponents; ++comp) {
			myType output = 0.0;
			for (int j = 0; j < num_elems_this_row; ++j) {
				// J or val arrays - can be put in shared memory
				// as the access is random and reused in next calls of gpuSpMV function.
				output += val[row_elem + j] * inputVecX[comp * num_rows + J[row_elem + j]];
			}

			outputVecY[comp * num_rows + i] -= output;
		}
	}
}

__global__ void Run(myType *eps, myType ExportTime, myType *field,
		myType *drift, myType *noise, int size, int *I, int *J, myType *vals, myType *maxDrift) {

	cg::grid_group grid = cg::this_grid();
	cg::thread_block cta = cg::this_thread_block();
//	myType t = 0.0;
	myType constexpr Kbar = 30.0;
	myType myEps = *eps;
//	while (t < ExportTime) {
		PotentialAndMass(field, drift, size);
//		cg::sync(grid);	// BIG difference between synchronising grid and cta...
		gpuSpMV(I, J, vals, size, field, drift);
		if (threadIdx.x == 0 && blockIdx.x == 0) maxDrift[0] = 0.0;
		cg::sync(grid);
        
		gpuMaxAbsReduce(drift, maxDrift, size);
        cg::sync(grid);
        
        if (threadIdx.x == 0 && blockIdx.x == 0){
            if (maxDrift[0] > 1e5) {
                printf("Max. drift exceeded %f \n", maxDrift[0]);
                maxDrift[0] = 1e5;
            } else if (maxDrift[0] < 1e-5) {
                printf("Min. drift exceeded %f \n", maxDrift[0]);
                maxDrift[0] = 1e-5;
            }
        }
        
        // ----------------------------------------------------------------------------------
        /*if (threadIdx.x == 0 && blockIdx.x == 0){
            if (maxDrift[0] > 1e3) {
                printf("Big drift: %f \n", maxDrift[0]);
                 for(int i=0; i<4*vol; i++) {
                    if (abs(drift[i]) > 1e3) printf ("%d %f \n", i, drift[i]);
                 }
                maxDrift[0] = 1e3;
            } else if (maxDrift[0] < 1e-3) {
                printf("Small drift: %f \n", maxDrift[0]);
                for(int i=0; i<4*vol; i++) {
                    if (abs(drift[i]) < 1e-3) printf ("%d %f \n", i, drift[i]);
                maxDrift[0] = 1e-3;
                }
            }
        }*/
        // ----------------------------------------------------------------------------------
        
		
		myEps = epsBar * Kbar / maxDrift[0];
		Evolve(drift, field, noise, size, myEps);
//		t += *eps;
//		cg::sync(grid);
/*		if (threadIdx.x == 0 && blockIdx.x == 0)*/ *eps = myEps;	// all threads writing?
//	}
}
