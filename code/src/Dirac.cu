#include "Dirac.cuh"

extern __constant__ double yukawa_coupling_gpu;
extern __constant__ double fermion_mass_gpu;
extern __constant__ thrust::complex<double> im_gpu;
extern __constant__ double sq2Kappa_gpu;
extern __constant__ double cutFraction_gpu;
extern __constant__ double WilsonParam_gpu;

template <typename T>
__host__ DiracOP<T>::DiracOP() : M(nullptr) {
    
    my2dArray idx;
    for(int i=0; i<vol; i++){
        idx = flatToVec(i);
        IUP.at[i][0] = 4*vecToFlat(PBC(idx[0]+1, Sizes[0]), idx[1]);
        IUP.at[i][1] = 4*vecToFlat(idx[0], PBC(idx[1]+1, Sizes[1]));
        IDN.at[i][0] = 4*vecToFlat(PBC(idx[0]-1, Sizes[0]), idx[1]);
        IDN.at[i][1] = 4*vecToFlat(idx[0], PBC(idx[1]-1, Sizes[1]));
    }
}

template <typename T>
void DiracOP<T>::applyD(cp<double> *in, cp<double> *out, MatrixType MType){
    int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, applyD_gpu);
	cudaDeviceSynchronize();
	auto dimGrid = dim3(nBlocks, 1, 1);
	auto dimBlock = dim3(nThreads, 1, 1);
	auto useDagger = MType;
	void *args[] = {(void*) &in, (void*) &out, (void*) &useDagger, (void*) &M, (void*) &IUP.at, (void*) &IDN.at};
    cudaLaunchCooperativeKernel((void*) applyD_gpu, dimGrid, dimBlock, args, 0, NULL);
	cudaDeviceSynchronize();
}

__global__ void applyD_gpu(cp<double> *in, cp<double> *out, MatrixType const useDagger, double *M, my2dArray *IUP, my2dArray *IDN){
    auto grid = cg::this_grid();
    for (int i = grid.thread_rank(); i < 4*vol; i += grid.size()) {
		out[i] = 0.0;
	}
	cg::sync(grid);
	applyDiagonal(in, out, useDagger, M);
    cg::sync(grid);
    applyHopping(in, out, useDagger, IUP, IDN);
    cg::sync(grid);
}

__device__ void applyDiagonal(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M){
    auto grid = cg::this_grid();
    
    thrust::complex<double> const two {2.0 * WilsonParam_gpu, 0.0};
    thrust::complex<double> const g = static_cast<thrust::complex<double>> (yukawa_coupling_gpu);
    thrust::complex<double> const mass = static_cast<thrust::complex<double>> (fermion_mass_gpu);
    thrust::complex<double> half {0.5, 0.0};
    
    for (int i = grid.thread_rank(); i < vol; i += grid.size()) {
        if (useDagger == MatrixType::Dagger){
            outVec[4*i]     += (two + mass + g * M[i]) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * M[i]) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * M[i]) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * M[i]) * inVec[4*i+3];
        } else{
            outVec[4*i]     += (two + mass + g * M[i]) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * M[i]) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * M[i]) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * M[i]) * inVec[4*i+3];
        }
    }
}

/*__device__ void applyHopping(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int nt, nx;
    //int IUP[2], IDN[2]; // neighbours up and down gor both directions
    double sgn[2]; // anti-periodic boundary conditions
	cp<double> psisum[2], psidiff[2];
    double r = WilsonParam_gpu;
    
    for (int i = grid.thread_rank(); i < vol; i += grid.size()) {

        auto idx = flatToVec(i);

        nt = idx[0];
        nx = idx[1];
        
        sgn[0] = (nt == (Sizes[0] - 1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        
		if (useDagger == MatrixType::Normal) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 0] - sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 0] 
								- 0.5 * (r * inVec[IUP[i][1] + 0] - inVec[IUP[i][1] + 1]) - 0.5 * (r * inVec[IDN[i][1] + 0] + inVec[IDN[i][1] + 1]); // f1s1
            outVec[4*i + 1] +=  sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 1] + sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 1] 
								- 0.5 * (inVec[IUP[i][1] + 0] - r * inVec[IUP[i][1] + 1]) - 0.5 * (inVec[IDN[i][1] + 0] + r * inVec[IDN[i][1] + 1]); // f1s2
            // flavour 2
            outVec[4*i + 2] +=  -sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 2] - sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 2] 
								- 0.5 * (r * inVec[IUP[i][1] + 2] - inVec[IUP[i][1] + 3]) - 0.5 * (r * inVec[IDN[i][1] + 2] + inVec[IDN[i][1] + 3]); // f1s1
            outVec[4*i + 3] +=  -sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 3] - sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 3] 
								+ 0.5 * (inVec[IUP[i][1] + 2] - r * inVec[IUP[i][1] + 3]) - 0.5 * (inVec[IDN[i][1] + 2] + r * inVec[IDN[i][1] + 3]); // f1s2
        } else if (useDagger == MatrixType::Dagger) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 0] - sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 0] 
								- 0.5 * (r * inVec[IDN[i][1] + 0] - inVec[IDN[i][1] + 1]) - 0.5 * (r * inVec[IUP[i][1] + 0] + inVec[IUP[i][1] + 1]); // f1s1
            outVec[4*i + 1] +=  -sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 1] - sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 1] 
								+ 0.5 * (inVec[IDN[i][1] + 0] - r * inVec[IDN[i][1] + 1]) - 0.5 * (inVec[IUP[i][1] + 0] + r * inVec[IUP[i][1] + 1]); // f1s2
            // flavour 2
            outVec[4*i + 2] +=  -sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 2] - sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 2] 
								- 0.5 * (r * inVec[IDN[i][1] + 2] - inVec[IDN[i][1] + 3]) - 0.5 * (r * inVec[IUP[i][1] + 2] + inVec[IUP[i][1] + 3]); // f1s1
            outVec[4*i + 3] +=  -sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 3] - sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 3] 
								+ 0.5 * (inVec[IDN[i][1] + 2] - r * inVec[IDN[i][1] + 3]) - 0.5 * (inVec[IUP[i][1] + 2] + r * inVec[IUP[i][1] + 3]); // f1s2
        }

        if (useDagger == MatrixType::Normal) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[1] * 0.5 * inVec[IDN[i][0] + 0] + sgn[0] * 0.5 * inVec[IUP[i][0] + 0] 
								        + 0.5 * inVec[IUP[i][1] + 1] -          0.5 * inVec[IDN[i][1] + 1]; // f1s1
            outVec[4*i + 1] +=  -sgn[1] * 0.5 * inVec[IDN[i][0] + 1] + sgn[0] * 0.5 * inVec[IUP[i][0] + 1] 
								        - 0.5 * inVec[IUP[i][1] + 0] +          0.5 * inVec[IDN[i][1] + 0]; // f1s1
            // flavour 2
            outVec[4*i + 2] +=  -sgn[1] * 0.5 * inVec[IDN[i][0] + 2] + sgn[0] * 0.5 * inVec[IUP[i][0] + 2] 
								        + 0.5 * inVec[IUP[i][1] + 3] -          0.5 * inVec[IDN[i][1] + 3]; // f1s1
            outVec[4*i + 3] +=  -sgn[1] * 0.5 * inVec[IDN[i][0] + 3] + sgn[0] * 0.5 * inVec[IUP[i][0] + 3] 
								        - 0.5 * inVec[IUP[i][1] + 2] +          0.5 * inVec[IDN[i][1] + 2]; // f1s1
        } else if (useDagger == MatrixType::Dagger) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[0] * 0.5 * inVec[IUP[i][0] + 0] + sgn[1] * 0.5 * inVec[IDN[i][0] + 0] 
								        + 0.5 * inVec[IDN[i][1] + 1] -          0.5 * inVec[IUP[i][1] + 1]; // f1s1
            outVec[4*i + 1] +=  -sgn[0] * 0.5 * inVec[IUP[i][0] + 1] + sgn[1] * 0.5 * inVec[IDN[i][0] + 1] 
								        - 0.5 * inVec[IDN[i][1] + 0] +          0.5 * inVec[IUP[i][1] + 0]; // f1s1
            // flavour 2
            outVec[4*i + 2] +=  -sgn[0] * 0.5 * inVec[IUP[i][0] + 2] + sgn[1] * 0.5 * inVec[IDN[i][0] + 2] 
								        + 0.5 * inVec[IDN[i][1] + 3] -          0.5 * inVec[IUP[i][1] + 3]; // f1s1
            outVec[4*i + 3] +=  -sgn[0] * 0.5 * inVec[IUP[i][0] + 3] + sgn[1] * 0.5 * inVec[IDN[i][0] + 3] 
								        - 0.5 * inVec[IDN[i][1] + 2] +          0.5 * inVec[IUP[i][1] + 2]; // f1s1
        }
        
    }  

}*/

__device__ void applyHopping(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int nt, nx;
    //int IUP[2], IDN[2]; // neighbours up and down gor both directions
    double sgn[2]; // anti-periodic boundary conditions
	cp<double> psisum[2], psidiff[2];
    double r = WilsonParam_gpu;
    
    for (int i = grid.thread_rank(); i < vol; i += grid.size()) {

        auto idx = flatToVec(i);

        nt = idx[0];
        nx = idx[1];
        
        sgn[0] = (nt == (Sizes[0] - 1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        
		if (useDagger == MatrixType::Normal) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 0] - sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 0] 
								- 0.5 * (r * inVec[IUP[i][1] + 0] - inVec[IUP[i][1] + 1]) - 0.5 * (r * inVec[IDN[i][1] + 0] + inVec[IDN[i][1] + 1]); // f1s1
            outVec[4*i + 1] +=  -sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 1] - sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 1] 
								+ 0.5 * (inVec[IUP[i][1] + 0] - r * inVec[IUP[i][1] + 1]) - 0.5 * (inVec[IDN[i][1] + 0] + r * inVec[IDN[i][1] + 1]); // f1s2
            // flavour 2
            outVec[4*i + 2] +=  -sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 2] - sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 2] 
								- 0.5 * (r * inVec[IUP[i][1] + 2] - inVec[IUP[i][1] + 3]) - 0.5 * (r * inVec[IDN[i][1] + 2] + inVec[IDN[i][1] + 3]); // f1s1
            outVec[4*i + 3] +=  -sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 3] - sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 3] 
								+ 0.5 * (inVec[IUP[i][1] + 2] - r * inVec[IUP[i][1] + 3]) - 0.5 * (inVec[IDN[i][1] + 2] + r * inVec[IDN[i][1] + 3]); // f1s2
        } else if (useDagger == MatrixType::Dagger) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 0] - sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 0] 
								- 0.5 * (r * inVec[IDN[i][1] + 0] - inVec[IDN[i][1] + 1]) - 0.5 * (r * inVec[IUP[i][1] + 0] + inVec[IUP[i][1] + 1]); // f1s1
            outVec[4*i + 1] +=  -sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 1] - sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 1] 
								+ 0.5 * (inVec[IDN[i][1] + 0] - r * inVec[IDN[i][1] + 1]) - 0.5 * (inVec[IUP[i][1] + 0] + r * inVec[IUP[i][1] + 1]); // f1s2
            // flavour 2
            outVec[4*i + 2] +=  -sgn[0] * 0.5 * (r + 1.0) * inVec[IUP[i][0] + 2] - sgn[1] * 0.5 * (r - 1.0) * inVec[IDN[i][0] + 2] 
								- 0.5 * (r * inVec[IDN[i][1] + 2] - inVec[IDN[i][1] + 3]) - 0.5 * (r * inVec[IUP[i][1] + 2] + inVec[IUP[i][1] + 3]); // f1s1
            outVec[4*i + 3] +=  -sgn[0] * 0.5 * (r - 1.0) * inVec[IUP[i][0] + 3] - sgn[1] * 0.5 * (r + 1.0) * inVec[IDN[i][0] + 3] 
								+ 0.5 * (inVec[IDN[i][1] + 2] - r * inVec[IDN[i][1] + 3]) - 0.5 * (inVec[IUP[i][1] + 2] + r * inVec[IUP[i][1] + 3]); // f1s2
        }
        
    }  

}

/*__device__ void applyDiagonal(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M){
    auto grid = cg::this_grid();
    
    thrust::complex<double> const two {2.0, 0.0};
    thrust::complex<double> const g = static_cast<thrust::complex<double>> (yukawa_coupling_gpu);
    thrust::complex<double> const mass = static_cast<thrust::complex<double>> (fermion_mass_gpu);
    thrust::complex<double> half {0.5, 0.0};
    
    for (int i = grid.thread_rank(); i < vol; i += grid.size()) {
        if (useDagger == MatrixType::Dagger){
            outVec[4*i]     += (two + mass + g * M[i]) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * M[i]) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * M[i]) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * M[i]) * inVec[4*i+3];
        } else{
            outVec[4*i]     += (two + mass + g * M[i]) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * M[i]) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * M[i]) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * M[i]) * inVec[4*i+3];
        }
    }
}

__device__ void applyHopping(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int nt, nx;
    //int IUP[2], IDN[2]; // neighbours up and down gor both directions
    double sgn[2]; // anti-periodic boundary conditions
	cp<double> psisum[2], psidiff[2];
    for (int i = grid.thread_rank(); i < vol; i += grid.size()) {

        auto idx = flatToVec(i);

        nt = idx[0];
        nx = idx[1];
        
        sgn[0] = (nt == (Sizes[0] - 1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        
		if (useDagger == MatrixType::Normal) {
            //flavour 1
			outVec[4*i + 0] +=  -sgn[1] * inVec[IDN[i][0] + 0] - 0.5 * (inVec[IUP[i][1] + 0] - inVec[IUP[i][1] + 1]) - 0.5 * (inVec[IDN[i][1] + 0] + inVec[IDN[i][1] + 1]); // f1s1
            outVec[4*i + 1] +=  -sgn[0] * inVec[IUP[i][0] + 1] + 0.5 * (inVec[IUP[i][1] + 0] - inVec[IUP[i][1] + 1]) - 0.5 * (inVec[IDN[i][1] + 0] + inVec[IDN[i][1] + 1]); // f1s2
            // flavour 2
            outVec[4*i + 2] +=  -sgn[1] * inVec[IDN[i][0] + 2] - 0.5 * (inVec[IUP[i][1] + 2] - inVec[IUP[i][1] + 3]) - 0.5 * (inVec[IDN[i][1] + 2] + inVec[IDN[i][1] + 3]); // f2s1
            outVec[4*i + 3] +=  -sgn[0] * inVec[IUP[i][0] + 3] + 0.5 * (inVec[IUP[i][1] + 2] - inVec[IUP[i][1] + 3]) - 0.5 * (inVec[IDN[i][1] + 2] + inVec[IDN[i][1] + 3]); // f2s2

        } else if (useDagger == MatrixType::Dagger) {
            // flavour 1
            outVec[4*i + 0] +=  -sgn[0] * inVec[IUP[i][0] + 0] - 0.5 * (inVec[IDN[i][1] + 0] - inVec[IDN[i][1] + 1]) - 0.5 * (inVec[IUP[i][1] + 0] + inVec[IUP[i][1] + 1]); // f1s1
            outVec[4*i + 1] +=  -sgn[1] * inVec[IDN[i][0] + 1] + 0.5 * (inVec[IDN[i][1] + 0] - inVec[IDN[i][1] + 1]) - 0.5 * (inVec[IUP[i][1] + 0] + inVec[IUP[i][1] + 1]); // f1s2
            // flavour 2
            outVec[4*i + 2] +=  -sgn[0] * inVec[IUP[i][0] + 2] - 0.5 * (inVec[IDN[i][1] + 2] - inVec[IDN[i][1] + 3]) - 0.5 * (inVec[IUP[i][1] + 2] + inVec[IUP[i][1] + 3]); // f2s1
            outVec[4*i + 3] +=  -sgn[1] * inVec[IDN[i][0] + 3] + 0.5 * (inVec[IDN[i][1] + 2] - inVec[IDN[i][1] + 3]) - 0.5 * (inVec[IUP[i][1] + 2] + inVec[IUP[i][1] + 3]); // f2s2
        }
        
    }  

}*/



/*__device__ void D_eo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    int nt;

    double sgn[2];

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {

{
    int n = i;
    int alpha = 0;
    if (n >= Sizes[0]*Sizes[1]/2) {
        alpha = 1;
        n -= Sizes[0]*Sizes[1]/2;
    }
    idx[0] = n / (Sizes[1]/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Sizes[1]/2))) + (1-alpha);
    else idx[1] = 2*((n % (Sizes[1]/2))) + alpha; 
}
    nt = idx[0];
    sgn[0] = (nt == (Sizes[0]-1)) ? -1.0 : 1.0;
    sgn[1] = (nt == 0) ? -1.0 : 1.0;

    thrust::complex<double> psisum[2], psidiff[2];

    double constexpr half {0.5};
    
    if (useDagger == MatrixType::Dagger) {
        
        psisum[0]  = inVec[4*IUP[i][1] + 0] + inVec[4*IUP[i][1] + 1];
        psisum[1]  = inVec[4*IUP[i][1] + 2] + inVec[4*IUP[i][1] + 3];
        psidiff[0] = inVec[4*IDN[i][1] + 0] - inVec[4*IDN[i][1] + 1];
        psidiff[1] = inVec[4*IDN[i][1] + 2] - inVec[4*IDN[i][1] + 3];

        outVec[4*i + 0] -=  sgn[0] * inVec[4*IUP[i][0] + 0] + half*psidiff[0] + half*psisum[0];
        outVec[4*i + 2] -=  sgn[0] * inVec[4*IUP[i][0] + 2] + half*psidiff[1] + half*psisum[1];
        outVec[4*i + 1] -=  sgn[1] * inVec[4*IDN[i][0] + 1] - half*psidiff[0] + half*psisum[0];
        outVec[4*i + 3] -=  sgn[1] * inVec[4*IDN[i][0] + 3] - half*psidiff[1] + half*psisum[1];

    } else {

        
        psisum[0]  = inVec[4*IDN[i][1] + 0] + inVec[4*IDN[i][1] + 1];
        psisum[1]  = inVec[4*IDN[i][1] + 2] + inVec[4*IDN[i][1] + 3];
        psidiff[0] = inVec[4*IUP[i][1] + 0] - inVec[4*IUP[i][1] + 1];
        psidiff[1] = inVec[4*IUP[i][1] + 2] - inVec[4*IUP[i][1] + 3];

        outVec[4*i + 0] -=  sgn[1] * inVec[4*IDN[i][0] + 0] + half*psisum[0] + half*psidiff[0];
        outVec[4*i + 2] -=  sgn[1] * inVec[4*IDN[i][0] + 2] + half*psisum[1] + half*psidiff[1];
        outVec[4*i + 1] -=  sgn[0] * inVec[4*IUP[i][0] + 1] + half*psisum[0] - half*psidiff[0];
        outVec[4*i + 3] -=  sgn[0] * inVec[4*IUP[i][0] + 3] + half*psisum[1] - half*psidiff[1];

    }

    }                            

}


__device__ void D_oe(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    double const half {0.5};

    double sgn[2];
    int nt;

    for (int j = grid.thread_rank(); j < vol/2; j += grid.size()) {
    int i = j + vol/2;
    
{
    int n = i;
    int alpha = 0;
    if (n >= Sizes[0]*Sizes[1]/2) {
        alpha = 1;
        n -= Sizes[0]*Sizes[1]/2;
    }
    idx[0] = n / (Sizes[1]/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Sizes[1]/2))) + (1-alpha);
    else idx[1] = 2*((n % (Sizes[1]/2))) + alpha; 
}

        nt = idx[0];

        sgn[0] = (nt == (Sizes[0]-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        thrust::complex<double> psisum[2], psidiff[2];
        
		i = j;
        if (useDagger == MatrixType::Dagger) {

            psisum[0]  = inVec[4*IUP[i+vol/2][1] + 0] + inVec[4*IUP[i+vol/2][1] + 1];
            psisum[1]  = inVec[4*IUP[i+vol/2][1] + 2] + inVec[4*IUP[i+vol/2][1] + 3];
            psidiff[0] = inVec[4*IDN[i+vol/2][1] + 0] - inVec[4*IDN[i+vol/2][1] + 1];
            psidiff[1] = inVec[4*IDN[i+vol/2][1] + 2] - inVec[4*IDN[i+vol/2][1] + 3];

            outVec[4*(i+vol/2) + 0] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 0] + half*(psidiff[0] + psisum[0]);
            outVec[4*(i+vol/2) + 2] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 2] + half*(psidiff[1] + psisum[1]);
            outVec[4*(i+vol/2) + 1] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 1] - half*(psidiff[0] - psisum[0]);
            outVec[4*(i+vol/2) + 3] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 3] - half*(psidiff[1] - psisum[1]);

        } else {

            psisum[0]  = inVec[4*IDN[i+vol/2][1] + 0] + inVec[4*IDN[i+vol/2][1] + 1];
            psisum[1]  = inVec[4*IDN[i+vol/2][1] + 2] + inVec[4*IDN[i+vol/2][1] + 3];
            psidiff[0] = inVec[4*IUP[i+vol/2][1] + 0] - inVec[4*IUP[i+vol/2][1] + 1];
            psidiff[1] = inVec[4*IUP[i+vol/2][1] + 2] - inVec[4*IUP[i+vol/2][1] + 3];

            outVec[4*(i+vol/2) + 0] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 0] + half*psisum[0] + half*psidiff[0];
            outVec[4*(i+vol/2) + 2] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 2] + half*psisum[1] + half*psidiff[1];
            outVec[4*(i+vol/2) + 1] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 1] + half*psisum[0] - half*psidiff[0];
            outVec[4*(i+vol/2) + 3] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 3] + half*psisum[1] - half*psidiff[1];


        }
    }
                                            
}*/


//template<> void D_oo<double>(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N);

template class DiracOP<double>;

