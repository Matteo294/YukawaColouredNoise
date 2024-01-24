#pragma once

#include <cooperative_groups.h>
#include "reductions.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "Dirac.cuh"
#include "reductions.cuh"


template <typename T>
using cp = thrust::complex<T>;

template <typename T>
class DiracOP;

//__global__ void gpuDotProduct(cp<double> *vecA, cp<double> *vecB, cp<double> *result, int size);
//__global__ void gpuSumSpinors(cp<double> *s1, cp<double> *s2, cp<double> *res, cp<double> c, int size); //  = s1 + c * s2;

__global__ void solve_kernel(cp<double>  *inVec, cp<double> *outVec, 
                             cp<double> *temp, cp<double> *temp2, cp<double> *r, cp<double> *p,
                             cp<double> *alpha, cp<double> *beta,
							 double *M,
							 my2dArray *IUP, my2dArray *IDN,
							 MatrixType Mtype, cp<double> *dot_res, double *rmodsq);
__device__ void gpuSumSpinors(cp<double> *s1, cp<double> *s2, cp<double> *res, cp<double> c, int size); //  = s1 + c * s2;
__device__ void gpuDotProduct(cp<double> *vecA, cp<double> *vecB, cp<double> *result, int size);
__device__ void applyD(cp<double> *in, cp<double> *out, int vol);
__device__ void setZeroGPU(thrust::complex<double> *v, int const vol);
__device__ void copyVec(thrust::complex<double> *v1,thrust::complex<double> *v2, int const vol);

class CGsolver{
    public:
        CGsolver();
        ~CGsolver(){cudaFree(rmodsq); cudaFree(dot_res); cudaFree(alpha); cudaFree(beta);}
        void solve(cp<double>  *inVec, cp<double> *outVec, DiracOP<double>& D, MatrixType Mtype);
        //void solveEO(cp<double> *inVec, cp<double> *outVec, DiracOP<double>& D, MatrixType Mtype=MatrixType::Normal);
    private:
        cp<double> *dot_res;
        Spinor<double> r, p, temp, temp2;
        dim3 dimGrid_dot, dimBlock_dot;
        dim3 dimGrid_zero, dimBlock_zero;
        dim3 dimGrid_sum, dimBlock_sum;
        dim3 dimGrid_copy, dimBlock_copy;
        void *dotArgs[4];
        void *setZeroArgs[2];
        void *sumArgs[5];
        void *copyArgs[3];
        cp<double> *beta, *alpha;
        int myvol;
        double *rmodsq;

};
