#pragma once

#include <array>
#include <complex>
#include <iostream>
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "CGsolver.cuh"
#include <cooperative_groups.h>
#include "params.h"

class CGsolver;

namespace cg = cooperative_groups;

template <typename T>
class DiracOP {
	public:
		DiracOP();
		~DiracOP(){;}
		void setScalar(T* phi){M = phi;}
		void applyD(cp<double> *in, cp<double> *out, MatrixType MType);
		T *M;
		LookUpTable IUP, IDN;


};


__global__ void applyD_gpu(cp<double> *in, cp<double> *out, MatrixType const useDagger, double *M, my2dArray *IUP, my2dArray *IDN);
__device__ void applyDiagonal(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M);
__device__ void applyHopping(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);

