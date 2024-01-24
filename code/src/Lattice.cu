#include "Lattice.cuh"
#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>

LookUpTable::LookUpTable(){cudaMallocManaged(&at, vol * sizeof(my2dArray));}
LookUpTable::~LookUpTable(){cudaFree(at);}

LookUpTableConv::LookUpTableConv(){cudaMallocManaged(&at, vol * sizeof(int));}
LookUpTableConv::~LookUpTableConv(){cudaFree(at);}



__host__ __device__ unsigned int PBC(int const n, int const N){
	return (n+N) % N;
}


__host__ __device__ unsigned int vecToFlat(int const nt, int const nx){
    return nt*Sizes[1] + nx;
}

__host__ __device__ my2dArray flatToVec(int n){
	my2dArray idx; // nt, nx
	idx[0] = n / Sizes[1];
    idx[1] = n % Sizes[1];
	return idx;
}
