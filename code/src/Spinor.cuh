#pragma once

#include <thrust/complex.h>
#include "params.h"


template <typename T>
struct Spinor {
    public:
		Spinor(){cudaMallocManaged(&val, sizeof(thrust::complex<T>) * 4*vol);}
		Spinor(int vol){cudaMallocManaged(&val, sizeof(thrust::complex<T>) * 4*vol);}
		~Spinor(){cudaFree(val);}
		cp<T>*& data(){return val;}
	private:
    	cp<T> *val;
};
