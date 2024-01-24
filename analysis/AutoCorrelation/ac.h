#ifndef AC_H
#define AC_H

#include <complex>
#include <numeric>
#include <vector>
#include <algorithm>
#include <fftw3.h>
//#include "complexTypes.h"
 
#ifdef SIMD_COMPLEX_AVAILABLE_XXXX

typedef compX Complex;

#else

typedef std::complex<double> Complex;

#endif

struct ACreturn {
	double average;
	double sigmaF;
	double sigmaF_error;
	double ACtime;
	double ACtimeError;
};

using iter = std::vector<double>::const_iterator;
ACreturn AutoCorrelation(iter const begin, iter const end);
//ACreturn AutoCorrelation(std::vector<double> const& v);

extern "C"
{
    ACreturn Wrapper(const double* dataSetReal, const long int n);
}

#endif
