#include "ac.h"
#include <iostream>

#include <cstdlib>
#include <new>
#include <limits>

template <class T>
struct Mallocator
{
  typedef T value_type;

  Mallocator () = delete;
  template <class U> constexpr Mallocator (const Mallocator <U>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t) noexcept;
};

template <>
struct Mallocator<double>
{
	typedef double T;
  typedef T value_type;

  Mallocator () = default;
  template <class U> constexpr Mallocator (const Mallocator <U>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
//	  std::cerr << "Custom allocation\n\n";
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    if (auto p = static_cast<T*>(fftw_malloc(n*sizeof(T))))
      return p;

    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { fftw_free(p); }//std::cerr << "Custom freeing\n\n"; }
};

template <>
struct Mallocator<std::complex<double>>
{
	typedef std::complex<double> T;
  typedef T value_type;

  Mallocator () = default;
  template <class U> constexpr Mallocator (const Mallocator <U>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
//	  std::cerr << "Custom allocation\n\n";
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    if (auto p = static_cast<T*>(fftw_malloc(n*sizeof(T))))
      return p;

    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { fftw_free(p); } //std::cerr << "Custom freeing\n\n"; }
};

template <class T, class U>
bool operator==(const Mallocator <T>&, const Mallocator <U>&) { return true; }
template <class T, class U>
bool operator!=(const Mallocator <T>&, const Mallocator <U>&) { return false; }


//ACreturn AutoCorrelation(std::vector<Complex> const& v)

ACreturn AutoCorrelation(iter const begin, iter const end) {
	auto const Stau = 1.5;

	auto const N = std::distance(begin, end);
	auto const abb = std::accumulate(begin, end, 0.0) / (1.0 * N);

	std::vector<double, Mallocator<double>> delpro(N);
	std::vector<double, Mallocator<double>> delpro2(N);
	std::transform(begin, end, delpro.begin(), [abb](auto const& v){ return v - abb; });
	auto const GammaFbb0 = std::accumulate(delpro.cbegin(), delpro.cend(), 0.0,
			[](double const sum, double const v){return sum + v*v;}) / (1.0 * N);

	if (GammaFbb0 == 0.0) return ACreturn{0.0, 0.0, 0.0, 0.0, 0.0};

	auto tmax = 0;
	bool flag = false;
	if (Stau != 0.0) {
		tmax = std::floor(N / 2);
		flag = true;
	}

	std::vector<double, Mallocator<double>> fft_in(N+tmax);
	std::copy(delpro.cbegin(), delpro.cend(), fft_in.begin());

	// N/2 + 1 because of some redundancy... this is the size FFTW itself suggests
	std::vector<std::complex<double>, Mallocator<std::complex<double>>> outfft((N+tmax)/2 + 1);
	auto forward_plan  = fftw_plan_dft_r2c_1d(N+tmax, fft_in.data(),
			reinterpret_cast<fftw_complex*>(outfft.data()), FFTW_ESTIMATE);
	auto backward_plan = fftw_plan_dft_c2r_1d(N+tmax,
			reinterpret_cast<fftw_complex*>(outfft.data()), fft_in.data(), FFTW_ESTIMATE);

	fftw_execute(forward_plan);
	/*
	std::cout << "outfft = " << '\n';
	for (auto e : outfft)
		std::cout << e << '\t';
	std::cout << '\n';
	*/
	std::transform(outfft.cbegin(), outfft.cend(), outfft.begin(),
			[N, tmax](auto const& el){ return std::norm(el) / (N+tmax); });
	fftw_execute(backward_plan);

	fftw_destroy_plan(forward_plan);
	fftw_destroy_plan(backward_plan);

	std::vector<double> GammaFbb(tmax+1);
	for(size_t i = 0; i < GammaFbb.size(); ++i)
		GammaFbb[i] = fft_in[i] / (N - i);

	/*
	std::cout << "GammaFbb = " << '\n';
	for (auto e : GammaFbb)
		std::cout << e << '\t';
	std::cout << '\n';
	*/

	auto const eps = 1e-5;

	std::vector<double> rho(N);
	auto const tempDiv = 1.0 / GammaFbb[0];

	std::transform(GammaFbb.cbegin(), GammaFbb.cend(), rho.begin(),
			[tempDiv](auto const& v){ return v * tempDiv; });

	auto Wopt = 0;
	auto Gint = 0.0;
	for (auto t = 1; t < tmax +1; ++t) {
		if (flag) {
			Gint += rho[t];
			auto const tauW = (Gint <= 0.0 ? eps : Stau / std::log((Gint+1)/Gint));
			auto const gW = exp(-t / tauW) - tauW / sqrt(t*N);
			if(gW < 0.0) {
				Wopt = t;
				tmax = std::min(tmax, 2*t);
				flag = false;
			}
		}
		else
			break;
	}

	if (flag) Wopt = tmax; // fail

//	std::cout << "Wopt = " << Wopt << '\n';

	auto const CFbbopt = GammaFbb[0]
		+ 2.0*std::accumulate(GammaFbb.cbegin()+1, GammaFbb.cbegin()+Wopt+1, 0.0);

//	std::cout << "CFbbopt = " << CFbbopt << '\n';
	
	if (CFbbopt < 0.0) std::cerr << "Gamma pathological: estimated error^2 < 0\n\n\n";

	for (auto& v : GammaFbb)
		v += CFbbopt/N;
	auto const CFbbopt_refined = GammaFbb[0]
		+ 2.0*std::accumulate(GammaFbb.cbegin()+1, GammaFbb.cbegin()+Wopt+1, 0.0);

	std::vector<double> acc(N);
	std::partial_sum(GammaFbb.cbegin(), GammaFbb.cend(), acc.begin());

    auto const tauint = acc[Wopt]/GammaFbb[0] - 0.5;
    auto const dtauint = tauint * 2.0 * std::sqrt((Wopt - tauint + 0.5)/N);

	auto const dvalue = sqrt(CFbbopt_refined / N);			// error
	auto const ddvalue = dvalue * sqrt((Wopt + 0.5) / N); 	// error of the error

//	std::cout << abb << '\t' << dvalue << '\t' << ddvalue
//		<< '\t' << tauint << '\t' << dtauint << '\n';

	return ACreturn{abb, dvalue, ddvalue, tauint, dtauint};
}

extern "C"
{
    ACreturn Wrapper(const double* dataSet, const long int n)
    {
		std::vector<double> data;

		data.assign(dataSet, dataSet + n);
		return AutoCorrelation(data.begin(), data.end());
    }
}
