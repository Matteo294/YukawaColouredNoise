#if !defined(USE_cuFFT) && !defined(USE_vkFFT)
#define USE_cUFFT
#endif

// Remember to add modify to drift instead of assign and remove magnetization

#include <cmath>
#include <atomic>
#include <vector>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "toml.hpp"
#include "params.h"
#include "Laplace.h"
#include "device_info.h"
#include "managedVector.h"
#include "colouredNoise.h"
#include "reductions.cuh"
#include "langevin_gpu_v2.cuh"

// ----------------------------------------------------------
#include "Dirac.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "CGsolver.cuh"
#include "FermionicDrift.cuh"

//#include <helper_cuda.h>  // helper function CUDA error checking and initialization
//#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
// ----------------------------------------------------------


__constant__ myType epsBar;
__constant__ myType m2;
__constant__ myType lambda;



// ----------------------------------------------------------
__constant__ double yukawa_coupling_gpu;
__constant__ double fermion_mass_gpu;
__constant__ thrust::complex<double> im_gpu;
__constant__ double cutFraction_gpu;
__constant__ double sq2Kappa_gpu;
__constant__ DriftMode driftMode_gpu;
__constant__ double WilsonParam_gpu;
// ----------------------------------------------------------




double FindKappa(double const m2, double const lambda0) {
	auto const Delta = (2.0*nDim + m2)*(2.0*nDim + m2) + 4.0 * lambda0 / 3.0;
	auto const k1 = (-(2.0*nDim + m2) + std::sqrt(Delta)) / (2.0 * lambda0 / 3.0);

	return k1;
}

namespace {
	std::atomic<bool> early_finish = false;
}

void signal_handler(int signal) {
	early_finish = true;
}

int main(int argc, char** argv) {

	std::signal(SIGUSR2, signal_handler);
    srand(1234);
    
    std::cout << "Nt: " << Sizes[0] << " Nx: " << Sizes[1] << std::endl;  

	if constexpr(nDim > 3)
		std::cout << "#Due do technical limitations, coloured noise is *DISABLED* for nDim > 3.\n\n";

	if (argc == 1) {
		std::cerr << "No input file provided.\nExiting.\n";
		exit(1);
	}

	int constexpr N = SIZE;
	auto lap = Laplace{N};
	lap.func2();

	auto ivec  = ManagedVector<myType>{N * nVectorComponents};
	for (auto &e : ivec)
		e = static_cast<myType>(1.0 - 2.0*drand48()); // little bias in the negative direction
//		e = static_cast<myType>(drand48());

	auto drift = ManagedVector<myType>{N * nVectorComponents};
	auto noise = ManagedVector<myType>{N * nVectorComponents};

	auto avg = ManagedVector<myType>{nVectorComponents};
	// timeSlices is organised with the field component as the *outer* index, and the time
	// coordinate as the inner index
	auto timeSlices = ManagedVector<myType>{nVectorComponents * Sizes[0]};

	// print out the parameters from the input file
	auto inFile = std::ifstream(argv[1]);
	std::string line;
	while (getline(inFile, line)) 
		std::cout << '#' << line << '\n';
	inFile.close();
	std::cout << "\n\n";

	auto const inputData = toml::parse(argv[1]);
	auto const& parameters = toml::find(inputData, "physics");
	auto const useMass = toml::find<std::string>(parameters, "useMass");
	myType my_m2, myLambda, kappa, Lambda;
	if (useMass == "true") {
		my_m2 = toml::find<myType>(parameters, "mass");
		myLambda = toml::find<myType>(parameters, "g");

		kappa = FindKappa(my_m2, myLambda);
		Lambda = kappa*kappa*myLambda/6.0;
	} else {
		kappa  = toml::find<myType>(parameters, "kappa");
		Lambda = toml::find<myType>(parameters, "lambda");

		my_m2 = ((1.0 - 2.0*Lambda) / kappa - 2.0*nDim);
		myLambda = (6.0 * Lambda / (kappa*kappa));
	}
	auto const sq2Kappa = std::sqrt(2.0 * kappa);
	auto const cutFraction = toml::find<myType>(parameters, "cutFraction");

	auto const& rndSection = toml::find(inputData, "random");
	int const seed = toml::find<int>(rndSection, "seed");

	auto const& langevin = toml::find(inputData, "langevin");
	auto	   myEpsBar = toml::find<double>(langevin, "averageEpsilon");
	auto const MaxLangevinTime = toml::find<double>(langevin, "MaxLangevinTime");
	auto const ExportTime = toml::find<double>(langevin, "ExportTime");
	auto const burnCount = toml::find<int>(langevin, "burnCount");
	auto const MeasureDriftCount = toml::find<int>(langevin, "MeasureDriftCount");

	auto const& ioSection = toml::find(inputData, "io");
	auto const outFileName = toml::find<std::string>(ioSection, "configFileName");
	auto const timeSliceFileName = toml::find<std::string>(ioSection, "timeSliceFileName");
	std::string startFileName = "";
	try {
		startFileName = toml::find<std::string>(ioSection, "startFileName");
	} catch (std::exception& e) {}
	bool exportHDF = false;
	try {
		exportHDF   = toml::find<bool>(ioSection, "export");
	} catch (std::exception& e) {}
	bool resumeRun = false;
	try {
		resumeRun = toml::find<bool>(ioSection, "resume");
	} catch (std::exception& e) {}

	// ----------------------------------------------------------

	myType sum2 = 0.0;
	auto const& fermionsSection = toml::find(inputData, "fermions");
	double const fermion_mass = toml::find<double>(fermionsSection, "fermion_mass");
    double const WilsonParam = toml::find<double>(fermionsSection, "WilsonParam");
	double yukawa_coupling;
	if (useMass == "true") 
		yukawa_coupling = toml::find<double>(fermionsSection, "yukawa_coupling");
	else 
		yukawa_coupling = toml::find<double>(fermionsSection, "yukawa_coupling") / sq2Kappa;
    auto driftMode = toml::find<int>(fermionsSection, "driftMode") == 0 ? DriftMode::Normal : DriftMode::Rescaled;
    
	Spinor<double> in, out, cpy, out2;
	DiracOP<double> Dirac;
	FermionicDrift fDrift(seed);
	CGsolver CG;
    double *trace; // trace D^-1
    ManagedVector<double> corr(Sizes[0]);
	
	cudaMallocManaged(&trace, sizeof(double));
    
    for(int i=0; i<vol; i++) {
        for(int j=0; j<4; j++) {
            in.data()[4*i+j] = 0.0;
            out.data()[4*i+j] = 0.0;
        }
    }
       
	int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, computeDrift);
	cudaDeviceSynchronize();
	auto dimGrid_drift = dim3(nBlocks, 1, 1);
	auto dimBlock_drift = dim3(nThreads, 1, 1);
    
    nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, gpuTraces);
	cudaDeviceSynchronize();
	auto dimGrid_traces = dim3(nBlocks, 1, 1);
	auto dimBlock_traces = dim3(nThreads, 1, 1);
    void *tracesArgs[] = {(void*) &drift.data(), (void*) &trace, (void*) &vol};

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, setZero_kernel);
	cudaDeviceSynchronize();
	auto dimGrid_setZero = dim3(nBlocks, 1, 1);
	auto dimBlock_setZero = dim3(nThreads, 1, 1);
    void *setZeroArgs[] = {(void*) &in.data(), (void*) &spinor_vol};
	
	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, gpuTimeSlices_spinors);
	cudaDeviceSynchronize();
	auto dimGrid_tsSpinors = dim3(nBlocks, 1, 1);
	auto dimBlock_tsSpinors = dim3(nThreads, 1, 1);
    void *tsSpinorsArgs[] = {(void*) &in.data(), (void*) &corr.data(), (void*) &Sizes[0]};
    
	// set up print files
	std::ofstream datafile, tracefile;
	datafile.open("data.csv");
	datafile << "corr" << "\n";
	std::string fname;
	fname.append("traces"); fname.append(".csv");
	tracefile.open(fname);
	tracefile << "tr,sigma,phi" << "\n";
	// ----------------------------------------------------------

	myType *maxDrift;
	myType *eps, elapsedLangevinTime;
	cudaMallocManaged(&maxDrift, sizeof(myType));
	cudaMalloc(&eps, sizeof(myType));
	myType *h_eps;
	h_eps = (myType*)malloc(sizeof(myType));
	//*eps = myEpsBar;
	cudaMemcpy(eps, &myEpsBar, sizeof(myType), cudaMemcpyHostToDevice);
	elapsedLangevinTime = 0.0;

	auto const kli = KernelLaunchInfo{Run};
	auto const kli_sMem = sizeof(myType) * std::max(kli.numThreads, 32);

	auto cn = ColouredNoise{noise, N, cutFraction, seed, kli};

	void *kMagnetisation[] = {(void*)&ivec, (void*)&(avg.data()), (void*)&N};

	auto const kTimeSlices = KernelLaunchInfo{gpuTimeSlices};
	void *timeSlicesArgs[] = {(void*)&ivec, (void*)&(timeSlices.data()), (void*)&N};

	// can't pass lap directly because it's not allocated on the device
	// is it worth it to allocate it on the device...? I, J, and cval are already there...
	// the only difference would be reducing the number of arguments here...
	void *kAll[] = {
		(void*)&eps,
		(void*)&ExportTime,
		(void*)&(ivec.data()),
		(void*)&(drift.data()),
		(void*)&(noise.data()),
		(void*)&N,
		(void*)&(lap.I),
		(void*)&(lap.J),
		(void*)&(lap.cval),
		(void*)&maxDrift};


	std::cout << std::endl;		// force a flush so we can see something on the screen before
								// actual computations start

	cudaMemcpyToSymbol(m2, &my_m2, sizeof(myType));
	cudaMemcpyToSymbol(lambda, &myLambda, sizeof(myType));
	cudaMemcpyToSymbol(epsBar, &myEpsBar, sizeof(myType));
	// -----------------------------------------------------------------
	cudaMemcpyToSymbol(yukawa_coupling_gpu, &yukawa_coupling, sizeof(double));
	cudaMemcpyToSymbol(fermion_mass_gpu, &fermion_mass, sizeof(double));
	cudaMemcpyToSymbol(im_gpu, &im, sizeof(thrust::complex<double>));
    cudaMemcpyToSymbol(cutFraction_gpu, &cutFraction, sizeof(double));
    cudaMemcpyToSymbol(sq2Kappa_gpu, &sq2Kappa, sizeof(double));
    cudaMemcpyToSymbol(driftMode_gpu, &driftMode, sizeof(DriftMode));
    cudaMemcpyToSymbol(WilsonParam_gpu, &WilsonParam, sizeof(double));
	// -----------------------------------------------------------------
    
	Dirac.setScalar(ivec.data());
    
    
    /*for(int i=0; i<4*vol; i++){
        in.data()[i] = i/10.0;
        cpy.data()[i] = in.data()[i];
    }
    for(int i=0; i<vol; i++) ivec.data()[i] = (double) 10.2/(i+1.3) + 0.4*i + (i-11.6);
    
    Dirac.applyD(in.data(), out.data(), MatrixType::Normal);
    Dirac.applyD(cpy.data(), out2.data(), MatrixType::Normal);
    
    for(int i=0; i<4*vol; i++) std::cout << out.data()[i] << "\t" << out2.data()[i] << "\n";*/
    
    
    
    
    
	
	// burn in a little bit, since the drift might be stronger at the beginning, since we are
	// likely far from the equilibrium state
	for (int burn = 0; burn < burnCount; ++burn) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();

			// ----------------------------------------------------------
			fDrift.getForce(drift.data(), Dirac, CG, dimGrid_drift, dimBlock_drift);
			// ----------------------------------------------------------

			kli.Run(kAll, kli_sMem);
			cudaDeviceSynchronize();
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		}
	}
	
	std::cout << "Thermalization done!" << std::endl;

	int nMeasurements = 0;
	int oldMeasurements = 0;
	elapsedLangevinTime = 0.0;
    
	if (MeasureDriftCount > 0) {
        myType epsSum = 0.0;
        while (elapsedLangevinTime < MeasureDriftCount * ExportTime) {
            myType t = 0.0;
            while (t < ExportTime) {
                cn();

                // ----------------------------------------------------------
                fDrift.getForce(drift.data(), Dirac, CG, dimGrid_drift, dimBlock_drift);
                // ----------------------------------------------------------

                kli.Run(kAll, kli_sMem);
                cudaDeviceSynchronize();
                cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                t += *h_eps;
            }
            elapsedLangevinTime += t;

            epsSum += *h_eps;
            nMeasurements++;
        }
        epsSum /= nMeasurements;
        std::cout << "#Average eps during drift measurement = " << epsSum << std::endl;
        myEpsBar *= myEpsBar/epsSum;	// update epsBar so that the average step size is roughly the
                                        // original value of epsBar provinded in the input file
	}
	cudaMemcpyToSymbol(epsBar, &myEpsBar, sizeof(myType));
    
    std::cout << "Drift measuring done!" << std::endl;

	// main loop
	elapsedLangevinTime = 0.0;
	nMeasurements = oldMeasurements;
	std::vector<myType> hostLattice(N*nVectorComponents);
	elapsedLangevinTime = nMeasurements * ExportTime;
	auto timeSliceFile = std::ofstream(timeSliceFileName);
	auto timerStart = std::chrono::high_resolution_clock::now();
    
    //elapsedLangevinTime = MaxLangevinTime;

	double avg_magnetisation = 0.0;
	int nConfig = 0;
    while (elapsedLangevinTime < MaxLangevinTime) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();

			// ----------------------------------------------------------
			fDrift.getForce(drift.data(), Dirac, CG, dimGrid_drift, dimBlock_drift);
			// ----------------------------------------------------------

			kli.Run(kAll, kli_sMem);
			cudaDeviceSynchronize();
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		}
		elapsedLangevinTime += t;

//		cudaMemPrefetchAsync(ivec.data(), N*nVectorComponents, cudaCpuDeviceId);
		cudaLaunchCooperativeKernel((void*)gpuMagnetisation, kli.dimGrid, kli.dimBlock,
				kMagnetisation, kli_sMem, NULL);
		cudaDeviceSynchronize();
        
        

		kTimeSlices.Run(timeSlicesArgs, kli_sMem);
		cudaDeviceSynchronize();

		for (int comp = 0; comp < nVectorComponents; ++comp) {
			for (int tt = 0; tt < nTimeSlices; ++tt)
				timeSliceFile << timeSlices[tt + nTimeSlices * comp] / SpatialVolume << '\t';
			timeSliceFile << '\n';
		}
		timeSliceFile << '\n';
		

		std::cout << elapsedLangevinTime << '\t' << *h_eps << '\t';
		sum2 = 0.0;
		for (auto& e : avg) {
			if (useMass == "false") {e /= sq2Kappa;}
			std::cout << e / N << '\t';
			sum2 += e*e;
		}

		// this explicit copy seems to peform slightly/marginally better
		// TODO: needs further investigation
		cudaMemcpy(hostLattice.data(), ivec.data(), N*nVectorComponents*sizeof(myType),
				cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
    
        
		// if the user provided kappa as input, we rescale the output field to dimensionless format
		if (useMass == "false")
			for (auto& e : hostLattice){
				// divide or multiply...?
				e /= sq2Kappa;
				//std::cout << e / N << '\t';
			}
		if (early_finish == true) {
			std::cout << "#Early termination signal received.\n#Wrapping up.\n";
			elapsedLangevinTime = MaxLangevinTime + 1.0;
		}
		std::stringstream ss;
		ss << "data/cnfg_" << std::setfill('0') << std::setw(8) << 
			(exportHDF == true ? nMeasurements : 1);
            
        nMeasurements++;
		nConfig ++;
		avg_magnetisation += abs(avg[0]);

		std::cout 	<< "abs. mag: " << (double) abs(avg[0]) / N << " \t "
                    << "cond: " << (double) abs(*trace) / N << " \t "
                    << "partial avg: " << (double) avg_magnetisation / (double) nMeasurements / (double) vol << "\n";
        
        // -------------------- extract fermion mass ----------------------------------
		setZeroArgs[0] = (void*) &in.data();
		cudaLaunchCooperativeKernel((void*) &setZero_kernel, dimGrid_setZero, dimBlock_setZero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();

		// Set source
        in.data()[0] = 1.0;
		in.data()[1] = 1.0;
		in.data()[2] = 0.0;
		in.data()[3] = 0.0;

        switch (CGmode) {
			
			case '0':

				CG.solve(in.data(), out.data(), Dirac, MatrixType::Normal);
				Dirac.applyD(out.data(), in.data(), MatrixType::Dagger);
				cudaDeviceSynchronize();

				break;
		}
		
		cudaLaunchCooperativeKernel((void*) &gpuTimeSlices_spinors, dimGrid_tsSpinors, dimBlock_tsSpinors, tsSpinorsArgs, 0, NULL);
		cudaDeviceSynchronize();
		

		for(int nt=0; nt<Sizes[0]; nt++){
			datafile << corr.data()[nt] << "\n";
		}
        
        // -->  compute condensates from drifts as they are proportional
        fDrift.getForce(drift.data(), Dirac, CG, dimGrid_drift, dimBlock_drift);        
        *trace = 0.0;
		cudaLaunchCooperativeKernel((void*) &gpuTraces, dimGrid_traces, dimBlock_traces, tracesArgs, 32 * sizeof(double), NULL);
		cudaDeviceSynchronize();
       	//std::cout << "Trace: " << *trace << std::endl; 
        if (yukawa_coupling != 0.0) { 
            *trace /= yukawa_coupling;
        } else {
            *trace = 0.0;
        }
		
        tracefile << (double) (*trace) / vol << "," << (double) (avg[0] / vol) << "," << (double) (std::sqrt(sum2) / vol) << "\n";
		// ------------------------------------------------------

		
	}

	std::cout << "Final abs. magnetisation: " << (double) avg_magnetisation / (double) nMeasurements / (double) vol << std::endl;
        
	auto timerStop  = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	timeSliceFile.close();

	std::cout << "#numSms = " << kli.numSms << '\n';
	std::cout << "#blocks per SM = " << kli.numBlocksPerSm << '\n';
	std::cout << "#theads = " << kli.numThreads << '\n';
	std::cout << "#blocks = " << kli.numBlocks << '\n';

	std::cout << "#Number of measurements: " << nMeasurements << '\n';

	std::cout << "#Run time for main loop: " << duration.count() / 1000.0 << "s\n";

	cudaFree(eps);
	cudaFree(maxDrift);
	free(h_eps);

	// ------------------------------------------------
	cudaFree(trace);
	// ------------------------------------------------

	std::cout << "Errors? " << cudaPeekAtLastError() << std::endl;
    
    
	return 0;
}

    
