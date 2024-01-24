#include <array>
#include <cmath>
#include <atomic>
#include <vector>
#include <chrono>
#include <csignal>
#include <complex>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "HDF5.h"
#include "toml.hpp"
#include "params.h"
#include "Laplace.h"
#include "device_info.h"
#include "managedVector.h"

extern __constant__ myType epsBar;
extern __constant__ myType m2;
extern __constant__ myType lambda;

namespace cg = cooperative_groups;

