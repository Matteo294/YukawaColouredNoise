include makefile_includes

INC_DIR=-I$(TOML) -I$(HDF_INC)
LIB_DIR=-L$(HDF_LIB)
LINK_ARGS=-Xlinker '-rpath,$(HDF_LIB)'

FLAGS=-O3 --std=c++17 -Xcompiler '-march=native' -MMD -MP -rdc=true -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 --expt-relaxed-constexpr
LIBS=-lcurand

ifeq ($(FFT), cuFFT)
	LIBS+= -lcufft
else ifeq ($(FFT), vkFFT)
	INC_DIR+=-I$(VKFFT_INC)
	LIBS+= -lnvrtc -lcuda
endif

#ifeq ($(USE_LTO), true)
#	LTO_FLAGS=-dc -dlto
#	LTO_LINK=-dlto
#else
#	LTO_FLAGS=-c
#	LTO_LINK=
#endif


LTO_FLAGS=-c
LTO_LINK=

all: out

BUILD_DIR:=./build

SRCS := FermionicDrift.cu Lattice.cu Dirac.cu CGsolver.cu Laplace.cu langevin_gpu_v2.cu reductions.cu main.cu colouredNoise.cpp
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

$(BUILD_DIR)/%.cu.o : src/%.cu Makefile makefile_includes
	mkdir -p $(dir $@)
	nvcc $(FLAGS) $(LTO_FLAGS) -DUSE_$(FFT) $(INC_DIR) $< -o $@

$(BUILD_DIR)/%.cpp.o : src/%.cpp Makefile makefile_includes
	mkdir -p $(dir $@)
	nvcc -x cu $(FLAGS) $(LTO_FLAGS) -DUSE_$(FFT) $(INC_DIR) $< -o $@

out: $(OBJS)
	nvcc $(LTO_LINK) -arch=$(ARCH) $(LIB_DIR) $(LINK_ARGS) $(OBJS) $(LIBS) -o out

clean:
	rm -rf build/ out

cleanitp:
	rm -rf *.csv *.o* *.e*

cleanbw:
	rm -rf *.csv *.out

-include $(DEPS)
