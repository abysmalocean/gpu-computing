# General makefile for cuda program
# Author: Dongwei Wang
# functionalities:
# make
# make clean
# echo variable
# echo include path
# echo lib path

################################################################################
#
# Build script for project
#
###############################################################################

BIN         := square
CXXFLAGS    := -g
LIBSUFFIX   := _x86_64
CUFLAGS	    := $(CFLAGS)

#cuda path( nvcc and include files and libs )
CUDA_INSTALL_PATH=/usr/local/cuda-4.0
CUDA_SDK_PATH=/usr/local/NVIDIA_GPU_Computing_SDK

NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc

INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./"
LIBS = -L"$(CUDA_INSTALL_PATH)/lib64" -lcuda -lcudart -L"$(CUDA_SDK_PATH)/C/common/lib" $(CUDA_SDK_PATH)/C/lib/libcutil$(LIBSUFFIX).a -lstdc++ -lpthread 
#files
HEADERS     := $(wildcard *.h)
CPP_SOURCES := $(wildcard *.cpp)
CU_SOURCES  := square.cu
CPP_OBJS    := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS     := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

NVCCFLAGS := --ptxas-options=-v -g -arch=sm_20

###############################################################################
# Rules and targets
%.cu_o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(INCD) $(LIBS)

# clean the files
clean:
	rm -f $(BIN) *.o *.cu_o tags

# echo the value of variables
var:
	@echo $(HEADERS)
	@echo $(CPP_SOURCES)
	@echo $(CU_SOURCES)
	@echo $(CPP_OBJS)
	@echo $(CU_OBJS)

# echo makefile include path
include:
	@echo $(INCD)

# echo makefile lib path
libs:
	@echo $(LIBS)
