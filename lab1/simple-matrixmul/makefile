# Makefile
#
# author      : Weijun Xiao <wxiao@umn.edu>
# version     : 0.2
# date        : 20130128
# discription : generic Makefile for making cuda programs
#

BIN					:= simplematrixmul 
CXXFLAGS			:= -O2
LIBSUFFIX			:= _x86_64

CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SDK_PATH = /usr/local/NVIDIA_GPU_Computing_SDK

NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc -I./
LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart -L$(CUDA_SDK_PATH)/C/common/lib $(CUDA_SDK_PATH)/C/lib/libcutil$(LIBSUFFIX).a -lstdc++ -lpthread

CUDA_SDK?=4
COMMONFLAGS = -DCUDA_SDK=$(CUDA_SDK)
NVCCFLAGS := --ptxas-options=-v -O2 -arch=sm_20 

# files
CPP_SOURCES       := simplematrixmul_gold.cpp 
CU_SOURCES        := simplematrixmul.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS)

%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(INCD)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCD)

clean:
	rm -f $(BIN) *.o *.cu_o gpufile.txt reference.txt tags
