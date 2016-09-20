# Makefile
#
# author      : Weijun Xiao <wxiao@umn.edu>
# version     : 0.1
# date        : 20100128
# discription : generic Makefile for making cuda programs
#

BIN               	:= pmmatrixmul 
CXXFLAGS          	:= -g
LIBSUFFIX	  		:= _x86_64

CUDA_INSTALL_PATH ?= /usr/local/cuda-4.0
CUDA_SDK_PATH ?= /usr/local/NVIDIA_GPU_Computing_SDK

NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./"
LIBS = -L"$(CUDA_INSTALL_PATH)/lib64" -lcuda -lcudart -L"$(CUDA_SDK_PATH)/C/common/lib" $(CUDA_SDK_PATH)/C/lib/libcutil$(LIBSUFFIX).a -lstdc++ -lpthread
CUDA_SDK?=4
COMMONFLAGS = -DCUDA_SDK=$(CUDA_SDK)
NVCCFLAGS := --ptxas-options=-v -O3 -arch=sm_20 

# files
CPP_SOURCES       := pmmatrixmul_gold.cpp 
CU_SOURCES        := pmmatrixmul.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu_o
