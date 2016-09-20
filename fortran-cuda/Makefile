all: complex_mul

LIBSUFFIX    :=  _x86_64 

CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SDK_PATH ?= /usr/local/cuda-3.2/sdk

NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./"
# Define Fortran compiler
#FC= ifort
#FC= g95
FC=gfortran


complex_mul: main.f90 Cuda_function.o
	$(FC) -o complex_mul main.f90 Cuda_function.o -L/usr/local/cuda/lib64  -lcudart

Cuda_function.o: Cuda_function.cu
	$(NVCC) -c -O3 Cuda_function.cu

clean: 
	rm complex_mul Cuda_function.o
