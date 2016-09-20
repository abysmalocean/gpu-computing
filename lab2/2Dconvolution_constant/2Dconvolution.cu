/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* 2D Convolution: C = A (*) B, A is the 5x5 kernel matrix, B is the image matrix.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <2Dconvolution_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix A, const Matrix B, Matrix C);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	Matrix  A;
	Matrix  B;
	Matrix  C;
	
	srand(2012);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
		B  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		C  = AllocateMatrix(B.height, B.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = NULL; 
		unsigned int data_read = 0;
		cutReadFilei(argv[1], &params, &data_read, true);
		if(data_read != 2)
		{
			printf("Error reading parameter file\n");
			cutFree(params);
			return 1;
		}

		A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		B  = AllocateMatrix(params[0], params[1], 0);		
		C  = AllocateMatrix(params[0], params[1], 0);
		cutFree(params);
		(void)ReadFile(&A, argv[2]);
		(void)ReadFile(&B, argv[3]);
	}

	// Convolution on the device
	ConvolutionOnDevice(A, B, C);
	// compute the matrix multiplication on the CPU for comparison
	Matrix reference = AllocateMatrix(C.height, C.width, 0);
	computeGold(reference.elements, A.elements, B.elements, B.height, B.width);
	// in this case check if the result is equivalent to the expected soluion
	CUTBoolean res = cutComparefe(reference.elements, C.elements, C.width * C.height, 0.0001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
	if(argc == 5)
	{
		WriteFile(C, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(C, argv[1]);
	}   

	// Free matrices
	FreeMatrix(&A);
	FreeMatrix(&B);
	FreeMatrix(&C);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
// __device__ __constant__ float* Ad;
void ConvolutionOnDevice(const Matrix A, const Matrix B, Matrix C)
{
    //Load A and B  to the device
    //Matrix Ad = AllocateDeviceMatrix(A);
    //CopyToDeviceMatrix(Ad, A);

    cudaMalloc((void**)&Ad, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
    cudaMemcpyToSymbol(Ad, A.elements, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
	
    Matrix Bd = AllocateDeviceMatrix(B);
    CopyToDeviceMatrix(Bd, B);

    //Allocate C on the device
    Matrix Cd = AllocateDeviceMatrix(C);
    CopyToDeviceMatrix(Cd, C); // Clear memory

    //Setup the execution configuration
    //Launch the device computation threads!
    int blocks = B.height;
    int threads = B.width; 

    cudaEvent_t start, stop;
    float elapsedTime=0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    ConvolutionKernel<<<blocks, threads>>>(Bd,Cd);    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("The execution time of GPU is :%f\n",elapsedTime); 
    //Read C from the device
    CopyFromDeviceMatrix(C, Cd); 

    FreeDeviceMatrix(&Bd);
    FreeDeviceMatrix(&Cd);
}

//Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

//Allocate a device matrix of dimensions height*width
//If init == 0, initialize to all zeroes.  
//If init == 1, perform random initialization.
//If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    //don't allocate memory on option 2
    if(init == 2)
	return M;
    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.height * M.width; i++)
    {
	M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	if(rand() % 2)
	    M.elements[i] = - M.elements[i];
    }
    return M;
}	

//Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,cudaMemcpyHostToDevice); 
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,cudaMemcpyDeviceToHost); 
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int data_read = M->height * M->width;
    cutReadFilef(file_name, &(M->elements), &data_read, true);
    return data_read;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    cutWriteFilef(file_name, M.elements, M.width*M.height, 0.0001f);
}
