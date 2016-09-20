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

/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "pmmatrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

__global__ void MatrixMulKernelA(Matrix M, Matrix N, Matrix P)
{
	__shared__ float Mds[THREAD_SIZE][THREAD_SIZE];
	__shared__ float Nds[THREAD_SIZE][THREAD_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * THREAD_SIZE + ty;
	int col = bx * THREAD_SIZE + tx;
	float sum = 0.0f;
	for( int m = 0; m<(M.width-1)/THREAD_SIZE	+ 1; m++)
	{
		if((m*THREAD_SIZE+tx)<M.width&&row<M.height)
			Mds[ty][tx] = M.elements[row*M.width+(m*THREAD_SIZE+tx)];
		else
			Mds[ty][tx] = 0.0;
		if((m*THREAD_SIZE+ty)<N.height&&col<N.width)
			Nds[ty][tx] = N.elements[(m*THREAD_SIZE+ty)*N.width+col];
		else
			Nds[ty][tx] = 0.0;
		__syncthreads();
		for(int n=0; n<THREAD_SIZE; n++)
		{
			sum += Mds[ty][n]*Nds[n][tx];
		}
		__syncthreads();
	}	
	if(row<P.height&&col<P.width)
		P.elements[row*P.width+col]=sum;
}

__global__ void MatrixMulKernelB(Matrix M, Matrix N, Matrix P,int streamid)
{
	__shared__ float Mds[THREAD_SIZE][THREAD_SIZE];
	__shared__ float Nds[THREAD_SIZE][THREAD_SIZE];

	int row_stream = blockIdx.y * THREAD_SIZE + threadIdx.y;
	int col_stream = blockIdx.x * THREAD_SIZE + threadIdx.x;

	int row_matrix = blockIdx.y * THREAD_SIZE + threadIdx.y + (streamid/2)*(M.height/2);
	int col_matrix = blockIdx.x * THREAD_SIZE + threadIdx.x + (streamid%2)*(N.width/2);
	float sum = 0.0f;

	for( int m = 0; m<(M.width-1)/THREAD_SIZE + 1; m++)
	{
		if((m*THREAD_SIZE+threadIdx.x)<M.width&&row_stream<(M.height/2))
			Mds[threadIdx.y][threadIdx.x] = M.elements[row_matrix*M.width+(m*THREAD_SIZE+threadIdx.x)];
		else
			Mds[threadIdx.y][threadIdx.x] = 0.0;
		if((m*THREAD_SIZE+threadIdx.y)<N.height&&col_stream<(N.width/2))
			Nds[threadIdx.y][threadIdx.x] = N.elements[(m*THREAD_SIZE+threadIdx.y)*N.width+col_matrix];
		else
			Nds[threadIdx.y][threadIdx.x] = 0.0;
		__syncthreads();
		for(int n=0; n<THREAD_SIZE; n++)
		{
			sum += Mds[threadIdx.y][n]*Nds[n][threadIdx.x];
		}
		__syncthreads();
	}	
	if(row_matrix<P.height&&col_matrix<P.width)
		P.elements[row_matrix*P.width+col_matrix]=sum;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
