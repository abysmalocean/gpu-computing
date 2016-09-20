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

#ifndef _MUL_KERNEL
#define _MUL_KERNEL

#include <stdio.h>
#include "mul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

__global__ void MatrixMulKernelA1(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices
	__shared__ float Mds[BLOCKSIZE][BLOCKSIZE];
	__shared__ float Nds[BLOCKSIZE][BLOCKSIZE];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float p_sum=0.0f;
	for(int m=0; m<(M.width-1)/BLOCKSIZE+1; m++)
	{
		if((m*BLOCKSIZE+threadIdx.x)<M.width&&row<M.height)
			Mds[threadIdx.y][threadIdx.x] = M.elements[row*M.width+(m*BLOCKSIZE+threadIdx.x)];
		else
			Mds[threadIdx.y][threadIdx.x] = 0.0;
		if((m*BLOCKSIZE+threadIdx.y)<N.height&&col<N.width)	
			Nds[threadIdx.y][threadIdx.x] = N.elements[(m*BLOCKSIZE+threadIdx.y)*N.width+col];
		else
			Nds[threadIdx.y][threadIdx.x] = 0.0;
		__syncthreads();
		for(int n=0; n<BLOCKSIZE; n++)
		{
			p_sum += Mds[threadIdx.y][n]*Nds[n][threadIdx.x];
		}
		__syncthreads(); 
	}
	if(row<P.height&&col<P.width)
		P.elements[ row*P.width+col] = p_sum;
}

__global__ void MatrixMulKernelA2(Matrix M, Matrix N, Matrix P)
{
        //Multiply the two matrices
	__shared__ float Mds[BLOCKSIZE][BLOCKSIZE];
	__shared__ float Nds[BLOCKSIZE][BLOCKSIZE];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float p_sum=0.0f;
	for( int m=0; m<M.width/BLOCKSIZE; m++ )
	{
		Mds[threadIdx.y][threadIdx.x] = M.elements[row*M.width+(m*BLOCKSIZE+threadIdx.x)];
		Nds[threadIdx.y][threadIdx.x] = N.elements[(m*BLOCKSIZE+threadIdx.y)*N.width+col];
		__syncthreads();
		for(int n=0; n<BLOCKSIZE; n++)
			p_sum += Mds[threadIdx.y][n]*Nds[n][threadIdx.x];
		__syncthreads(); 
	}
	P.elements[ row*P.width+col ] = p_sum;
}

__global__ void MatrixMulKernelB(Matrix M, Matrix N, Matrix P)
{
        __shared__ float Mds[bm][bk];
        __shared__ float Nds[bk][bn];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float p_sum=0.0f;
	for(int m=0; m<(M.width-1)/bm+1; m++)
	{
		if( threadIdx.x<bk )
		{
			if((m*bm+threadIdx.x)<M.width&&row<M.height&&threadIdx.x<bk)
				Mds[threadIdx.y][threadIdx.x] = M.elements[row*M.width+(m*bm+threadIdx.x)];
			else
				Mds[threadIdx.y][threadIdx.x] = 0.0;
		}

		if((m*bm+threadIdx.y)<N.height&&col<N.width)	
			Nds[threadIdx.y][threadIdx.x] = N.elements[(m*bm+threadIdx.y)*N.width+col];
		else
			Nds[threadIdx.y][threadIdx.x] = 0.0;
		__syncthreads();
		for(int n=0; n<bk; n++)
		{
			p_sum += Mds[threadIdx.y][n]*Nds[n][threadIdx.x];
		}
		__syncthreads(); 
	}
	if(row<P.height&&col<P.width)
		P.elements[ row*P.width+col] = p_sum;
}

#endif 
