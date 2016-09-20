// Matrix multiplication: P = M * N.
// Device code.

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float p_sum = 0.0, ca=0.0, cb=0.0;

    for( int i=0; i<MATRIX_SIZE; i++ )
    {
        ca = M.elements[MATRIX_SIZE * tx + i];
        cb = N.elements[MATRIX_SIZE * i + ty];
        p_sum += ca*cb;
    }
    P.elements[tx*MATRIX_SIZE+ty]=p_sum;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
