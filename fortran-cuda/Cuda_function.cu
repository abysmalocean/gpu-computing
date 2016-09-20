#include <stdio.h>
#include <cuComplex.h>
#include "cuda.h"


/* Define complex multiply operation */
__device__ cuComplex ComplexMul(cuComplex a, cuComplex b)
{
    cuComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;

}

/* Define CUDA kernel that squares the input complex array */
__global__ void  square_complex(cuComplex *in, cuComplex *out, int N)
{
 unsigned int index   = blockIdx.x*blockDim.x+threadIdx.x;
 if( index<N ) 
  {
   out[index] = ComplexMul(in[index], in[index]);
  }

}


/* 
   Fortran subroutine arguments are passed by references.   
   call fun( array_a, array_b, N) will be mapped to
   function (*a, *b, *N);
*/
extern "C" void cudafunction_(cuComplex *a, cuComplex *b,  int *Np)
{
  int block_size=4;
  cuComplex *a_d;
  int N=*Np;
 
  /* Allocate complex array on device */
  cudaMalloc ((void **) &a_d , sizeof(cuComplex)*N);
  
  /* Copy array from host memory to device memory */
  cudaMemcpy( a_d, a,  sizeof(cuComplex)*N   ,cudaMemcpyHostToDevice);

  /* Compute execution configuration */
   dim3 dimBlock(block_size);
   dim3 dimGrid (N/dimBlock.x);
   if( N % block_size != 0 ) dimGrid.x+=1;

  /* Execute the kernel */
  square_complex<<<dimGrid,dimBlock>>>(a_d,a_d,N);

  /* Copy the result back */
   cudaMemcpy( b, a_d, sizeof(cuComplex)*N,cudaMemcpyDeviceToHost);  

  /* Free memory on the device */
  cudaFree(a_d);

  return;
}

