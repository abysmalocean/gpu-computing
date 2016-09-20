#include <stdio.h>
#include <cutil.h>
#include <cuda.h>
#include "square_kernel.cu"

//function declaration
int checkDeviceProp( cudaDeviceProp p );
void square_with_pinned_memory( int n );

int main( int argc, char* argv[] )
{
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev,0);
	int success = checkDeviceProp(dev);
	if(success != 0 )
		square_with_pinned_memory(32);
	return 0;
}

int checkDeviceProp( cudaDeviceProp p )
{
	int support = p.canMapHostMemory;
	if( support == 0 )
		printf("%s does not support mapping host memory. \n", p.name);
	else
		printf("%s supports mapping host memory. \n", p.name);
	return support;
}

void square_with_pinned_memory( int n )
{
	float *h_data=NULL;


	size_t sz=n*sizeof(float);
	h_data=(float*)malloc(sz);
	cudaError_t error;
	for( int i=0; i<n; i++ )
		h_data[i] = (float)(i+1);
	float* d_data = NULL;
	error = cudaMalloc((void**)&d_data,sz );
	error = cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice);
	if( error != 0 )
	{
		printf("There is error!\n");
		return;
	}
	square<<<1,n>>>(d_data);
	cudaMemcpy(h_data, d_data, sz, cudaMemcpyDeviceToHost);
	printf("After squaring %d numbers!\n", n);
	for( int i=0; i<n; i++ )
		printf("%d\n", (int)h_data[i]);
	cudaFreeHost(h_data);
}
