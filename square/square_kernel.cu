__global__ void square( float *x )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	x[i]=x[i]*x[i];
}
