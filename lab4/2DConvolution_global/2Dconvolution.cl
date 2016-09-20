__kernel
void convolution(__global float* A,
				 __global float* B,
				 __global float* C,
							int  height, 
							int  width)
{
	int th = get_global_id(1);
	int tw = get_global_id(0);
	float sum = 0.0f;
	int mbegin = (th<2)?(2-th):0;
	int mend = (th>=(height-2))?(height-th+2):5;
	int nbegin = (tw<2)?(2-tw):0;
	int nend = (tw>=(width-2))?(width-tw+2):5;
	int m,n;
	for(m=mbegin;m<mend;m++)
	{
		for(n=nbegin;n<nend;n++)
		{
			sum+=A[m*5+n]*B[(th+m-2)*width+tw+n-2];
		}
	}
	C[th*width+tw]=sum;
	return;
}
