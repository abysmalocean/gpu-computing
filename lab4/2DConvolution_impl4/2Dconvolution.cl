__kernel
void convolution(__global float* A,
		 __global float* B,
		 __global float* C,
		  __local float* localmem,
		            int  height, 
		            int  width,
			    int  kernel_size,
			    int  tile_size,
			    int  work_group_size)
{
	int bh = get_group_id(1);
	int bw = get_group_id(0);
	int th = get_local_id(1);
	int tw = get_local_id(0);	
	int row_o = bh*tile_size+th;
	int col_o = bw*tile_size+tw;
	int row_i = row_o-2;
	int col_i = col_o-2;

	if((row_i >= 0)&&(row_i < height)&&(col_i >= 0)&&(col_i < width))
	{
		localmem[th*work_group_size+tw] = B[row_i*width + col_i];
	}
	else
	{
		localmem[th*work_group_size+tw] = 0.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum=0.0f;
	int m,n;
	if((th<tile_size)&&(tw<tile_size))
	{
		for(m=0;m<5;m++)
		{
			for(n=0;n<5;n++)
				sum+=A[m*kernel_size+n]*localmem[(th+m)*work_group_size+tw+n];
		}
		if((row_o<height)&&(col_o<width))
		C[row_o*width+col_o]=sum;
	}
	return;
}
