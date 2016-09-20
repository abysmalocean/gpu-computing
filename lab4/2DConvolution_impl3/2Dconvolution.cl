__kernel
void convolution(__global float* A,
		 __global float* B,
		 __global float* C,
		  __local float* localmem,
		            int  height, 
		            int  width,
			    int  kernel_size,
			    int  block_size)
{
	int bh = get_group_id(1);
	int bw = get_group_id(0);
	int th = get_local_id(1);
	int tw = get_local_id(0);	
	int row = bh*get_local_size(1)+th;
	int col = bw*get_local_size(0)+tw;
	if((row>=height)||(col>=width))
		return;

	localmem[(th+2)*(block_size+4)+tw+2] = B[(bh*block_size+th+2)*(width+4)+bw*block_size+tw+2];

	//top two lines
	if((th==0)||(th==1))
	{
		localmem[th*(block_size+4)+tw+2]=B[(bh*block_size+th)*(width+4)+bw*block_size+tw+2];
	}
	//bottom two lines
	if((th==(block_size-1))||(th==(block_size-2)))
	{
		localmem[(th+4)*(block_size+4)+tw+2]=B[(bh*block_size+th+4)*(width+4)+bw*block_size+tw+2];
	}
	//left two lines
	if((tw==0)||(tw==1))
	{
		localmem[(th+2)*(block_size+4)+tw]=B[(bh*block_size+th+2)*(width+4)+bw*block_size+tw];
	}
	//right two lines
	if((tw==(block_size-1))||(tw==(block_size-2)))
	{
		localmem[(th+2)*(block_size+4)+tw+4]=B[(bh*block_size+th+2)*(width+4)+bw*block_size+tw+4];	
	}
	//left top
	if((th<2)&&(tw<2))
	{
		localmem[th*(block_size+4)+tw]=B[(bh*block_size+th)*(width+4)+bw*block_size+tw];
	}
	//right top
	if((th<2)&&(tw>(block_size-3)))
	{
		localmem[th*(block_size+4)+tw+4]=B[(bh*block_size+th)*(width+4)+bw*block_size+tw+4];
	}
	//left bottom
	if((th>(block_size-3))&&(tw<2))
	{
		localmem[(th+4)*(block_size+4)+tw]=B[(bh*block_size+th+4)*(width+4)+bw*block_size+tw];
	}
	//right bottom
	if((th>(block_size-3))&&(tw>(block_size-3)))
	{
		localmem[(th+4)*(block_size+4)+tw+4]=B[(bh*block_size+th+4)*(width+4)+bw*block_size+tw+4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	int m,n;
	for(m=0;m<kernel_size;m++)
	{
		for(n=0;n<kernel_size;n++)
		{
			sum+=A[m*kernel_size+n]*localmem[(th+m)*(block_size+4)+tw+n];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	C[row*width+col]=sum;
	return;
}
