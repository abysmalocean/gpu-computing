/*	if(bh == 0)
	{
		//top
		for(i=0;i<block_size+4;i++)
			localmem[i]=localmem[block_size+4+i]=0.0f;
	}
	else
	{
		//not top
		for(i=0;i<block_size+4;i++)
		{
			localmem[i]		=B[((bh*block_size)-2)*width+bw*block_size-2+i];
			localmem[block_size+4+i]=B[((bh*block_size)-1)*width+bw*block_size-2+i];
		}
	}

	if(bh == grp_cnt_height)
	{
		//bottom
		for(i=0;i<block_size+4;i++)
			localmem[(block_size+2)*(block_size+4)+i]=localmem[(block_size+3)*(block_size+4)+i]=0.0f;
	}
	else
	{
		//not bottom
		for(i=0;i<block_size+4;i++)
		{
			localmem[(block_size+2)*(block_size+4)+i]=B[((bh+1)*block_size+1)*width+bw*block_size-2+i];
			localmem[(block_size+3)*(block_size+4)+i]=B[((bh+1)*block_size+2)*width+bw*block_size-2+i];
		}
	}

	
	if( bw == 0 )
	{
		//left
		for(i=0;i<block_size+4;i++)
			localmem[i*(block_size+4)]=localmem[i*(block_size+4)+1]=0.0f;
	}
	else
	{	
		//not left
		for(i=0;i<block_size+4;i++)
		{
			localmem[i*(block_size+4)]=B[(bh*block_size+i-2)*width+bw*block_size-2];
			localmem[i*(block_size+4)+1]=B[(bh*block_size+i-2)*width+bw*block_size-1];
		}
	}

	if(bw == grp_cnt_width)
	{
		//right
		for(i=0;i<block_size+4;i++)
			localmem[(i+1)*(block_size+4)-2] = localmem[(i+1)*(block_size+4)-1]=0.0f;
	}
	else
	{
		//not right
		for(i=0;i<block_size+4;i++)
		{
			localmem[(i+1)*(block_size+4)-2]=B[(bh*block_size+i-2)*width+(bw+1)*block_size];
			localmem[(i+1)*(block_size+4)-1]=B[(bh*block_size+i-2)*width+(bw+1)*block_size+1];
		}
	}*/

__kernel
void convolution(__global float* A,
		 __global float* B,
		 __global float* C,
		  __local float* localmem,
	  	  __local float* locala,
		            int  height, 
		            int  width,
			    int  grp_cnt_height,
			    int  grp_cnt_width,	
			    int  kernel_size,
			    int  block_size)
{
	int col = get_group_id(0)*get_local_size(0)+get_local_id(0);
	int row = get_group_id(1)*get_local_size(1)+get_local_id(1);  
	if((row>=height)||(col>=width))
		return;

	//initialize matrix A
	int i,j;		
	for( i=0;i<kernel_size*kernel_size;i++)
		locala[i]=A[i];

	//initialize the boundary value of local memory
	int bh = get_group_id(1);
	int bw = get_group_id(0);

	for(i=0;i<block_size+4;i++)
	{
		for(j=0;j<block_size+4;j++)
			localmem[i*(block_size+4)+j]=B[(bh*block_size+i)*(width+4)+bw*block_size+j];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	int m,n;
	for(m=0;m<kernel_size;m++)
	{
		for(n=0;n<kernel_size;n++)
		{
			sum+=locala[m*kernel_size+n]*localmem[(get_local_id(1)+m)*(block_size+4)+get_local_id(0)+n];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	C[row*width+col]=sum;
	return;
}
