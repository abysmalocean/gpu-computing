// 2D Convolution: C = A (*) B, A is the 5x5 kernel matrix, B is the image matrix.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <time.h>
#include <inttypes.h>
#include "2Dconvolution.h"
#define PLATFORM_TO_USE 0

//extern "C"
void chk(cl_int status, const char* cmd);
unsigned int roundup(unsigned int value, unsigned int multiple);
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);
Matrix AllocateMatrix(int height, int width, int init);
int compareMatrix(float* ref, float* maxtrix, int size, float tolerance);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeMatrix(Matrix* M);
void PaddingMatrixB(float* paddingb,const float* hb, int height, int width);
char* readSource(char* kernelPath);
void ConvolutionOnDevice(const Matrix A, const Matrix B, Matrix C);
int readParam(const char* filename, int* data);
void WriteToFile(const char* filename,float* data,int height, int width);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	Matrix  A;
	Matrix  B;
	Matrix  C;
	
	srand(2012);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
		B  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		C  = AllocateMatrix(B.height, B.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int params[2] = {0}; 
		if(ReadParam(argv[1],params) != 2)
		{
			printf("Error reading parameter file\n");
			return 1;
		}
		A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		B  = AllocateMatrix(params[0], params[1], 0);		
		C  = AllocateMatrix(params[0], params[1], 0);
		(void)ReadFile(&A, argv[2]);
		(void)ReadFile(&B, argv[3]);
	}

	// Convolution on the device
	ConvolutionOnDevice(A, B, C);
	WriteToFile("opencl",C.elements, C.height,C.width);
	// compute the matrix multiplication on the CPU for comparison
	Matrix reference = AllocateMatrix(C.height, C.width, 0);
	computeGold(reference.elements, A.elements, B.elements, B.height, B.width);
	WriteToFile("cpu",reference.elements,reference.height,reference.width);
	// in this case check if the result is equivalent to the expected soluion
	int res = compareMatrix(reference.elements, C.elements, C.width*C.height, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
	if(argc == 5)
	{
		WriteFile(C, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(C, argv[1]);
	}   

	//Free matrices
	FreeMatrix(&A);
	FreeMatrix(&B);
	FreeMatrix(&C);
	return 0;
}

void chk(cl_int status, const char* cmd)
{
	if( status != CL_SUCCESS)
	{
		printf("%s failed (%d)\n", cmd, status);
		exit(-1);
	}
}

unsigned int roundup(unsigned int value, unsigned int multiple)
{
	unsigned int remainder = value%multiple;
	if(remainder!= 0)
	{
		value+=(multiple-remainder);
	}
	return value;
}

void computeGold(float* C, const float* A, const float* B, unsigned int hB, unsigned int wB)
{
	unsigned int i,j,m,n;	
	// For each element in the result matrix matrix
	for (i = 0; i < hB; ++i)
	{
        	for (j = 0; j < wB; ++j) 
		{
			double sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int mbegin = (i < 2)? 2 - i : 0;
			unsigned int mend = (i > (hB - 3))?hB - i + 2 : 5;
			unsigned int nbegin = (j < 2)? 2 - j : 0;
			unsigned int nend = (j > (wB - 3))?(wB-j) + 2 : 5;
			// overlay A over B centered at element (i,j).  For each 
			//  overlapping element, multiply the two and accumulate
			for(m = mbegin; m < mend; ++m) 
			{
				for(n = nbegin; n < nend; n++) 
				{
					sum += A[m * 5 + n]*B[wB*(i + m - 2) + (j+n - 2)];
				}
			}
			// store the result
			C[i*wB + j] = (float)sum;
        	}
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix A, const Matrix B, Matrix C)
{
	cl_int status;
	//discovery platform
	cl_platform_id platforms[2];
	cl_platform_id platform;
	status = clGetPlatformIDs(2,platforms, NULL);
	printf("status is : %d\n",status);
	chk(status,"clGetPlatformIDs");
	platform = platforms[PLATFORM_TO_USE];

	//discovery device
	cl_device_id device;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	chk(status,"clGetDeviceIDs");
	printf("status is : %d\n",status);

	//setup the context of opencl
	cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
	cl_context context;
	printf("status is : %d\n",status);
	context = clCreateContext(props, 1, &device, NULL, NULL, &status);
	chk(status,"clCreateContext");
	
	//create command queue
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	printf("status is : %d\n",status);
	chk(status, "clCreateCommandQueue");

	//create memory buffers
	cl_mem dma;
	cl_mem dmb;
	cl_mem dmc;
	dma = clCreateBuffer(context, CL_MEM_READ_ONLY, A.height*A.width*sizeof(float), NULL, &status);
	printf("status is : %d\n",status);
	chk(status, "clCreateBuffer");
	dmb = clCreateBuffer(context, CL_MEM_READ_ONLY, B.height*B.width*sizeof(float), NULL, &status);
	printf("status is : %d\n",status);
	chk(status, "clCreateBuffer");
	dmc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, C.height*C.width*sizeof(float), NULL, &status);
	printf("status is : %d\n",status);
	chk(status, "clCreateBuffer");

	//write data to the cl_mem
	status=clEnqueueWriteBuffer(queue, dma, CL_TRUE, 0, A.height*A.width*sizeof(float), A.elements, 0, NULL, NULL);
	printf("status is : %d\n",status);
	chk(status, "clEnqueueWriteBuffer");
	status=clEnqueueWriteBuffer(queue, dmb, CL_TRUE, 0, B.height*B.width*sizeof(float), B.elements, 0, NULL, NULL);
	printf("status is : %d\n",status);
	chk(status, "clEnqueueWriteBuffer");

	//read the kernel file
	char* source = readSource("2Dconvolution.cl");

	//create program
	cl_program program;
 	program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);
	printf("status is : %d\n",status);
	chk(status, "clCreateProgramWithSource");
	char* infobuf[KERNEL_BUF_SIZE];
	size_t infosize=0;
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	cl_int info = clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG,KERNEL_BUF_SIZE,infobuf,&infosize);
	//printf("%s\n",infobuf);
	//printf("Returned info size is: %d\n",infosize);
	printf("status is : %d\n",status);
	chk(status,"clBuildProgram");

	//create kernel
	cl_kernel kernel;
	//create kernel
	kernel = clCreateKernel(program, "convolution", &status);
	printf("status is : %d\n",status);
	chk(status, "clCreateKernel");
	
	int kernel_size=KERNEL_SIZE;
	int tile_size=TILE_SIZE;
	int work_group_size=WORK_GROUP_SIZE;
	size_t localsize[2] ={work_group_size,work_group_size};
	int group_item_1 = (roundup(C.height,TILE_SIZE)/TILE_SIZE)*WORK_GROUP_SIZE;
	int group_item_0 = (roundup(C.width,TILE_SIZE)/TILE_SIZE)*WORK_GROUP_SIZE;

	size_t globalsize[2] ={group_item_0,group_item_1};
	size_t localmem = WORK_GROUP_SIZE*WORK_GROUP_SIZE*sizeof(float);	

	//get total local memory size in a work group	
	cl_ulong local_size;
	status = clGetDeviceInfo(device,CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),&local_size,0);
	chk(status, "clGetDeviceInfo");

	//printf("Local memmory size is:%" PRIu64 "\n",local_size);
	//printf("Assigned the local memory size is:%d\n",localmem);
	
	//pass the parameter to kernel function
	status =clSetKernelArg(kernel, 0, sizeof(cl_mem), &dma);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 1, sizeof(cl_mem), &dmb);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 2, sizeof(cl_mem), &dmc);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 3, localmem, NULL);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 4, sizeof(int), &C.height);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 5, sizeof(int), &C.width);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 6, sizeof(int), &kernel_size);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 7, sizeof(int), &tile_size);
	printf("status is : %d\n",status);
	status|=clSetKernelArg(kernel, 8, sizeof(int), &work_group_size);
	printf("status is : %d\n",status);
	chk(status,"clSetKernekArg");

	cl_event event;
	//pass the work-item and work-group dimension to kernel
	status=clEnqueueNDRangeKernel(queue,kernel,2,NULL,globalsize,localsize,0,NULL,&event);
	printf("status is : %d\n",status);
	chk(status, "clEnqueueNDRange");
	clFinish(queue);
	//get the data back to host from device
	status=	clEnqueueReadBuffer(queue, dmc, CL_TRUE, 0, C.height*C.width*sizeof(float), C.elements, 0, NULL, NULL);
	chk(status, "clEnqueueReadBuffer");
	clFinish(queue);
	clWaitForEvents(1,&event);
	cl_ulong time_start, time_end;
	double total_time;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start),&time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end),&time_end, NULL);
	total_time= time_end-time_start;
	printf("Execution time in milliseconds = %0.4f ms\n",(total_time/1000000.0));
	
	//release cl_mem, kernel, queue, context
	clReleaseMemObject(dma); 	
	clReleaseMemObject(dmb);
	clReleaseMemObject(dmc);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return;
}

//Allocate a device matrix of dimensions height*width
//If init == 0, initialize to all zeroes.  
//If init == 1, perform random initialization.
//If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    //don't allocate memory on option 2
    if(init == 2)
	return M;
    M.elements = (float*) malloc(size*sizeof(float));

    unsigned int i;
    for(i = 0; i < M.height * M.width; i++)
    {
	M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	if(rand() % 2)
	    M.elements[i] = - M.elements[i];
    }
    return M;
}	

int compareMatrix(float* ref, float* maxtrix, int size, float tolerance)
{
	int i;
	for( i=0; i<size; i++ )
	{
		if(abs(ref[i]-maxtrix[i])>tolerance)
			return 0;
	}
	return 1;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

void PaddingMatrixB(float* paddingb, const float* hb, int height, int width)
{
	int a,b;
	for(a=0;a<height+4;a++)
	{
		for(b=0;b<width+4;b++)
			paddingb[a*(width+4)+b]=0.0f;
	}
	for(a=0;a<height;a++)
	{
		for( b=0;b<width;b++)
			paddingb[(a+2)*(width+4)+b+2]=hb[a*width+b];
	}
	WriteToFile("paddingb",paddingb,height+4,width+4);
}


// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height * M->width;
	
	size_t result;
	FILE *p;
	p = fopen(file_name, "rb");
	if( p == NULL )
	{
		printf("Can NOT open the file!\n");
		exit(1);
	}
	result = fread(M->elements, 1, data_read, p);
	if( result != data_read )
	{
		printf("Does not read enough data to matrix!\n");
		exit(3);
	}
	fclose(p);
	return data_read;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
	//cutWriteFilef(file_name, M.elements, M.width*M.height, 0.0001f);
	WriteToFile(file_name, M.elements, M.height, M.width);
}

char* readSource(char* kernelPath)
{
	cl_int status;
	FILE *fp;
	char *source;
	long int size;
//	printf("Program file is: %s\n", kernelPath);
	fp = fopen(kernelPath, "rb");
	if(!fp)
	{
		printf("Could not open kernel file!\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if(status != 0)
	{
		printf("Error seeking to end of file!\n");
		exit(-1);
	}

	size = ftell(fp);
	if(size < 0)
	{
		printf("Error getting file position!\n");
		exit(-1);
	}

	rewind(fp);
	source = (char*)malloc(size + 1);
	int i;
	for(i=0; i<size+1; i++)
	{
		source[i] = '\0';
	}
	if( source == NULL )
	{
		printf("Error allocating space for the kernel source\n");
		exit(-1);
	}
	fread(source, 1, size, fp);
	source[size] = '\0';
	return source;
}

int ReadParam(const char* filename, int* data)
{
	int cnt=0;
	FILE *fp;
	fp=fopen(filename, "r");
	if(fp != NULL)
	{
		while(fscanf(fp,"%d",&data[cnt])!=EOF)
			cnt++;
	}
	return cnt;
}

void WriteToFile(const char* filename,float* data,int height, int width)
{
	FILE *fp = fopen(filename,"w");
	if( fp == NULL )
	{
		printf("Openfile %s failed!\n",filename);
		return;
	}
	int i,j;
	for( i=0; i<height; i++ )
	{
		for(j=0;j<width;j++)
			fprintf(fp,"%8.4g\t",data[i*width+j]);	
		fprintf(fp,"\n",NULL);
	}
	fclose(fp);
}
