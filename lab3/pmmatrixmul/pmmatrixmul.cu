
/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include <cuda.h>
#include <cutil_inline.h>

// includes, kernels
#include <pmmatrixmul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

#define MACRO_VARIABLE 10

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrixLock(const Matrix M);
Matrix AllocateMatrixLock(int height, int width, int init);
void FreeMatrixLock(Matrix *M);
void MatrixReset(Matrix M);
void WriteFile(Matrix M, char* filename);
bool cudaPropChk();
void MatrixMulOnDeviceA(const Matrix M, const Matrix N, Matrix P);
void MatrixMulOnDeviceB(const Matrix M, const Matrix N, Matrix P);

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	//Matrices for the program
	Matrix  M;
	Matrix  N;
	Matrix  P;

	if (argc==4) 
	{
		HM = atoi(argv[1]); //height of Matrix M
		WM = atoi(argv[2]); //Width of Matrix M
		WN = atoi(argv[3]); //Width of Matrix N
		HN = WM;
		HP = HM;
		WP = WN;
	}

	// Number of elements in the solution matrix
	// Assuming square matrices, so the sizes of M, N and P are equal
	unsigned int size_elements = WP * HP;
	
	srand(2012);
	
	// Check command line for input matrix files
	// No inputs provided
	// Allocate and initialize the matrices
	M  = AllocateMatrixLock(HM, WM, 1);
	N  = AllocateMatrixLock(HN, WN, 1);
	P  = AllocateMatrixLock(HP, WP, 0);
    	
	// compute the matrix multiplication on the CPU for comparison
    	Matrix reference = AllocateMatrixLock(HP, WP, 0);
	clock_t st=clock();
    	computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);
	
	st = clock()-st;
	printf("CPU executation is %.4f\n",(double)st/CLOCKS_PER_SEC);
    	
	if( cudaPropChk() )
	{
		printf("The kernel size is:%d\n",THREAD_SIZE);
		MatrixMulOnDeviceA(M, N, P);
		// check if the device result is equivalent to the expected solution
    	CUTBoolean res = cutComparefe(reference.elements, P.elements, size_elements, 0.001f);
    	printf("Part A Test %s\n", (1 == res) ? "PASSED" : "FAILED");
		MatrixReset(P);
		MatrixMulOnDeviceB(M, N, P);
    	
		// check if the device result is equivalent to the expected solution
    		res = cutComparefe(reference.elements, P.elements, size_elements, 0.001f);
    		printf("Part B Test %s\n", (1 == res) ? "PASSED" : "FAILED");
	}
	else
		printf("The GPU device can NOT support memory map or overlap execution!\n");

	// Free host matrices
	FreeMatrixLock(&M);
	FreeMatrixLock(&N);
	FreeMatrixLock(&P);
	return 0;
}

//cuda capable device properities check
bool cudaPropChk()
{
	printf("/*********************************************Device information check!*******************************************/\n");
	int devCount = 0;  //the number of GPU device
	cudaDeviceProp prop;
	//get the number of GPU device
	cudaGetDeviceCount(&devCount);	
	printf("The total number of GPU device is:%d\n", devCount);

	for(int i=0; i<devCount; i++)
	{
		//get cuda capable device properities				
		cudaGetDeviceProperties(&prop, i);
		//output the number of GPU device
		printf("The index of GPU device is:%d!\n", i+1);
		//output the name of GPU device
		printf("The device is:%s!\n",prop.name);
		//check if the GPU device is integrated	
		if( prop.integrated == true )
			printf("This is an integrated GPU!\n");
		else
			printf("This is an discrete GPU!\n");
		//check if the device support memory map
		if(prop.canMapHostMemory != 1)
		{
			printf("Device can NOT map memory!\n");
			return false;
		}
		else
			printf("Device can map memory!\n");
		//printf("%d", prop.deviceOverlap);
		if(prop.deviceOverlap != 1)
		{
			printf("Device can NOT execute overlap!\n");
			return false;
		}
		else
			printf("Device can execute overlap!\n");
		//place the runtime into a state where it will be able to allocate zero-copy buffers
		cudaSetDeviceFlags(cudaDeviceMapHost);
	}
	printf("/*********************************************Check complete!*****************************************************/\n");
	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
//zero-copy
void MatrixMulOnDeviceA(const Matrix M, const Matrix N, Matrix P)
{
	//Interface host call to the device kernel code and invoke the kernel
	cudaEvent_t start, stop;
	float elapsedTime=0.0f;
	//bind the host pointer with device pointer
	Matrix Md=AllocateDeviceMatrixLock(M);
	Matrix Nd=AllocateDeviceMatrixLock(N);
	Matrix Pd=AllocateDeviceMatrixLock(P);
	//start the cuda event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//launch the kernel
	int blky = (P.height%THREAD_SIZE==0)?P.height/THREAD_SIZE:P.height/THREAD_SIZE+1;
	int blkx = (P.width%THREAD_SIZE==0)?P.width/THREAD_SIZE:P.width/THREAD_SIZE+1;
	dim3 block2D(blkx, blky);
	dim3 thread2D(THREAD_SIZE, THREAD_SIZE);
	MatrixMulKernelA<<<block2D, thread2D>>>(Md, Nd, Pd);
	//cudaThreadSynchronize();
	cudaDeviceSynchronize();
	//stop the cuda event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//output the execution time
	printf("GPU execution time of kernel A is:%f\n",elapsedTime);
	return;
}

void MatrixMulOnDeviceB(const Matrix M, const Matrix N, Matrix P)
{
	if( M.height%2 !=0 || M.width%2 != 0 || N.height%2 != 0 || N.width%2 != 0 )
	{
		printf("The input is not right!~Please reinput the dimension of matrix M and N!\n"); 
		return;
	}
	printf("The dimension of Matrix M is %d*%d\n", M.height, M.width);
	printf("The dimension of Matrix N is %d*%d\n", N.height, N.width);
	//allocate device matrix
	Matrix Md=AllocateDeviceMatrixLock(M);
	Matrix Nd=AllocateDeviceMatrixLock(N);
	Matrix Pd=AllocateDeviceMatrixLock(P);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0f;
	//cuda event create
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//cuda stream declaration and creation
	cudaStream_t stream[STREAMS_COUNT];
	for( int i=0; i<STREAMS_COUNT; i++ )
	{
		cudaStreamCreate(&(stream[i]));
	}
	//define the dimension of block and thread
	int blky = ((P.height/2)%THREAD_SIZE==0)?P.height/(THREAD_SIZE*2):P.height/(THREAD_SIZE*2)+1;
	int blkx = ((P.width/2)%THREAD_SIZE==0)?P.width/(THREAD_SIZE*2):P.width/(THREAD_SIZE*2)+1;
	dim3 block2D(blkx, blky);
	dim3 thread2D(THREAD_SIZE, THREAD_SIZE);
	//Interface host call to the device kernel code and invoke the kernel
	//launch 4 kernels 
	for(int i=0; i<STREAMS_COUNT; i++)
	{
		MatrixMulKernelB<<<block2D, thread2D, 0, stream[i]>>>(Md, Nd, Pd, i);
	}
	//destroy 4 streams
	for(int i=0; i<STREAMS_COUNT; i++)
	{
		cudaStreamSynchronize(stream[i]);	
	}
	//cuda event stop and synchronize
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU execution time of kernel B is:%f\n",elapsedTime);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrixLock(Matrix M)
{
	Matrix Mdevice = M;
	cudaHostGetDevicePointer((void **)&(Mdevice.elements), M.elements,0);
	return Mdevice;
}

//Allocate a matrix of dimensions height*width
//If init == 0, initialize to all zeroes.  
//If init == 1, perform random initialization.
Matrix AllocateMatrixLock(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;
	cudaHostAlloc((void**)&M.elements, M.height*M.width*sizeof(float), cudaHostAllocMapped);
	cudaHostRegister(M.elements, size*sizeof(float),CU_MEMHOSTALLOC_DEVICEMAP);
	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
	return M;
}	

void FreeMatrixLock(Matrix *M)
{
	cudaHostUnregister(M->elements);
	cudaFreeHost(M->elements);
	M->elements = NULL;
}

void MatrixReset(Matrix M)
{
	for (int i=0;i<M.width*M.height;i++)
		M.elements[i] = 0;
}

void WriteFile(Matrix M, char* filename)
{
	cutWriteFilef(filename, M.elements, M.height*M.width, 0.0001f);
}
