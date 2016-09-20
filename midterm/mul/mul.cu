
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
#include <mul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void MatrixReset(Matrix M);
Matrix MatrixPadding(Matrix M);
void MatrixPadRemoving(Matrix M,Matrix M2);

void MatrixMulOnDeviceA1(const Matrix M, const Matrix N, Matrix P);
void MatrixMulOnDeviceA2(const Matrix M, const Matrix N, Matrix P);
void MatrixMulOnDeviceB(const Matrix M, const Matrix N, Matrix P);
void WriteFile(Matrix M, char* filename);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	// Matrices for the program
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
	M  = AllocateMatrix(HM, WM, 1);
	N  = AllocateMatrix(HN, WN, 1);
	P  = AllocateMatrix(HP, WP, 0);
    	
	//compute the matrix multiplication on the CPU for comparison
    	Matrix reference = AllocateMatrix(HP, WP, 0);
	clock_t st=clock();
    	computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);
        st = clock()-st;
	printf("CPU executation is %.4f\n",(double)st/CLOCKS_PER_SEC);
	
	CUTBoolean res;
	printf("/******************A1 begins*******************/\n");
	MatrixMulOnDeviceA1(M, N, P);
    	res = cutComparefe(reference.elements, P.elements, size_elements, 0.001f);
    	printf("Part A1 Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	MatrixReset(P);
	printf("/******************A2 begins*******************/\n");
	MatrixMulOnDeviceA2(M, N, P);
	res = cutComparefe(reference.elements, P.elements, size_elements, 0.001f);
	printf("Part A2 Test %s\n", (1 == res) ? "PASSED" : "FAILED");
   
	MatrixReset(P);
	printf("/******************B begins*******************/\n");
	MatrixMulOnDeviceB(M, N, P);
	res = cutComparefe(reference.elements, P.elements, size_elements, 0.001f);
	printf("Part B Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// Free host matrices
	FreeMatrix(&M);
	FreeMatrix(&N);
	FreeMatrix(&P);
	return 0;
}

void MatrixMulOnDeviceA1(const Matrix M, const Matrix N, Matrix P)
{
	//Allocate device matrices
	Matrix Md = AllocateDeviceMatrix(M);
	Matrix Nd = AllocateDeviceMatrix(N);
	Matrix Pd = AllocateDeviceMatrix(P);
	//copy data from host to device
	CopyToDeviceMatrix(Md, M);
	CopyToDeviceMatrix(Nd, N);
	CopyToDeviceMatrix(Pd, P); 
	//Setup the execution configuration
	int blky = (P.height%BLOCKSIZE==0)?P.height/BLOCKSIZE:P.height/BLOCKSIZE+1;
	int blkx = (P.width%BLOCKSIZE==0)?P.width/BLOCKSIZE:P.width/BLOCKSIZE+1;
	dim3 block2D(blkx, blky);
	dim3 thread2D(BLOCKSIZE, BLOCKSIZE);
	//cuda event start
	cudaEvent_t start, stop;
	float elapsedTime=0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// Launch the device computation threads!
	MatrixMulKernelA1<<<block2D,thread2D>>>(Md, Nd, Pd);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("The execution time of GPU for A1 is:%.4f\n",elapsedTime);
	// Read P from the device
	CopyFromDeviceMatrix(P, Pd); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

void MatrixMulOnDeviceA2(const Matrix M, const Matrix N, Matrix P)
{
	//Matrix Padding
	Matrix padM = MatrixPadding(M);
	Matrix padN = MatrixPadding(N);
	Matrix padP = MatrixPadding(P);
	//Allocate device matrices
	Matrix Md = AllocateDeviceMatrix(padM);
	Matrix Nd = AllocateDeviceMatrix(padN);
	Matrix Pd = AllocateDeviceMatrix(padP);
	//copy data from host to device
	CopyToDeviceMatrix(Md, padM);
	CopyToDeviceMatrix(Nd, padN);
	CopyToDeviceMatrix(Pd, padP); 
	//kernel configuration	
	int blky = padP.height/BLOCKSIZE;
	int blkx = padP.width/BLOCKSIZE;
	dim3 block2D(blkx, blky);
	dim3 thread2D(BLOCKSIZE, BLOCKSIZE);
	//cuda event start
	cudaEvent_t start, stop;
	float elapsedTime=0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Launch the device computation threads!
	MatrixMulKernelA2<<<block2D,thread2D>>>(Md, Nd, Pd);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("The execution time of GPU for A2 is:%.4f\n",elapsedTime);
	//Read P from the device
	CopyFromDeviceMatrix(padP, Pd);
	MatrixPadRemoving(P, padP); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
	FreeMatrix(&padM);
	FreeMatrix(&padN);
	FreeMatrix(&padP);
}

void MatrixMulOnDeviceB(const Matrix M, const Matrix N, Matrix P)
{
	//Allocate device matrices
	Matrix Md = AllocateDeviceMatrix(M);
	Matrix Nd = AllocateDeviceMatrix(N);
	Matrix Pd = AllocateDeviceMatrix(P);
	//copy data from host to device
	CopyToDeviceMatrix(Md, M);
	CopyToDeviceMatrix(Nd, N);
	CopyToDeviceMatrix(Pd, P); 
	//Setup the execution configuration
	int blky = (P.height%BLOCKSIZE==0)?P.height/BLOCKSIZE:P.height/BLOCKSIZE+1;
	int blkx = (P.width%BLOCKSIZE==0)?P.width/BLOCKSIZE:P.width/BLOCKSIZE+1;
	dim3 block2D(blkx, blky);
	dim3 thread2D(bn, bm);
	//cuda event start
	cudaEvent_t start, stop;
	float elapsedTime=0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// Launch the device computation threads!
	MatrixMulKernelB<<<block2D,thread2D>>>(Md, Nd, Pd);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("The execution time of GPU for B is:%.4f\n",elapsedTime);
	// Read P from the device
	CopyFromDeviceMatrix(P, Pd); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

// Allocate a device matrix of dimensions height*width
// If init == 0, initialize to all zeroes.  
// If init == 1, perform random initialization.
// If init == 2, initialize matrix parameters, but do not allocate memory 
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
        for(unsigned int i = 0; i < M.height * M.width; i++)
        {
                M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
        }
	return M;
}

//return a matrix which is used to padding in host
Matrix MatrixPadding(Matrix M)
{
	//assign the height and width of the matrix for padding
	int height = ((M.height-1)/BLOCKSIZE+1)*BLOCKSIZE;
	int width = ((M.width-1)/BLOCKSIZE+1)*BLOCKSIZE;
	Matrix M2 =AllocateMatrix(height,width,0);
	//assign the value of elements in matrix M
	for (int i=0;i<M.height;i++)
	{
		for (int j=0;j<M.width;j++)
			M2.elements[i*M2.width+j] = M.elements[i*M.width+j];
	}
	return M2;
}

void MatrixPadRemoving(Matrix M, Matrix M2)
{
	for (int i=0;i<M.height;i++)
	{
		for (int j=0;j<M.width;j++)
			M.elements[i*M.width+j] = M2.elements[i*M2.width+j];
	}
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
	cudaFree(M->elements);
	M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
	free(M->elements);
	M->elements = NULL;
}

void MatrixReset(Matrix M)
{
	for (int i=0;i<M.width*M.height;i++)
		M.elements[i] = 0;
}

void WriteFile(Matrix M, char* filename)
{
	cutWriteFilef(filename, M.elements, M.width*M.height, 0.0001f);
}
