/*
 * Purpose: Demonstrate and compare run time of the native matrix multiplication
 * on the GPU compared with Level-1 CUBLAS libraries (cublas_ddot & cublas_daxpy)
 *
 *
 * Author: Petros Apostolou
 * Date: 11/20/2018
 *
 * modules to load: 1)gcc/5.4.0  2)lapack/3.7.0  3)cuda
 *
 * to compile: gcc -O3 -gencode arch=compute_61,code=[compute_61,sm_61] gpu_matrixMultiply.c -lcublas -o gpu_matrixMultiply.exe -lm
 * 
 * to execute: ./gpu_matrixMultiply.exe
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include "cublas.h"
#include <cublas_v2.h>
#include "timer.h"
#include "timer_nv.h"
#include <cuda_runtime.h>

#define SIZE 5000 
#define A_NROW SIZE
#define A_NCOL SIZE
#define B_NROW A_NCOL
#define B_NCOL SIZE
#define C_NROW A_NROW
#define C_NCOL B_NCOL

#define BLOCKSIZE 16



/*--------------let's print the first rows of each matrix----------------*/
void printMatrix(double *matrix)
{
	int i, j, idx;

	int nrow = 6; // C_NROW
	int ncol = 6; // C_NCOL

	for (i = 0; i < nrow; i++) {
		for (j = 0; j < ncol; j++) {
			idx = j + i * ncol;
			printf("%8.2f ; ", matrix[idx]);
		}
		printf("\n");
	}
}


/*----CPU-----------Native matrix multiplication C = A * B-----------------*/
void matrixMultiplyCPU(double *a, double *b, double *c)
{
	int i, j, k, idx;

	// initialize matrices a & b

	for (i = 0; i < A_NROW; i++) {
		for (k = 0; k < A_NCOL; k++) {
			idx    = k + i * A_NCOL;
			a[idx] = rand() % 10 +1;
			// printf("%8.2f ; ", a[idx]);
		}
		// printf("\n");
	}
	// printf("==================\n");
	for (k = 0; k < B_NROW; k++) {
		for (j = 0; j < B_NCOL; j++) {
			idx    = j + k * B_NCOL;
			b[idx] = rand() % 10 + 1;
			// printf("%8.2f ; ", b[idx]);
		}
		// printf("\n");
	}
	// printf("==================\n");
	// this function does the following matrix multiplication c = a * b
	// a(i x k); b(k x j); c(i x j)

	for (i = 0; i < A_NROW; i++) {
		for (j = 0; j < B_NCOL; j++) {
			double sum = 0.;
			for (k = 0; k < A_NCOL; k++) {
				double aa = a[k + i * A_NCOL];
				double bb = b[j + k * B_NCOL];
				sum += aa * bb;
			}
			c[j + i * C_NCOL] = sum;
		}
	}
}


/*----GPU-----------Native matrix multiplication C = A * B-----------------*/
__global__ void matrixMultiplyGPU_gl(double *a, double *b, double *c)
{
	// Block index

	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Row index of matrices a and c

	int row = by * BLOCKSIZE + ty;

	// Column index of matrices a and b
	int col = bx * BLOCKSIZE + tx;

	double C_temp = 0.;

	for (int k = 0; k < A_NCOL; k++)
		C_temp += a[k + row * A_NCOL] * b[col + k * B_NCOL];

	c[col + row * C_NCOL] = C_temp;
}



/*---GPU------CBLAS LEVEL-1:: inner product of A rows * columns of B (DDOT)--*/
__inline__ void gpu_blas_ddot(double *a, double *b, double *c)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
        int i,j;
	for (i = 0; i < C_NROW; i++) {
		for (j = 0; j < C_NCOL; j++) {
                        int idx = (i * C_NCOL) + j;
			// Create a handle for CUBLAS
			cublasDdot(handle, B_NROW, &a[i * B_NROW], 1, &b[j], C_NCOL, &c[idx]);
			// Destroy the handle
		}
	}
	cublasDestroy(handle);
}



/*--GPU------CBLAS LEVEL-1:: linear combination of columns of arrays A and C using
------------------------------------array B as a scalar (C = B*A + A) (DAXPY)---*/
__inline__ void gpu_blas_daxpy(double *a, double *b, double *c)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
        int i,j;
	for (i = 0; i < C_NROW; i++) {
		for (j = 0; j < C_NCOL; j++) {
                       // int idx = (i * C_NCOL) + j;
			// Create a handle for CUBLAS
			cublasDaxpy(handle, C_NROW, &b[i + j*B_NCOL], &a[j], B_NROW, &c[i], C_NROW);
			// Destroy the handle
		}
	}
	cublasDestroy(handle);
}


/*------------CBLAS LEVEL-3:: C = alpha (A * B) + beta (DGEMM)----------------------*/
__inline__ void gpu_blas_dgemm(double *a, double *b, double *c)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
        
        const double   alf = 1;
        const double   bet = 0;
        const double   *alpha = &alf;
        const double   *beta  = &bet;


/*--make sure to set the right order of matrix sizes, here: A[mxn], B[n,k], C[mxk]--*/
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,\
                   A_NCOL, A_NROW, B_NCOL, alpha, b, B_NCOL, a, A_NCOL, beta, c, B_NROW);                 

	cublasDestroy(handle);
}



/*---GPU------Matrix Multiplication using Shared Memory (not used in this case)-------*/
__global__ void matrixMultiplyGPU_sh(double *a, double *b, double *c)
{
	// shared memory for submatrices
	__shared__ double a_sh[BLOCKSIZE][BLOCKSIZE];
	__shared__ double b_sh[BLOCKSIZE][BLOCKSIZE];

	// Block index

	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Row index of matrices a and c

	int row = by * BLOCKSIZE + ty;

	// Column index of matrices a and b
	int col = bx * BLOCKSIZE + tx;

	double C_temp = 0.;

	for (int n = 0; n < (A_NCOL / BLOCKSIZE); n++) {
		a_sh[ty][tx] = a[row * A_NCOL + (n * BLOCKSIZE + tx)];
		b_sh[ty][tx] = b[(n * BLOCKSIZE + ty) * B_NCOL + col];
		__syncthreads();

		for (int k = 0; k < BLOCKSIZE; k++)
			C_temp += a_sh[ty][k] * b_sh[k][tx];
		__syncthreads();
	}
	c[row * C_NCOL + col] = C_temp;
}


/*-------------GPU version starts here-----------------------------------------*/

/*--------let's start calling functions into the main program------------------*/
int main(int argc, char *argv[])
{
	int gpuCount;

	cudaGetDeviceCount(&gpuCount);

//	printf("======================================================\n");
//	printf("Number of GPUs = %d\n", gpuCount);

/*--------let's check the device properties-----------------------------------*/

	cudaDeviceProp gpuSpecs;

	for (int i = 0; i < gpuCount; i++) {
	    cudaGetDeviceProperties(&gpuSpecs, i);

//	    printf("GPU Name: %s\n", gpuSpecs.name);
//	    printf("Total Global Memory: %ld\n", gpuSpecs.totalGlobalMem);
//	    printf("Compute Capability: %d.%d\n", gpuSpecs.major, gpuSpecs.minor);
	}

	// let's make sure that we use the device that we want. There can be multiple
	// GPUs on a computer

	gpuSpecs.totalGlobalMem = 5032706048;
	int myDevice;

	cudaGetDevice(&myDevice);

//	printf("The ID of the current GPU: %d\n", myDevice);

	cudaChooseDevice(&myDevice, &gpuSpecs);

	cudaSetDevice(myDevice);
//	printf("Total Global Memory: %ld\n", gpuSpecs.totalGlobalMem);
//	printf("======================================================\n");



/*------let's let the NVIDIA to take care allocation of matrices using the Unified Memory API----------*/
	double *a, *b, *c;
	double *a_d, *b_d, *c_d;

        cudaMallocManaged(&a,   SIZE*SIZE* sizeof (*a));
        cudaMallocManaged(&a_d, SIZE*SIZE* sizeof (*a));
        cudaMallocManaged(&b,   SIZE*SIZE* sizeof (*a));
        cudaMallocManaged(&b_d, SIZE*SIZE* sizeof (*a));
        cudaMallocManaged(&c,   SIZE*SIZE* sizeof (*a));
        cudaMallocManaged(&c_d, SIZE*SIZE* sizeof (*a));


/*-----let's give some values to our matrices and implement the native matrix multiplication on the CPU first----*/
       matrixMultiplyCPU( a, b, c );



/*----let's copy A & B matrices to the device (GPU)---------------------------------------------*/
      cudaMemcpy( a_d, a, sizeof(double)*A_NROW*A_NCOL, cudaMemcpyHostToDevice );
      cudaMemcpy( b_d, b, sizeof(double)*B_NROW*B_NCOL, cudaMemcpyHostToDevice );


/*----let's define the BLOCKS and GRID for the GPU computations---------------------------------*/
      dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
      dim3 dimGrid((C_NCOL + BLOCKSIZE -1)/dimBlock.x, (C_NROW + BLOCKSIZE -1)/dimBlock.y);
 
/*----let's get ready to time each function (native, ddot and daxpy) of mat/mult for the GPU----*/
      float elapsedTime_gpu_native; 
      float elapsedTime_gpu_ddot;   
      float elapsedTime_gpu_daxpy;  
      float elapsedTime_gpu_dgemm;  


/*****Function for Shared Memory checked only for the native matrix multiplication but is not included in this work********
********matrixMultiplyGPU_sh<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);   // A little bit faster but loosing some accuracy*****
**************************************************************************************************************************/

/*-----let's time the kernel for the native multiplication----------------------------------------------*/
       cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device

      cudaEventCreate(&timeStart);
      cudaEventCreate(&timeStop);
      cudaEventRecord(timeStart, 0);

      matrixMultiplyGPU_gl<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);

      cudaEventRecord(timeStop, 0);
      cudaEventSynchronize(timeStop);
      cudaEventElapsedTime(&elapsedTime_gpu_native, timeStart, timeStop);

/*----let's time the GPU for the Cublas Ddot LEVEL-1 Library--------------------------------------------*/
      cudaEventCreate(&timeStart);
      cudaEventCreate(&timeStop);
      cudaEventRecord(timeStart, 0);

      gpu_blas_ddot(a_d, b_d, c_d);

      cudaEventRecord(timeStop, 0);
      cudaEventSynchronize(timeStop);
      cudaEventElapsedTime(&elapsedTime_gpu_ddot, timeStart, timeStop);



/*----let's time the GPU for the Cublas Daxpy LEVEL-1 Library--------------------------------------------*/
      cudaEventCreate(&timeStart);
      cudaEventCreate(&timeStop);
      cudaEventRecord(timeStart, 0);

      gpu_blas_daxpy(a_d, b_d, c_d);

      cudaEventRecord(timeStop, 0);
      cudaEventSynchronize(timeStop);
      cudaEventElapsedTime(&elapsedTime_gpu_daxpy, timeStart, timeStop);


/*----let's time the GPU for the Cublas Dgemm LEVEL-3 Library--------------------------------------------*/
      cudaEventCreate(&timeStart);
      cudaEventCreate(&timeStop);
      cudaEventRecord(timeStart, 0);

      gpu_blas_dgemm(a_d, b_d, c_d);

      cudaEventRecord(timeStop, 0);
      cudaEventSynchronize(timeStop);
      cudaEventElapsedTime(&elapsedTime_gpu_dgemm, timeStart, timeStop);

/*-----let's Destroy the all the events----------------------------------------------*/
      cudaEventDestroy(timeStart);
      cudaEventDestroy(timeStop);

/*--let's copy the result matrix C back to host memory-------------------------------------------------*/
      cudaMemcpy( c, c_d, sizeof(double)*C_NROW*C_NCOL, cudaMemcpyDeviceToHost );


	    // let's print some rows of each matrix
	//printf("=============MATRIX A=============\n");
	//printMatrix(a);
	//printf("=============MATRIX B=============\n");
	//printMatrix(b);
	//printf("=============MATRIX C=============\n");
	//printMatrix(c);


/*----let's see some wall timings in the screen--------------------------*/
    printf("GPU Wall Time in seconds (s) measurements with respect to the square Matrix Size (n) \n");
    //printf("Matrix Size (n) = %4d  CUBLAS_DGEMM = %4.4f\n", SIZE, elapsedTime_gpu_dgemm*0.001);
    //printf("Matrix Size (n) = %4d  CUBLAS_NATIVE = %4.4f  CUBLAS_DGEMM = %4.4f\n",\
                  SIZE, elapsedTime_gpu_native*0.001, elapsedTime_gpu_dgemm*0.001);
    printf("Matrix Size (n)  Kernel   cublasDdot  cublasDaxpy  cublasDgemm\n");
    printf("%4d  %4.4f  %4.4f  %4.4f %4.4f\n", SIZE, elapsedTime_gpu_native*0.001, elapsedTime_gpu_ddot*0.001, elapsedTime_gpu_daxpy*0.001, elapsedTime_gpu_dgemm*0.001);
	

/*---let's measure Wall Time for different square matrix sizes (n)----*/
    //FILE *output;
    //output = fopen("GPUsizeVStime.dat", "w");    // !Wall time in seconds 

    //fprintf(output,"#GPU Wall Time in seconds (s) measurements with respect to the square Matrix Size (n) \n");
    //fprintf(output,"Matrix Size (n) = %4d  Native = %4.4f  CUBLAS_DDOT = %4.4f  CUBLAS_DAXPY = %4.4f CUBLAS_DGEMM = %4.4f\n",\
      SIZE, elapsedTime_gpu_native*0.001, elapsedTime_gpu_ddot*0.001, elapsedTime_gpu_daxpy*0.001, elapsedTime_gpu_dgemm*0.001);

    //fclose(output);



/*--------let's clean up the memory-----------------------------------*/
     cudaFree(a);
     cudaFree(b);
     cudaFree(c);
     cudaFree(a_d);
     cudaFree(b_d);
     cudaFree(c_d);

return EXIT_SUCCESS;
}
