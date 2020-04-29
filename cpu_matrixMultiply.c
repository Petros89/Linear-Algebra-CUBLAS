/*
 * Purpose: Demonstrate and compare run time of the native matrix multiplication
 * on the CPU compared with Level-1 CBLAS (cblas_ddot & cblas_daxpy) and Level-3
 * CBLAS (cblas_dgemm) libraries.
 *
 *
 * Author: Petros Apostolou
 * Date: 11/29/2018
 *
 * modules to load: 1)gcc/6.3.0  2)lapack/3.8.0
 *
 * to compile: gcc -O3 cpu_matrixMultiply.c -lcblas -o cpu_matrixMultiply.exe -lm
 * 
 * to execute: ./cpu_matrixMultiply.exe < m > < n > < k >
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include "timer.h"
#include "cblas.h"





/*--------------let's print the first rows of each matrix----------------*/
void printMatrix (double *matrix)
{
    int i, j, idx;
    int nrow = 6; // Matrix(A or B or C)_NROW
    int ncol = 6; // Matrix(A or B or C)_NCOL

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            idx = j + i * ncol;
            printf("%8.2lf ; ", matrix[idx]);
        }
        printf("\n");
    }
    printf("\n");
}


/*--------------let's print the matrices for a small [5x5] problem----------------*/
void print_Tiny_Matrix (int rows, int cols, double *matrix)
{
    int i, j, idx;
//    int nrow = 6; // Matrix(A or B or C)_NROW
//    int ncol = 6; // Matrix(A or B or C)_NCOL

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            idx = j + i * cols;
            printf("%8.2lf ; ", matrix[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

/*--------------let's give some values to our matrices------------------*/
void InitializeMatrices(int m, int n, int k, double *a, double *b)
{
    int i, j, p, idx;

    // initialize matrices a & b
    for (i = 0; i < m; i++) {
        for (p = 0; p < n; p++) {
            idx      = p + i * n;
            a[ idx ] = rand()  % 10 + 1;  /* Be careful--if not +1 ==> rand() returns pseudo-random numbers in [0,9] */
        }
    }

    for (p = 0; p < n; p++) {
        for (j = 0; j < k; j++) {
            idx      = j + p * k;
            b[ idx ] = rand()  % 10 + 1;
        }
    }
}


/*---------------Native matrix multiplication C = A * B-----------------*/
void matrixMultiply(int m, int n, int k, double *a, double *b, double *c)
{
    // this function does the following matrix multiplication c = a * b
    // a(i x p); b(p x j); c(i x j)

    int   i, j, p, idx;
    double sum = 0.0;
    // multiply the matrices C=A*B
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            for (p = 0; p < n; p++) {
                sum += a[ p + i * n ] * b[ j + p * k ];
            }
            c[ j + i * k ] = sum;
            sum            = 0.0;
        }
    }
}


/*------------CBLAS LEVEL-1:: inner product of A rows * columns of B (DDOT)--*/
void CBLAS_DDOT(int m, int n, int k, double *a, double *b, double *c)
{
    int i, j, idx;

    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
           idx = j + i * k;
           c[idx] = cblas_ddot(n, a + i*n, 1, b+j, k);
        }
    }


}


/*-----------CBLAS LEVEL-1:: linear combination of columns of arrays A and C using
------------------------------------array B as a scalar (C = B*A + A) (DAXPY)---*/
void CBLAS_DAXPY(int m, int n, int k, double *a, double *b, double *c)
{
    int i, j;
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
        cblas_daxpy(m, b[i + j * k], a + j, n, c + i, m);
        }
    }
}


/*------------CBLAS LEVEL-3:: C = alpha (A * B) + beta (DGEMM)----------------------*/
void CBLAS_DGEMM(int m, int n, int k, double *a, double *b, double *c)
{
    int i, j, idx;
    double alpha = 1.0;
    double beta  = 0.0;

/*--make sure to set the right order of matrix sizes, here: A[mxn], B[n,k], C[mxk]--*/
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\
                       m, k, n, alpha, a, n, b, k, beta, c, k);                 
}


/*--------let's start calling functions into the main program------------------*/
int main(int argc, char *argv[])
{
    if (argc < 4) {
        perror("Command-line usage: executableName < m > < n > < k >");
        exit(1);
    }

    int m = atof(argv[ 1 ]);  // Enter # of A rows
    int n = atof(argv[ 2 ]);  // Enter # of B rows = # of A columns
    int k = atof(argv[ 3 ]);  // Enter # of C columns
 


/*-----------let's do some casting in memmory allocation-----------------------*/
    double *a =  (double*) malloc(sizeof(*a) * m * n);
    double *b =  (double*) malloc(sizeof(*b) * n * k);
    double *c =  (double*) calloc(m * k, sizeof(*c));



/*----------here matrices A and B are filled up with real values---------------*/
    InitializeMatrices(m, n, k, a, b);

    //printf("# This file demonstrates the matrix-matrix multiplication using Level-3 BLAS (cblas_dgemm) library #\n");

/*---------let's see the matrix elments----------------------------------------*/
    //printf("================================================MATRIX A===================================================\n");
    //print_Tiny_Matrix(m, n, a);
    //printMatrix(a);

    //printf("================================================MATRIX B===================================================\n");
    //print_Tiny_Matrix(n, k, b);
    //printMatrix(b);


/*--let's get ready to time each function (native, ddot and daxpy) of mat/mult-*/
    double start, finish, elapsedTime_native, elapsedTime_ddot, elapsedTime_daxpy, elapsedTime_dgemm;


/*------------------Native Matrix Multiplication----------------------------------*/
    GET_TIME(start);
    matrixMultiply(m, n, k, a, b, c);
    GET_TIME(finish);   //  Timing measurements ends here !Be careful the time units
    elapsedTime_native = finish - start;


/*-----------------CBLAS_DDOT-----------------------------------------------------*/
    GET_TIME(start);
    CBLAS_DDOT(m, n, k, a, b, c);
    GET_TIME(finish);   //  Timing measurements ends here !Be careful the time units
    elapsedTime_ddot = finish - start;
    

/*-----------------CBLAS_DAXPY----------------------------------------------------*/
    GET_TIME(start);
    CBLAS_DAXPY(m, n, k, a, b, c);
    GET_TIME(finish);   //  Timing measurements ends here !Be careful the time units
    elapsedTime_daxpy = finish - start;


/*-----------------CBLAS_DGEMM----------------------------------------------------*/
    GET_TIME(start);
    CBLAS_DGEMM(m, n, k, a, b, c);
    GET_TIME(finish);   //  Timing measurements ends here !Be careful the time units
    elapsedTime_dgemm = finish - start;


/*----------let's make sure the multiplication implemented correctly------------*/
    //printf("================================================MATRIX C===================================================\n");
    //print_Tiny_Matrix(m, k, c);
    //printMatrix(c);


/*---------let's print the wall time in screen----------------------------------*/
    //printf("#Wall Time in seconds (s) measurements for a small square matrix size (N=10)\n");
    printf("Matrix Size (n) = %4d Native = %4.4f s DDOT = %4.4f s DAXPY = %4.4f s DGEMM = %4.4f s \n", n,\
           elapsedTime_native, elapsedTime_ddot, elapsedTime_daxpy, elapsedTime_dgemm);


/*--------let's clean up the memory---------------------------------------------*/
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
