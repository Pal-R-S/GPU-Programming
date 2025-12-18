#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "help.cu"

__global__ void kernel_matrix_mult(unsigned int n, unsigned int n1,unsigned int n2,unsigned int n3, int *data_a, int *data_b, int*data_mult){
    int tid =gettid_1D();
    if (    tid < n) {
        int ind_matr1=tid/n3;
        int row_1=ind_matr1/(n2);
        int col_1=(ind_matr1%n2);
        int col_2=tid%n3;
        atomicAdd(&data_mult[row_1*n3+col_2],data_a[ind_matr1]*data_b[col_1*n3+col_2]);
    }
    
}
    
int main(){
    unsigned int N1 = 2u;
    unsigned int N2 = 3u;
    unsigned int N3 = 2u;

    int *matrx_a, *matrx_b;
    matrx_a = (int*)malloc((N1*N2) * sizeof(int));
    // Allocate memory on the host
    int *d_matrx_a;
    cudaMalloc((void**)&d_matrx_a, (N1*N2) * sizeof(int));
    matrx_b = (int*)malloc((N3*N2) * sizeof(int));
    int *d_matrx_b;
    cudaMalloc((void**)&d_matrx_b, (N3*N2) * sizeof(int));
    int *mult_matrx;
    mult_matrx = (int*)malloc((N1*N3) * sizeof(int));
    int *d_mult_matrx;
    cudaMalloc((void**)&d_mult_matrx, (N1*N3) * sizeof(int));
    //initialize the data on device

    matrx_a[0] = 3;
    matrx_a[1] = 7;
    matrx_a[2] = 2;
     matrx_a[3] = 1;
    matrx_a[4] = 4;
    matrx_a[5] = 5;
    // init_int_seq(N1*N2, matrx_a);
    matrx_b[0] = 4;
    matrx_b[1] = 5;
    matrx_b[2] = 2;
    matrx_b[3] = 3;
    matrx_b[4] = 6;
    matrx_b[5] = 7;
    
    // init_int_seq(N3*N2, matrx_b);
    printf("Matrix A:\n");
    for (unsigned int i = 0; i < N1*N2; i++) {
        printf("%i ", matrx_a[i]);
    }
    printf("\n");
    printf("Matrix B:\n");
    for (unsigned int i = 0; i < N3*N2; i++) {
        printf("%i ", matrx_b[i]);
    }
    H2D(matrx_a, N1*N2, int);
    H2D(matrx_b, N3*N2, int);
    
    time_kernel_call(kernel_matrix_mult,N1*N2*N3, N1,N2,N3, d_matrx_a, d_matrx_b, d_mult_matrx);
    // Copy data back to host
    D2H(mult_matrx, N1*N3, int);
    for (unsigned int i = 0; i < 10 && i < N1*N3; i++) {
        printf("%i ", mult_matrx[i]);
    }
    printf("\n");
    
    free(matrx_a);
    free(matrx_b);
    free(mult_matrx);
    cudaFree(d_matrx_a);
    cudaFree(d_matrx_b);
    cudaFree(d_mult_matrx);
    
    return 0;
}
