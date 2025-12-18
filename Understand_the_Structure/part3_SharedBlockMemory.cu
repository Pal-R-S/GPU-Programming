#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "help.cu"
// The Aim behind this file is to understand blocks, threads, warps and the way they execute
// we design several kernels and execute them to see how they behave 
// the conclusions from the observed behaviour will be documented in the report file
__global__ void kernel_good_localization(unsigned int N, int *data){
    gettid_1D();
    if(tid < N){
        data[tid] = data[tid] * 2;
    }
}
__global__ void kernel_bad_localization(unsigned int N, int *data){
    gettid_1D();
    unsigned int idx=(tid%32)*(N/32)+(tid/32);
    if(idx < N){
        data[idx] = data[idx] * 2;
    }
}


    
int main(){
    
    /*Shred Memory allocations are done at a thread block level 
    */
     int *a;
    int *d_a;
    unsigned int N = 1<<30; 
    a = (int*)malloc( N*sizeof(int));
    init_int_seq(N, a);
    cudaMalloc((void**)&d_a, N*sizeof(int));
    H2D(a, N, int);
    printf("Executing kernel with good memory localization\n");
    time_kernel_call(kernel_good_localization, N, 32, d_a);
    init_int_seq(N, a);
    H2D(a, N, int);
    printf("Executing kernel with bad memory localization\n");
    time_kernel_call(kernel_bad_localization, N, 32, d_a);
        return 0;
    /*
    Executing kernel with good memory localization
    Kernel execution time: 22.576448 ms
    Executing kernel with bad memory localization
    Kernel execution time: 147.358978 ms

    Also note this results don't agree with our expectations for small values of N
    */

}


