#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "help.cu"
// The Aim behind this file is to understand blocks, threaads, warps and the way they execute
// we design several kernels and execute them to see how they behave 
// the conclusions from the observed behaviour will be documented in the report file



    
int main(){
    //We are going to explore the observe the ill effects of Warp Divergence and how much it can affect performance

    //problem1 
    //if a number is odd multiply it by 2 add 2 then multiply by 3 else if its even divide by 2 add 2 then multiply by 4.
    int* a;
    unsigned int N= 1<<20;
    a = (int*)malloc(N * sizeof(int));
    init_int_seq(N, a);
    int* d_a;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    H2D(a, N, int);
    time_kernel_call(kernel_diverge_problem1, N, 1024, d_a);
    init_int_seq(N, a);
    H2D(a, N,int);
    time_kernel_call(kernel_no_diverge_problem1, N, 1024, d_a);
    D2H(a, N, int);
    free_dual(a);

    /*We observe time for execution 
    Kernel execution time: ~0.250368 ms -- for warp divergence
    Kernel execution time: ~0.069408 ms -- for no warp Divergence
    Note that we have taken care to reduce anylatency caused by memory acceses by a warp of threads*/

    /*Since the results from one probelem cannot be considered convincing enough, 
    Problem two is designed such that there is even more divergence in a warp, vs no divergence in a warp.*/

    //Problem 2:
    //We design the extreme case where every thread enters a different branch. We totally avoid share memory problems here
    unsigned int N2=1<<10;
    int* a2 = (int*)malloc(N2 * sizeof(int));
    init_eq(N2, 5, a2);
    int* d_a2;
    cudaMalloc((void**)&d_a2, N2 * sizeof(int));
    H2D(a2, N2, int);
   
    time_kernel_call(kernel_diverge_problem2, N2, 1024, d_a2);
    D2H(a2, N2, int);
    //Print first 10 elements
    for (int i = 0; i < 10; i++) {
        printf("%d ", a2[i]);
    }
    printf("\n");


    init_eq(N2, 5, a2);
    H2D(a2, N2, int);
    time_kernel_call(kernel_less_diverge_problem2, N2, 1024, d_a2);
    D2H(a2, N2, int);
    //Print first 10 elements
    for (int i = 0; i < 10; i++) {
        printf("%d ", a2[i]);
    }
    printf("\n");


    init_eq(N2, 5, a2);
    H2D(a2, N2, int);
    time_kernel_call(kernel_no_diverge_problem2, N2, 1024, d_a2);
    D2H(a2, N2, int);
    //Print first 10 elements
    for (int i = 0; i < 10; i++) {
        printf("%d ", a2[i]);
    }
    printf("\n");

    /*Results are ase follows:
    Most divergence:0.036672 ms
    Less divergence:0.021216 ms
    No divergence:0.019328 ms
    */
    free_dual(a2);
        return 0;
}

