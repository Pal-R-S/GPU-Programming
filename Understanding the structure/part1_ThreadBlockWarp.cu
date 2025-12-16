#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "help.cu"
// The Aim behind this file is to understand blocks, threaads, warps and the way they execute
// we design several kernels and execute them to see how they behave 
// the conclusions from the observed behaviour will be documented in the report file

__global__ void kernel_1(unsigned int n){
    gettid_1D();
    if (tid < n) {
        printf("Hello from threadID  %d\n and of block %d\n", threadIdx.x, blockIdx.x);
    }
    
}
__global__ void kernel_2(unsigned int n){
    gettid_1D();
    if (tid < n) {
        printf("tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);
        if (tid % 2 == 0) {
            printf("EVEN %d %d\n", threadIdx.x, blockIdx.x);
        } else {
            printf("ODD %d %d\n", threadIdx.x, blockIdx.x);
        }
    }
    
}
__global__ void kernel_3(unsigned int n){
    gettid_1D();
    if (tid < n) {
        printf(" 1111.   tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);
        printf(" 2222.   tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);
        printf(" 3333.   tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);

        
    }
    
}
__global__ void kernel_4(unsigned int n){
    gettid_1D();
    if (tid < n) {
        unsigned int mod_val = threadIdx.x % 3;
        if (mod_val == 0) {
            // First warp
            printf("mod=0 warp - tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);
        } else if (mod_val == 1) {
            // Second warp
            printf("mod=1 warp - tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);
        } else if (mod_val == 2) {
            // Third warp
            printf("mod=2 warp - tid: %d threadidx: %d  block %d\n ", tid, threadIdx.x, blockIdx.x);
        } else {
            //do nothing 
            }
    }
    
}
    
int main(){
    make_kernel_call(kernel_1, 120,40); //120 threads , 40 threads per block

    // Observe that thread idx goes from 0 uptil 39. There are 40 threads per block. The first 32 of them grp together in execution. 
    //such a execution unit is called a warp=>(max) 32 threads within the same block that execute together

    make_kernel_call(kernel_1, 100, 50); //100 threads , 50 threads per block

    //Observe the same behaviour here. First 32 threads grp together, then the next 18 threads grp together. Thread id of a block goes from 0 to 49
    
    make_kernel_call(kernel_2, 120,32);
    
    //Upuntil the first printf statements, the behaviour is ame as before.
    // Observe how all the odd threads send thier out put first and then all the even threads. 
    //Also note the scheduling order for the blocks is different for the three print statements. 
    //this raises the question does the scheduling happen for each statement of the kernel separately?
    //we can device a simple test for this (NOTE: we are yet to check how warps within a block behave in this regard)

    make_kernel_call(kernel_3, 96,32);

    //This much simpler kernel with no coonditional statements shows that the scheduling between happens at the level of individual print statements.
    //All the blocks execute statement 1 then all execute statement 2 and then all execute statement 3.
    //Lets now try to explore what happens if we have multiple warps within a block.
    //We continue to keep away the complication that comes with conditional statements for now.

    make_kernel_call(kernel_3, 42,42); //2 warps within 1 block
    make_kernel_call(kernel_3, 80,80); //3 warps within 1 block

    //Here we observe that there is line wise execution of the print statements but the order of execution of the warps at each statement is different

    make_kernel_call(kernel_3, 100,40); //2 warps in a block and there are 2 blocks 

    //We observe that the different blocks may interleave thier execution in units of warps. 
    //Simply put, the sheduling seems to happen at the warp unit level and not the block unit level.

    //We now try to analyse the behaviour of execution when threads diverge.
    //We design kernel 4 that depicts divergence of threads in a warp

    make_kernel_call(kernel_4, 32  ,32); //1 block , 1 warp per block
    make_kernel_call(kernel_4, 128  ,32); //4 blocks , 1 warp per block

    //We observe that the warps of each block execute the first print statement together, then the second print statement together and then the third print statement together.
    //threads from differnt warps dont interleave in the execution of a single statement.

    make_kernel_call(kernel_4, 128  ,64); //2 blocks and 2 warps per block

    //the above gives us rather interesting data. 
    //Key observations include that block 0 and block 1 seem to have similar scheduling - we are yet to test this.
    //Also that Different warps seem to execute the 3 if statements in a different order>>> Again lets explore this more 

    return 0;
}

