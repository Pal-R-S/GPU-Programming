#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "help.cu"
// The Aim behind this file is to understand blocks, threaads, warps and the way they execute
// we design several kernels and execute them to see how they behave 
// the conclusions from the observed behaviour will be documented in the report file
int main(){
    int count=0;
    cudaGetDeviceCount(&count);
    printf("Number of devices: %d\n", count);
    for (int i=0;i<count;i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        printf("Device Number: %d\n",i);
        printf("Device Name: %s\n",prop.name);
        printf("Compute Capability: %d.%d\n",prop.major,prop.minor);
        printf("Total Global Memory: %lu bytes\n",prop.totalGlobalMem);
        printf("Multiprocessor Count: %d\n",prop.multiProcessorCount);
        printf("Max Threads per Multiprocessor: %d\n",prop.maxThreadsPerMultiProcessor);
        printf("Max Threads per Block: %d\n",prop.maxThreadsPerBlock);
        printf("Warp Size: %d\n",prop.warpSize);
        printf("Max Threads Dimensions: [%d, %d, %d]\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
        printf("Max Grid Size: [%d, %d, %d]\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
        //minor, clockrate, textureAlignment, deviceOverlap, Maxtexture1D, MaxTexture2D, MaxTexture3D, integrated, canMapHostMemory, computeMode, concurrentKernels, mempitch
        printf("Minor Version: %d\n",prop.minor);
        printf("Clock Rate: %d kHz\n",prop.clockRate);
        printf("Texture Alignment: %lu bytes\n",prop.textureAlignment);
        printf("Device Overlap: %d\n",prop.deviceOverlap);
        printf("Max Texture 1D: %d\n",prop.maxTexture1D);
        printf("Max Texture 2D: [%d, %d]\n",prop.maxTexture2D[0],prop.maxTexture2D[1]);
        printf("Max Texture 3D: [%d, %d, %d]\n",prop.maxTexture3D[0],prop.maxTexture3D[1],prop.maxTexture3D[2]);
        printf("Integrated: %d\n",prop.integrated);
        printf("Can Map Host Memory: %d\n",prop.canMapHostMemory);
        printf("Compute Mode: %d\n",prop.computeMode);
        printf("Concurrent Kernels: %d\n",prop.concurrentKernels);
        printf("Memory Pitch: %lu bytes\n",prop.memPitch);
        printf("-------------------------------\n");
    }
    return 0;

}