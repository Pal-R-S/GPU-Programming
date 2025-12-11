#define H2D(x, n, T) cudaMemcpy(d_##x, x, n*sizeof(T), cudaMemcpyHostToDevice);
#define D2H(x,n, T) cudaMemcpy(x, d_##x, n*sizeof(T), cudaMemcpyDeviceToHost);
#define free_dual(x) { \
    cudaFree(d_##x); \
    free(x); \
}
#define gettid_1D() unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x

template <typename T>
void init_eq(unsigned int n, T value, T *data) {
    for (unsigned int i = 0; i < n; i++) {
        data[i] = value;
    }
}

void init_int_seq(unsigned int n, int *data) {
    for (unsigned int i = 0; i < n; i++) {
        data[i] = (int) i;
    }
}
template <typename Func, typename... Args>
void make_kernel_call(Func kernel_name, unsigned int N,unsigned int number_threads, Args... args) {
    unsigned int num_threads = number_threads > 0 ? number_threads : 256;
    unsigned int num_blocks = (unsigned int)(N + num_threads - 1) / num_threads;
    kernel_name<<<num_blocks, num_threads>>>(N, args...);
    cudaDeviceSynchronize();
}
template <typename Func, typename... Args>
void time_kernel_call(Func kernel_name, unsigned int N,unsigned int num_threads, Args... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    make_kernel_call(kernel_name, N, num_threads, args...);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

 
