#include <iostream>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>

#define it 10000
#define size 65536
#define B_SIZE 1024

using namespace std;

void device_memory_alloc() {
    
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        double *m_device;

        cudaMalloc(&m_device, size * sizeof(double));

        cudaFree(m_device);
        cudaDeviceSynchronize();
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "device memory allocation time " << t_time << endl; 
}

__global__ void kernel() {

}

void kernel_offload() {
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        kernel<<<B_SIZE, B_SIZE>>>();
        cudaDeviceSynchronize();
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "kernel offload time " << t_time << endl;
}

void host_to_dev_copy() {
    int *host_arr, *dev_arr;
    host_arr = (int *)malloc(sizeof(int) * size);
    cudaMalloc(&dev_arr, sizeof(int) * size);
    
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        cudaMemcpy(dev_arr, host_arr, sizeof(int) * size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    cudaFree(dev_arr);

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "host to device memcpy time " << t_time << endl;
}

__global__ void kernel_add(int *x) {
    atomicAdd(x, 1);
}

void atomic_add_time() {
    int *x;
    cudaMalloc(&x, sizeof(int));

    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        kernel_add<<<B_SIZE, B_SIZE>>>(x);
        cudaDeviceSynchronize();
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "atomic add time " << t_time << endl;
}

int main(int argc, char *argv[]) {

    cudaSetDevice(4);

    device_memory_alloc();

    kernel_offload();

    host_to_dev_copy();

    atomic_add_time();

    return 0;
}