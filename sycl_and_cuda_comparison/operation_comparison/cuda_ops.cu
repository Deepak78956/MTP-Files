#include <iostream>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>
#include <fstream>

#define it 10000
#define size 65536
#define B_SIZE 1024

// parameters for shared_memory kernel
#define use_prefetch 1
#define instr_mix 10
#define kernel_launches 1000

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

__global__ void DRAM_kernel(long int gb, float *input_dev_arr, float *output_dev_arr) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < gb) {
        output_dev_arr[id] = input_dev_arr[id];
    }
}

void DRAM(){
    long int gb = 1024*1024*1024;
    float *input_dev_arr, *output_dev_arr;

    float *arr;
    arr = (float *)malloc(sizeof(float) * gb);

    for(int i = 0; i < gb; i++) {
        arr[i] = i;
    }

    cudaMalloc(&input_dev_arr, sizeof(float) * gb);
    cudaMalloc(&output_dev_arr, sizeof(float) * gb);

    cudaMemcpy(input_dev_arr, arr, sizeof(float) * gb, cudaMemcpyHostToDevice);

    unsigned nBlocks = ceil((float)gb / B_SIZE);

    clock_t timer;
    timer = clock();

    DRAM_kernel<<<nBlocks, B_SIZE>>>(gb, input_dev_arr, output_dev_arr);
    cudaDeviceSynchronize();

    timer = clock() - timer;
    float t_time = ((float)timer) / CLOCKS_PER_SEC;

    float bw = (sizeof(float) * gb) / t_time;

    int div = 1000000000;
    cout << "Bandwidth achieved: " << bw / div << " GBps" << endl;

    cudaFree(input_dev_arr);
    cudaFree(output_dev_arr);
}

__global__ void shared_memory_kernel(float *arr) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    const auto num_ops = size * instr_mix;
    if (id < size) {
        for (size_t i = id, j = 0; i < num_ops; i += size, j++) {
            arr[(id + j) % size] += id;
        }
    }
}

void shared_memory(){
    float *arr;
    // cudaMalloc(&arr, sizeof(double) * size);

    cudaMallocManaged(&arr, sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }

    clock_t timer;
    timer = clock();

    for (int iter = 0; iter < kernel_launches; iter++) {
        int deviceId; 
        cudaGetDevice(&deviceId); 

        if (use_prefetch) cudaMemPrefetchAsync(arr, size * sizeof(float), deviceId);

        unsigned nBlocks = ceil((float)size / B_SIZE);
        shared_memory_kernel<<<nBlocks, B_SIZE>>>(arr);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < size; i++) {
            arr[i] -= i;
        }
    }

    timer = clock() - timer;
    double t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;

    cout << "Shared memory kernel: " << t_time << endl;
}

void readNumbersFromFile(const string& filename, int* randoms) {
    ifstream inFile(filename);

    if (!inFile.is_open()) {
        cerr << "Error opening file." << endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        if (!(inFile >> randoms[i])) {
            cerr << "Error reading from file." << endl;
            return;
        }
    }

    inFile.close();
}

__global__ void random_accesses_kernel(int *randoms_dev, int *output_arr) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) {
        output_arr[randoms_dev[id]] = id * 2; 
    }
}

void random_accesses(){
    int* randoms = static_cast<int*>(malloc(size * sizeof(int))); 

    if (randoms == nullptr) {
        cerr << "Memory allocation failed." << endl;
        return;
    }

    readNumbersFromFile("random_numbers.txt", randoms);

    int *randoms_dev, *output_arr;
    cudaMalloc(&randoms_dev, sizeof(int) * size);
    cudaMalloc(&output_arr, sizeof(int) * size);
    cudaMemcpy(randoms_dev, randoms, sizeof(int) * size, cudaMemcpyHostToDevice);

    unsigned nBlocks = ceil((float)size / B_SIZE);

    clock_t timer;
    timer = clock();

    for (int i = 0; i < it; i++) {
        random_accesses_kernel<<<nBlocks, B_SIZE>>>(randoms_dev, output_arr);
        cudaDeviceSynchronize();
    }

    timer = clock() - timer;
    double t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;

    cout << "Random accesses kernel: " << t_time << endl;
}

int main(int argc, char *argv[]) {

    // cudaSetDevice(4);

    // device_memory_alloc();

    // kernel_offload();

    // host_to_dev_copy();

    // atomic_add_time();

    // DRAM();

    // shared_memory();

    // random_accesses();

    return 0;
}