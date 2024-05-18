#include <iostream>
#include <math.h>
#include <vector>
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>
#include <fstream>

#define it 1
#define size (1 << 28) // 2^28
#define B_SIZE 1024
#define randomArrSize (1 << 18)

using namespace std;

// parameters for shared_memory kernel
#define use_prefetch 1
#define instr_mix 1
#define kernel_launches 100

void device_memory_alloc() {
    
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        double *m_device;

        // 1GB of data allocation, 2^27 * 2 ^ 3
        hipMalloc(&m_device, size * sizeof(double));

        hipFree(m_device);
        hipDeviceSynchronize();
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
        hipDeviceSynchronize();
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "kernel offload time " << t_time << endl;
}

void host_to_dev_copy() {
    // 1GB copy
    double *host_arr, *dev_arr;
    host_arr = (double *)malloc(sizeof(double) * size);
    hipMalloc(&dev_arr, sizeof(double) * size);
    
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        hipMemcpy(dev_arr, host_arr, sizeof(double) * size, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
    }

    hipFree(dev_arr);

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "host to device memcpy time " << t_time << endl;
}

__global__ void kernel_add(int *x) {
    atomicAdd(x, 1);
}

void atomic_add_time() {
    int *x;
    hipMalloc(&x, sizeof(int));

    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        kernel_add<<<B_SIZE, B_SIZE>>>(x);
        hipDeviceSynchronize();
    }

    int *host_x;
    host_x = (int *)malloc(sizeof(int));

    hipMemcpy(host_x, x, sizeof(int), hipMemcpyDeviceToHost);

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "atomic add time " << t_time << endl;
    // cout << host_x[0] << endl;
}

__global__ void DRAM_kernel(size_t gb, int *input_dev_arr, int *output_dev_arr) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < gb) {
        output_dev_arr[id] = input_dev_arr[id];
    }
}

void DRAM(){
    size_t gb = static_cast<size_t>(10) * 1024 * 1024 * 1024 / sizeof(int);
    int *input_dev_arr, *output_dev_arr;

    int *arr;
    arr = (int *)malloc(sizeof(int) * gb);

    for(int i = 0; i < gb; i++) {
        arr[i] = i;
    }

    hipMalloc(&input_dev_arr, sizeof(int) * gb);
    hipMalloc(&output_dev_arr, sizeof(int) * gb);

    hipMemcpy(input_dev_arr, arr, sizeof(int) * gb, hipMemcpyHostToDevice);

    unsigned nBlocks = ceil((float)gb / B_SIZE);

    clock_t timer;
    timer = clock();

    DRAM_kernel<<<nBlocks, B_SIZE>>>(gb, input_dev_arr, output_dev_arr);
    hipDeviceSynchronize();

    timer = clock() - timer;
    float t_time = ((float)timer) / CLOCKS_PER_SEC;

    float bw = (sizeof(float) * gb) / t_time;

    int div = 1000000000;
    cout << "Bandwidth achieved: " << bw / div << " GBps" << endl;

    hipFree(input_dev_arr);
    hipFree(output_dev_arr);
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

    hipMallocManaged(&arr, sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }

    clock_t timer;
    timer = clock();

    for (int iter = 0; iter < kernel_launches; iter++) {
        int deviceId; 
        hipGetDevice(&deviceId); 

        if (use_prefetch) hipMemPrefetchAsync(arr, size * sizeof(float), deviceId);

        unsigned nBlocks = ceil((float)size / B_SIZE);
        shared_memory_kernel<<<nBlocks, B_SIZE>>>(arr);
        hipDeviceSynchronize();

        for (size_t i = 0; i < randomArrSize; i++) {
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
    int* randoms = static_cast<int*>(malloc((randomArrSize) * sizeof(int))); 

    if (randoms == nullptr) {
        cerr << "Memory allocation failed." << endl;
        return;
    }

    readNumbersFromFile("random_numbers.txt", randoms);

    int *randoms_dev, *output_arr;
    hipMalloc(&randoms_dev, sizeof(int) * randomArrSize);
    hipMalloc(&output_arr, sizeof(int) * size);
    hipMemcpy(randoms_dev, randoms, sizeof(int) * randomArrSize, hipMemcpyHostToDevice);

    unsigned nBlocks = ceil((float)randomArrSize / B_SIZE);

    clock_t timer;
    timer = clock();

    for (int i = 0; i < it; i++) {
        random_accesses_kernel<<<nBlocks, B_SIZE>>>(randoms_dev, output_arr);
        hipDeviceSynchronize();
    }

    timer = clock() - timer;
    double t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;

    cout << "Random accesses kernel: " << t_time << endl;
}

int main(int argc, char *argv[]) {

    device_memory_alloc();

    // kernel_offload();

    host_to_dev_copy();

    atomic_add_time();

    // DRAM();

    // shared_memory();

    random_accesses();

    return 0;
}