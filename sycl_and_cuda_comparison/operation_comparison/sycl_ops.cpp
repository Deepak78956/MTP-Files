#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <sycl/sycl.hpp>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>

#define it 10000
#define size 65536
#define B_SIZE 1024

// parameters for shared_memory kernel
#define use_prefetch 1
#define instr_mix 10
#define kernel_launches 1000

using namespace std;

void device_memory_alloc(sycl::queue &Q) {
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        auto m_device = sycl::malloc_device<double>(65536*65536,Q);
        Q.wait();

        sycl::free(m_device, Q);
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "device memory allocation time " << t_time << endl; 
}

void kernel_offload(sycl::queue &Q) {
    auto totalItems = sycl::range<1>(1024 * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);

    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        Q.parallel_for(sycl::nd_range<1>(totalItems, itemsInWG), [=](sycl::nd_item<1> item){

        }).wait();
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "kernel offload time " << t_time << endl;
}

void host_to_dev_copy(sycl::queue &Q) {
    int *host_arr, *dev_arr;
    host_arr = (int *)malloc(sizeof(int) * size);
    dev_arr = sycl::malloc_device<int>(sizeof(int) * size, Q);
    Q.wait();

    clock_t timer;
    double t_time;
    
    timer = clock();

    for (int i = 0; i < it; i++) {
        Q.memcpy(dev_arr, host_arr, sizeof(int) * size);
        Q.wait();
    }

    free(dev_arr, Q);

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "host to device memcpy time " << t_time << endl;
}

void atomic_add_time(sycl::queue &Q) {
    auto totalItems = sycl::range<1>(1024 * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);

    int *x;
    x = sycl::malloc_device<int>(1, Q);

    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        Q.parallel_for(sycl::nd_range<1>(totalItems, itemsInWG), [=](sycl::nd_item<1> item){
            sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(x)).fetch_add(1);
        }).wait();
    }

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "atomic add time " << t_time << endl;
}

void DRAM(sycl::queue &Q){
    long int gb = 1024*1024*1024;
    float *input_dev_arr, *output_dev_arr;

    float *arr;
    arr = (float *)malloc(sizeof(float) * gb);

    for(int i = 0; i < gb; i++) {
        arr[i] = i;
    }

    input_dev_arr = sycl::malloc_device<float>(gb * sizeof(float), Q);
    Q.wait();
    output_dev_arr = sycl::malloc_device<float>(gb * sizeof(float), Q);
    Q.wait();

    Q.memcpy(input_dev_arr, arr, sizeof(float) * gb);
    Q.wait();

    auto totalItems = sycl::range<1>(gb);
    auto itemsInWG = sycl::range<1>(B_SIZE);

    clock_t timer;
    timer = clock();

    Q.parallel_for(sycl::nd_range<1>(totalItems, itemsInWG), [=](sycl::nd_item<1> item){
        unsigned id = item.get_global_id(0);
        if (id < gb) {
            output_dev_arr[id] = input_dev_arr[id];
        }
    }).wait();

    timer = clock() - timer;
    double t_time = ((double)timer) / CLOCKS_PER_SEC;
    
    double bw = (sizeof(double) * gb) / t_time;

    int div = 1000000000;
    cout << "Bandwidth achieved: " << bw / div << " GBps" << endl;

    free(output_dev_arr, Q);
    free(input_dev_arr, Q);
}

void shared_memory(sycl::queue &Q) {
    float *arr;

    arr = sycl::malloc_shared<float>(sizeof(float) * size, Q);

    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }

    clock_t timer;
    timer = clock();

    for (int iter = 0; iter < kernel_launches; iter++) {
        if (use_prefetch) Q.prefetch(arr, sizeof(float) * size);

        auto globalRange = sycl::range<1>(size);
        auto localRange = sycl::range<1>(B_SIZE);

        Q.parallel_for(sycl::nd_range<1>(globalRange, localRange), [=](sycl::nd_item<1> item){
            unsigned id = item.get_global_id(0);
            const auto num_ops = size * instr_mix;
            if (id < size) {
                for (size_t i = id, j = 0; i < num_ops; i += size, j++) {
                    arr[(id + j) % size] += id;
                }
            }
        }).wait();

        for (size_t i = 0; i < size; i++) {
            arr[i] -= i;
        }
    }

    timer = clock() - timer;
    double t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;

    cout << "Shared memory kernel: " << t_time << endl;

    sycl::free(arr, Q);
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

void random_accesses(sycl::queue &Q){
    int* randoms = static_cast<int*>(malloc(size * sizeof(int))); 

    if (randoms == nullptr) {
        cerr << "Memory allocation failed." << endl;
        return;
    }

    readNumbersFromFile("random_numbers.txt", randoms);

    int *randoms_dev, *output_arr;
    randoms_dev = sycl::malloc_device<int>(size * sizeof(int), Q);
    output_arr = sycl::malloc_device<int>(size * sizeof(int), Q);

    Q.memcpy(randoms_dev, randoms, sizeof(int) * size);

    auto globalRange = sycl::range<1>(size);
    auto localRange = sycl::range<1>(B_SIZE);

    clock_t timer;
    timer = clock();

    for (int i = 0; i < it; i++){
        Q.parallel_for(sycl::nd_range<1>(globalRange, localRange), [=](sycl::nd_item<1> item){
            unsigned id = item.get_global_id(0);
            if (id < size) {
                output_arr[randoms_dev[id]] = id * 2; 
            }
        }).wait();
    }

    timer = clock() - timer;
    double t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;

    cout << "Random accesses kernel: " << t_time << endl;

    sycl::free(randoms_dev, Q);
    sycl::free(output_arr, Q);
    free(randoms);
}

int main(int argc, char *argv[]) {
    // sycl::queue Q(sycl::gpu_selector_v);
    
    // device_memory_alloc(Q);

    // kernel_offload(Q);

    // host_to_dev_copy(Q);

    // atomic_add_time(Q);

    // DRAM(Q);

    // shared_memory(Q);

    return 0;
}