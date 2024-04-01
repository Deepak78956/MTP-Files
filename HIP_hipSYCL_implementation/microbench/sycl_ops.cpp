#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>

#define it 10000
#define size 1024
#define B_SIZE 1024

using namespace std;

void device_memory_alloc(sycl::queue &Q) {
    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        auto m_device = sycl::malloc_device<double>(size,Q);
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
    double *host_arr, *dev_arr;
    host_arr = (double *)malloc(sizeof(double) * size);
    dev_arr = sycl::malloc_device<double>(sizeof(double) * size, Q);
    Q.wait();

    clock_t timer;
    double t_time;
    
    timer = clock();

    for (int i = 0; i < it; i++) {
        Q.memcpy(dev_arr, host_arr, sizeof(double) * size);
        Q.wait();
    }

    free(dev_arr, Q);

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "host to device memcpy time " << t_time << endl;
}

void atomic_add_time(sycl::queue &Q) {
    auto totalItems = sycl::range<1>(B_SIZE * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);

    int *x;
    x = sycl::malloc_device<int>(1, Q);

    Q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item){
        x[0] = 0;
    }).wait();

    clock_t timer;
    double t_time;

    timer = clock();

    for (int i = 0; i < it; i++) {
        Q.parallel_for(sycl::nd_range<1>(totalItems, itemsInWG), [=](sycl::nd_item<1> item){
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_x(*x);
            atomic_x += 1;
            // sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(x)).fetch_add(1);
            // sycl::atomic_fetch_add<int>(sycl::atomic<int>(sycl::global_ptr<int>(x)), 1);
        }).wait();
    }

    int *host_x;
    host_x = (int *)malloc(sizeof(int));

    Q.memcpy(host_x, x, sizeof(int));
    Q.wait();

    timer = clock() - timer;
    t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    cout << "atomic add time " << t_time << endl;
    cout << host_x[0] << endl;
}

void DRAM(sycl::queue &Q){
    long int gb = 1024;
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
    double t_time = ((double)timer) / CLOCKS_PER_SEC * 1000;
    
    double bw = (sizeof(float) * gb) / t_time;

    int div = 1000000000;
    cout << "Bandwidth achieved: " << bw / div << " GBps" << endl;

    free(output_dev_arr, Q);
    free(input_dev_arr, Q);
}

int main(int argc, char *argv[]) {
    sycl::queue Q{sycl::gpu_selector{}}; 
    
    // device_memory_alloc(Q);

    // kernel_offload(Q);

    // host_to_dev_copy(Q);

    atomic_add_time(Q);

    // DRAM(Q);

    return 0;
}