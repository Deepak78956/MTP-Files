#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>

#define it 10000
#define size 134217728
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
            sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(x)).fetch_add(1);
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

int main(int argc, char *argv[]) {
    sycl::queue Q{sycl::gpu_selector{}}; 
    
    // device_memory_alloc(Q);

    // kernel_offload(Q);

    // host_to_dev_copy(Q);

    atomic_add_time(Q);

    return 0;
}