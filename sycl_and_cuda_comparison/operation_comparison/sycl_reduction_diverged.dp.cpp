// This program computes a sum reduction algortithm with warp divergence

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define SHMEM_SIZE 1024

void sumReduction(int *v, int *v_r, const sycl::nd_item<3> &item_ct1, int *partial_sum) {

    int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

    // Load elements into shared memory
    partial_sum[item_ct1.get_local_id(2)] = v[tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Iterate of log base 2 the block dimension
    for (int s = 1; s < item_ct1.get_local_range(2); s *= 2) {
        // Reduce the threads performing work by half previous the previous
        // iteration each cycle
        if (item_ct1.get_local_id(2) % (2 * s) == 0) {
            partial_sum[item_ct1.get_local_id(2)] += partial_sum[item_ct1.get_local_id(2) + s];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
    if (item_ct1.get_local_id(2) == 0) {
            v_r[item_ct1.get_group(2)] = partial_sum[0];
    }
}

int main() {
    sycl::device dev_ct1;
    sycl::queue q_ct1(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
    
    int N = 1 << 30;
	size_t bytes = N * sizeof(int);

	vector<int> h_v(N);
	vector<int> h_v_r(N);

//   generate(begin(h_v), end(h_v), [](){ return rand() % 10; });
    for (int i = 0; i < N; i++) {
        h_v[i] = 1;
    }

	int *d_v, *d_v_r;
    d_v = (int *)sycl::malloc_device(bytes, q_ct1);
    d_v_r = (int *)sycl::malloc_device(bytes, q_ct1);

    q_ct1.memcpy(d_v, h_v.data(), bytes).wait();

        // TB Size
	const int TB_SIZE = 1024;

	// (No padding)
	int GRID_SIZE = N / TB_SIZE;

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<int, 1> partial_sum_acc_ct1(
                sycl::range<1>(1024 /*SHMEM_SIZE*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, GRID_SIZE) *
                                        sycl::range<3>(1, 1, TB_SIZE),
                                    sycl::range<3>(1, 1, TB_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                        sumReduction(d_v, d_v_r, item_ct1,
                                        partial_sum_acc_ct1.get_pointer());
            });
    });

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<int, 1> partial_sum_acc_ct1(
                sycl::range<1>(1024 /*SHMEM_SIZE*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, TB_SIZE) *
                                        sycl::range<3>(1, 1, TB_SIZE),
                                    sycl::range<3>(1, 1, TB_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                        sumReduction(d_v_r, d_v, item_ct1,
                                        partial_sum_acc_ct1.get_pointer());
                });
        });

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<int, 1> partial_sum_acc_ct1(
                sycl::range<1>(1024 /*SHMEM_SIZE*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, TB_SIZE),
                                    sycl::range<3>(1, 1, TB_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                        sumReduction(d_v, d_v_r, item_ct1,
                                        partial_sum_acc_ct1.get_pointer());
                });
        });

        q_ct1.memcpy(h_v_r.data(), d_v_r, bytes).wait();

        // assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

    if (h_v_r[0] == N) printf("Correct\n");
	cout << "COMPLETED SUCCESSFULLY\n";

	return 0;
}