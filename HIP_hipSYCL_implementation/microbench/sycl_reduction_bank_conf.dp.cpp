// This program performs sum reduction with an optimization
// removing warp divergence

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::vector;

#define SHMEM_SIZE 1024

void sumReduction(int *v, int *v_r, const sycl::nd_item<3> &item_ct1,
                  int *partial_sum) {

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  partial_sum[item_ct1.get_local_id(2)] = v[tid];
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (int s = 1; s < item_ct1.get_local_range(2); s *= 2) {
    // Change the indexing to be sequential threads
    int index = 2 * s * item_ct1.get_local_id(2);

    // Each thread does work unless the index goes off the block
    if (index < item_ct1.get_local_range(2)) {
      partial_sum[index] += partial_sum[index + s];
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

  // Initialize the input data
//   generate(begin(h_v), end(h_v), []() { return rand() % 10; });
  for (int i = 0; i < N; i++) {
        h_v[i] = 1;
    }

  int *d_v, *d_v_r;
  d_v = (int *)sycl::malloc_device(bytes, q_ct1);
  d_v_r = (int *)sycl::malloc_device(bytes, q_ct1);

  q_ct1.memcpy(d_v, h_v.data(), bytes).wait();

  const int TB_SIZE = 1024;

  // Grid Size (No padding)
  int GRID_SIZE = N / TB_SIZE;

  clock_t t;

  t = clock();
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> partial_sum_acc_ct1(
        sycl::range<1>(1024 /*SHMEM_SIZE*/), cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, GRID_SIZE) *
                                           sycl::range<3>(1, 1, TB_SIZE),
                                       sycl::range<3>(1, 1, TB_SIZE)),
                     [=](sycl::nd_item<3> item_ct1) {
                       sumReduction(d_v, d_v_r, item_ct1,
                                    partial_sum_acc_ct1.get_pointer());
                     });
  }).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> partial_sum_acc_ct1(
        sycl::range<1>(1024 /*SHMEM_SIZE*/), cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, TB_SIZE) *
                                           sycl::range<3>(1, 1, TB_SIZE),
                                       sycl::range<3>(1, 1, TB_SIZE)),
                     [=](sycl::nd_item<3> item_ct1) {
                       sumReduction(d_v_r, d_v, item_ct1,
                                    partial_sum_acc_ct1.get_pointer());
                     });
  }).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> partial_sum_acc_ct1(
        sycl::range<1>(1024 /*SHMEM_SIZE*/), cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, TB_SIZE),
                                       sycl::range<3>(1, 1, TB_SIZE)),
                     [=](sycl::nd_item<3> item_ct1) {
                       sumReduction(d_v, d_v_r, item_ct1,
                                    partial_sum_acc_ct1.get_pointer());
                     });
  }).wait();

  t = clock() - t;
  double t_time = ((double)t) / CLOCKS_PER_SEC * 1000;

  q_ct1.memcpy(h_v_r.data(), d_v_r, bytes).wait();

//   assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
  if (h_v_r[0] == N) printf("bank conf, Correct\n");

  std::cout << "COMPLETED SUCCESSFULLY, time taken: " << t_time << std::endl;

  return 0;
}