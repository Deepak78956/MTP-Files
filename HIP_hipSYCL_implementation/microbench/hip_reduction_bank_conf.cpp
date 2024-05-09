// This program performs sum reduction with an optimization
// removing warp divergence


#include <hip/hip_runtime.h>
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

__global__ void sumReduction(int *v, int *v_r) {
  __shared__ int partial_sum[SHMEM_SIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  partial_sum[threadIdx.x] = v[tid];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    // Change the indexing to be sequential threads
    int index = 2 * s * threadIdx.x;

    // Each thread does work unless the index goes off the block
    if (index < blockDim.x) {
      partial_sum[index] += partial_sum[index + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is inexed by this block
  if (threadIdx.x == 0) {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

int main() {
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
  hipMalloc(&d_v, bytes);
  hipMalloc(&d_v_r, bytes);

  hipMemcpy(d_v, h_v.data(), bytes, hipMemcpyHostToDevice);

  const int TB_SIZE = 1024;

  // Grid Size (No padding)
  int GRID_SIZE = N / TB_SIZE;

  clock_t t;

  t = clock();
  sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
  hipDeviceSynchronize();

  sumReduction<<<TB_SIZE, TB_SIZE>>>(d_v_r, d_v);
  hipDeviceSynchronize();

  sumReduction<<<1, TB_SIZE>>>(d_v, d_v_r);
  hipDeviceSynchronize();
  t = clock() - t;

  double t_time = ((double)t) / CLOCKS_PER_SEC * 1000;

  hipMemcpy(h_v_r.data(), d_v_r, bytes, hipMemcpyDeviceToHost);

//   assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
  if (h_v_r[0] == N) printf("bank conf, Correct\n");

  std::cout << "COMPLETED SUCCESSFULLY, time taken: " << t_time << std::endl;

  return 0;
}