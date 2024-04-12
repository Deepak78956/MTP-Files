#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <hip/hip_runtime.h>

using namespace std;

__global__ void matrixMul(const int *a, const int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = 0;
    for (int k = 0; k < N; k++)
    {
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int tmp = 0;
            for (int k = 0; k < N; k++)
            {
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result
            assert(tmp == c[i * N + j]);
        }
    }
}

int main()
{
    int M, N;

    ifstream inputFile("output.txt");

    if (!inputFile)
    {
        cerr << "Error opening the input file." << endl;
        return 1;
    }

    inputFile >> M >> N;

    size_t bytes = N * N * sizeof(int);

    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    for (int i = 0; i < N * N; i++)
    {
        inputFile >> h_a[i];
    }

    for (int i = 0; i < N * N; i++)
    {
        inputFile >> h_b[i];
    }

    int *d_a, *d_b, *d_c;


    // Memory allocation and memcpy timing record
    clock_t memTime, calcTime;
    memTime = clock();
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    memTime = clock() - memTime;

    int THREADS = 32;

    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Kernel execution timing record
    calcTime = clock();
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    calcTime = clock() - calcTime;

    hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, N);

    cout << "COMPLETED SUCCESSFULLY\n";
    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << t_time << endl;
    cout << endl;

    return 0;
}