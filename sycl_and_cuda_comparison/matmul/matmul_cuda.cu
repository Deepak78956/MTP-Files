#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>

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

    clock_t memTime, calcTime;
    memTime = clock();
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    memTime = clock() - memTime;

    int THREADS = 32;

    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    calcTime = clock();
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    calcTime = clock() - calcTime;

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, N);

    cout << "COMPLETED SUCCESSFULLY\n";
    double t_time = ((double)calcTime + (double)memTime) / CLOCKS_PER_SEC * 1000;
    cout << t_time << endl;
    cout << endl;

    return 0;
}