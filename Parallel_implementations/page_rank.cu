#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <cuda.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024

__device__ float d = 0.85;

__global__ void init(float *pr, int num_vertices)
{
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_vertices)
    {
        pr[id] = 1 / num_vertices;
    }
}

__global__ void pagerank(int *dev_row_ptr, int *dev_col_ind, float *pr, int num_vertices, float *val, int p)
{
    unsigned t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t < num_vertices)
    {
        for (int i = dev_row_ptr[t]; i < dev_row_ptr[t + 1]; i++)
        {
            int neigh = dev_col_ind[i];
            if (neigh == p && (dev_row_ptr[t + 1] - dev_row_ptr[t] != 0))
            {
                int outDeg = dev_row_ptr[t + 1] - dev_row_ptr[t];
                atomicAdd(val, (pr[t] / outDeg));
            }
        }
    }
}

__global__ void computePR(int *dev_row_ptr, int *dev_col_ind, float *pr, int num_vertices)
{
    unsigned p = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0;

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    // For in-neighbors of p (using Dynamic Parallelism here)
    pagerank<<<nBlocks_for_vertices, B_SIZE>>>(dev_row_ptr, dev_col_ind, pr, num_vertices, &val, p);
    __syncthreads();

    pr[p] = val * d + (1 - d) / num_vertices;
}

__global__ void printPR(float *pr, int vertices)
{
    printf("HEy\n");
    for (int i = 0; i < vertices; i++)
    {
        printf("Hey\n");
        printf("%d ", pr[i]);
    }
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size;
    if (directed)
        size = num_edges;
    else
    {
        cout << "Directed graph is required" << endl;
        exit(0);
    }
    if (weighted)
    {
        cout << "Non weighted graph is required" << endl;
        exit(0);
    }

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    int *row_ptr, *col_index;
    row_ptr = (int *)malloc(sizeof(int) * (num_vertices + 1));
    col_index = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        row_ptr[i] = csr.row_ptr[i];
    }

    for (int i = 0; i < size; i++)
    {
        col_index[i] = csr.col_ind[i];
    }

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, row_ptr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, col_index, sizeof(int) * size, cudaMemcpyHostToDevice);

    float *pr;
    cudaMalloc(&pr, sizeof(float) * num_vertices);
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init<<<nBlocks_for_vertices, B_SIZE>>>(pr, num_vertices);
    cudaDeviceSynchronize();

    computePR<<<nBlocks_for_vertices, B_SIZE>>>(dev_row_ptr, dev_col_ind, pr, num_vertices);
    cudaDeviceSynchronize();

    cout << "here" << endl;

    printPR<<<1, 1>>>(pr, num_vertices);
    cout << "here" << endl;
    cudaDeviceSynchronize();
    cout << "here" << endl;
    return 0;
}