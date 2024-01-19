#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <cuda.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 0
#define inf 10000000

__global__ void init_dist(int *dist, int vertices) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < vertices) {
        if (id == 0) {
            dist[id] = 0;
        }
        else {
            dist[id] = 1000000;
        }
    }
}

__global__ void print_dist(int *dist, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        printf("node i = %d, dist = %d\n", i, dist[i]);
    }
}

__global__ void BFS(int *dist, int *src, int *dest, int num_vertices, int *changed) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_vertices) {
        int u = id;
        
        for (int i = src[u]; i < src[u + 1]; i++) {
            int v = dest[i];
            if(dist[v] > dist[u] + 1){
                atomicMin(&dist[v], dist[u] + 1);
                changed[0] = 1;
            }
        }
    }
}

int main() {
    // if (argc != 2)
    // {
    //     printf("Usage: %s <input_file>\n", argv[0]);
    //     return 1;
    // }

    // string fileName = argv[1];

    string fileName = "file.txt";
    ifstream fin(fileName);
    string line;
    while (getline(fin, line))
    {
        if (line[0] == '%')
        {
            continue;
        }
        else
        {
            break;
        }
    }

    istringstream header(line);
    int num_vertices, num_edges, x;
    header >> num_vertices >> x >> num_edges;

    int size = num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    int *row_ptr, *col_index;
    row_ptr = (int *)malloc(sizeof(int) * (num_vertices + 1));
    col_index = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        row_ptr[i] = csr.offsetArr[i];
    }

    for (int i = 0; i < size; i++)
    {
        col_index[i] = csr.edgeList[i];
    }

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, row_ptr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, col_index, sizeof(int) * size, cudaMemcpyHostToDevice);

    int *dist;
    cudaMalloc(&dist, sizeof(int) * num_vertices);

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init_dist<<<nBlocks_for_vertices, B_SIZE>>>(dist, num_vertices);
    cudaDeviceSynchronize();

    int *changed;
    cudaMalloc(&changed, sizeof(int));
    cudaMallocManaged(&changed, sizeof(int));

    while(true) {
        changed[0] = 0;
        unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);

        BFS<<<nBlocks_for_vertices, B_SIZE>>>(dist, dev_row_ptr, dev_col_ind, num_vertices, changed);
        cudaDeviceSynchronize();

        if (changed[0] == 0) break;
    }

    print_dist<<<1, 1>>>(dist, num_vertices);
    cudaDeviceSynchronize();

    return 0;
}