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
#define weighted 1
#define inf 10000000

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
    // num_vertices += 1;

    int size;
    if (directed)
        size = num_edges;
    
    int *src, *dest, *weights;
    int *dev_src, *dev_dest, *dev_weights;
    src = (int *)malloc(sizeof(int) * (num_vertices + 1));
    dest = (int *)malloc(sizeof(int) * num_edges);
    weights = (int *)malloc(sizeof(int) * num_edges);

    struct WeightCSR csr;
    csr = CSRWeighted(num_vertices, num_edges, directed, fin);

    cudaMalloc(&dev_src, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_dest, sizeof(int) * num_edges);
    cudaMalloc(&dev_weights, sizeof(int) * num_edges);

    for (int i = 0; i < size; i++) {
        dest[i] = csr.col_ind[i];
        weights[i] = csr.weights[i];
    }

    for (int i = 0; i < num_vertices + 1; i++) {
        src[i] = csr.row_ptr[i];
    }

    cudaMemcpy(dev_src, src, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dest, dest, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weights, weights, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

    int *dist;
    cudaMalloc(&dist, sizeof(int) * num_vertices);

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init_dist<<<nBlocks_for_vertices, B_SIZE>>>(dist, num_vertices);
    cudaDeviceSynchronize();

    int *changed;
    cudaMalloc(&changed, sizeof(int));
    cudaMallocManaged(&changed, sizeof(int));

    return 0;
}