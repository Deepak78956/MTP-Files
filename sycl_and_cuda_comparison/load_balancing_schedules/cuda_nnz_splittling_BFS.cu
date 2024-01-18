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

struct atomRange {
    long int start, end;
};

struct NonWeightCSR convertToCSR(string fileName) {
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
    num_vertices += 1;

    int size;
    if (directed)
        size = num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    return csr;
}

__global__ void init_dist(int *dist, int vertices) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < vertices) {
        if (id == 0) {
            dist[id] = 0;
        }
        else {
            dist[id] = inf;
        }
    }
}

__global__ void print_dist(int *dist, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        printf("node i = %d, dist = %d\n", i, dist[i]);
    }
}

__device__ struct atomRange getAtomRange(unsigned t_id, long int totalWork, long int totalThreads) {
    long int workToEachThread;
    workToEachThread = totalWork / totalThreads;

    struct atomRange range;
    range.start = t_id * workToEachThread;
    if (t_id == totalThreads - 1) {
        range.end = totalWork;
    }
    else {
        range.end = range.start + workToEachThread;
    }

    return range;
}

__device__ int binarySearch(long int searchItem, long int num_vertices, int *rowOffset) {
    long int start = 0, end = num_vertices - 1, index = end, mid;
    while (start <= end) {
        mid = (start + end) / 2;
        if (rowOffset[mid] > searchItem) {
            end = mid - 1;
        } 
        else {
            index = mid;
            start = mid + 1;
        }
    }

    return index;
}

__global__ void BFS(int *dist, int *src, int *dest, int num_vertices, int num_edges, int *changed) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_vertices) {
        struct atomRange range = getAtomRange(id, num_edges, num_vertices);
        long int u = binarySearch(range.start, num_vertices, src); // get tile

        for (int i = range.start; i < range.end; i++) {
            int v = dest[i];

            // Check if assigned atom goes out of row offset range, if so.. then update the tile
            if (i >= src[u + 1]) {
                u = binarySearch(i, num_vertices, src);
            }

            if(dist[v] > dist[u] + 1){
                atomicMin(&dist[v], dist[u] + 1);
                changed[0] = 1;
            }
        }
        
        // for (int i = src[u]; i < src[u + 1]; i++) {
        //     int v = dest[i];
        //     if(dist[v] > dist[u] + 1){
        //         atomicMin(&dist[v], dist[u] + 1);
        //         changed[0] = 1;
        //     }
        // }
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
    
    struct NonWeightCSR csr = convertToCSR(fileName);
    int size = csr.num_edges;

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (csr.num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, csr.row_ptr, sizeof(int) * (csr.num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, csr.col_ind, sizeof(int) * size, cudaMemcpyHostToDevice);

    int *dist;
    cudaMalloc(&dist, sizeof(int) * csr.num_vertices);

    unsigned nBlocks_for_vertices = ceil((float)csr.num_vertices / B_SIZE);
    init_dist<<<nBlocks_for_vertices, B_SIZE>>>(dist, csr.num_vertices);
    cudaDeviceSynchronize();

    int *changed;
    cudaMalloc(&changed, sizeof(int));
    cudaMallocManaged(&changed, sizeof(int));

    while(true) {
        changed[0] = 0;
        unsigned nBlocks_for_vertices = ceil((float)csr.num_vertices / B_SIZE);

        BFS<<<nBlocks_for_vertices, B_SIZE>>>(dist, dev_row_ptr, dev_col_ind, csr.num_vertices, csr.num_edges, changed);
        cudaDeviceSynchronize();

        if (changed[0] == 0) break;
    }

    print_dist<<<1, 1>>>(dist, csr.num_vertices);
    cudaDeviceSynchronize();

    return 0;
}