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
#define qSize 10000000

__device__ int insertCounter = 0;
__device__ int consumeCounter = 0;

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

__global__ void initialLaunch(int src, int *Q) {
    Q[insertCounter] = src;
    insertCounter += 1;
}

__global__ void setThreadsToLaunchZero(int *tl) {
    tl[0] = 0;
}

__global__ void SSSP_worklist(int *Q, int *dist, int *src, int *dest, int *weights, int threadsToLaunch, int *threadsToLaunch_dev) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < threadsToLaunch) {
        int u = Q[consumeCounter];
        atomicAdd(&consumeCounter, 1);
        
        for (int i = src[u]; i < src[u + 1]; i++) {
            int v = dest[i];
            int w = weights[i];
            if(dist[v] > dist[u] + w){
                atomicMin(&dist[v], dist[u] + w);
                Q[insertCounter] = v;
                atomicAdd(&insertCounter, 1);
                atomicAdd(&threadsToLaunch_dev, 1);
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

    int *threadsToLaunch_dev;
    cudaMalloc(&threadsToLaunch_dev, sizeof(int));

    int *threadsToLaunch;
    threadsToLaunch = (int *)malloc(sizeof(int));
    threadsToLaunch[0] = 1;

    int *Q;
    cudaMalloc(&Q, sizeof(int) * qSize);

    int startVertex = 0;
    initialLaunch<<<1, 1>>>(startVertex, Q);
    cudaDeviceSynchronize();

    while(threadsToLaunch[0] > 0) {
        unsigned nBlocks = ceil((float)threadsToLaunch[0] / B_SIZE);
        if (nBlocks == 0) nBlocks = 1;

        setThreadsToLaunchZero<<<1, 1>>>(threadsToLaunch_dev);
        cudaDeviceSynchronize();

        SSSP_worklist<<<nBlocks, B_SIZE>>>(Q, dist, dev_src, dev_dest, dev_weights, threadsToLaunch[0], threadsToLaunch_dev);
        cudaDeviceSynchronize();

        cudaMemcpy(threadsToLaunch, threadsToLaunch_dev, sizeof(int), cudaMemcpyDeviceToHost);
    }

    return 0;
}