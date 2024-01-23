#include <iostream>
#include <vector>
#include <cstdlib> 
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <cuda.h>
#include <queue>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 1
#define inf 10000000
#define qSize 10000000

__device__ int insertCounter = -1;

__global__ void init_dist(int *dist, int vertices, int src) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < vertices) {
        if (id == src) {
            dist[src] = 0;
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

__global__ void initialLaunch(int src, int *Q) {
    Q[0] = src;
}

__global__ void initQ(int *Q) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < qSize) Q[id] = -1;
}

__global__ void setThreadsToLaunchZero(int *tl) {
    tl[0] = 0;
    insertCounter = -1;
}

__global__ void SSSP_worklist(int *Q, int *Q_copy, int *dist, int *src, int *dest, int *weights, int threadsToLaunch, int *threadsToLaunch_dev) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    int index;
    if (id < threadsToLaunch) {
        // lock is required while accessing the queue
        int u = Q[id];
        
        for (int i = src[u]; i < src[u + 1]; i++) {
            int v = dest[i];
            int w = weights[i];
            if(dist[v] > dist[u] + w){
                atomicMin(&dist[v], dist[u] + w);
                // atomicAdd(&insertCounter, 1); // Synchronization needed for "threadsToLaunch_dev"
                
                index = atomicAdd(threadsToLaunch_dev, 1);
                Q_copy[index] = v;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    string fileName = argv[1];
    // string fileName = "file.txt";
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
    csr = CSRWeighted(num_vertices, num_edges, directed, fin, fileName);

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

    // for (int i = 0; i < 5; ++i) {
    //     printf("%d ", src[i]);
    // }
    // cout << endl;

    // for (int i = 0; i < 5; ++i) {
    //     printf("%d ", dest[i]);
    // }
    // cout << endl;

    // for (int i = 0; i < 5; ++i) {
    //     printf("%d ", weights[i]);
    // }
    // cout << endl;

    cudaMemcpy(dev_src, src, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dest, dest, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weights, weights, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

    int *dist;
    cudaMalloc(&dist, sizeof(int) * num_vertices);

    srand(time(0));
    int startVertex = rand() % num_vertices;
    printf("source = %d\n", startVertex);

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init_dist<<<nBlocks_for_vertices, B_SIZE>>>(dist, num_vertices, startVertex);
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

    int *Q_copy;
    cudaMalloc(&Q_copy, sizeof(int) * qSize);

    // unsigned temp = ceil((float)qSize / B_SIZE);
    // initQ<<<temp, B_SIZE>>>(Q, Q_copy);
    // cudaDeviceSynchronize();

    initialLaunch<<<1, 1>>>(startVertex, Q);
    cudaDeviceSynchronize();

    clock_t calcTime;
    calcTime = clock();

    bool launch_A = true;
    while(threadsToLaunch[0] > 0) {
        unsigned nBlocks = ceil((float)threadsToLaunch[0] / B_SIZE);
        if (nBlocks == 0) nBlocks = 1;

        setThreadsToLaunchZero<<<1, 1>>>(threadsToLaunch_dev);
        cudaDeviceSynchronize();

        if (launch_A) {
            SSSP_worklist<<<nBlocks, B_SIZE>>>(Q, Q_copy, dist, dev_src, dev_dest, dev_weights, threadsToLaunch[0], threadsToLaunch_dev);
            launch_A = false;
        }
        else {
            SSSP_worklist<<<nBlocks, B_SIZE>>>(Q_copy, Q, dist, dev_src, dev_dest, dev_weights, threadsToLaunch[0], threadsToLaunch_dev);
            launch_A = true;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(threadsToLaunch, threadsToLaunch_dev, sizeof(int), cudaMemcpyDeviceToHost);
    }

    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;

    // check answer
    int *check_dist;
    check_dist = (int *)malloc(sizeof(int) * num_vertices);
    for (int i = 0; i < num_vertices; i++) {
        check_dist[i] = inf;
    }
    check_dist[startVertex] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, startVertex});

    while(!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (int i = csr.row_ptr[u]; i < csr.row_ptr[u + 1]; ++i) {
            int v = csr.col_ind[i];
            int w = csr.weights[i];

            if (check_dist[u] + w < check_dist[v]) {
                check_dist[v] = check_dist[u] + w;
                pq.push({check_dist[v], v});
            }
        }
    }

    // for (int i = 0; i < num_vertices; ++i) {
    //     if (check_dist[i] == inf)
    //         cout << "Vertex " << i << ": INF\n";
    //     else
    //         cout << "Vertex " << i << ": " << check_dist[i] << "\n";
    // }

    int *deviceCopiedDist;
    deviceCopiedDist = (int *)malloc(sizeof(int) * num_vertices);

    cudaMemcpy(deviceCopiedDist, dist, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    bool flag = false;
    for (int i = 0; i < num_vertices; ++i) {
        if (check_dist[i] != deviceCopiedDist[i]) {
            printf("Wrong ans, Expected = %d, Actual = %d on vertex: %d\n", check_dist[i], deviceCopiedDist[i], i);
            flag = true;
            break;
        }
    }
    if (!flag) cout << "Correct ans, Time taken = " << t_time << endl;
    cout << endl;
    return 0;
}