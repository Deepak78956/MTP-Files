#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <hip/hip_runtime.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 0
#define inf 10000000

__global__ void init_dist(int *dist, int vertices, int s) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < vertices) {
        if (id == s) {
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

int main(int argc, char *argv[]) {
    hipSetDevice(0);
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

    vector<string> keywords = {"kron", "file"};

    bool keywordFound = false;

    for (const string& keyword : keywords) {
        // Check if the keyword is present in the filename
        if (fileName.find(keyword) != string::npos) {
            // Set the flag to true indicating the keyword is found
            keywordFound = true;
            break;
        }
    }

    int size = num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin, keywordFound);

    // int *row_ptr, *col_index;
    // row_ptr = (int *)malloc(sizeof(int) * (num_vertices + 1));
    // col_index = (int *)malloc(sizeof(int) * size);

    // for (int i = 0; i < num_vertices + 1; i++)
    // {
    //     row_ptr[i] = csr.offsetArr[i];
    // }

    // for (int i = 0; i < size; i++)
    // {
    //     col_index[i] = csr.edgeList[i];
    // }

    // for (int i = 0; i < 4; i++) {
    //     printf("%d ", row_ptr[i]);
    // }

    // cout << endl;

    // for (int i = 0; i < 4; i++) {
    //     printf("%d ", col_index[i]);
    // }
    // cout << endl;

    int *dev_row_ptr, *dev_col_ind;
    hipMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    hipMalloc(&dev_col_ind, sizeof(int) * size);
    hipMemcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1), hipMemcpyHostToDevice);
    hipMemcpy(dev_col_ind, csr.edgeList, sizeof(int) * size, hipMemcpyHostToDevice);

    int *dist;
    hipMalloc(&dist, sizeof(int) * num_vertices);

    int source = num_vertices / 2;
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init_dist<<<nBlocks_for_vertices, B_SIZE>>>(dist, num_vertices, source);
    hipDeviceSynchronize();

    int *changed;
    hipMalloc(&changed, sizeof(int));
    hipMallocManaged(&changed, sizeof(int));

    clock_t calcTime;
    calcTime = clock();

    while(true) {
        changed[0] = 0;
        unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);

        BFS<<<nBlocks_for_vertices, B_SIZE>>>(dist, dev_row_ptr, dev_col_ind, num_vertices, changed);
        hipDeviceSynchronize();

        if (changed[0] == 0) break;
    }

    // print_dist<<<1, 1>>>(dist, num_vertices);
    // cudaDeviceSynchronize();
    ////

    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;

    cout << "On graph " << fileName << " Time taken = " << t_time << endl;
    hipDeviceSynchronize();
    cout << endl;

    return 0;
}
