#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include <cooperative_groups.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 0
#define inf 10000000

using namespace std;
namespace cg = cooperative_groups;

struct ArgsStruct {
    int *dist;
    int *dev_row_ptr;
    int *dev_col_ind;
    int num_vertices;
    int *changed;
};

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

__device__ void BFS_util(int *dist, int *src, int *destination, int num_vertices, int *changed) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_vertices) {
        int u = id;
        
        for (int i = src[u]; i < src[u + 1]; i++) {
            int v = destination[i];
            if(dist[v] > dist[u] + 1){
                atomicMin(&dist[v], dist[u] + 1);
                changed[0] = 1;
            }
        }
    }
}

__global__ void BFS(int *dist, int *dev_row_ptr, int *dev_col_ind, int num_vertices, int *changed, int *launchedKernel){
    // ArgsStruct *args;
    // args = (ArgsStruct *)para;
    cg::grid_group grid = cg::this_grid();

    launchedKernel[0] = 1;
    
    while (true) {
        BFS_util(dist, dev_row_ptr, dev_col_ind, num_vertices, changed);
        grid.sync();
        if (changed[0] == 0) {
            break;
        }
    }
}

__global__ void setParams(int *dist, int *dev_row_ptr, int *dev_col_ind, int num_vertices, int *changed, void **para){
    ArgsStruct *args;
    args = (ArgsStruct *)para;

    (*args).dist = dist;
    (*args).dev_row_ptr = dev_row_ptr;
    (*args).dev_col_ind = dev_col_ind;
    (*args).num_vertices = num_vertices;
    (*args).changed = changed;
}

int main(int argc, char *argv[]){
    int deviceId = 0;
    cudaSetDevice(deviceId);
    
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
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, csr.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);

    int *dist;
    cudaMalloc(&dist, sizeof(int) * num_vertices);

    int source = num_vertices / 2;
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init_dist<<<nBlocks_for_vertices, B_SIZE>>>(dist, num_vertices, source);
    cudaDeviceSynchronize();

    int *changed;
    cudaMalloc(&changed, sizeof(int));
    cudaMemset(changed, 0, sizeof(int));

    int *launchedKernel;
    cudaMallocManaged(&launchedKernel, sizeof(int));
    launchedKernel[0] = 0;

    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaError_t result = cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

    if (result == cudaSuccess) {
    if (supportsCoopLaunch) {
      printf("Cooperative launches are supported on device %d\n", dev);
    } else {
      // Cooperative launches are not supported on this device
      printf("Cooperative launches are not supported on device %d\n", dev);
    }
  } else {
    printf("cudaDeviceGetAttribute failed with error: %s\n", cudaGetErrorName(result));
  }

    dim3 blockSize(256,1,1);
    dim3 gridSize(1024,1,1);

    ArgsStruct *para;
    cudaMalloc(&para, sizeof(ArgsStruct));
    
    // setParams<<<1,1>>>(dist, dev_row_ptr, dev_col_ind, num_vertices, changed, (void **)para);
    // cudaDeviceSynchronize();

    void *kernelArgs[] = {(void *)&dist, (void *)&dev_row_ptr, (void *)&dev_col_ind, (void *)&num_vertices, (void *)&changed, (void *)&launchedKernel};

    // cout << "here" << endl;
    cudaLaunchCooperativeKernel((void*)BFS, gridSize, blockSize, kernelArgs);
    auto error = cudaDeviceSynchronize();

    int *dist_copy;
    dist_copy = (int *)malloc(sizeof(int) * num_vertices);

    cudaMemcpy(dist_copy, dist, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_vertices; i++) {
        cout << dist_copy[i] << " ";
    }
    cout << endl;

    cout << launchedKernel[0] << endl;

    return 0;
}