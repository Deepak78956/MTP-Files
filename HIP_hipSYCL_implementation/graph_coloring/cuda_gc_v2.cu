// This one is a little modification of the GC algorithm mentioned in the Graph analytics book
//and it required the edges to be in sorted order.

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 0

__device__ int isNeigh(int *offsets, int *values, int u, int v) {
    int neigh = 0;
    for (int i = offsets[u]; i < offsets[u + 1]; i++) {
        if (values[i] > v) break;
        else if (values[i] == v) {
            neigh = 1;
            break;
        }
    }

    return neigh;
}

__device__ void assignColor(int k, int *offsets, int *values, int *colors) {
    int v = k;
    colors[v] = 1;
    for (int i = 0; i < k; i++) {
        if (isNeigh(offsets, values, v, i) && colors[v] == colors[i]) {
            atomicAdd(&colors[v], colors[i] + 1);
        }
    }
}

__global__ void graph_coloring_kernel(int n, int *offsets, int *values, int *colors) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        // colors[id] = 0; // try color 1
        for (int i = 0; i < n; i++) {
            assignColor(i, offsets, values, colors);
        }
    }
}

int* graph_coloring(int n, int *offsets, int *values) {
    int nt = B_SIZE;
    int nb =  ceil((float)n / nt);

    int *colors;
    cudaMalloc(&colors, sizeof(int)*n);
    cudaMemset(colors, 0, sizeof(int) * n);

    clock_t t_time;

    t_time = clock();
    graph_coloring_kernel<<<nb, nt>>>(n, offsets, values, colors);
    cudaDeviceSynchronize();
    t_time = clock() - t_time;

    double final_time = ((double)t_time) / CLOCKS_PER_SEC * 1000;

    std::cout << "Time taken: " << final_time << std::endl;

    return colors;
}

void check_ans(int *colorsArr, int *offsets, int *values, int n) {
    int breakLoop = 0;
    for (int i = 0; i < n; i++) {
        int color_u = colorsArr[i];
        for (int j = offsets[i]; j < offsets[i + 1]; j++) {
            int color_v = colorsArr[values[j]];
            if (color_u == color_v) {
                printf("Wrong ans on vertex %d, same color %d with vertex %d\n", i, colorsArr[i], values[j]);
                breakLoop = 1;
                break;
            }
        }
        if (breakLoop) break;
    }

    if (!breakLoop) std::cout << "Correct ans" << std::endl;
}

int main(int argc, char *argv[]){
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

    // for (int i = 0; i < 15; i++) {
    //     printf("%d ", csr.offsetArr[i]);
    // }

    // cout << endl;

    // for (int i = 0; i < 15; i++) {
    //     printf("%d ", csr.edgeList[i]);
    // }
    // cout << endl;

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, csr.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);

    std::cout << "On graph " << fileName << std::endl;

    int *dev_colors;
    dev_colors = graph_coloring(num_vertices, dev_row_ptr, dev_col_ind);

    int *colors;
    colors = (int *)malloc(sizeof(int) * num_vertices);

    cudaMemcpy(colors, dev_colors, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    check_ans(colors, csr.offsetArr, csr.edgeList, num_vertices);
    cout << endl;

    cudaFree(dev_colors);
    cudaFree(dev_col_ind);
    cudaFree(dev_row_ptr);


    return 0;
}