#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include <thrust/count.h>
#include <cooperative_groups.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 0

__global__ void graph_coloring_kernel(int n, int c, int *offsets, int *values, int *randoms, int *colors){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = id; i < n; i += blockDim.x * gridDim.x) {
        bool f = true; // true iff you have max random

        if ((colors[i] != -1)) continue; // ignore nodes colored earlier

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = offsets[i]; k < offsets[i + 1]; k++) {
            int j = values[k];
            int jc = colors[j];

            // ignore nodes colored earlier (and yourself)
            if (((jc != -1) && (jc != c)) || (i == j)) continue;

            int jr = randoms[j];
            if (ir <= jr) f = false;
        }

        // assign color if you have the maximum random number
        if (f) colors[i] = c;
    }
}

void graph_coloring(int n, int *offsets, int *values) {
    int *randoms; // have to allocate and init randoms
    
    int *colors;
    cudaMalloc(&colors, sizeof(int)*n);
    thrust::fill(colors, colors + n, -1);
    randoms = (int *)malloc(sizeof(int) * n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, n);

    for (int i = 0; i < n; i++) {
        int randNum = dis(gen);
        randoms[i] = randNum;
    }

    int *dev_randoms;
    cudaMalloc(&dev_randoms, sizeof(int) * n);
    cudaMemcpy(dev_randoms, randoms, sizeof(int)*n, cudaMemcpyHostToDevice);

    clock_t t_time = 0, temp_time;
    long int iterations = 0;

    for (int c = 0; c < n; c++) {
        int nt = B_SIZE;
        int nb =  ceil((float)n / nt);
        iterations += 1;

        temp_time = clock();
        graph_coloring_kernel<<<nb, nt>>>(n, c, offsets, values, dev_randoms, colors);
        cudaDeviceSynchronize();
        temp_time = clock() - temp_time;

        t_time += temp_time;

        int left = (int)thrust::count(colors, colors + n, -1);
        
        if (left == 0) break;
    }

    t_time = ((double)t_time) / CLOCKS_PER_SEC * 1000;

    std::cout << "Iterations: " << iterations << std::endl; 
    std::cout << "Time taken: " << t_time << std::endl;
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

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, csr.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);

    std::cout << "Time taken on graph " << fileName << std::endl;
    graph_coloring(num_vertices, dev_row_ptr, dev_col_ind);

    return 0;
}