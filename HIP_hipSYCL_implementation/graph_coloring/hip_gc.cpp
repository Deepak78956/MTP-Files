#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <hip/hip_runtime.h>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 0

__global__ void check(int *offsets, int *values) {
    // for (int i = offsets[23]; i < offsets[24]; i++) {
    //     printf("%d ", values[i]);
    // }
    // printf("\n");

    for (int i = 0; i < 39; i++) {
        printf("For vertex %d:\n", i);
        for (int j = offsets[i]; j < offsets[i + 1]; j++) {
            printf("%d ", values[j]);
        }
        printf("\n");
    }
}

__global__ void graph_coloring_kernel(int n, int c, int *offsets, int *values, int *randoms, int *colors){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = id; i < n; i += blockDim.x * gridDim.x) {
        int f = 1; // true iff you have max random

        if ((colors[i] != -1)) continue; // ignore nodes colored earlier

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = offsets[i]; k < offsets[i + 1]; k++) {
            int j = values[k];
            int jc = colors[j];

            // ignore nodes colored earlier (and yourself)
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            
            
            int jr = randoms[j];
            if (ir < jr) f = 0;
        }

        // assign color if you have the maximum random number
        if (f) colors[i] = c;
        // printf("id = %d\n", id);
    }
}

__global__ void countm1(int n, int *left, int *colors) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        if (colors[id] == -1) atomicAdd(left, 1);
    }
}

int* graph_coloring(int n, int *offsets, int *values) {
    int *randoms; // have to allocate and init randoms
    int *colors;
    hipMalloc(&colors, sizeof(int)*n);
    // thrust::fill(colors, colors + n, -1);
    hipMemset(colors, -1, sizeof(int) * n);
    randoms = (int *)malloc(sizeof(int) * n);

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<int> dis(0, n);

    for (int i = 0; i < n; i++) {
        // int randNum = dis(gen);
        int randNum = i;
        randoms[i] = randNum;
    }

    int *dev_randoms;
    hipMalloc(&dev_randoms, sizeof(int) * n);
    hipMemcpy(dev_randoms, randoms, sizeof(int)*n, hipMemcpyHostToDevice);

    clock_t t_time = 0, temp_time;
    long int iterations = 0;

    int *left_dev, *left_host;
    hipMalloc(&left_dev, sizeof(int));
    hipMemset(left_dev, 0, sizeof(int));
    left_host = (int *)malloc(sizeof(int));
    left_host[0] = 0; 

    for (int c = 0; c < n; c++) {
        int nt = B_SIZE;
        int nb =  ceil((float)n / nt);
        iterations += 1;

        temp_time = clock();
        graph_coloring_kernel<<<nb, nt>>>(n, c, offsets, values, dev_randoms, colors);
        // graph_coloring_kernel<<<1, 1>>>(n, c, offsets, values, dev_randoms, colors);
        hipDeviceSynchronize();
        temp_time = clock() - temp_time;

        t_time += temp_time;

        countm1<<<nb, nt>>>(n, left_dev, colors);
        hipDeviceSynchronize();

        // std::cout << iterations << std::endl;
        hipMemcpy(left_host, left_dev, sizeof(int), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        // std::cout << left_host[0] << std::endl;

        if (left_host[0] == 0) break;
    }

    double final_time = ((double)t_time) / CLOCKS_PER_SEC * 1000;

    std::cout << "Iterations: " << iterations << std::endl; 
    std::cout << "Time taken: " << final_time << std::endl;

    return colors;
}

void check_ans(int *colorsArr, int *offsets, int *values, int n) {
    int breakLoop = 0;
    for (int i = 0; i < n; i++) {
        int color_u = colorsArr[i];
        // std::cout << "vertex is " << i << std::endl;
        for (int j = offsets[i]; j < offsets[i + 1]; j++) {
            int color_v = colorsArr[values[j]];
            // std::cout << values[j] << std::endl;
            if (color_u == color_v) {
                printf("Wrong ans on vertex %d, same color %d with vertex %d\n", i, color_v, values[j]);
                breakLoop = 1;
                break;
            }
        }
        if (breakLoop) break;
        // std::cout << std::endl;
    }

    if (!breakLoop) std::cout << "Correct ans" << std::endl;
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

    int size;
    if (directed) size = num_edges;
    else size = 2 * num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin, keywordFound);

    int *dev_row_ptr, *dev_col_ind;
    hipMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    hipMalloc(&dev_col_ind, sizeof(int) * size);
    hipMemcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1), hipMemcpyHostToDevice);
    hipMemcpy(dev_col_ind, csr.edgeList, sizeof(int) * size, hipMemcpyHostToDevice);

    // for (int i = 34; i < 36; i++) {
    //     printf("%d ", csr.offsetArr[i]);
    // }

    // cout << endl;

    // for (int i = csr.offsetArr[23]; i < csr.offsetArr[24]; i++) {
    //     printf("%d ", csr.edgeList[i]);
    // }
    // cout << endl;

    // check<<<1,1>>>(dev_row_ptr, dev_col_ind);
    // cudaDeviceSynchronize();

    std::cout << "On graph " << fileName << std::endl;

    int *dev_colors;
    dev_colors = graph_coloring(num_vertices, dev_row_ptr, dev_col_ind);

    int *colors;
    colors = (int *)malloc(sizeof(int) * num_vertices);

    hipMemcpy(colors, dev_colors, sizeof(int) * num_vertices, hipMemcpyDeviceToHost);

    check_ans(colors, csr.offsetArr, csr.edgeList, num_vertices);
    std::cout << std::endl;

    hipFree(dev_colors);
    hipFree(dev_col_ind);
    hipFree(dev_row_ptr);

    return 0;
}