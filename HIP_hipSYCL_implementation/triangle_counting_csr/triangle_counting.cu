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
#define weighted 0

using namespace std;

__device__ int isNeigh(int *offset, int * values, int t, int r)
{
    for (int i = offset[r]; i < offset[r + 1]; i++) {
        int temp = values[i];
        if (temp == t) return 1;
    }
    return 0;
}

__device__ int tc = 0;

__global__ void countTriangles(int *offset, int *values, int n)
{
    unsigned p = blockDim.x * blockIdx.x + threadIdx.x;

    if (p < n)
    {
        for (int i = offset[p]; i < offset[p+1]; i++) {
            int t = values[i];
            for (int j = offset[p]; j < offset[p+1]; j++){
                int r = values[j];
                if (t != r && isNeigh(offset, values, t, r)){
                    atomicAdd(&tc, 1);
                }
            }
        }
    }
}

__global__ void printTc()
{
    printf("Triangles got %d\n", tc / 6);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    string fileName = argv[1];
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

    int num_vertices, num_edges, x;
    istringstream header(line);
    header >> num_vertices >> x >> num_edges;
    // num_vertices += 1;

    vector<string> keywords = {"kron", "file"};

    bool keywordFound = false;

    for (const string& keyword : keywords) {
        // Check if the keyword is present in the filename//
        if (fileName.find(keyword) != string::npos) {
            // Set the flag to true indicating the keyword is found
            keywordFound = true;
            break;
        }
    }

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin, keywordFound);

    int size = 2 * num_edges;

    if (DEBUG == true)
    {
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << csr.offsetArr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.edgeList[i] << " ";
        }
        cout << endl;
    }

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

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, csr.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);

    unsigned nBlocks_for_edges = ceil((float)size / B_SIZE);
    clock_t calcTime;

    calcTime = clock();
    countTriangles<<<nBlocks_for_vertices, B_SIZE>>>(dev_row_ptr, dev_col_ind, num_vertices);
    cudaDeviceSynchronize();
    calcTime = clock() - calcTime;

     printTc<<<1, 1>>>();
     cudaDeviceSynchronize();
    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    return 0;
}
