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

using namespace std;

__device__ float d = 0.85;

struct CSR
{
    int *offsetArr;
    int *edgeList;
    int num_vertices;
    int num_edges;
};

__global__ void assignToCSR(int *dev_offsetArr, int *dev_edgeList, int num_vertices, int num_edges, struct CSR *dev_csr)
{
    dev_csr->num_edges = num_edges;
    dev_csr->num_vertices = num_vertices;
    dev_csr->edgeList = dev_edgeList;
    dev_csr->offsetArr = dev_offsetArr;
}

__global__ void init(float *pr, int num_vertices)
{
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    if (DEBUG)
    {
        if (id == 0)
            printf("Inside init with num vertices = %d \n", num_vertices);
    }
    if (id < num_vertices)
    {
        pr[id] = 1.0 / num_vertices;
        if (DEBUG)
            printf("id = %d, val = %f, actual value = %f\n", id, pr[id], 1.0 / num_vertices);
    }
}

__global__ void checkPR(float *pr, int num_vertices)
{
    for (int i = 0; i < num_vertices; i++)
    {
        printf("%lf ", pr[i]);
    }
    printf("\n");
}

__global__ void computePR(struct CSR *csr, struct CSR *in_csr, float *oldPr, float *newPr)
{
    float val = 0.0;
    unsigned p = blockDim.x * blockIdx.x + threadIdx.x;

    if (DEBUG && p == 0)
        printf("Inside PR, value of d = %f\n", d);

    if (p < csr->num_vertices)
    {
        for (int i = in_csr->offsetArr[p]; i < in_csr->offsetArr[p + 1]; i++)
        {
            unsigned t = in_csr->edgeList[i];
            unsigned out_deg_t = csr->offsetArr[t + 1] - csr->offsetArr[t];

            if (out_deg_t != 0)
            {
                // pr[t] can have race conditions because of pr[p], use of atomics is needed
                // val += pr[t] / out_deg_t;
                float temp = oldPr[t] / out_deg_t;
                val += oldPr[t] / out_deg_t;
                // atomicAdd(&val, temp);
                if (DEBUG)
                    printf("%f\n", val);
            }
        }

        newPr[p] = val * d + (1 - d) / csr->num_vertices; // Need of lock here
    }
}

__global__ void printPR(float *pr, int vertices)
{
    for (int i = 0; i < vertices; i++)
    {
        printf("%lf ", pr[i]);
    }
    printf("\n");
}

__global__ void checkAssignment(struct CSR *csr)
{
    printf("Checking correct assignment to GPU \n");
    for (int i = 0; i < csr->num_vertices + 1; i++)
    {
        printf("%d ", csr->offsetArr[i]);
    }
    printf("\n");

    for (int i = 0; i < csr->num_edges; i++)
    {
        printf("%d ", csr->edgeList[i]);
    }
    printf("\n");
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

    istringstream header(line);
    int num_vertices, num_edges, x;
    header >> num_vertices >> x >> num_edges;
    // num_vertices += 1;

    int size;
    if (directed)
        size = num_edges;
    else
    {
        cout << "Directed graph is required" << endl;
        exit(0);
    }
    if (weighted)
    {
        cout << "Non weighted graph is required" << endl;
        exit(0);
    }

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

    vector<vector<int>> edges(size, vector<int>(2, 0));
    vector<vector<int>> in_neigh(size, vector<int>(2, 0));
    for (int i = 0; i < num_edges; i++)
    {
        int u, v, w;
        if (keywordFound) {
            fin >> u >> v >> w;
        }
        else {
            fin >> u >> v;
        }
        edges[i][0] = u - 1;
        edges[i][1] = v - 1;

        in_neigh[i][0] = v - 1;
        in_neigh[i][1] = u - 1;
    }

    sort(edges.begin(), edges.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[0] < b[0]; });

    sort(in_neigh.begin(), in_neigh.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[0] < b[0]; });

    int *edgeList, *offsetArr;
    edgeList = (int *)malloc(sizeof(int) * size);
    offsetArr = (int *)malloc(sizeof(int) * num_vertices + 1);

    int *in_edgeList, *in_offsetArr;
    in_edgeList = (int *)malloc(sizeof(int) * size);
    in_offsetArr = (int *)malloc(sizeof(int) * num_vertices + 1);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        offsetArr[i] = 0;
        in_offsetArr[i] = 0;
    }

    for (int i = 0; i < num_edges; i++)
    {
        int u, v;
        u = edges[i][0];
        v = edges[i][1];

        int vertex = in_neigh[i][0];
        int vertex_in_neigh = in_neigh[i][1];

        edgeList[i] = v;
        offsetArr[u + 1] += 1;

        in_edgeList[i] = vertex_in_neigh;
        in_offsetArr[vertex + 1] += 1;
    }

    for (int i = 1; i < num_vertices + 1; i++)
    {
        offsetArr[i] += offsetArr[i - 1];
        in_offsetArr[i] += in_offsetArr[i - 1];
    }

    if (DEBUG == true)
    {
        cout << "For normal CSR" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << offsetArr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << edgeList[i] << " ";
        }
        cout << endl;
        cout << "For in neigh CSR" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << in_offsetArr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << in_edgeList[i] << " ";
        }
        cout << endl;
    }

    struct CSR csr = {offsetArr, edgeList, num_vertices, num_edges};
    struct CSR in_csr = {in_offsetArr, in_edgeList, num_vertices, num_edges};

    int *dev_offsetArr, *dev_edgeList;
    int *dev_in_offsetArr, *dev_in_edgeList;

    clock_t calcTime, assignTime, initTime, initialMemOP, prMem;

    initialMemOP = clock();
    cudaMalloc(&dev_offsetArr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_in_offsetArr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_edgeList, sizeof(int) * size);
    cudaMalloc(&dev_in_edgeList, sizeof(int) * size);

    cudaMemcpy(dev_offsetArr, csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_in_offsetArr, in_csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_in_edgeList, in_csr.edgeList, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_edgeList, csr.edgeList, sizeof(int) * num_edges, cudaMemcpyHostToDevice);

    struct CSR *dev_csr, *dev_in_csr;
    cudaMalloc(&dev_csr, sizeof(struct CSR));
    cudaMalloc(&dev_in_csr, sizeof(struct CSR));
    initialMemOP = clock() - initialMemOP;

    assignTime = clock();
    assignToCSR<<<1, 1>>>(dev_offsetArr, dev_edgeList, num_vertices, num_edges, dev_csr);
    assignToCSR<<<1, 1>>>(dev_in_offsetArr, dev_in_edgeList, num_vertices, num_edges, dev_in_csr);
    cudaDeviceSynchronize();
    assignTime = clock() - assignTime;

    if (DEBUG == true)
    {
        checkAssignment<<<1, 1>>>(dev_csr);
        checkAssignment<<<1, 1>>>(dev_in_csr);
        cudaDeviceSynchronize();
    }

    float *pr, *prCopy;

    prMem = clock();
    cudaMalloc(&pr, sizeof(float) * num_vertices);
    cudaMalloc(&prCopy, sizeof(float) * num_vertices);
    prMem = clock() - prMem;

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);

    initTime = clock();
    init<<<nBlocks_for_vertices, B_SIZE>>>(pr, num_vertices);
    init<<<nBlocks_for_vertices, B_SIZE>>>(prCopy, num_vertices);
    cudaDeviceSynchronize();
    initTime = clock() - initTime;

    // if (DEBUG == true)
    // {
    //     checkPR<<<1, 1>>>(pr, num_vertices);
    //     cudaDeviceSynchronize();
    // }

    int max_iter = 10;
    // 3rd and 4th param for oldPr and newpr
    calcTime = clock();

    for (int i = 1; i < max_iter + 1; i++)
    {
        if (i % 2 == 0)
        {
            computePR<<<nBlocks_for_vertices, B_SIZE>>>(dev_csr, dev_in_csr, pr, prCopy);
        }
        else
        {
            computePR<<<nBlocks_for_vertices, B_SIZE>>>(dev_csr, dev_in_csr, prCopy, pr);
        }
        cudaDeviceSynchronize();
    }

    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime + (double)initTime + (double)assignTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    // if (max_iter % 2 == 0)
    // {
    //     printPR<<<1, 1>>>(prCopy, num_vertices);
    //     cudaDeviceSynchronize();
    // }
    // else
    // {
    //     printPR<<<1, 1>>>(pr, num_vertices);
    //     cudaDeviceSynchronize();
    // }

    return 0;
}