#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <cuda.h>
#define DEBUG true
#define B_SIZE 1024

using namespace std;

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

__global__ void computePR(struct CSR *csr, struct CSR *in_csr, float *pr, float d)
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
                float temp = pr[t] / out_deg_t;
                atomicAdd(&val, temp);
                if (DEBUG)
                    printf("%f\n", val);
            }
        }

        pr[p] = val * d + (1 - d) / csr->num_vertices;
    }
}

__global__ void printPR(float *pr, int vertices)
{
    printf("hey\n");
    for (int i = 0; i < vertices; i++)
    {
        printf("%lf ", pr[i]);
    }
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

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

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

    vector<vector<int>> edges(size, vector<int>(2, 0));
    vector<vector<int>> in_neigh(size, vector<int>(2, 0));
    for (int i = 0; i < num_edges; i++)
    {
        int u, v;
        fin >> u >> v;
        edges[i][0] = u;
        edges[i][1] = v;

        in_neigh[i][0] = v;
        in_neigh[i][1] = u;
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

    assignToCSR<<<1, 1>>>(dev_offsetArr, dev_edgeList, num_vertices, num_edges, dev_csr);
    assignToCSR<<<1, 1>>>(dev_in_offsetArr, dev_in_edgeList, num_vertices, num_edges, dev_in_csr);
    cudaDeviceSynchronize();

    if (DEBUG == true)
    {
        checkAssignment<<<1, 1>>>(dev_csr);
        checkAssignment<<<1, 1>>>(dev_in_csr);
        cudaDeviceSynchronize();
    }

    float *pr;
    cudaMalloc(&pr, sizeof(float) * num_vertices);
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init<<<nBlocks_for_vertices, B_SIZE>>>(pr, num_vertices);
    cudaDeviceSynchronize();

    // if (DEBUG == true)
    // {
    //     checkPR<<<1, 1>>>(pr, num_vertices);
    //     cudaDeviceSynchronize();
    // }

    float d = 0.85;
    computePR<<<nBlocks_for_vertices, B_SIZE>>>(dev_csr, dev_in_csr, pr, d);
    cudaDeviceSynchronize();

    printPR<<<1, 1>>>(pr, num_vertices);
    cudaDeviceSynchronize();

    return 0;
}