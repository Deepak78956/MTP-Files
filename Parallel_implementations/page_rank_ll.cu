#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <cuda.h>
#define DEBUG false
#define B_SIZE 1024

using namespace std;

struct Node
{
    int data;
    struct Node *next;
};

struct Graph
{
    int numVertices;
    struct Node **adjLists;
};

struct CSR
{
    int numVertices;
    int *offsetArr;
    int *edgeList;
};

__global__ void initGraph(struct Graph *graph, int vertices, struct Node **adjLists)
{
    graph->numVertices = vertices;
    graph->adjLists = adjLists;

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    if (DEBUG)
    {
        printf("In init graph func\n");
        printf("vertices got %d\n", vertices);
    }

    if (id < vertices)
    {
        graph->adjLists[id] = NULL;
    }

    if (DEBUG)
    {
        printf("id = %d and its val %d\n", id, graph->adjLists[id]);
    }
}

__global__ void initEdgeList(struct Node *edgeList, int *dev_col_ind, int size)
{
    if (DEBUG)
    {
        printf("In init edgelist func\n");
        printf("size got %d\n", size);
    }
    // for (int i = 0; i < size; i++)
    // {
    //     if (DEBUG)
    //     {
    //         printf("loop in initEdge %d\n", i);
    //     }

    //     edgeList[i]->data = dev_col_ind[i];

    //     if (DEBUG)
    //     {
    //         printf("data of dev_col_ind i got %d\n", dev_col_ind[i]);
    //         printf("data of Edgelist i got %d\n", edgeList[i]->data);
    //     }
    // }
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        edgeList[id].data = dev_col_ind[id];
    }
}

__global__ void makeD_LL(struct Node *edgeList, int *dev_row_ptr, struct Graph *graph, int size)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < graph->numVertices)
    {
        int start = dev_row_ptr[id];
        int end = dev_row_ptr[id + 1];
        // printf("For vertex number %d Edges are: ", id);
        for (int v = start; v < end; v++)
        {
            edgeList[v].next = graph->adjLists[id];
            graph->adjLists[id] = &edgeList[v];
            // printf(" %d ", graph->adjLists[id]);
        }
        // printf("\n");
    }
    // for (int u = 0; u < graph->numVertices; u++)
    // {
    //     int start = dev_row_ptr[u];
    //     int end = dev_row_ptr[u + 1];
    //     for (int v = start; v < end; v++)
    //     {
    //         edgeList[v]->next = graph->adjLists[u];
    //         graph->adjLists[u] = edgeList[v];
    //     }
    // }
}

struct CSR returnCSR(vector<vector<int>> &edges, int size, int num_vertices)
{
    sort(edges.begin(), edges.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[0] < b[0]; });

    int *edgeList, *offsetArr;
    edgeList = (int *)malloc(sizeof(int) * size);
    offsetArr = (int *)malloc(sizeof(int) * num_vertices + 1);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        offsetArr[i] = 0;
    }

    for (int i = 0; i < size; i++)
    {
        offsetArr[edges[i][0] + 1] += 1;
        edgeList[i] = edges[i][1];
    }

    for (int i = 1; i < num_vertices + 1; i++)
    {
        offsetArr[i] += offsetArr[i - 1];
    }

    if (DEBUG == true)
    {
        cout << "CSR" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << offsetArr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << edgeList[i] << " ";
        }
    }

    struct CSR csr = {num_vertices, offsetArr, edgeList};

    return csr;
}

struct Graph *makeGraph(struct CSR csr, int num_vertices, int size)
{
    int *dev_offset_arr, *dev_edge_list;

    cudaMalloc(&dev_offset_arr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_edge_list, sizeof(int) * size);

    cudaMemcpy(dev_offset_arr, csr.offsetArr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_edge_list, csr.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);

    struct Graph *graph;
    struct Node **adjLists;
    cudaMalloc(&graph, sizeof(struct Graph));
    cudaMalloc(&adjLists, sizeof(struct Node *) * num_vertices);

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    initGraph<<<nBlocks_for_vertices, B_SIZE>>>(graph, num_vertices, adjLists);
    cudaDeviceSynchronize();

    struct Node *edgeList;
    cudaMalloc((struct Node **)&edgeList, size * sizeof(struct Node));

    unsigned nBlocks_for_edges = ceil((float)size / B_SIZE);
    initEdgeList<<<nBlocks_for_edges, B_SIZE>>>(edgeList, dev_edge_list, size);
    cudaDeviceSynchronize();

    makeD_LL<<<nBlocks_for_vertices, B_SIZE>>>(edgeList, dev_offset_arr, graph, size);
    cudaDeviceSynchronize();

    return graph;
}

__global__ void printD_LL(struct Graph *graph)
{
    int vertices = graph->numVertices;
    for (int u = 0; u < vertices; u++)
    {
        struct Node *temp = graph->adjLists[u];
        printf("For vertex %d its neighbors are: ", u);
        while (temp)
        {
            printf("%d ", temp->data);
            temp = temp->next;
        }
        printf("\n");
    }
}

__device__ float d = 0.85;

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

__global__ void computePR(struct Graph *inNeigh, int *outDeg, float *oldPr, float *newPr)
{
    float val = 0.0;
    unsigned p = blockDim.x * blockIdx.x + threadIdx.x;

    if (DEBUG && p == 0)
        printf("Inside PR, value of d = %f\n", d);

    if (p < inNeigh->numVertices)
    {
        struct Node *t = inNeigh->adjLists[p];
        // printf("p: %d\n", p);
        while (t)
        {
            // printf("p: %d, t: %d, outdeg: %d\n", p, t->data, outDeg[t->data]);
            if (outDeg[t->data] != 0)
            {
                float temp = oldPr[t->data] / (float)outDeg[t->data];
                val += temp;
                // printf("outdeg: %d, val: %f, temp: %f\n", outDeg[t->data], val, temp);
            }
            t = t->next;
        }

        newPr[p] = val * d + (1 - d) / inNeigh->numVertices; // Need of lock here
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

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size = num_edges;

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

    struct CSR mainCSR = returnCSR(edges, size, num_vertices);
    struct CSR inNeighCSR = returnCSR(in_neigh, size, num_vertices);

    int *outDeg;
    outDeg = (int *)malloc(num_vertices * sizeof(int));
    cout << "printing outdegs" << endl;
    for (int i = 0; i < num_vertices; i++)
    {
        outDeg[i] = mainCSR.offsetArr[i + 1] - mainCSR.offsetArr[i];
    }

    // struct Graph *graph = makeGraph(mainCSR, num_vertices, size);
    struct Graph *inNeigh = makeGraph(inNeighCSR, num_vertices, size);

    int *dev_outDeg;
    cudaMalloc(&dev_outDeg, sizeof(int) * num_vertices);
    cudaMemcpy(dev_outDeg, outDeg, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);

    float *pr, *prCopy;
    cudaMalloc(&pr, sizeof(float) * num_vertices);
    cudaMalloc(&prCopy, sizeof(float) * num_vertices);
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    init<<<nBlocks_for_vertices, B_SIZE>>>(pr, num_vertices);
    init<<<nBlocks_for_vertices, B_SIZE>>>(prCopy, num_vertices);
    cudaDeviceSynchronize();

    int max_iter = 1;
    // 3rd and 4th param for oldPr and newpr
    for (int i = 1; i < max_iter + 1; i++)
    {
        if (i % 2 == 0)
        {
            computePR<<<nBlocks_for_vertices, B_SIZE>>>(inNeigh, dev_outDeg, pr, prCopy);
        }
        else
        {
            computePR<<<nBlocks_for_vertices, B_SIZE>>>(inNeigh, dev_outDeg, prCopy, pr);
        }
        cudaDeviceSynchronize();
    }

    if (max_iter % 2 == 0)
    {
        printPR<<<1, 1>>>(prCopy, num_vertices);
        cudaDeviceSynchronize();
    }
    else
    {
        printPR<<<1, 1>>>(pr, num_vertices);
        cudaDeviceSynchronize();
    }

    return 0;
}