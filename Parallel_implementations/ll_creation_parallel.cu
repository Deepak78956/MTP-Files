#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"
#include <cuda.h>
#define DEBUG false
#define B_SIZE 1024

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
    // for (int i = 0; i < vertices; i++)
    // {
    //     graph->adjLists[i] = NULL;
    // }
    if (DEBUG)
    {
        printf("id = %d and its val %d\n", id, graph->adjLists[id]);
    }
}

__global__ void initEdgeList(struct Node **edgeList, int *dev_col_ind, int size)
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
        edgeList[id]->data = dev_col_ind[id];
    }
}

__global__ void makeD_LL(struct Node **edgeList, int *dev_row_ptr, struct Graph *graph, int size)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < graph->numVertices)
    {
        int start = dev_row_ptr[id];
        int end = dev_row_ptr[id + 1];
        // printf("For vertex number %d Edges are: ", id);
        for (int v = start; v < end; v++)
        {
            edgeList[v]->next = graph->adjLists[id];
            graph->adjLists[id] = edgeList[v];
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

__global__ void allocate(struct Node *node, struct Node **edgeList, int i)
{
    edgeList[i] = node;
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size;
    if (!directed)
        size = 2 * num_edges;
    else
    {
        cout << "Un-Directed graph is required" << endl;
        exit(0);
    }
    if (weighted)
    {
        cout << "Non weighted graph is required" << endl;
        exit(0);
    }
    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);
    if (DEBUG == true)
    {
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << csr.row_ptr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.col_ind[i] << " ";
        }
        cout << endl;
    }

    int *row_ptr, *col_index;
    row_ptr = (int *)malloc(sizeof(int) * (num_vertices + 1));
    col_index = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        row_ptr[i] = csr.row_ptr[i];
    }

    for (int i = 0; i < size; i++)
    {
        col_index[i] = csr.col_ind[i];
    }

    int *dev_row_ptr, *dev_col_ind;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (num_vertices + 1));
    cudaMalloc(&dev_col_ind, sizeof(int) * size);
    cudaMemcpy(dev_row_ptr, row_ptr, sizeof(int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ind, col_index, sizeof(int) * size, cudaMemcpyHostToDevice);

    struct Graph *graph;
    struct Node **adjLists;
    cudaMalloc(&graph, sizeof(struct Graph));
    cudaMalloc(&adjLists, sizeof(struct Node *) * num_vertices);

    clock_t initClock, initEdgeClock, makeD_LLClock;

    initClock = clock();
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    initGraph<<<nBlocks_for_vertices, B_SIZE>>>(graph, num_vertices, adjLists);
    cudaDeviceSynchronize();
    initClock = clock() - initClock;

    struct Node **edgeList;
    cudaMalloc(&edgeList, size * sizeof(struct Node *));
    for (int i = 0; i < size; i++)
    {
        struct Node *node;
        cudaMalloc(&node, sizeof(struct Node));
        allocate<<<1, 1>>>(node, edgeList, i);
        if (DEBUG)
        {
            printf("Allocated for edge %d\n", i);
        }
    }

    initEdgeClock = clock();
    unsigned nBlocks_for_edges = ceil((float)size / B_SIZE);
    initEdgeList<<<nBlocks_for_edges, B_SIZE>>>(edgeList, dev_col_ind, size);
    cudaDeviceSynchronize();
    initEdgeClock = clock() - initEdgeClock;

    makeD_LLClock = clock();
    makeD_LL<<<nBlocks_for_vertices, B_SIZE>>>(edgeList, dev_row_ptr, graph, size);
    cudaDeviceSynchronize();
    makeD_LLClock = clock() - makeD_LLClock;

    // printD_LL<<<1, 1>>>(graph);
    // cudaDeviceSynchronize();

    cout << endl;
    cout << "Total time taken: " << ((double)(makeD_LLClock + initEdgeClock + initClock)) / CLOCKS_PER_SEC << endl;

    return 0;
}