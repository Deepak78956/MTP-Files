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

    if (DEBUG)
    {
        printf("id = %d and its val %d\n", id, graph->adjLists[id]->data);
    }
}

__global__ void initEdgeList(struct Node *edgeList, int *dev_col_ind, int size)
{
    if (DEBUG)
    {
        printf("In init edgelist func\n");
        printf("size got %d\n", size);
    }

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

        for (int v = start; v < end; v++)
        {
            edgeList[v].next = graph->adjLists[id];
            graph->adjLists[id] = &edgeList[v];
        }
    }
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

__device__ int isNeigh(struct Graph *graph, int t, int r)
{
    struct Node *temp = graph->adjLists[r];
    while (temp)
    {
        if (temp->data == t)
            return 1;
        temp = temp->next;
    }
    return 0;
}

__device__ int tc = 0;

__global__ void countTriangles(struct Graph *graph)
{
    unsigned p = blockDim.x * blockIdx.x + threadIdx.x;

    if (p < graph->numVertices)
    {
        struct Node *t = graph->adjLists[p];
        while (t)
        {
            struct Node *r = graph->adjLists[p];
            while (r)
            {
                if (t != r && isNeigh(graph, t->data, r->data))
                {
                    atomicAdd(&tc, 1);
                }
                r = r->next;
            }
            t = t->next;
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
    initEdgeList<<<nBlocks_for_edges, B_SIZE>>>(edgeList, dev_col_ind, size);
    cudaDeviceSynchronize();

    makeD_LL<<<nBlocks_for_vertices, B_SIZE>>>(edgeList, dev_row_ptr, graph, size);
    cudaDeviceSynchronize();

    clock_t calcTime;

    calcTime = clock();
    countTriangles<<<nBlocks_for_vertices, B_SIZE>>>(graph);
    cudaDeviceSynchronize();
    calcTime = clock() - calcTime;

     printTc<<<1, 1>>>();
     cudaDeviceSynchronize();
    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    return 0;
}
