// Batch processing for linked list deletions
#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <cuda.h>

#define DEBUG false
#define BATCH_SIZE 100000
#define BATCH_SIZE_FOR_DEL 100000
#define BLOCK_SIZE 1024

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
    int *vertexArr, *offsetArr, *edgeList, entries;
};

struct CSR_del
{
    int *offsetArr, *edgeList;
};

__global__ void setNone(int vertices, struct Node **list)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < vertices)
    {
        list[id] = NULL;
    }
}

__global__ void copyAdjlist(struct Graph *graph, struct Node **tempLst)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < graph->numVertices)
    {
        tempLst[id] = graph->adjLists[id];
    }
}

__global__ void initGraph(struct Graph *graph, int vertices, struct Node **list)
{
    graph->numVertices = vertices;
    graph->adjLists = list;
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

__global__ void makeD_LL(struct Node *edgeList, int *dev_row_ptr, int *dev_vertex_arr, struct Graph *graph, int num_vertices)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_vertices)
    {
        int start = dev_row_ptr[id];
        int end = dev_row_ptr[id + 1];

        int u = dev_vertex_arr[id];
        for (int v = start; v < end; v++)
        {
            edgeList[v].next = graph->adjLists[u];
            graph->adjLists[u] = &edgeList[v];
        }
    }
}

__global__ void printD_LL(struct Graph *graph)
{
    int vertices = graph->numVertices;
    int edges = 0;
    for (int u = 0; u < vertices; u++)
    {
        struct Node *temp = graph->adjLists[u];
        printf("For vertex %d its neighbors are: ", u);
        while (temp)
        {
            printf("%d ", temp->data);
            temp = temp->next;
            edges++;
        }
        printf("\n");
    }
    printf("Total Vertices: %d\n", vertices);
    printf("Totak Edges: %d\n", edges);
}

__global__ void del_edge(struct Graph *graph, int *vertexArr, int *offsetArr, int *edgeList, int num_vertices)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_vertices)
    {
        int vertex = vertexArr[id];
        for (unsigned j = offsetArr[id]; j < offsetArr[id + 1]; j++)
        {
            int toDel = edgeList[j];
            struct Node *temp = graph->adjLists[vertex];

            while (temp)
            {
                if (temp->data == toDel)
                {
                    temp->data = -1;
                    break;
                }
                temp = temp->next;
            }
        }
    }
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, total_edges, directed, weighted;
    fin >> num_vertices >> total_edges >> directed >> weighted;

    struct Graph *graph;
    int dev_ll_size = 0;

    cudaMalloc(&graph, sizeof(struct Graph));

    clock_t setNoneClk, copyAdjClk, edgeBuffClk;
    double totalTime = 0;

    while (total_edges > 0)
    {
        int size, num_edges_cur_batch;

        // Defining edges in current batch size
        if (BATCH_SIZE > total_edges)
        {
            num_edges_cur_batch = total_edges;
            total_edges = 0;
        }
        else
        {
            num_edges_cur_batch = BATCH_SIZE;
            total_edges -= BATCH_SIZE;
        }

        // 2 * size for undirected edges
        size = 2 * num_edges_cur_batch;
        vector<vector<int>> edges(size, vector<int>(2, 0));

        // Reading from file
        for (int i = 0; i < num_edges_cur_batch; i++)
        {
            int u, v;
            fin >> u >> v;
            edges[i][0] = u;
            edges[i][1] = v;
            if (!directed)
            {
                edges[num_edges_cur_batch + i][0] = v;
                edges[num_edges_cur_batch + i][1] = u;
            }
        }

        sort(edges.begin(), edges.end(), [](const vector<int> &a, const vector<int> &b)
             { return a[0] < b[0]; });

        if (DEBUG == true)
        {
            cout << "sorted output after file reading: " << endl;
            for (int i = 0; i < num_edges_cur_batch; i++)
            {
                cout << edges[i][0] << " " << edges[i][1] << endl;
            }
            cout << endl;
        }

        int *edgeList, *offsetArr, *vertexArr;

        // Making and Filling EdgeList while also maintaing frequency map for offest array
        edgeList = (int *)malloc(sizeof(int) * size);
        map<int, int> hmap;

        int i = 0, maximum = INT_MIN, mapSize = 0;
        for (const auto &edge : edges)
        {
            int u, v;
            u = edge[0];
            v = edge[1];

            hmap[u]++;

            maximum = max(maximum, u);

            edgeList[i] = v;
            i++;
        }

        mapSize = hmap.size();

        // making vertexArray and offsetArray
        vertexArr = (int *)malloc(sizeof(int) * mapSize);
        offsetArr = (int *)malloc(sizeof(int) * (mapSize + 1));
        offsetArr[0] = 0;

        map<int, int>::iterator it = hmap.begin();

        i = 0;
        while (it != hmap.end())
        {
            vertexArr[i] = it->first;
            // Frequency fill leaving index 0 for 0
            offsetArr[i + 1] = it->second;
            ++i;
            ++it;
        }

        for (int i = 1; i < mapSize + 1; i++)
        {
            offsetArr[i] += offsetArr[i - 1];
        }

        struct CSR csr = {vertexArr, offsetArr, edgeList, mapSize};

        if (DEBUG == true)
        {
            cout << "Vertex array: ";
            for (int i = 0; i < mapSize; i++)
            {
                cout << csr.vertexArr[i] << " ";
            }
            cout << endl;
            cout << "Offset array: ";
            for (int i = 0; i < mapSize + 1; i++)
            {
                cout << csr.offsetArr[i] << " ";
            }
            cout << endl;
            cout << "Edge list: ";
            for (int i = 0; i < size; i++)
            {
                cout << csr.edgeList[i] << " ";
            }
            cout << endl;
        }

        int *dev_offset_arr, *dev_edge_list, *dev_vertex_arr;
        cudaMalloc(&dev_offset_arr, sizeof(int) * (mapSize + 1));
        cudaMalloc(&dev_edge_list, sizeof(int) * size);
        cudaMalloc(&dev_vertex_arr, sizeof(int) * mapSize);
        cudaMemcpy(dev_offset_arr, csr.offsetArr, sizeof(int) * (mapSize + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edge_list, csr.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vertex_arr, csr.vertexArr, sizeof(int) * mapSize, cudaMemcpyHostToDevice);

        if (dev_ll_size < maximum)
        {
            // make space for bigger adjlist
            int req_size = maximum * 2;

            struct Node **tempList;

            setNoneClk = clock();
            cudaMalloc(&tempList, sizeof(struct Node *) * req_size);
            unsigned nBlocks_for_setNone = ceil((float)req_size / BLOCK_SIZE);
            setNone<<<nBlocks_for_setNone, BLOCK_SIZE>>>(req_size, tempList);
            cudaDeviceSynchronize();
            setNoneClk = clock() - setNoneClk;
            totalTime += ((double)setNoneClk) / CLOCKS_PER_SEC * 1000;

            if (dev_ll_size != 0)
            {
                // copy old adjlist
                unsigned nBlock_for_copy_adjlist = ceil((float)dev_ll_size / BLOCK_SIZE);
                copyAdjClk = clock();
                copyAdjlist<<<nBlock_for_copy_adjlist, BLOCK_SIZE>>>(graph, tempList);
                cudaDeviceSynchronize();
                copyAdjClk = clock() - copyAdjClk;
                totalTime += ((double)copyAdjClk) / CLOCKS_PER_SEC * 1000;
            }

            // assign new number of vertices to graph and assign tempList to adjList of graph
            initGraph<<<1, 1>>>(graph, req_size, tempList);
            cudaDeviceSynchronize();

            dev_ll_size = req_size;
        }

        edgeBuffClk = clock();
        struct Node *edgeList_buffer;
        cudaMalloc((struct Node **)&edgeList_buffer, size * sizeof(struct Node));

        unsigned nBlocks_for_edges = ceil((float)size / BLOCK_SIZE);
        initEdgeList<<<nBlocks_for_edges, BLOCK_SIZE>>>(edgeList_buffer, dev_edge_list, size);
        cudaDeviceSynchronize();

        unsigned nBlocks_for_vertices = ceil((float)mapSize / BLOCK_SIZE);
        makeD_LL<<<nBlocks_for_vertices, BLOCK_SIZE>>>(edgeList_buffer, dev_offset_arr, dev_vertex_arr, graph, mapSize);
        cudaDeviceSynchronize();
        edgeBuffClk = clock() - edgeBuffClk;

        totalTime += ((double)edgeBuffClk) / CLOCKS_PER_SEC * 1000;
    }

    // printD_LL<<<1, 1>>>(graph);
    // cudaDeviceSynchronize();

    // cout << "Total time taken by kernels to Execute: " << totalTime << endl;

    // Batch Delete start
    ifstream fdel("file.txt");
    fdel >> num_vertices >> total_edges >> directed >> weighted;

    double total_Time = 0;
    clock_t calcTime;

    while (total_edges > 0)
    {
        int size, num_edges_cur_batch;

        // Defining edges in current batch size
        if (BATCH_SIZE_FOR_DEL > total_edges)
        {
            num_edges_cur_batch = total_edges;
            total_edges = 0;
        }
        else
        {
            num_edges_cur_batch = BATCH_SIZE_FOR_DEL;
            total_edges -= BATCH_SIZE_FOR_DEL;
        }

        size = 2 * num_edges_cur_batch;
        vector<vector<int>> edges(size, vector<int>(2, 0));

        // Reading from file
        for (int i = 0; i < num_edges_cur_batch; i++)
        {
            int u, v;
            fdel >> u >> v;
            edges[i][0] = u;
            edges[i][1] = v;
            if (!directed)
            {
                edges[num_edges_cur_batch + i][0] = v;
                edges[num_edges_cur_batch + i][1] = u;
            }
        }

        sort(edges.begin(), edges.end(), [](const vector<int> &a, const vector<int> &b)
             { return a[0] < b[0]; });

        int *edgeList, *offsetArr, *vertexArr;
        edgeList = (int *)malloc(sizeof(int) * size);
        map<int, int> hmap;

        int i = 0, maximum = INT_MIN, mapSize = 0;
        for (const auto &edge : edges)
        {
            int u, v;
            u = edge[0];
            v = edge[1];

            hmap[u]++;

            maximum = max(maximum, u);

            edgeList[i] = v;
            i++;
        }

        mapSize = hmap.size();

        vertexArr = (int *)malloc(sizeof(int) * mapSize);
        offsetArr = (int *)malloc(sizeof(int) * (mapSize + 1));
        offsetArr[0] = 0;

        map<int, int>::iterator it = hmap.begin();

        i = 0;
        while (it != hmap.end())
        {
            vertexArr[i] = it->first;
            // Frequency fill leaving index 0 for 0
            offsetArr[i + 1] = it->second;
            ++i;
            ++it;
        }

        for (int i = 1; i < mapSize + 1; i++)
        {
            offsetArr[i] += offsetArr[i - 1];
        }

        struct CSR csr_del = {vertexArr, offsetArr, edgeList, mapSize};

        int *dev_offset_arr, *dev_edge_list, *dev_vertex_arr;

        calcTime = clock();
        cudaMalloc(&dev_offset_arr, sizeof(int) * (mapSize + 1));
        cudaMalloc(&dev_edge_list, sizeof(int) * size);
        cudaMalloc(&dev_vertex_arr, sizeof(int) * mapSize);
        cudaMemcpy(dev_offset_arr, csr_del.offsetArr, sizeof(int) * (mapSize + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edge_list, csr_del.edgeList, sizeof(int) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vertex_arr, csr_del.vertexArr, sizeof(int) * mapSize, cudaMemcpyHostToDevice);

        unsigned nBlocks_for_vertices = ceil((float)mapSize / BLOCK_SIZE);
        del_edge<<<nBlocks_for_vertices, BLOCK_SIZE>>>(graph, dev_vertex_arr, dev_offset_arr, dev_edge_list, mapSize);
        cudaDeviceSynchronize();
        calcTime = clock() - calcTime;

        total_Time += ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    }

    // cout << endl;

    // printD_LL<<<1, 1>>>(graph);
    // cudaDeviceSynchronize();

    cout << "Total time taken: " << total_Time << endl;

    return 0;
}
