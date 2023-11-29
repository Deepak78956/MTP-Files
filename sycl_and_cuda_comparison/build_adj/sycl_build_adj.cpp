#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <fstream>

#include "make_csr.hpp"

#define DEBUG false
#define B_SIZE 1024
#define directed 0
#define weighted 0
#define USE_RANGE 1
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

class UsingRangeFunctions
{
public:
    void initGraph(sycl::queue &Q, struct Graph *graph, int num_vertices, struct Node **adjLists)
    {
        Q.submit([&](sycl::handler &cgh)
        {
            unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
            cgh.parallel_for(sycl::range<1>{nBlocks_for_vertices * B_SIZE}, [=](sycl::id<1> globalID){
                unsigned id = globalID[0];

                if (id < num_vertices) {
                    graph->adjLists[id] = NULL;
                    
                    if(DEBUG) printf("In init graph, %d\n", graph->adjLists[id]);
                }
            }); 
        }).wait();
    }

    void initEdgeList(sycl::queue &Q, struct Node *edgeList, int *dev_col_ind, int size)
    {
        Q.submit([&](sycl::handler &cgh)
                 {
            unsigned nBlocks_for_edges = ceil((float)size / B_SIZE);
            cgh.parallel_for(sycl::range<1>{nBlocks_for_edges * B_SIZE}, [=](sycl::id<1> globalID){
                unsigned id = globalID[0];

                if (id < size) {
                    edgeList[id].data = dev_col_ind[id];

                    if (DEBUG) printf("In init edge, id: %d, got: %d, actual: %d\n", id, edgeList[id].data, dev_col_ind[id]);
                }
            }); }).wait();
    }

    void make_DLL(sycl::queue &Q, struct Node *edgeList, int *dev_row_ptr, struct Graph *graph, int num_vertices)
    {
        Q.submit([&](sycl::handler &cgh){
            unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
            cgh.parallel_for(sycl::range<1>{nBlocks_for_vertices * B_SIZE}, [=](sycl::id<1> globalID){
                unsigned id = globalID[0];
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
            });
        }).wait();

        if (DEBUG) {
            Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
                for (int i = 0; i < graph->numVertices; i++) {
                    int start = dev_row_ptr[i];
                    int end = dev_row_ptr[i + 1];
                    printf("Inside make dll, For vertex %d where start = %d, end = %d\n", i, start, end);
                    for (int v = start; v < end; v++)
                    {
                        edgeList[v].next = graph->adjLists[i];
                        graph->adjLists[i] = &edgeList[v];
                        printf("%d ", graph->adjLists[i]->data);
                    }
                    printf("\n");
                }
            }).wait();
            printf("out of make dll func\n");
        }
        
    }
};

class UsingND_RangeFunctions
{
    public:
    void initGraph(sycl::queue &Q, struct Graph *graph, int num_vertices, struct Node **adjLists)
    {
        Q.submit([&](sycl::handler &cgh){
            unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
            unsigned globalSize = nBlocks_for_vertices * B_SIZE;
            constexpr unsigned localSize = B_SIZE;
            sycl::nd_range<1> executionRange{sycl::range<1>{globalSize}, sycl::range<1>{localSize}};

            cgh.parallel_for(executionRange, [=](sycl::nd_item<1> item){
                unsigned id = item.get_global_id(0);

                if (id < num_vertices) {
                    graph->adjLists[id] = NULL;
                }
            }); 
        }).wait();
    }

    void initEdgeList(sycl::queue &Q, struct Node *edgeList, int *dev_col_ind, int size)
    {
        Q.submit([&](sycl::handler &cgh){
            unsigned nBlocks_for_edges = ceil((float)size / B_SIZE);
            unsigned globalSize = nBlocks_for_edges * B_SIZE;
            constexpr unsigned localSize = B_SIZE;
            sycl::nd_range<1> executionRange{sycl::range<1>{globalSize}, sycl::range<1>{localSize}};

            cgh.parallel_for(executionRange, [=](sycl::nd_item<1> item){
                unsigned id = item.get_global_id(0);

                if (id < size) {
                    edgeList[id].data = dev_col_ind[id];
                }
            }); 
        }).wait();
    }

    void make_DLL(sycl::queue &Q, struct Node *edgeList, int *dev_row_ptr, struct Graph *graph, int num_vertices)
    {
        Q.submit([&](sycl::handler &cgh){
            unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
            unsigned globalSize = nBlocks_for_vertices * B_SIZE;
            constexpr unsigned localSize = B_SIZE;
            sycl::nd_range<1> executionRange{sycl::range<1>{globalSize}, sycl::range<1>{localSize}};

            cgh.parallel_for(executionRange, [=](sycl::nd_item<1> item){
                unsigned id = item.get_global_id(0);

                if (id < num_vertices) {
                    int start = dev_row_ptr[id];
                    int end = dev_row_ptr[id + 1];

                    for (int v = start; v < end; v++)
                    {
                        edgeList[v].next = graph->adjLists[id];
                        graph->adjLists[id] = &edgeList[v];
                    }
                }
            }); 
        }).wait();
    }
};

void printDLL(sycl::queue &Q, struct Graph *graph){
    Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
        for (int u = 0; u < graph->numVertices; u++)
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
    }).wait();
}

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    sycl::queue Q{sycl::gpu_selector{}};

    sycl::device dev = Q.get_device();
    cout << "Device Name: " << dev.get_info<sycl::info::device::name>() << endl;
    cout << "Device Vendor: " << dev.get_info<sycl::info::device::vendor>() << endl;

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

    int num_vertices, num_edges, x;
    istringstream header(line);
    header >> num_vertices >> x >> num_edges;
    // cout << num_vertices << " " << x << " " << num_edges << endl;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    int size = 2 * num_edges;

    int *dev_row_ptr, *dev_col_ind;
    dev_row_ptr = sycl::malloc_device<int>(num_vertices + 1, Q);
    dev_col_ind = sycl::malloc_device<int>(size, Q);

    Q.memcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1)).wait();
    Q.memcpy(dev_col_ind, csr.edgeList, sizeof(int) * size).wait();

    if (DEBUG) {
        for (int i = 0; i < num_vertices + 1; i++)
        {
            printf("%d ", csr.offsetArr[i]);
        }
        printf("\n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", csr.edgeList[i]);
        }
        printf("\n");
    }

    // struct NonWeightCSR *dev_csr;
    // dev_csr = sycl::malloc_device<struct NonWeightCSR>(1, Q);

    struct Graph *graph;
    struct Node **adjLists;
    graph = sycl::malloc_device<struct Graph>(1, Q);
    adjLists = sycl::malloc_device<struct Node *>(num_vertices, Q);

    Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx)
    {
        graph->numVertices = num_vertices;
        graph->adjLists = adjLists; 
    }).wait();

    if (USE_RANGE) {
        clock_t initClock, initEdgeClock, makeD_LLClock;

        UsingRangeFunctions rangeQueries;
        initClock = clock();
        rangeQueries.initGraph(Q, graph, num_vertices, adjLists);
        initClock = clock() - initClock;

        struct Node *edgeList;
        edgeList = sycl::malloc_device<struct Node>(size, Q);

        initEdgeClock = clock();
        rangeQueries.initEdgeList(Q, edgeList, dev_col_ind, size);
        initEdgeClock = clock() - initEdgeClock;
        
        makeD_LLClock = clock();
        rangeQueries.make_DLL(Q, edgeList, dev_row_ptr, graph, num_vertices);
        makeD_LLClock = clock() - makeD_LLClock;

        cout << fileName << endl;
        cout << "Total time taken: " << ((double)(makeD_LLClock + initEdgeClock + initClock)) / CLOCKS_PER_SEC * 1000 << endl;
        cout << endl;
    }
    else {
        clock_t initClock, initEdgeClock, makeD_LLClock;

        UsingND_RangeFunctions ndQueries;
        initClock = clock();
        ndQueries.initGraph(Q, graph, num_vertices, adjLists);
        initClock = clock() - initClock;

        struct Node *edgeList;
        edgeList = sycl::malloc_device<struct Node>(size, Q);

        initEdgeClock = clock();
        ndQueries.initEdgeList(Q, edgeList, dev_col_ind, size);
        initEdgeClock = clock() - initEdgeClock;

        makeD_LLClock = clock();
        ndQueries.make_DLL(Q, edgeList, dev_row_ptr, graph, num_vertices);
        makeD_LLClock = clock() - makeD_LLClock;

        cout << fileName << endl;
        cout << "Total time taken: " << ((double)(makeD_LLClock + initEdgeClock + initClock)) / CLOCKS_PER_SEC * 1000 << endl;
        cout << endl;
    }

    // printDLL(Q, graph);
    
    return 0;
}