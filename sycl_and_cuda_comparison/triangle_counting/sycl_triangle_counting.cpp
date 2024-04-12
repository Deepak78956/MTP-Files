#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include "make_csr.hpp"
using namespace std;

#define B_SIZE 1024
#define DEBUG false
#define directed 0
#define weighted 0
#define USE_RANGE 1

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

bool isNeigh(Graph* graph, int t, int r) {
    Node* temp = graph->adjLists[r];
    while (temp) {
        if (temp->data == t)
            return true;
        temp = temp->next;
    }
    return false;
}

void countTriangles(sycl::queue &Q, struct Graph *graph, int num_vertices) {
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));
    
    float *tc;
    tc = sycl::malloc_device<float>(1, Q);
    Q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned p = item.get_global_id(0);
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
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_tc(*tc);
                        atomic_tc += 1.0;
                        // sycl::atomic<float, sycl::access::address_space::global_space>(sycl::global_ptr<float>(tc)).fetch_add(1);
                        
                    }
                    r = r->next;
                }
                t = t->next;
            }
        }
    }).wait();
    
    // Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx){
    //     printf("Triangles got %d\n", tc[0] / 6);
    // }).wait();
}

void countTriangles_usingRange(sycl::queue &Q, struct Graph *graph, int num_vertices) {
    auto range = sycl::range<1>(num_vertices);
    float *tc;
    tc = sycl::malloc_device<float>(1, Q);

    Q.parallel_for(range, [=](sycl::id<1> idx){
        unsigned p = idx[0];
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
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_tc(*tc);
                        atomic_tc += 1.0;
                        // sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(tc)).fetch_add(1);
                    }
                    r = r->next;
                }
                t = t->next;
            }
        }
    }).wait();
    
    // Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx){
    //     printf("Triangles got %d\n", tc[0] / 6);
    // }).wait();
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

    sycl::queue Q{sycl::gpu_selector{}};

    // sycl::device dev = Q.get_device();
    // cout << "Device Name: " << dev.get_info<sycl::info::device::name>() << endl;
    // cout << "Device Vendor: " << dev.get_info<sycl::info::device::vendor>() << endl;

    int *dev_row_ptr, *dev_col_ind;
    dev_row_ptr = sycl::malloc_device<int>(num_vertices + 1, Q);
    dev_col_ind = sycl::malloc_device<int>(size, Q);

    Q.memcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1)).wait();
    Q.memcpy(dev_col_ind, csr.edgeList, sizeof(int) * size).wait();

    struct Graph *graph;
    struct Node **adjLists;
    graph = sycl::malloc_device<struct Graph>(1, Q);
    adjLists = sycl::malloc_device<struct Node *>(num_vertices, Q);

    Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx)
    {
        graph->numVertices = num_vertices;
        graph->adjLists = adjLists; 
    }).wait();

    UsingRangeFunctions rangeQueries;
    rangeQueries.initGraph(Q, graph, num_vertices, adjLists);

    struct Node *edgeList;
    edgeList = sycl::malloc_device<struct Node>(size, Q);

    rangeQueries.initEdgeList(Q, edgeList, dev_col_ind, size);
    
    rangeQueries.make_DLL(Q, edgeList, dev_row_ptr, graph, num_vertices);

    clock_t calcTime;

    calcTime = clock();
    if (USE_RANGE == 0) countTriangles(Q, graph, num_vertices);
    else countTriangles_usingRange(Q, graph, num_vertices);
    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    return 0;
}