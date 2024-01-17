#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include "make_csr.hpp"
using namespace std;

#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 0
#define inf 1000000

void init_dist(sycl::queue &Q, int *dist, int num_vertices) {
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

    Q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned id =  item.get_global_id(0);
        if (id < num_vertices) {
            if (id == 0) {
                dist[id] = 0;
            }
            else {
                dist[id] = inf;
            }
        }
    }).wait();
}

void BFS(sycl::queue &Q, int *dist, int *src, int *dest, int num_vertices, int *changed){
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

    Q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned id =  item.get_global_id(0);
        if (id < num_vertices) {
            int u = id;
            
            for (int i = src[u]; i < src[u + 1]; i++) {
                int v = dest[i];

                if(dist[v] > dist[u] + 1){
                    sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(&dist[v])).fetch_min(dist[u] + 1);
                    changed[0] = 1;
                }
            }
        }
    }).wait();
}

int main(int argc, char *argv[]) {
    // if (argc != 2)
    // {
    //     printf("Usage: %s <input_file>\n", argv[0]);
    //     return 1;
    // }

    // string fileName = argv[1];
    string fileName = "file.txt";
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
    
    int *src, *dest, *weights;
    int *dev_src, *dev_dest, *dev_weights;
    src = (int *)malloc(sizeof(int) * (num_vertices + 1));
    dest = (int *)malloc(sizeof(int) * num_edges);
    weights = (int *)malloc(sizeof(int) * num_edges);

    struct NonWeightCSR csr;
    csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    int *row_ptr, *col_index;
    row_ptr = (int *)malloc(sizeof(int) * (num_vertices + 1));
    col_index = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        row_ptr[i] = csr.offsetArr[i];
    }

    for (int i = 0; i < size; i++)
    {
        col_index[i] = csr.edgeList[i];
    }

    sycl::queue Q{sycl::gpu_selector{}};
    dev_src = sycl::malloc_device<int>(num_vertices + 1, Q);
    dev_dest = sycl::malloc_device<int>(num_edges, Q);

    Q.memcpy(dev_src, row_ptr, sizeof(int) * (num_vertices + 1)).wait();
    Q.memcpy(dev_dest, col_index, sizeof(int) * num_edges).wait();

    int *dist;
    dist = sycl::malloc_device<int>(num_vertices, Q);
    
    init_dist(Q, dist, num_vertices);

    int *changed;
    changed = sycl::malloc_shared<int>(1, Q);

    while(true) {
        changed[0] = 0;
        BFS(Q, dist, dev_src, dev_dest, num_vertices, changed);

        if (changed[0] == 0) break;
    }

    // To print distances
    Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
        for (int i = 0; i < num_vertices; i++) {
            printf("node i = %d, dist = %d\n", i, dist[i]);
        }
    }).wait();
    return 0;
}