#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include "make_csr.hpp"
using namespace std;

#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 1
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

void sssp(sycl::queue &Q, int *dist, int *src, int *dest, int *weights, int num_vertices, int *changed){
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

    Q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned id =  item.get_global_id(0);
        if (id < num_vertices) {
            int u = id;
            
            for (int i = src[u]; i < src[u + 1]; i++) {
                int v = dest[i];
                int w = weights[i];

                // printf("here, dist[v] = %d, dist[u] + w = %d\n", dist[v], dist[u] + w);

                if(dist[v] > dist[u] + w){
                    // printf("th = %d,    Before %d %d\n", u, dist[v], dist[u] + w);
                    sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(&dist[v])).fetch_min(dist[u] + w);
                    // printf("th = %d,    After %d %d\n", u, dist[v], dist[u] + w);
                    changed[0] = 1;
                }
            }
        }
    }).wait();
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

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

    struct WeightCSR csr;
    csr = CSRWeighted(num_vertices, num_edges, directed, fin);

    for (int i = 0; i < size; i++) {
        dest[i] = csr.col_ind[i];
        weights[i] = csr.weights[i];
    }

    for (int i = 0; i < num_vertices + 1; i++) {
        src[i] = csr.row_ptr[i];
    }

    sycl::queue Q{sycl::gpu_selector{}};
    dev_src = sycl::malloc_device<int>(num_vertices + 1, Q);
    dev_dest = sycl::malloc_device<int>(num_edges, Q);
    dev_weights = sycl::malloc_device<int>(num_edges, Q);

    Q.memcpy(dev_src, src, sizeof(int) * (num_vertices + 1)).wait();
    Q.memcpy(dev_dest, dest, sizeof(int) * num_edges).wait();
    Q.memcpy(dev_weights, weights, sizeof(int) * num_edges);

    int *dist;
    dist = sycl::malloc_device<int>(num_vertices, Q);
    
    init_dist(Q, dist, num_vertices);

    int *changed;
    changed = sycl::malloc_shared<int>(1, Q);

    while(true) {
        changed[0] = 0;
        sssp(Q, dist, dev_src, dev_dest, dev_weights, num_vertices, changed);

        if (changed[0] == 0) break;
    }

    Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
        for (int i = 0; i < num_vertices; i++) {
            printf("node i = %d, dist = %d\n", i, dist[i]);
        }
    }).wait();
    return 0;
}