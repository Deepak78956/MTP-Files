#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include <sycl/sycl.hpp>
#include "make_csr.hpp"
#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 0
#define inf 10000000

struct atomRange {
    long int start, end;
};

struct NonWeightCSR convertToCSR(string fileName) {
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

    int size;
    if (directed)
        size = num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    return csr;
}

void init_dist(int *dist, int num_vertices, sycl::queue &q, int s){
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

    q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned id = item.get_global_id(0);
        if (id < num_vertices) {
            if (id == s) {
                printf("source is %d\n", s);
                dist[id] = 0;
            }
            else {
                dist[id] = inf;
            }
        }
    }).wait();
}

struct atomRange getAtomRange(unsigned t_id, long int totalWork, long int totalThreads, int ttl) {
    long int workToEachThread;
    
    workToEachThread = totalWork / ttl;

    struct atomRange range;
    range.start = t_id * workToEachThread;
    if (t_id == ttl - 1) {
        range.end = totalWork;
    }
    else {
        range.end = range.start + workToEachThread;
    }

    if (DEBUG) printf("Inside atom range, worktoeachth = %d, id = %d, range = %d %d\n", workToEachThread, t_id, range.start, range.end);

    return range;
}

int binarySearch(long int searchItem, long int num_vertices, int *rowOffset) {
    long int start = 0, end = num_vertices - 1, index = end, mid;
    while (start <= end) {
        mid = (start + end) / 2;
        if (rowOffset[mid] > searchItem) {
            end = mid - 1;
        } 
        else {
            index = mid;
            start = mid + 1;
        }
    }

    return index;
}

int updateTileIfReq(int i, int prevTile, int num_vertices, int *src) {
    if (i >= src[prevTile + 1]) {
        prevTile = binarySearch(i, num_vertices, src);
    }

    return prevTile;
}

void BFS(unsigned id, int *dist, int *src, int *dest, int num_vertices, int num_edges, int *changed, int TTL) {
    if (id < TTL) {
        struct atomRange range = getAtomRange(id, num_edges, num_vertices, TTL);
        long int u = binarySearch(range.start, num_vertices, src); // get tile

        if (DEBUG) printf("Inside BFS, t_id: %d, index = %d, range = %d %d\n", id, u, range.start, range.end);

        for (int i = range.start; i < range.end; i++) {
            int v = dest[i];

            // Check if assigned atom goes out of row offset range, if so.. then update the tile
            u = updateTileIfReq(i, u, num_vertices, src);

            if(dist[v] > dist[u] + 1){
                sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(&dist[v])).fetch_min(dist[u] + 1);
                changed[0] = 1;
            }
        }
    }
}

int main() {
    // if (argc != 2)
    // {
    //     printf("Usage: %s <input_file>\n", argv[0]);
    //     return 1;
    // }

    // string fileName = argv[1];
    string fileName = "file.txt";
    
    struct NonWeightCSR csr = convertToCSR(fileName);
    int size = csr.edges;

    sycl::queue q{sycl::gpu_selector{}};

    int *dev_row_ptr, *dev_col_ind;
    dev_row_ptr = sycl::malloc_device<int>(sizeof(int) * (csr.vertices + 1), q);
    dev_col_ind = sycl::malloc_device<int>(sizeof(int) * size, q);
    
    q.memcpy(dev_row_ptr, csr.row_ptr, sizeof(int) * (csr.vertices + 1)).wait();
    q.memcpy(dev_col_ind, csr.col_ind, sizeof(int) * size).wait();

    int *dist;
    dist = sycl::malloc_device<int>(sizeof(int) * (csr.vertices), q);

    int source = csr.vertices / 2;
    init_dist(dist, csr.vertices, q, source);

    int *changed;
    changed = sycl::malloc_shared<int>(1, q);

    int TTL = min(csr.vertices, csr.edges);

    while(true) {
        changed[0] = 0;

        unsigned nBlocks_for_vertices = ceil((float)csr.vertices / B_SIZE);

        auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

        q.parallel_for(range, [=](sycl::nd_item<1> item){
            unsigned id = item.get_global_id(0);
            BFS(id, dist, dev_row_ptr, dev_col_ind, csr.vertices, csr.edges, changed, TTL);
        }).wait();

        if (changed[0] == 0) break;
    }

    // To print distances
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
        for (int i = 0; i < csr.vertices; i++) {
            printf("node i = %d, dist = %d\n", i, dist[i]);
        }
    }).wait();
    
    return 0;
}