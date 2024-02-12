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
#define weighted 1
#define inf 10000000

struct WeightCSR convertToCSR(string fileName, bool keywordFound) {
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

    struct WeightCSR csr = CSRWeighted(num_vertices, num_edges, directed, fin, keywordFound);

    return csr;
}

void init_dist(sycl::queue &Q, int *dist, int num_vertices, int source) {
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

    Q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned id =  item.get_global_id(0);
        if (id < num_vertices) {
            if (id == source) {
                dist[id] = 0;
            }
            else {
                dist[id] = inf;
            }
        }
    }).wait();
}

void sssp(sycl::queue &Q, int *dist, int *src, int *dest, int *weights, int num_edges, int *changed){
    unsigned nBlocks_for_edges = ceil((float)num_edges / B_SIZE);
    auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_edges * B_SIZE), sycl::range<1>(B_SIZE));

    Q.parallel_for(range, [=](sycl::nd_item<1> item){
        unsigned id =  item.get_global_id(0);
        if (id < num_edges) {
            int u = src[id];
            int v = dest[id];
            int w = weights[id];

            if(dist[v] > dist[u] + w){
                sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(&dist[v])).fetch_min(dist[u] + w);
                changed[0] = 1;
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

    sycl::queue Q{sycl::gpu_selector{}};

    int size;
    if (directed)
        size = num_edges;
    
    int *src, *dest, *weights;
    int *dev_src, *dev_dest, *dev_weights;
    src = (int *)malloc(sizeof(int) * num_edges);
    dest = (int *)malloc(sizeof(int) * num_edges);
    weights = (int *)malloc(sizeof(int) * num_edges);
    dev_src = sycl::malloc_device<int>(num_edges, Q);
    dev_dest = sycl::malloc_device<int>(num_edges, Q);
    dev_weights = sycl::malloc_device<int>(num_edges, Q);

    vector<string> keywords = {"kron", "file"};

    bool keywordFound = false;

    for (const string& keyword : keywords) {
        // Check if the keyword is present in the filename
        if (fileName.find(keyword) != string::npos) {
            // Set the flag to true indicating the keyword is found
            keywordFound = true;
            break;
        }
    }

    if (keywordFound == true) {
        // Code to execute if "kron" is found
        for (int i = 0; i < num_edges; i++) {
            int u, v, w;
            fin >> u >> v >> w;
        
            src[i] = u - 1;
            dest[i] = v - 1;
            weights[i] = w;
        }
    } else {
        for (int i = 0; i < num_edges; i++) {
            int u, v;
            fin >> u >> v;
        
            src[i] = u - 1;
            dest[i] = v - 1;
            weights[i] = 1;
        }
    }

    Q.memcpy(dev_src, src, sizeof(int) * num_edges).wait();
    Q.memcpy(dev_dest, dest, sizeof(int) * num_edges).wait();
    Q.memcpy(dev_weights, weights, sizeof(int) * num_edges).wait();

    int *dist;
    dist = sycl::malloc_device<int>(num_vertices, Q);
    
    int source = num_vertices / 2;
    init_dist(Q, dist, num_vertices, source);

    int *changed;
    changed = sycl::malloc_shared<int>(1, Q);

    clock_t calcTime;
    calcTime = clock();
    while (true) {
        changed[0] = 0;
        sssp(Q, dist, dev_src, dev_dest, dev_weights, num_edges, changed);
        if (changed[0] == 0) break;
    }
    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    // To print the distances
    // Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
    //     for (int i = 0; i < num_vertices; i++) {
    //         printf("node i = %d, dist = %d\n", i, dist[i]);
    //     }
    // }).wait();

    return 0;
}