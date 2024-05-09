#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"
#include <cmath>

#include <time.h>

#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 1
#define inf 10000000

void ker(int *dist, const sycl::stream &stream_ct1) {
    for (int i = 0; i < 3; i++) {
        /*
        DPCT1015:0: Output needs adjustment.
        */
        stream_ct1 << "%d ";
    }
    stream_ct1 << "\n";
}

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


void init_dist(int *dist, int vertices, int source,
               const sycl::nd_item<3> &item_ct1) {
    unsigned id = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (id < vertices) {
        if (id == source) {
            dist[id] = 0;
        }
        else {
            dist[id] = inf;
        }
    }
}

void sssp(int *dist, int *src, int *dest, int *weights, int num_edges, int *changed,
          const sycl::nd_item<3> &item_ct1) {
    unsigned id = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (id < num_edges) {
        int u = src[id];
        int v = dest[id];
        int w = weights[id];

        // printf("%d %d %d %d\n", u, v, w, dist[u]);
        // int newVal = 0;
        // atomicAdd(&newVal, dist[u]);
        // atomicAdd(&newVal, w);
        
        if(dist[v] > dist[u] + w){
            sycl::atomic_fetch_min<sycl::access::address_space::generic_space>(
                &dist[v], dist[u] + w);
            // printf("%d\n", dist[v]);
            changed[0] = 1;
        }
        // if (dist[v] > newVal) {
        //     dist[v] = newVal;
        //     changed = true;
        // }
    }
}

void print_dist(int *dist, int num_vertices, const sycl::stream &stream_ct1) {
    for (int i = 0; i < num_vertices; i++) {
        /*
        DPCT1015:1: Output needs adjustment.
        */
        stream_ct1 << "node i = %d, dist = %d\n";
    }
}

int main(int argc, char *argv[])
{
    device_ext &dev_ct1 = get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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
    src = (int *)malloc(sizeof(int) * num_edges);
    dest = (int *)malloc(sizeof(int) * num_edges);
    weights = (int *)malloc(sizeof(int) * num_edges);
    dev_src = sycl::malloc_device<int>(num_edges, q_ct1);
    dev_dest = sycl::malloc_device<int>(num_edges, q_ct1);
    dev_weights = sycl::malloc_device<int>(num_edges, q_ct1);

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

    q_ct1.memcpy(dev_src, src, sizeof(int) * num_edges).wait();
    q_ct1.memcpy(dev_dest, dest, sizeof(int) * num_edges).wait();
    q_ct1.memcpy(dev_weights, weights, sizeof(int) * num_edges).wait();

    int *dist;
    dist = sycl::malloc_device<int>(num_vertices, q_ct1);

    int source = num_vertices / 2;

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks_for_vertices) *
                              sycl::range<3>(1, 1, B_SIZE),
                          sycl::range<3>(1, 1, B_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            init_dist(dist, num_vertices, source, item_ct1);
        });
    dev_ct1.queues_wait_and_throw();

    int *changed;
    changed = sycl::malloc_device<int>(1, q_ct1);
    changed = sycl::malloc_shared<int>(1, q_ct1);

    clock_t calcTime;
    /*
    DPCT1008:4: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    calcTime = clock();
    while (true) {
        changed[0] = 0;
        unsigned nBlocks_for_edges = ceil((float)num_edges / B_SIZE);
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks_for_edges) *
                                  sycl::range<3>(1, 1, B_SIZE),
                              sycl::range<3>(1, 1, B_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                sssp(dist, dev_src, dev_dest, dev_weights, num_edges, changed,
                     item_ct1);
            });
        dev_ct1.queues_wait_and_throw();

        if (changed[0] == 0) break;
    }
    /*
    DPCT1008:5: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;
    // printf("here\n");
    // print_dist<<<1, 1>>>(dist, num_vertices);
    // cudaDeviceSynchronize();


    return 0;
}