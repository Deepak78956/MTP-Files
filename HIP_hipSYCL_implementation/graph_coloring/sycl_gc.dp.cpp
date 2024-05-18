#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <sstream>
#include <algorithm>
#include <numeric>
#include "make_csr.hpp"
#include <cmath>

#include <time.h>

#define DEBUG false
#define B_SIZE 1024
#define directed 0

void check(int *offsets, int *values, const sycl::stream &stream_ct1) {
    // for (int i = offsets[23]; i < offsets[24]; i++) {
    //     printf("%d ", values[i]);
    // }
    // printf("\n");

    for (int i = 0; i < 39; i++) {
        /*
        DPCT1015:0: Output needs adjustment.
        */
        stream_ct1 << "For vertex %d:\n";
        for (int j = offsets[i]; j < offsets[i + 1]; j++) {
            /*
            DPCT1015:1: Output needs adjustment.
            */
            stream_ct1 << "%d ";
        }
        stream_ct1 << "\n";
    }
}

void graph_coloring_kernel(int n, int c, int *offsets, int *values, int *randoms, int *colors,
                           const sycl::nd_item<3> &item_ct1){
    int id = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    for (int i = id; i < n; i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        int f = 1; // true iff you have max random

        if ((colors[i] != -1)) continue; // ignore nodes colored earlier

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = offsets[i]; k < offsets[i + 1]; k++) {
            int j = values[k];
            int jc = colors[j];

            // ignore nodes colored earlier (and yourself)
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            
            
            int jr = randoms[j];
            if (ir < jr) f = 0;
        }

        // assign color if you have the maximum random number
        if (f) colors[i] = c;
        // printf("id = %d\n", id);
    }
}

void countm1(int n, int *left, int *colors, const sycl::nd_item<3> &item_ct1) {
    int id = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (id < n) {
        if (colors[id] == -1){
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_tc(*left);
            atomic_tc += 1;
        }
    }
}

int *graph_coloring(int n, int *offsets, int *values, sycl::queue &q_ct1) {

    int *randoms; // have to allocate and init randoms
    int *colors;
    colors = sycl::malloc_device<int>(n, q_ct1);
    // thrust::fill(colors, colors + n, -1);
    q_ct1.memset(colors, -1, sizeof(int) * n).wait();
    randoms = (int *)malloc(sizeof(int) * n);

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<int> dis(0, n);

    for (int i = 0; i < n; i++) {
        // int randNum = dis(gen);
        int randNum = i;
        randoms[i] = randNum;
    }

    int *dev_randoms;
    dev_randoms = sycl::malloc_device<int>(n, q_ct1);
    q_ct1.memcpy(dev_randoms, randoms, sizeof(int) * n).wait();

    clock_t t_time = 0, temp_time;
    long int iterations = 0;

    int *left_dev, *left_host;
    left_dev = sycl::malloc_device<int>(1, q_ct1);
    q_ct1.memset(left_dev, 0, sizeof(int)).wait();
    left_host = (int *)malloc(sizeof(int));
    left_host[0] = 0; 

    for (int c = 0; c < n; c++) {
        int nt = B_SIZE;
        int nb =  ceil((float)n / nt);
        iterations += 1;

        temp_time = clock();
        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                 sycl::range<3>(1, 1, nt),
                                             sycl::range<3>(1, 1, nt)),
                           [=](sycl::nd_item<3> item_ct1) {
                               graph_coloring_kernel(n, c, offsets, values,
                                                     dev_randoms, colors,
                                                     item_ct1);
                           }).wait_and_throw();
        temp_time = clock() - temp_time;

        t_time += temp_time;

        q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                 sycl::range<3>(1, 1, nt),
                                             sycl::range<3>(1, 1, nt)),
                           [=](sycl::nd_item<3> item_ct1) {
                               countm1(n, left_dev, colors, item_ct1);
                           }).wait_and_throw();

        // std::cout << iterations << std::endl;
        q_ct1.memcpy(left_host, left_dev, sizeof(int)).wait();
        // std::cout << left_host[0] << std::endl;

        if (left_host[0] == 0) break;
    }

    double final_time = ((double)t_time) / CLOCKS_PER_SEC * 1000;

    std::cout << "Iterations: " << iterations << std::endl; 
    std::cout << "Time taken: " << final_time << std::endl;

    return colors;
}

void check_ans(int *colorsArr, int *offsets, int *values, int n) {
    int breakLoop = 0;
    for (int i = 0; i < n; i++) {
        int color_u = colorsArr[i];
        // std::cout << "vertex is " << i << std::endl;
        for (int j = offsets[i]; j < offsets[i + 1]; j++) {
            int color_v = colorsArr[values[j]];
            // std::cout << values[j] << std::endl;
            if (color_u == color_v) {
                printf("Wrong ans on vertex %d, same color %d with vertex %d\n", i, color_v, values[j]);
                breakLoop = 1;
                break;
            }
        }
        if (breakLoop) break;
        // std::cout << std::endl;
    }

    if (!breakLoop) std::cout << "Correct ans" << std::endl;
}

int main(int argc, char *argv[]) {
    sycl::queue q_ct1{sycl::gpu_selector{}};

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

    int size;
    if (directed) size = num_edges;
    else size = 2 * num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin, keywordFound);

    int *dev_row_ptr, *dev_col_ind;
    dev_row_ptr = sycl::malloc_device<int>((num_vertices + 1), q_ct1);
    dev_col_ind = sycl::malloc_device<int>(size, q_ct1);
    q_ct1.memcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1))
        .wait();
    q_ct1.memcpy(dev_col_ind, csr.edgeList, sizeof(int) * size).wait();

    // for (int i = 34; i < 36; i++) {
    //     printf("%d ", csr.offsetArr[i]);
    // }

    // cout << endl;

    // for (int i = csr.offsetArr[23]; i < csr.offsetArr[24]; i++) {
    //     printf("%d ", csr.edgeList[i]);
    // }
    // cout << endl;

    // check<<<1,1>>>(dev_row_ptr, dev_col_ind);
    // cudaDeviceSynchronize();

    std::cout << "On graph " << fileName << std::endl;

    int *dev_colors;
    dev_colors = graph_coloring(num_vertices, dev_row_ptr, dev_col_ind, q_ct1);

    int *colors;
    colors = (int *)malloc(sizeof(int) * num_vertices);

    q_ct1.memcpy(colors, dev_colors, sizeof(int) * num_vertices).wait();

    check_ans(colors, csr.offsetArr, csr.edgeList, num_vertices);
    std::cout << std::endl;

    sycl::free(dev_colors, q_ct1);
    sycl::free(dev_col_ind, q_ct1);
    sycl::free(dev_row_ptr, q_ct1);

    return 0;
}