#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include "make_csr.hpp"
#include <cmath>

#include <time.h>

#define DEBUG false
#define B_SIZE 1024
#define directed 0
#define weighted 0

using namespace std;

int isNeigh(int *offset, int * values, int t, int r)
{
    for (int i = offset[r]; i < offset[r + 1]; i++) {
        int temp = values[i];
        if (temp == t) return 1;
    }
    return 0;
}


void countTriangles(int *offset, int *values, int n,
                    const sycl::nd_item<3> &item_ct1, int *tc)
{
    unsigned p = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                 item_ct1.get_local_id(2);

    if (p < n)
    {
        for (int i = offset[p]; i < offset[p+1]; i++) {
            int t = values[i];
            for (int j = offset[p]; j < offset[p+1]; j++){
                int r = values[j];
                if (t != r && isNeigh(offset, values, t, r)){
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_tc(*tc);
                        atomic_tc += 1;
                }
            }
        }
    }
}

void printTc(const sycl::stream &stream_ct1, int &tc)
{
    /*
    DPCT1015:0: Output needs adjustment.
    */
    stream_ct1 << "Triangles got %d\n";
}

int main(int argc, char *argv[])
{
    sycl::queue q_ct1{sycl::gpu_selector{}};

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
    dev_row_ptr = sycl::malloc_device<int>((num_vertices + 1), q_ct1);
    dev_col_ind = sycl::malloc_device<int>(size, q_ct1);
    q_ct1.memcpy(dev_row_ptr, csr.offsetArr, sizeof(int) * (num_vertices + 1))
        .wait();
    q_ct1.memcpy(dev_col_ind, csr.edgeList, sizeof(int) * size).wait();

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    clock_t calcTime;

    int *tc;
    tc = sycl::malloc_device<int>(1, q_ct1);
    
    calcTime = clock();
    
    q_ct1.submit([&](sycl::handler &cgh) {
        

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks_for_vertices) *
                                  sycl::range<3>(1, 1, B_SIZE),
                              sycl::range<3>(1, 1, B_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                countTriangles(dev_row_ptr, dev_col_ind, num_vertices, item_ct1,
                               tc);
            });
    }).wait_and_throw();
    /*
    DPCT1008:3: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    return 0;
}
