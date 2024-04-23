#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <queue>
#include "make_csr.hpp"
#include <cmath>

#define DEBUG false
#define B_SIZE 1024
#define directed 1
#define weighted 0
#define inf 10000000

using namespace std;

struct ArgsStruct {
    int *dist;
    int *dev_row_ptr;
    int *dev_col_ind;
    int num_vertices;
    int *changed;
};

dpct::global_memory<int, 0> changed(0);

void init_dist(int *dist, int vertices, int s, const sycl::nd_item<3> &item_ct1) {
    unsigned id = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (id < vertices) {
        if (id == s) {
            dist[id] = 0;
        }
        else {
            dist[id] = inf;
        }
    }
}

void print_dist(int *dist, int num_vertices, const sycl::stream &stream_ct1) {
    for (int i = 0; i < num_vertices; i++) {
        /*
        DPCT1015:0: Output needs adjustment.
        */
        stream_ct1 << "node i = %d, dist = %d\n";
    }
}

void BFS_util(int *dist, int *src, int *destination, int num_vertices,
              const sycl::nd_item<3> &item_ct1, int &changed) {
    unsigned id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (id < num_vertices) {
        int u = id;
        
        for (int i = src[u]; i < src[u + 1]; i++) {
            int v = destination[i];
            if(dist[v] > dist[u] + 1){
                dpct::atomic_fetch_min<
                    sycl::access::address_space::generic_space>(&dist[v],
                                                                dist[u] + 1);
                changed = 1;
            }
        }
    }
}

void BFS(int *dist, int *dev_row_ptr, int *dev_col_ind, int num_vertices,
         const sycl::nd_item<3> &item_ct1, int &changed){
    // ArgsStruct *args;
    // args = (ArgsStruct *)para;
    /*
    DPCT1087:1: SYCL currently does not support cross group synchronization. You
    can specify "--use-experimental-features=nd_range_barrier" to use the dpct
    helper function nd_range_barrier to migrate this_grid().
    */
    cg::grid_group grid = cg::this_grid();

    while (true) {
        changed = 0;
        /*
        DPCT1087:2: SYCL currently does not support cross group synchronization.
        You can specify "--use-experimental-features=nd_range_barrier" to use
        the dpct helper function nd_range_barrier to migrate grid.sync().
        */
        grid.sync();

        BFS_util(dist, dev_row_ptr, dev_col_ind, num_vertices, item_ct1,
                 changed);
        /*
        DPCT1087:3: SYCL currently does not support cross group synchronization.
        You can specify "--use-experimental-features=nd_range_barrier" to use
        the dpct helper function nd_range_barrier to migrate grid.sync().
        */
        grid.sync();
        if (changed == 0) {
            break;
        }
    }
}

void setParams(int *dist, int *dev_row_ptr, int *dev_col_ind, int num_vertices, int *changed, void **para){
    ArgsStruct *args;
    args = (ArgsStruct *)para;

    (*args).dist = dist;
    (*args).dev_row_ptr = dev_row_ptr;
    (*args).dev_col_ind = dev_col_ind;
    (*args).num_vertices = num_vertices;
    (*args).changed = changed;
}

void verifyDistances(struct NonWeightCSR csr, int num_vertices, int source, int *dev_dist) {
    queue<int> q;
    int dist[num_vertices];

    for (int i = 0; i < num_vertices; i++) {
        dist[i] = inf;
    }
    dist[source] = 0;

    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int i = csr.offsetArr[u]; i < csr.offsetArr[u + 1]; i++) {
            int v = csr.edgeList[i];
            if (dist[v] == inf) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    int correct = 1;
    for (int i = 0; i < num_vertices; i++) {
        if (dist[i] != dev_dist[i]) {
            printf("Distance mismatch for node %d, expected: %d, actual: %d\n", i, dist[i], dev_dist[i]);
            correct = 0;
            break;
        }
    }

    if (correct == 1) printf("Answer correct\n");
    return;
}

int main(int argc, char *argv[]) try {
    int deviceId = 0;
    /*
    DPCT1093:6: The "deviceId" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(deviceId);

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

    int size = num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin, keywordFound);

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

    // for (int i = 0; i < 4; i++) {
    //     printf("%d ", row_ptr[i]);
    // }

    // cout << endl;

    // for (int i = 0; i < 4; i++) {
    //     printf("%d ", col_index[i]);
    // }
    // cout << endl;

    int *dev_row_ptr, *dev_col_ind;
    dev_row_ptr = sycl::malloc_device<int>((num_vertices + 1),
                                           dpct::get_in_order_queue());
    dev_col_ind = sycl::malloc_device<int>(size, dpct::get_in_order_queue());
    dpct::get_in_order_queue().memcpy(dev_row_ptr, csr.offsetArr,
                                      sizeof(int) * (num_vertices + 1));
    dpct::get_in_order_queue()
        .memcpy(dev_col_ind, csr.edgeList, sizeof(int) * size)
        .wait();

    int *dist;
    dist = sycl::malloc_device<int>(num_vertices, dpct::get_in_order_queue());

    int source = num_vertices / 2;
    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks_for_vertices) *
                              sycl::range<3>(1, 1, B_SIZE),
                          sycl::range<3>(1, 1, B_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            init_dist(dist, num_vertices, source, item_ct1);
        });
    dpct::get_current_device().queues_wait_and_throw();

    // int *changed;
    // cudaMalloc(&changed, sizeof(int));
    // cudaMemset(changed, 0, sizeof(int));

    int *launchedKernel;
    launchedKernel = sycl::malloc_shared<int>(1, dpct::get_in_order_queue());
    launchedKernel[0] = 0;

    int dev = 0;
    int supportsCoopLaunch = 0;
    dpct::err0 result = cudaDeviceGetAttribute(
        &supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

    if (result == 0) {
    if (supportsCoopLaunch) {
      printf("Cooperative launches are supported on device %d\n", dev);
    } else {
      // Cooperative launches are not supported on this device
      printf("Cooperative launches are not supported on device %d\n", dev);
    }
  } else {
    /*
    DPCT1009:7: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced by a placeholder string. You need to rewrite
    this code.
    */
    printf("cudaDeviceGetAttribute failed with error: %s\n",
           "<Placeholder string>");
  }

    sycl::range<3> blockSize(1, 1, 1024);
    sycl::range<3> gridSize(1, 1, nBlocks_for_vertices);

    ArgsStruct *para;
    para = sycl::malloc_device<ArgsStruct>(1, dpct::get_in_order_queue());

    // setParams<<<1,1>>>(dist, dev_row_ptr, dev_col_ind, num_vertices, changed, (void **)para);
    // cudaDeviceSynchronize();

    void *kernelArgs[] = {(void *)&dist, (void *)&dev_row_ptr, (void *)&dev_col_ind, (void *)&num_vertices, (void *)&launchedKernel};

    // cout << nBlocks_for_vertices << endl;
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        changed.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            auto changed_ptr_ct1 = changed.get_ptr();

            int *dist_ct0 = *(int **)kernelArgs[0];
            int *dev_row_ptr_ct1 = *(int **)kernelArgs[1];
            int *dev_col_ind_ct2 = *(int **)kernelArgs[2];
            int num_vertices_ct3 = *(int *)kernelArgs[3];

            cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                             [=](sycl::nd_item<3> item_ct1) {
                                 BFS(dist_ct0, dev_row_ptr_ct1, dev_col_ind_ct2,
                                     num_vertices_ct3, item_ct1,
                                     *changed_ptr_ct1);
                             });
        });
    }
    auto error =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());

    int *dist_copy;
    dist_copy = (int *)malloc(sizeof(int) * num_vertices);

    dpct::get_in_order_queue()
        .memcpy(dist_copy, dist, sizeof(int) * num_vertices)
        .wait();

    verifyDistances(csr, num_vertices, source, dist_copy);
    cout << endl;

    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}