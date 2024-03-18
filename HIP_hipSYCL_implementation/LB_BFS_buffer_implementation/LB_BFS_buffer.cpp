#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
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

struct NonWeightCSR convertToCSR(std::string fileName, bool keywordFound) {
    std::ifstream fin(fileName);
    std::string line;
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

    std::istringstream header(line);
    int num_vertices, num_edges, x;
    header >> num_vertices >> x >> num_edges;

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin, keywordFound);

    return csr;
}

void init_dist(sycl::buffer<int, 1>& dist_buf, int num_vertices, sycl::queue& q, int s) {
    q.submit([&](sycl::handler& cgh) {
        auto dist_acc = dist_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(num_vertices), [=](sycl::id<1> idx) {
            int id = idx.get(0);
            if (id == s) {
                dist_acc[id] = 0;
            } else {
                dist_acc[id] = inf;
            }
        });
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

    if (DEBUG) printf("Inside atom range, worktoeachth = %ld, id = %d, range = %ld %ld\n", workToEachThread, t_id, range.start, range.end);

    return range;
}

int binarySearch(long int searchItem, long int num_vertices, sycl::accessor<int, 1, sycl::access::mode::read> rowOffset_acc) {
    long int start = 0, end = num_vertices - 1, index = end, mid;
    while (start <= end) {
        mid = (start + end) / 2;
        if (rowOffset_acc[mid] > searchItem) {
            end = mid - 1;
        } 
        else {
            index = mid;
            start = mid + 1;
        }
    }

    return index;
}

int updateTileIfReq(int i, int prevTile, int num_vertices, sycl::accessor<int, 1, sycl::access::mode::read> src_acc) {
    if (i >= src_acc[prevTile + 1]) {
        prevTile = binarySearch(i, num_vertices, src_acc);
    }

    return prevTile;
}

void BFS(unsigned id, sycl::buffer<int, 1>& dist_buf, sycl::accessor<int, 1, sycl::access::mode::read> src_acc, sycl::accessor<int, 1, sycl::access::mode::read> dest_acc, int num_vertices, int num_edges, sycl::buffer<int, 1>& changed_buf, int TTL) {
    sycl::accessor<int, 1, sycl::access::mode::write> dist_acc(dist_buf);

    if (id < TTL) {
        struct atomRange range = getAtomRange(id, num_edges, num_vertices, TTL);
        long int u = binarySearch(range.start, num_vertices, src_acc); // get tile

        if (DEBUG) printf("Inside BFS, t_id: %d, index = %ld, range = %ld %ld\n", id, u, range.start, range.end);

        for (int i = range.start; i < range.end; i++) {
            int v = dest_acc[i];

            // Check if assigned atom goes out of row offset range, if so.. then update the tile
            u = updateTileIfReq(i, u, num_vertices, src_acc);

            if(dist_acc[v] > dist_acc[u] + 1){
                sycl::atomic<int, sycl::access::address_space::global_space>(sycl::global_ptr<int>(&dist_acc[v])).fetch_min(dist_acc[u] + 1);
                changed_buf[0] = 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    std::string fileName = argv[1];
    // string fileName = "file.txt";

    std::vector<std::string> keywords = {"kron", "file"};

    bool keywordFound = false;

    for (const std::string& keyword : keywords) {
        // Check if the keyword is present in the filename
        if (fileName.find(keyword) != std::string::npos) {
            // Set the flag to true indicating the keyword is found//
            keywordFound = true;
            break;
        }
    }
    
    struct NonWeightCSR csr = convertToCSR(fileName, keywordFound);
    int size = csr.edges;

    sycl::queue q{sycl::gpu_selector{}};

    sycl::buffer<int, 1> dev_row_ptr_buf(csr.vertices + 1);
    sycl::buffer<int, 1> dev_col_ind_buf(size);
    
    q.submit([&](sycl::handler& cgh) {
        auto dev_row_ptr_acc = dev_row_ptr_buf.get_access<sycl::access::mode::write>(cgh);
        auto dev_col_ind_acc = dev_col_ind_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(csr.vertices + 1), [=](sycl::id<1> idx) {
            dev_row_ptr_acc[idx] = csr.row_ptr[idx];
        });
        cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            dev_col_ind_acc[idx] = csr.col_ind[idx];
        });
    }).wait();

    sycl::buffer<int, 1> dist_buf(csr.vertices);
    int source = csr.vertices / 2;
    init_dist(dist_buf, csr.vertices, q, source);

    int *changed;
    changed = sycl::malloc_shared<int>(1, q);

    int TTL = std::min(csr.vertices, csr.edges);

    clock_t calcTime;
    calcTime = clock();

    while(true) {
        changed[0] = 0;

        unsigned nBlocks_for_vertices = ceil((float)csr.vertices / B_SIZE);

        auto range = sycl::nd_range<1>(sycl::range<1>(nBlocks_for_vertices * B_SIZE), sycl::range<1>(B_SIZE));

        q.submit([&](sycl::handler& cgh) {
            auto changed_acc = changed_buf.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                unsigned id = item.get_global_id(0);
                BFS(id, dist_buf, dev_row_ptr_buf.get_access<sycl::access::mode::read>(cgh), dev_col_ind_buf.get_access<sycl::access::mode::read>(cgh), csr.vertices, csr.edges, changed_buf, TTL);
            });
        }).wait();

        if (changed[0] == 0) break;
    }

    calcTime = clock() - calcTime;

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;

    std::cout << "Time taken = " << t_time << std::endl;
    std::cout << std::endl;

    return 0;
}
