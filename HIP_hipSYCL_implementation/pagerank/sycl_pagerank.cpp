#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
using namespace std;

#define directed 1
#define weighted 0
#define DEBUG false
#define B_SIZE 1024
#define USE_RANGE 0

struct CSR
{
    int *offsetArr;
    int *edgeList;
    int num_vertices;
    int num_edges;
};

void assignToCSR(sycl::queue &q, struct CSR *csr, int *offsetArr, int *edgeList, int num_vertices, int num_edges) {
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx){
        csr->offsetArr = offsetArr;
        csr->edgeList = edgeList;
        csr->num_vertices = num_vertices;
        csr->num_edges = num_edges;
    }).wait();
    q.wait_and_throw();
}

void checkAssignment(sycl::queue &q, struct CSR *csr) {
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
        printf("Checking correct assignment to GPU \n");
        for (int i = 0; i < csr->num_vertices + 1; i++)
        {
            printf("%d ", csr->offsetArr[i]);
        }
        printf("\n");

        for (int i = 0; i < csr->num_edges; i++)
        {
            printf("%d ", csr->edgeList[i]);
        }
        printf("\n");
    }).wait();
    q.wait_and_throw();
}

clock_t init(sycl::queue &q, float *pr, int num_vertices, unsigned blocks) {
    // sycl::range<1> totalItems{blocks * B_SIZE};
    // sycl::range<1> itemsInWG{B_SIZE};
    auto totalItems = sycl::range<1>(blocks * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);

    clock_t initTime;
    initTime = clock();

    q.parallel_for(sycl::nd_range<1>(totalItems, itemsInWG), [=](sycl::nd_item<1> item){
        unsigned id = item.get_global_id(0);

        if (DEBUG) {
            if (id == 0)
                printf("Inside init with num vertices = %d \n", num_vertices);
        }

        if (id < num_vertices) {
            pr[id] = 1.0f / num_vertices;

            if (DEBUG)
                printf("id = %d, val = %f, actual value = %f\n", id, pr[id], 1.0 / num_vertices);
        }
    }).wait();
    q.wait_and_throw();

    initTime = clock() - initTime;
    return initTime;
}

clock_t init_range(sycl::queue &q, float *pr, int num_vertices, unsigned blocks) {
    auto totalItems = sycl::range<1>(blocks * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);

    clock_t initTime;
    initTime = clock();

    q.parallel_for(sycl::range<1>(totalItems), [=](sycl::id<1> id){
        unsigned global_id = id[0];

        if (DEBUG) {
            if (global_id == 0)
                printf("Inside init with num vertices = %d \n", num_vertices);
        }

        if (global_id < num_vertices) {
            pr[global_id] = 1.0f / num_vertices;

            if (DEBUG)
                printf("id = %d, val = %f, actual value = %f\n", global_id, pr[global_id], 1.0 / num_vertices);
        }
    }).wait();
    q.wait_and_throw();

    initTime = clock() - initTime;
    return initTime;
}


clock_t computePR(sycl::queue &q, struct CSR *csr, struct CSR *in_csr, float *oldPr, float *newPr, unsigned blocks){
    auto totalItems = sycl::range<1>(blocks * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);
    float d = 0.85;

    clock_t calcTime;
    calcTime = clock();

    q.parallel_for(sycl::nd_range<1>(totalItems, itemsInWG), [=](sycl::nd_item<1> item){
        unsigned p = item.get_global_id(0);
        float val = 0.0;

        if (DEBUG && p == 0) printf("Inside PR, value of d = %f\n", d);
        if (p < csr->num_vertices)
        {
            for (int i = in_csr->offsetArr[p]; i < in_csr->offsetArr[p + 1]; i++)
            {
                unsigned t = in_csr->edgeList[i];
                unsigned out_deg_t = csr->offsetArr[t + 1] - csr->offsetArr[t];

                if (out_deg_t != 0)
                {
                    float temp = oldPr[t] / out_deg_t;
                    val += oldPr[t] / out_deg_t;
                    
                    if (DEBUG)
                        printf("%f\n", val);
                }
            }
            newPr[p] = val * d + (1 - d) / csr->num_vertices;
        }
    }).wait();
    calcTime = clock() - calcTime;
    return calcTime;
}

clock_t computePR_range(sycl::queue &q, struct CSR *csr, struct CSR *in_csr, float *oldPr, float *newPr, unsigned blocks){
    auto totalItems = sycl::range<1>(blocks * B_SIZE);
    auto itemsInWG = sycl::range<1>(B_SIZE);
    float d = 0.85;

    clock_t calcTime;
    calcTime = clock();
    q.parallel_for(sycl::range<1>(totalItems), [=](sycl::id<1> p){
        unsigned global_id = p[0];
        float val = 0.0;

        if (DEBUG && global_id == 0) printf("Inside PR, value of d = %f\n", d);

        if (global_id < csr->num_vertices) {
            for (int i = in_csr->offsetArr[global_id]; i < in_csr->offsetArr[global_id + 1]; i++) {
                unsigned t = in_csr->edgeList[i];
                unsigned out_deg_t = csr->offsetArr[t + 1] - csr->offsetArr[t];

                if (out_deg_t != 0) {
                    float temp = oldPr[t] / out_deg_t;
                    val += oldPr[t] / out_deg_t;

                    if (DEBUG) printf("%f\n", val);
                }
            }
            newPr[global_id] = val * d + (1 - d) / csr->num_vertices;
        }
    }).wait();
    calcTime = clock() - calcTime;
    return calcTime;
}


void printPR(sycl::queue &q, float *pr, int vertices) {
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
        for (int i = 0; i < vertices; i++)
        {
            printf("%lf ", pr[i]);
        }
        printf("\n");
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

    int size;
    if (directed)
        size = num_edges;
    else
    {
        cout << "Directed graph is required" << endl;
        exit(0);
    }
    if (weighted)
    {
        cout << "Non weighted graph is required" << endl;
        exit(0);
    }

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

    vector<vector<int>> edges(size, vector<int>(2, 0));
    vector<vector<int>> in_neigh(size, vector<int>(2, 0));
    for (int i = 0; i < num_edges; i++)
    {
        int u, v, w;
        if (keywordFound) {
            fin >> u >> v >> w;    
        }
        else {
            fin >> u >> v;
        }
        edges[i][0] = u - 1;
        edges[i][1] = v - 1;

        in_neigh[i][0] = v - 1;
        in_neigh[i][1] = u - 1;
    }

    sort(edges.begin(), edges.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[0] < b[0]; });

    sort(in_neigh.begin(), in_neigh.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[0] < b[0]; });

    int *edgeList, *offsetArr;
    edgeList = (int *)malloc(sizeof(int) * size);
    offsetArr = (int *)malloc(sizeof(int) * num_vertices + 1);

    int *in_edgeList, *in_offsetArr;
    in_edgeList = (int *)malloc(sizeof(int) * size);
    in_offsetArr = (int *)malloc(sizeof(int) * num_vertices + 1);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        offsetArr[i] = 0;
        in_offsetArr[i] = 0;
    }

    for (int i = 0; i < num_edges; i++)
    {
        int u, v;
        u = edges[i][0];
        v = edges[i][1];

        int vertex = in_neigh[i][0];
        int vertex_in_neigh = in_neigh[i][1];

        edgeList[i] = v;
        offsetArr[u + 1] += 1;

        in_edgeList[i] = vertex_in_neigh;
        in_offsetArr[vertex + 1] += 1;
    }

    for (int i = 1; i < num_vertices + 1; i++)
    {
        offsetArr[i] += offsetArr[i - 1];
        in_offsetArr[i] += in_offsetArr[i - 1];
    }

    if (DEBUG == true)
    {
        cout << "For normal CSR" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << offsetArr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << edgeList[i] << " ";
        }
        cout << endl;
        cout << "For in neigh CSR" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << in_offsetArr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << in_edgeList[i] << " ";
        }
        cout << endl;
    }

    struct CSR csr = {offsetArr, edgeList, num_vertices, num_edges};
    struct CSR in_csr = {in_offsetArr, in_edgeList, num_vertices, num_edges};

    int *dev_offsetArr, *dev_edgeList;
    int *dev_in_offsetArr, *dev_in_edgeList;

    clock_t calcTime, assignTime, initTime, initialMemOP, prMem;

    sycl::queue q{sycl::gpu_selector{}};

    sycl::device dev = q.get_device();

    //Get information about the device
    // cout << "Device Name: " << dev.get_info<sycl::info::device::name>() << endl;
    // cout << "Device Vendor: " << dev.get_info<sycl::info::device::vendor>() << endl;
    // cout << "Device Type: " << dev.is_gpu() << endl;


    initialMemOP = clock();
    dev_offsetArr = sycl::malloc_device<int>(num_vertices + 1, q);
    dev_in_offsetArr = sycl::malloc_device<int>(num_vertices + 1, q);

    dev_edgeList = sycl::malloc_device<int>(size, q);
    dev_in_edgeList = sycl::malloc_device<int>(size, q);

    q.memcpy(dev_offsetArr, csr.offsetArr, sizeof(int) * (num_vertices + 1)).wait();
    q.memcpy(dev_in_offsetArr, in_csr.offsetArr, sizeof(int) * (num_vertices + 1)).wait();

    q.memcpy(dev_in_edgeList, in_csr.edgeList, sizeof(int) * size).wait();
    q.memcpy(dev_edgeList, csr.edgeList, sizeof(int) * size).wait();

    struct CSR *dev_csr, *dev_in_csr;
    dev_csr = sycl::malloc_device<struct CSR>(1, q);
    dev_in_csr = sycl::malloc_device<struct CSR>(1, q);
    initialMemOP = clock() - initialMemOP;

    assignTime = clock();
    assignToCSR(q, dev_csr, dev_offsetArr, dev_edgeList, num_vertices, num_edges);
    assignToCSR(q, dev_in_csr, dev_in_offsetArr, dev_in_edgeList, num_vertices, num_edges);
    assignTime = clock() - assignTime;

    if (DEBUG == true) {
        checkAssignment(q, dev_csr);
        checkAssignment(q, dev_in_csr);
    }

    float *pr, *prCopy;

    prMem = clock();
    pr = sycl::malloc_device<float>(num_vertices, q);
    prCopy = sycl::malloc_device<float>(num_vertices, q);
    
    prMem = clock() - prMem;

    unsigned nBlocks_for_vertices = ceil((float)num_vertices / B_SIZE);

    
    if (USE_RANGE == 0) {
        initTime = init(q, pr, num_vertices, nBlocks_for_vertices);    
        initTime += init(q, prCopy, num_vertices, nBlocks_for_vertices); 
    }
    else {
        initTime = init_range(q, pr, num_vertices, nBlocks_for_vertices);
        initTime += init_range(q, prCopy, num_vertices, nBlocks_for_vertices); 
    }

    int max_iter = 3;

    
    calcTime = clock();
    for (int i = 1; i < max_iter + 1; i++)
    {
        if (i % 2 == 0)
        {
            if (USE_RANGE == 0) computePR(q, dev_csr, dev_in_csr, pr, prCopy, nBlocks_for_vertices);
            else computePR_range(q, dev_csr, dev_in_csr, pr, prCopy, nBlocks_for_vertices);
        }
        else
        {
            if (USE_RANGE == 0) computePR(q, dev_csr, dev_in_csr, prCopy, pr, nBlocks_for_vertices);
            else computePR_range(q, dev_csr, dev_in_csr, prCopy, pr, nBlocks_for_vertices);
        }
    }
    calcTime = clock() - calcTime;

    if (DEBUG) {
        if (max_iter % 2 == 0)
        {
            printPR(q, prCopy, num_vertices);
        }
        else
        {
            printPR(q, pr, num_vertices);
        }
    }

    double t_time = ((double)calcTime) / CLOCKS_PER_SEC * 1000;
    cout << "On graph: " << fileName << ", Time taken: " << t_time << endl;
    cout << endl;

    return 0;
}