#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
using namespace std;

#define THREADS 32

void matrix_mutilply(sycl::queue &q, int *A, int *B, int *C, unsigned int N)
{
    // q.submit([&](sycl::handler &cgh){
    //     sycl::range<2> globalRange{ N , N };
    //     sycl::range<2> localRange{ THREADS, THREADS };

    //     cgh.parallel_for(sycl::nd_range<2>(globalRange, localRange), [=](sycl::nd_item<2> item){
    //         size_t i = item.get_group(1) * THREADS + item.get_local_id(1);
    //         size_t j = item.get_group(0) * THREADS + item.get_local_id(0);

    //         float sum = 0.0f;
    //         for (size_t k = 0; k < N; ++k) {
    //             sum += A[i * N + k] * B[k * N + j];
    //         }

    //         C[i * N + j] = sum;
    //     });
    // });
    // q.wait_and_throw();
    // q.submit([&](sycl::handler &h)
    //          { h.parallel_for(sycl::range{N, N}, [=](sycl::id<2> index)
    //                           {
    //         int i = index[0];
    //         int j = index[1];

    //         for (int k = 0; k < N; k++){
    //             C[i * N + j] += A[i * N + k] * B[k * N + j];
                
    //                     } }); })
    //     .wait();
    q.submit([&](sycl::handler &h) {
        h.parallel_for<class matrix_mul>(sycl::nd_range<2>(sycl::range<2>{N, N}, sycl::range<2>{THREADS, THREADS}), [=](sycl::nd_item<2> itemID) {
            int i = itemID.get_global_id(0);
            int j = itemID.get_global_id(1);

            if (i < N && j < N) {
                for (int k = 0; k < N; ++k) {
                    C[i * N + j] += A[i * N + k] * B[k * N + j];
                }
            }
        });
    }).wait();
}

int main()
{

    int M, N;

    // Open the file for reading
    std::ifstream inputFile("output.txt");

    if (!inputFile)
    {
        std::cerr << "Error opening the input file." << std::endl;
        return 1;
    }

    // Read the header from the file
    inputFile >> M >> N;

    int *mat_A, *mat_B, *mat_C, *dev_res;

    sycl::queue q{sycl::gpu_selector{}};

    mat_A = (int *)malloc(sizeof(int) * (N * N));
    mat_B = (int *)malloc(sizeof(int) * (N * N));
    mat_C = (int *)malloc(sizeof(int) * (N * N));
    dev_res = (int *)malloc(sizeof(int) * (N * N));

    // Read the values of matrix A from the file
    for (int i = 0; i < N * N; i++)
    {
        inputFile >> mat_A[i];
    }

    // Read the values of matrix B from the file
    for (int i = 0; i < N * N; i++)
    {
        inputFile >> mat_B[i];
    }

    inputFile.close();

    int *dev_mat_A, *dev_mat_B, *dev_mat_C;

    clock_t memTime, calcTime;
    memTime = clock();
    dev_mat_A = sycl::malloc_device<int>((N * N), q);
    dev_mat_B = sycl::malloc_device<int>((N * N), q);
    dev_mat_C = sycl::malloc_device<int>((N * N), q);

    q.memcpy(dev_mat_A, mat_A,  N * N * sizeof(int)).wait();
    q.memcpy(dev_mat_B, mat_B,  N * N * sizeof(int)).wait();
    memTime = clock() - memTime;

    // q.submit([&](sycl::handler &cgh)
    //          { cgh.memcpy(dev_mat_A, mat_A, N * N * sizeof(int)); });
    // q.submit([&](sycl::handler &cgh)
    //          { cgh.memcpy(dev_mat_B, mat_B, N * N * sizeof(int)); });

    calcTime = clock();
    matrix_mutilply(q, dev_mat_A, dev_mat_B, dev_mat_C, N);
    calcTime = clock() - calcTime;

    q.memcpy(dev_res, dev_mat_C,  N * N * sizeof(int)).wait();

    // q.submit([&](sycl::handler &cgh)
    //          { cgh.memcpy(dev_res, dev_mat_C, N * N * sizeof(int)); });

    int flag = 0;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            mat_C[i * N + j] = 0;
            for (int k = 0; k < N; ++k)
            {
                mat_C[i * N + j] += mat_A[i * N + k] * mat_B[k * N + j];
            }

            if (mat_C[i * N + j] != dev_res[i * N + j])
            {
                flag = 1;
                printf("Wrong on: %d, %d  Expected = %d, Actual = %d\n", i, j, mat_C[i * N + j], dev_res[i * N + j]);
            }
        }
    }

    if (flag == 1)
    {
        cout << "Check Complete: Wrong Answer" << endl;
    }
    else
    {
        cout << "Check Complete: Correct Answer" << endl;
    }

    cout << "COMPLETED SUCCESSFULLY\n";
    double t_time = ((double)calcTime + (double)memTime) / CLOCKS_PER_SEC * 1000;
    cout << t_time << endl;
    cout << endl;

    auto maxWorkGroupSize = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    std::cout << "Maximum Work-Group Size: " << maxWorkGroupSize << std::endl;

    return 0;
}