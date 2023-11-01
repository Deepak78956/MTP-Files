#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#define N 100
using namespace std;

void matrix_mutilply(sycl::queue &q, int *A, int *B, int *C)
{
    q.submit([&](sycl::handler &h)
             { h.parallel_for(sycl::range{N, N}, [=](sycl::id<2> index)
                              {
            int i = index[0];
            int j = index[1];

            for (int k = 0; k < N; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
                
                        } }); })
        .wait();
}

void fillWithRandom(int *arr)
{
    int numRange = 10;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arr[i * N + j] = rand() % numRange;
        }
    }
    cout << endl;
}

int main()
{
    int *mat_A, *mat_B, *mat_C, *dev_res;

    sycl::queue q{sycl::gpu_selector{}};

    mat_A = (int *)malloc(sizeof(int) * (N * N));
    mat_B = (int *)malloc(sizeof(int) * (N * N));
    mat_C = (int *)malloc(sizeof(int) * (N * N));
    dev_res = (int *)malloc(sizeof(int) * (N * N));

    fillWithRandom(mat_A);
    fillWithRandom(mat_B);

    int *dev_mat_A, *dev_mat_B, *dev_mat_C;

    dev_mat_A = sycl::malloc_device<int>((N * N), q);
    dev_mat_B = sycl::malloc_device<int>((N * N), q);
    dev_mat_C = sycl::malloc_device<int>((N * N), q);

    q.submit([&](sycl::handler &cgh)
             { cgh.memcpy(dev_mat_A, mat_A, N * N * sizeof(int)); });
    q.submit([&](sycl::handler &cgh)
             { cgh.memcpy(dev_mat_B, mat_B, N * N * sizeof(int)); });

    matrix_mutilply(q, dev_mat_A, dev_mat_B, dev_mat_C);
    cout << endl;

    q.submit([&](sycl::handler &cgh)
             { cgh.memcpy(dev_res, dev_mat_C, N * N * sizeof(int)); });

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

    return 0;
}