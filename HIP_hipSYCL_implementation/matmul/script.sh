#!/bin/bash

# Define matrix sizes # max matrix file size goes to 1 GB 16384
sizes=(64 128 256 512 1024 2048 4096)

# Run generate_matrices.cpp for each matrix size
for size in "${sizes[@]}"; do
    ./matrix_generator $size

    # for i in 1 2 3 4 5; do
    #     ./cuda_output >> cuda_res/output_cuda_$size.txt
    # done

    for i in 1 2 3 4 5; do
        ./sycl_output >> sycl_res/output_$size.txt
    done

    # for i in 1 2 3 4 5; do
    #     ./hip_output >> hip_res/output_$size.txt
    # done
done