SYCL_OUTPUT := sycl_output
$(SYCL_OUTPUT): clang++ -O3 -fsycl -funroll-loops -fsycl-targets=nvptx64-nvidia-cuda ./matmul_sycl.cpp -o $(SYCL_OUTPUT)

run_sycl: ./$(SYCL_OUTPUT)