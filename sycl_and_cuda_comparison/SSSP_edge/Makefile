INPUT_DIR := ./sample_graphs

FILENAMES := $(wildcard $(INPUT_DIR)/*.mtx)

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output

$(CUDA_OUTPUT): cuda_sssp_edge.o make_csr.o
	nvcc cuda_sssp_edge.o make_csr.o -o $(CUDA_OUTPUT)

cuda_sssp_edge.o: cuda_sssp_edge.cu $(FILENAMES)
	nvcc -c -DINPUT_FILENAME=\"$<\" cuda_sssp_edge.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

$(SYCL_OUTPUT): sycl_sssp_edge.cpp make_csr.hpp make_csr.cpp
	clang++ -fsycl sycl_sssp_edge.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

run_cuda: $(CUDA_OUTPUT)
	@for file in $(FILENAMES); do \
		echo "Running ./$(CUDA_OUTPUT) $$file:"; \
		./$(CUDA_OUTPUT) "$$file"; \
	done

run_sycl: $(SYCL_OUTPUT)
	@for file in $(FILENAMES); do \
		echo "Running ./$(SYCL_OUTPUT) $$file:"; \
		./$(SYCL_OUTPUT) "$$file"; \
	done

clean:
	rm -f *.o $(CUDA_OUTPUT) $(SYCL_OUTPUT)