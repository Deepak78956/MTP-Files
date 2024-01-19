INPUT_DIR := ./sample_graphs

FILENAMES := file.txt

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output

$(CUDA_OUTPUT): cuda_nnz_splitting_BFS.o make_csr.o
	nvcc cuda_nnz_splitting_BFS.o make_csr.o -o $(CUDA_OUTPUT)

cuda_nnz_splitting_BFS.o: cuda_nnz_splitting_BFS.cu $(FILENAMES)
	nvcc -c -DINPUT_FILENAME=\"$<\" cuda_nnz_splitting_BFS.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

$(SYCL_OUTPUT): sycl_BFS.cpp make_csr.hpp make_csr.cpp
	clang++ -fsycl sycl_BFS.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

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