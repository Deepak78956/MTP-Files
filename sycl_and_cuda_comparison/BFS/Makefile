INPUT_DIR := ./sample_graphs

FILENAMES := $(wildcard $(INPUT_DIR)/*.mtx)

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output

$(CUDA_OUTPUT): cuda_BFS.o make_csr.o
	nvcc -O3 cuda_BFS.o make_csr.o -o $(CUDA_OUTPUT)

cuda_BFS.o: cuda_BFS.cu $(FILENAMES)
	nvcc -c -O3 -DINPUT_FILENAME=\"$<\" cuda_BFS.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c -O3 make_csr.cpp

$(SYCL_OUTPUT): sycl_BFS.cpp make_csr.hpp make_csr.cpp
	clang++ -O3 -fsycl sycl_BFS.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

run_cuda: $(CUDA_OUTPUT)
	for i in 1 2 3; do \
		echo "Running iteration $$i"; \
		for file in $(FILENAMES); do \
			echo "Running ./$(CUDA_OUTPUT) $$file:"; \
			./$(CUDA_OUTPUT) "$$file"; \
		done \
	done

run_sycl: $(SYCL_OUTPUT)
	for i in {1..3}; do \
		for file in $(FILENAMES); do \
			echo "Running ./$(SYCL_OUTPUT) $$file:"; \
			./$(SYCL_OUTPUT) "$$file"; \
		done \
	done 

clean:
	rm -f *.o $(CUDA_OUTPUT) $(SYCL_OUTPUT)