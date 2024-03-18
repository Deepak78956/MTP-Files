INPUT_DIR := ./sample_graphs

FILENAMES := $(wildcard $(INPUT_DIR)/*.mtx)

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output
HIP_OUTPUT := hip_output

$(CUDA_OUTPUT): cuda_BFS.o make_csr.o
	nvcc -O3 cuda_BFS.o make_csr.o -o $(CUDA_OUTPUT)

cuda_BFS.o: cuda_BFS.cu $(FILENAMES)
	nvcc -c -O3 -DINPUT_FILENAME=\"$<\" cuda_BFS.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c -O3 make_csr.cpp

# $(SYCL_OUTPUT): sycl_BFS.cpp make_csr.hpp make_csr.cpp
# 	clang++ -O3 -fsycl sycl_BFS.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

$(SYCL_OUTPUT): LB_BFS_buffer.cpp make_csr.hpp make_csr.cpp
	acpp -O3 LB_BFS_buffer.cpp make_csr.cpp -o $(SYCL_OUTPUT)

$(HIP_OUTPUT): hip_BFS.cpp make_csr.hpp make_csr.cpp
	hipcc -O3 hip_BFS.cpp make_csr.cpp -o $(HIP_OUTPUT)

# run_cuda: $(CUDA_OUTPUT)
# 	for i in 1 2 3; do \
# 		echo "Running iteration $$i"; \
# 		for file in $(FILENAMES); do \
# 			echo "Running ./$(CUDA_OUTPUT) $$file:"; \
# 			./$(CUDA_OUTPUT) "$$file"; \
# 		done \
# 	done

run_sycl: $(SYCL_OUTPUT)
	for i in 1 2 3 4 5; do \
		echo "Running iteration $$i"; \
		for file in $(FILENAMES); do \
			echo "Running ./$(SYCL_OUTPUT) $$file:"; \
			./$(SYCL_OUTPUT) "$$file"; \
		done \
	done 

# run_hip: $(HIP_OUTPUT)
# 	for i in 1 2 3 4 5; do \
# 		for file in $(FILENAMES); do \
# 			echo "Running ./$(HIP_OUTPUT) $$file:"; \
# 			./$(HIP_OUTPUT) "$$file"; \
# 		done \
# 	done 

clean:
	rm -f *.o $(SYCL_OUTPUT)