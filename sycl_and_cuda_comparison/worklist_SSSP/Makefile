INPUT_DIR := ./sample_graphs

FILENAMES := file.txt

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output

$(CUDA_OUTPUT): cuda_worklist_sssp.o make_csr.o
	nvcc cuda_worklist_sssp.o make_csr.o -o $(CUDA_OUTPUT)

cuda_worklist_sssp.o: cuda_worklist_sssp.cu $(FILENAMES)
	nvcc -c -DINPUT_FILENAME=\"$<\" cuda_worklist_sssp.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

$(SYCL_OUTPUT): sycl_sssp.cpp make_csr.hpp make_csr.cpp
	clang++ -fsycl sycl_sssp.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

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