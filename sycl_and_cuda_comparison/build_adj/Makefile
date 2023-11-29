INPUT_DIR := ./sample_graphs

FILENAMES := delaunay_n10.mtx delaunay_n12.mtx inf-luxembourg_osm.mtx inf-belgium_osm.mtx rgg_n_2_15_s0.mtx chesapeake.mtx

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output

$(CUDA_OUTPUT): cuda_build_adj.o make_csr.o
	nvcc cuda_build_adj.o make_csr.o -o $(CUDA_OUTPUT)

cuda_build_adj.o: cuda_build_adj.cu $(FILENAMES)
	nvcc -c -DINPUT_FILENAME=\"$<\" cuda_build_adj.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

$(SYCL_OUTPUT): sycl_build_adj.cpp make_csr.hpp make_csr.cpp
	clang++ -fsycl sycl_build_adj.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

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

