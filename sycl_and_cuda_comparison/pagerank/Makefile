INPUT_DIR := ./sample_graphs

FILENAMES := delaunay_n10.mtx delaunay_n12.mtx inf-luxembourg_osm.mtx inf-belgium_osm.mtx rgg_n_2_15_s0.mtx chesapeake.mtx

CUDA_OUTPUT := output
SYCL_OUTPUT := sycl_output

$(CUDA_OUTPUT): page_rank_CSR.o make_csr.o
	nvcc page_rank_CSR.o make_csr.o -o $(CUDA_OUTPUT)

page_rank_CSR.o: page_rank_CSR.cu $(FILENAMES)
	nvcc -c -DINPUT_FILENAME=\"$<\" page_rank_CSR.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

$(SYCL_OUTPUT): sycl_pagerank.cpp
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda $< -o $(SYCL_OUTPUT)

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
