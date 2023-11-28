SYCL_OUTPUT := sycl_output

$(SYCL_OUTPUT): sycl_build_adj.cpp make_csr.hpp make_csr.cpp
	clang++ -fsycl sycl_build_adj.cpp -fsycl-targets=nvptx64-nvidia-cuda make_csr.cpp -o $(SYCL_OUTPUT)

run_sycl: $(SYCL_OUTPUT)
	@echo "Running SYCL program: ./$(SYCL_OUTPUT)"
	./$(SYCL_OUTPUT)

clean:
	rm -f *.o $(SYCL_OUTPUT)