output: ll_creation_parallel.o make_csr.o
	nvcc ll_creation_parallel.o make_csr.o -o output

ll_creation_parallel.o: ll_creation_parallel.cu file.txt
	nvcc -c ll_creation_parallel.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

clean:
	rm *.o output