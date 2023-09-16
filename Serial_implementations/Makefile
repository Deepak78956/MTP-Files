output: csr_to_adj.o make_csr.o
	nvcc csr_to_adj.o make_csr.o -o output

csr_to_adj.o: csr_to_adj.cpp file.txt
	nvcc -c csr_to_adj.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

clean:
	rm *.o output