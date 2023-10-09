output: page_rank.o make_csr.o
	nvcc page_rank.o make_csr.o -o output

page_rank.o: page_rank.cu file.txt
	nvcc -c page_rank.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

clean:
	rm *.o output