output: triangle_counting.o make_csr.o
	nvcc triangle_counting.o make_csr.o -o output

triangle_counting.o: triangle_counting.cu file.txt
	nvcc -c triangle_counting.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

clean:
	rm *.o output