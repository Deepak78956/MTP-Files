output: make_csr.o make_csr.o
	nvcc make_csr.o make_csr.o -o output

make_csr.o: make_csr.cpp file.txt
	nvcc -c make_csr.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

clean:
	rm *.o output