output: k_core.o make_csr.o
	g++ k_core.o make_csr.o -o output

k_core.o: k_core.cpp file.txt
	g++ -c k_core.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	g++ -c make_csr.cpp

clean:
	rm *.o output