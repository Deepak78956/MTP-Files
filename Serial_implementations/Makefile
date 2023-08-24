output: triangle_counting.o make_csr.o
	g++ triangle_counting.o make_csr.o -o output

triangle_counting.o: triangle_counting.cpp file.txt
	g++ -c triangle_counting.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	g++ -c make_csr.cpp

clean:
	rm *.o output