output: main.o make_csr.o
	g++ main.o make_csr.o -o output

main.o: main.cpp
	g++ -c main.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	g++ -c make_csr.cpp

clean:
	rm *.o output