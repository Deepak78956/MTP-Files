output: Boruvkas_MST.o make_csr.o
	g++ Boruvkas_MST.o make_csr.o -o output

Boruvkas_MST.o: Boruvkas_MST.cpp
	g++ -c Boruvkas_MST.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	g++ -c make_csr.cpp

clean:
	rm *.o output