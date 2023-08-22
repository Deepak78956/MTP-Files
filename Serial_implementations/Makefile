output: MIS.o make_csr.o
	g++ MIS.o make_csr.o -o output

MIS.o: MIS.cpp file.txt
	g++ -c MIS.cpp

make_csr.o: make_csr.cpp make_csr.hpp
	g++ -c make_csr.cpp

clean:
	rm *.o output