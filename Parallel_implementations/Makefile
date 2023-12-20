INPUT_DIR := ./sample_graphs

# FILENAMES := delaunay_n10.mtx delaunay_n23.mtx delaunay_n24.mtx inf-luxembourg_osm.mtx inf-road_usa.mtx rgg_n_2_24_s0.mtx rgg_n_2_23_s0.mtx
FILENAMES := delaunay_n10.mtx

OUTPUTS := $(addprefix output_, $(notdir $(FILENAMES)))

output: sssp_vertex_centric.o make_csr.o
	nvcc sssp_vertex_centric.o make_csr.o -o output

sssp_vertex_centric.o: sssp_vertex_centric.cu $(FILENAMES)
	nvcc -c -DINPUT_FILENAME=\"$<\" sssp_vertex_centric.cu

make_csr.o: make_csr.cpp make_csr.hpp
	nvcc -c make_csr.cpp

run: output
	@for file in $(FILENAMES); do \
		./output "$(INPUT_DIR)/$$file"; \
	done

clean:
	rm -f *.o output $(OUTPUTS)