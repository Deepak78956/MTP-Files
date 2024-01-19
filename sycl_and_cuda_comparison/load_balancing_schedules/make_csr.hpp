#ifndef MAKE_CSR_HPP
#define MAKE_CSR_HPP
using namespace std;

struct WeightCSR
{
    int *row_ptr;
    int *col_ind;
    int *weights;
};

struct NonWeightCSR
{
    int *row_ptr, *col_ind, vertices, edges;
};

struct NonWeightCSR CSRNonWeighted(int num_vertices, int num_edges, int directed, ifstream &fin);
struct WeightCSR CSRWeighted(int num_vertices, int num_edges, int directed, ifstream &fin);
#endif