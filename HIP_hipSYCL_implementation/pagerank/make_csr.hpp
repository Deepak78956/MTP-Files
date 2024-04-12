#ifndef MAKE_CSR_HPP
#define MAKE_CSR_HPP
using namespace std;

struct WeightCSR
{
    vector<int> row_ptr;
    vector<int> col_ind;
    vector<int> weights;
};

struct NonWeightCSR
{
    vector<int> row_ptr;
    vector<int> col_ind;
};

struct NonWeightCSR CSRNonWeighted(int num_vertices, int num_edges, int directed, ifstream &fin);
struct WeightCSR CSRWeighted(int num_vertices, int num_edges, int directed, ifstream &fin);
#endif
