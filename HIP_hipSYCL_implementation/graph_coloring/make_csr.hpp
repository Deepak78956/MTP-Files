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
    int *offsetArr;
    int *edgeList;
};

struct NonWeightCSR CSRNonWeighted(int num_vertices, int num_edges, int directed, ifstream &fin, bool keywordFound);
struct WeightCSR CSRWeighted(int num_vertices, int num_edges, int directed, ifstream &fin);
#endif