#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"

using namespace std;

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size;
    if (directed)
        size = num_edges;
    else
        size = 2 * num_edges;

    if (weighted)
    {
        struct WeightCSR csr = CSRWeighted(num_vertices, num_edges, directed, fin);
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << csr.row_ptr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.col_ind[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.weights[i] << " ";
        }
        cout << endl;
    }
    else
    {
        struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << csr.row_ptr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.col_ind[i] << " ";
        }
        cout << endl;
    }
    return 0;
}
