#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"

void triangleCounting(struct NonWeightCSR csr, int vertices)
{
    int tc = 0;
    for (int i = 0; i < vertices; i++)
    {
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++)
        {
            int t = csr.col_ind[j];
            for (int k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; k++)
            {
                int r = csr.col_ind[k];
                if (t != r)
                {
                    for (int q = csr.row_ptr[r]; q < csr.row_ptr[r + 1]; q++)
                    {
                        int neighOfr = csr.col_ind[q];
                        if (t == neighOfr)
                        {
                            tc += 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    cout << "Triangles in Graph is " << tc / 6 << endl;
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size;
    if (!directed)
        size = 2 * num_edges;
    else
    {
        cout << "Un-Directed graph is required" << endl;
        exit(0);
    }
    if (weighted)
    {
        cout << "Non weighted graph is required" << endl;
        exit(0);
    }
    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);
    triangleCounting(csr, num_vertices);

    return 0;
}