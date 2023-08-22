#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"
using namespace std;
#define DEBUG false

vector<int> MIS(vector<int> rowPtr, vector<int> dest, int vertices)
{
    vector<int> I;
    vector<int> next(vertices);
    iota(next.begin(), next.end(), 1);
    for (int i = 0; i < vertices; i++)
    {
        int v = i;
        if (next[v] != -1)
        {
            I.push_back(v);
            for (int j = rowPtr[v]; j < rowPtr[v + 1]; j++)
            {
                int neigh = dest[j];
                next[neigh] = -1;
            }
        }
    }

    return I;
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size = num_edges * 2;

    if (directed == 1)
    {
        cout << "un-directed graph is required" << endl;
        exit(0);
    }

    if (weighted == 1)
    {
        cout << "Non-Weighted graph is required" << endl;
        exit(0);
    }

    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);

    if (DEBUG == true)
    {
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

    vector<int> ans = MIS(csr.row_ptr, csr.col_ind, num_vertices);
    cout << "Maximal independent set length is " << ans.size() << endl;
    for (int i = 0; i < ans.size(); i++)
    {
        cout << ans[i] << " ";
    }
    cout << endl;

    return 0;
}