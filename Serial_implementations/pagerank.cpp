#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"

float d = 0.85;

void pageRank(int p, struct NonWeightCSR csr, int vertices, vector<float> &pr)
{
    float val = 0.0;
    for (int i = 0; i < vertices; i++)
    {
        if (i == p)
            continue;

        int t = i, flag = 0, outDegT = csr.row_ptr[i + 1] - csr.row_ptr[i];

        // Calculating in-neighbours of p
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++)
        {
            if (csr.col_ind[j] == p)
            {
                flag = 1;
                break;
            }
        }
        if (flag == 0)
            continue;

        if (outDegT != 0)
        {
            val += (pr[t] / outDegT);
        }
        pr[p] = val * d + (1 - d) / vertices;
    }
}

void computePR(struct NonWeightCSR csr, int max_itr, int vertices)
{
    vector<float> pr(vertices);
    for (int i = 0; i < vertices; i++)
    {
        pr[i] = 1 / vertices;
    }
    for (int j = 0; j < max_itr; j++)
    {
        for (int i = 0; i < vertices; i++)
        {
            pageRank(i, csr, vertices, pr);
        }
    }

    // Dispalying computed page-ranks
    for (int i = 0; i < vertices; i++)
    {
        printf("For vertex %d page rank is %f\n", i, pr[i]);
    }
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size;
    if (directed)
        size = num_edges;
    else
    {
        cout << "Directed graph is required" << endl;
        exit(0);
    }
    if (weighted)
    {
        cout << "Non weighted graph is required" << endl;
        exit(0);
    }
    struct NonWeightCSR csr = CSRNonWeighted(num_vertices, num_edges, directed, fin);
    computePR(csr, 1, num_vertices);

    return 0;
}
