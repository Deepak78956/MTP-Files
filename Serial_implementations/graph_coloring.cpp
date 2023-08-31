#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"
#define DEBUG true

void assignColor(int k, vector<int> &color, struct NonWeightCSR csr)
{
    int v = k;
    color[v] = 1;
    for (int i = 0; i < k; i++)
    {
        for (int j = csr.row_ptr[v]; j < csr.row_ptr[v + 1]; j++)
        {
            int neigh = csr.col_ind[j];
            // if (DEBUG)
            // {
            //     printf("Before condition check\n");
            //     printf("node %d has color %d and its neigh %d has color %d\n", v, color[v], neigh, color[neigh]);
            // }
            if (neigh == i && color[v] == color[i])

            {
                if (DEBUG)
                {
                    printf("Neighbor of node %d is node %d with same color %d\n", v, neigh, color[neigh]);
                }
                color[v] = color[i] + 1;
                if (DEBUG)
                {
                    printf("Color changed\n");
                    printf("Now node %d has color %d and its neigh %d has color %d\n", v, color[v], neigh, color[neigh]);
                    cout << endl;
                }
            }
        }
    }
}

void gColor(struct NonWeightCSR csr, int vertices)
{
    vector<int> color(vertices);
    for (int v = 0; v < vertices; v++)
    {
        color[v] = 0;
        for (int i = 0; i < vertices; i++)
        {
            assignColor(i, color, csr);
        }
    }

    for (int i = 0; i < vertices; i++)
    {
        printf("Color of vertex %d is %d\n", i, color[i]);
    }
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
    gColor(csr, num_vertices);

    return 0;
}