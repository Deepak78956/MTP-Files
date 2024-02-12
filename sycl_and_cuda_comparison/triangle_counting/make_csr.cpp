#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"

struct WeightCSR CSRWeighted(int num_vertices, int num_edges, int directed, ifstream &fin)
{
    int size;
    struct WeightCSR csr;
    if (directed)
        size = num_edges;
    else
        size = 2 * num_edges;
    vector<vector<int>> edges(size, vector<int>(3, 0));
    for (int i = 0; i < num_edges; i++)
    {
        int u, v, w;
        fin >> u >> v >> w;
        edges[i][0] = u;
        edges[i][1] = v;
        edges[i][2] = w;
        if (!directed)
        {
            edges[num_edges + i][0] = v;
            edges[num_edges + i][1] = u;
            edges[num_edges + i][2] = w;
        }
    }

    sort(edges.begin(), edges.end());

    vector<int> row_ptr(num_vertices + 1);
    vector<int> col_ind;
    vector<int> weights;

    for (const auto &edge : edges)
    {
        int u, v, w;
        u = edge[0];
        v = edge[1];
        w = edge[2];

        row_ptr[u + 1] += 1;

        col_ind.push_back(v);
        weights.push_back(w);
    }

    for (int i = 1; i < num_vertices + 1; i++)
    {
        row_ptr[i] += row_ptr[i - 1];
    }

    // for (int i = 0; i < num_vertices + 1; i++)
    // {
    //     cout << row_ptr[i] << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < size; i++)
    // {
    //     cout << col_ind[i] << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < size; i++)
    // {
    //     cout << weights[i] << " ";
    // }
    // cout << endl;
    csr = {row_ptr, col_ind, weights};
    return csr;
}

struct NonWeightCSR CSRNonWeighted(int num_vertices, int num_edges, int directed, ifstream &fin, bool keywordFound)
{
    int size;
    struct NonWeightCSR csr;
    if (directed)
        size = num_edges;
    else
        size = 2 * num_edges;

    vector<vector<int>> edges(size, vector<int>(2, 0));
    for (int i = 0; i < num_edges; i++)
    {
        int u, v, w;
        if (keywordFound) {
            fin >> u >> v >> w;
        }
        else{
            fin >> u >> v;
        }
        edges[i][0] = u - 1;
        edges[i][1] = v - 1;
        if (!directed)
        {
            edges[num_edges + i][0] = v - 1;
            edges[num_edges + i][1] = u - 1;
        }
    }

    sort(edges.begin(), edges.end(), [](const vector<int> &a, const vector<int> &b)
         { return a[0] < b[0]; });

    int *edgeList, *offsetArr;
    edgeList = (int *)malloc(sizeof(int) * size);
    offsetArr = (int *)malloc(sizeof(int) * num_vertices + 1);

    for (int i = 0; i < num_vertices + 1; i++)
    {
        offsetArr[i] = 0;
    }

    for (int i = 0; i < size; i++)
    {
        int u, v;
        u = edges[i][0];
        v = edges[i][1];

        edgeList[i] = v;
        offsetArr[u + 1] += 1;
    }

    for (int i = 1; i < num_vertices + 1; i++)
    {
        offsetArr[i] += offsetArr[i - 1];
    }

    // for (int i = 0; i < num_vertices + 1; i++)
    // {
    //     cout << row_ptr[i] << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < size; i++)
    // {
    //     cout << col_ind[i] << " ";
    // }
    // cout << endl;
    csr = {offsetArr, edgeList};
    return csr;
}