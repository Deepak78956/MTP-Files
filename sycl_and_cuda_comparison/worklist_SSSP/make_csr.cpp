#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"

struct WeightCSR CSRWeighted(int num_vertices, int num_edges, int directed, ifstream &fin, string filename)
{
    int size;
    struct WeightCSR csr;
    if (directed)
        size = num_edges;
    else
        size = 2 * num_edges;
    vector<vector<int>> edges(size, vector<int>(3, 0));

    // // manually add weight if not present in the input file
    // vector<string> keywords = {"kron", "file"};

    // bool keywordFound = false;

    // for (const string& keyword : keywords) {
    //     // Check if the keyword is present in the filename
    //     if (filename.find(keyword) != string::npos) {
    //         // Set the flag to true indicating the keyword is found
    //         keywordFound = true;
    //         break;
    //     }
    // }

    for (int i = 0; i < num_edges; i++)
    {
        int u, v, w;

        // if (keywordFound == false){
        //     fin >> u >> v >> w;
        // } 
        // else {
        //     fin >> u >> v;
        //     w = 1;
        // }
        fin >> u >> v;
        w = 1;
        edges[i][0] = u - 1;
        edges[i][1] = v - 1;
        edges[i][2] = 1;
        if (!directed)
        {
            edges[num_edges + i][0] = v - 1;
            edges[num_edges + i][1] = u - 1;
            edges[num_edges + i][2] = 1;
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

struct NonWeightCSR CSRNonWeighted(int num_vertices, int num_edges, int directed, ifstream &fin)
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
        int u, v;
        fin >> u >> v;
        edges[i][0] = u;
        edges[i][1] = v;
        if (!directed)
        {
            edges[num_edges + i][0] = v;
            edges[num_edges + i][1] = u;
        }
    }

    sort(edges.begin(), edges.end());

    vector<int> row_ptr(num_vertices + 1);
    vector<int> col_ind;

    for (const auto &edge : edges)
    {
        int u, v;
        u = edge[0];
        v = edge[1];

        row_ptr[u + 1] += 1;
        col_ind.push_back(v);
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
    csr = {row_ptr, col_ind};
    return csr;
}