#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"
using namespace std;
#define DEBUG false

int clk = 0;

struct props
{
    int color, component, start, etime, pred;
};

struct NonWeightCSR convertToUndirectedCSR(struct NonWeightCSR directedCSR, int vertices)
{
    int num_edges = directedCSR.col_ind.size();
    int size = num_edges * 2;
    vector<vector<int>> edges(size, vector<int>(2, 0));

    for (int i = 0; i < vertices; i++)
    {
        for (int j = directedCSR.row_ptr[i]; j < directedCSR.row_ptr[i + 1]; j++)
        {
            int u = i;
            int v = directedCSR.col_ind[j];
            edges[i][0] = u;
            edges[i][1] = v;
            edges[num_edges + i][0] = v;
            edges[num_edges + i][1] = u;
        }
    }

    sort(edges.begin(), edges.end());

    vector<int> row_ptr(vertices + 1);
    vector<int> col_ind;

    for (const auto &edge : edges)
    {
        int u, v;
        u = edge[0];
        v = edge[1];

        row_ptr[u + 1] += 1;
        col_ind.push_back(v);
    }

    struct NonWeightCSR undirectedCSR;
    undirectedCSR.col_ind = col_ind;
    undirectedCSR.row_ptr = row_ptr;

    return undirectedCSR;
}

void Traverse(int p, vector<int> rowPtr, vector<int> dest, int cn, struct props *prop)
{
    prop[p].start = ++clk;
    prop[p].component = cn;
    prop[p].color = 1;

    int ptr = rowPtr[p];
    while (ptr < rowPtr[p + 1])
    {
        int t = dest[ptr];
        if (prop[t].color == 0)
        {
            prop[t].pred = p;
            Traverse(t, rowPtr, dest, cn, prop);
        }
        ptr++;
    }
    prop[p].color = 2;
    prop[p].etime = ++clk;
}

void DFS_CC(vector<int> rowPtr, vector<int> dest)
{
    int cn = 0;
    int N = rowPtr.size() - 1;
    struct props prop[N];
    for (int i = 0; i < N; i++)
    {
        prop[i].pred = -1;
        prop[i].color = 0;
        prop[i].component = 0;
        prop[i].start = -1;
        prop[i].etime = -1;
    }
    for (int i = 0; i < N; i++)
    {
        if (prop[i].color == 0)
        {
            Traverse(i, rowPtr, dest, cn, prop);
            cn += 1;
        }
    }

    for (int i = 0; i < N; i++)
    {
        printf("Node %d, component %d\n", i + 1, prop[i].component);
    }
}

void WCC(struct NonWeightCSR directedCSR, int vertices)
{
    struct NonWeightCSR undirectedCSR = convertToUndirectedCSR(directedCSR, vertices);
    DFS_CC(undirectedCSR.row_ptr, undirectedCSR.col_ind);
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size = num_edges;

    if (directed == 0)
    {
        cout << "directed graph is required" << endl;
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

    WCC(csr, num_vertices);

    return 0;
}