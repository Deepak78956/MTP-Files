#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"
#define DEBUG false

bool DFSUtil(int v, vector<bool> &visited, vector<int> &vDegree, int k, struct NonWeightCSR csr)
{
    // Mark the current node as visited and print it
    visited[v] = true;

    // Recur for all the vertices adjacent to this vertex
    for (int i = csr.row_ptr[v]; i < csr.row_ptr[v + 1]; i++)
    {
        // degree of v is less than k, then degree of adjacent
        // must be reduced
        int neigh = csr.col_ind[i];
        if (vDegree[v] < k)
            vDegree[neigh]--;

        // If adjacent is not processed, process it
        if (!visited[neigh])
        {
            // If degree of adjacent after processing becomes
            // less than k, then reduce degree of v also.
            DFSUtil(neigh, visited, vDegree, k, csr);
        }
    }
    // Return true if degree of v is less than k
    return (vDegree[v] < k);
}

void printKCores(int k, int V, struct NonWeightCSR csr)
{
    // INITIALIZATION
    // Mark all the vertices as not visited and not
    // processed.
    vector<bool> visited(V, false);
    vector<bool> processed(V, false);

    int mindeg = INT_MAX;
    int startvertex;

    // Store degrees of all vertices
    vector<int> vDegree(V);
    for (int i = 0; i < V; i++)
    {
        vDegree[i] = csr.row_ptr[i + 1] - csr.row_ptr[i];

        if (vDegree[i] < mindeg)
        {
            mindeg = vDegree[i];
            startvertex = i;
        }
    }

    DFSUtil(startvertex, visited, vDegree, k, csr);

    // If Graph is disconnected.
    for (int i = 0; i < V; i++)
        if (visited[i] == false)
            DFSUtil(i, visited, vDegree, k, csr);

    // Considering Edge Case
    for (int v = 0; v < V; v++)
    {
        if (vDegree[v] >= k)
        {
            int cnt = 0;

            for (int i = csr.row_ptr[v]; i < csr.row_ptr[v + 1]; i++)
            {
                int neigh = csr.col_ind[i];
                if (vDegree[neigh] >= k)
                    cnt++;
            }

            if (cnt < k)
                vDegree[v] = cnt;
        }
    }

    // PRINTING K CORES
    cout << "K-Cores : \n";
    for (int v = 0; v < V; v++)
    {
        // Only considering those vertices which have degree
        // >= K after DFS
        if (vDegree[v] >= k)
        {
            cout << "\n[" << v << "]";

            // Traverse adjacency list of v and print only
            // those adjacent which have vDegree >= k after
            // DFS.
            for (int i = csr.row_ptr[v]; i < csr.row_ptr[v + 1]; i++)
            {
                int neigh = csr.col_ind[i];
                if (vDegree[neigh] >= k)
                    cout << " -> " << neigh;
            }
        }
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

    int k = 3;
    printKCores(k, num_vertices, csr);

    return 0;
}