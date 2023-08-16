#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"
using namespace std;
#define DEBUG true

struct Edge
{
    int src, dest, weight;
};

class DisjointUnionSet
{
private:
    vector<int> rank, parent;

public:
    int setLen;
    DisjointUnionSet(int n)
    {
        setLen = n;
        for (int i = 0; i < n; i++)
        {
            parent.push_back(i);
            rank.push_back(0);
        }
    }

    int myFind(int x)
    {
        if (parent[x] != x)
        {
            parent[x] = myFind(parent[x]);
        }

        return parent[x];
    }

    bool isLoop(int x, int y)
    {
        int xRoot = myFind(x);
        int yRoot = myFind(y);

        if (xRoot == yRoot)
            return true;

        return false;
    }

    void myUnion(int x, int y)
    {
        int xRoot = myFind(x);
        int yRoot = myFind(y);

        if (xRoot == yRoot)
            return;

        if (rank[xRoot] < rank[yRoot])
            parent[xRoot] = yRoot;

        else if (rank[yRoot] < rank[xRoot])
            parent[yRoot] = xRoot;

        else
        {
            parent[yRoot] = xRoot;
            rank[xRoot] = rank[xRoot] + 1;
        }

        setLen -= 1;
    }
};

void computeMST(vector<int> rowPtr, vector<int> dest, vector<int> weights, int vertices)
{
    DisjointUnionSet set = DisjointUnionSet(vertices);
    int mst_cost = 0;
    vector<Edge> MST;

    while (set.setLen > 1)
    {
        if (DEBUG == true)
        {
            cout << "Set len is: " << set.setLen << endl;
        }
        for (int i = 0; i < vertices; i++)
        {
            // Calculating smallest outgoing edge from a set component (vertices with same parent)
            int parent = set.myFind(i), smallest, u, v, flag = 0, update = false;
            for (int j = 0; j < vertices; j++)
            {
                if (parent == set.myFind(j))
                {
                    for (int k = rowPtr[j]; k < rowPtr[j + 1]; j++)
                    {
                        int adj = dest[k];
                        if (parent == set.myFind(adj))
                            continue;
                        update = true;
                        if (flag == 0)
                        {
                            smallest = weights[k];
                            v = adj;
                            u = j;
                            flag = 1;
                        }
                        else
                        {
                            if (smallest > weights[k])
                            {
                                smallest = weights[k];
                                v = adj;
                                u = k;
                            }
                        }
                    }
                }
            }

            if (DEBUG == true)
            {
                cout << "Smallest edge for vertex " << u << " is " << smallest << endl;
            }

            if (update)
            {
                struct Edge e;
                e.src = u;
                e.dest = v;
                e.weight = smallest;
                MST.push_back(e);
                set.myUnion(u, v);
                mst_cost += smallest;
            }
        }
    }

    cout << "MST cost: " << mst_cost << endl;
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    int size;
    size = 2 * num_edges;

    if (directed)
    {
        cout << "Undirected graph is required" << endl;
        exit(0);
    }

    if (weighted == 0)
    {
        cout << "Weighted graph is required" << endl;
        exit(0);
    }

    struct WeightCSR csr = CSRWeighted(num_vertices, num_edges, directed, fin);

    if (DEBUG == true)
    {
        cout << "Acquired Graph:" << endl;
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

    computeMST(csr.row_ptr, csr.col_ind, csr.weights, num_vertices);

    return 0;
}