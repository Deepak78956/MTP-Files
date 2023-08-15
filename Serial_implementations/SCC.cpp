#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"
using namespace std;
#define DEBUG false

struct node
{
    int color;
    int start;
    int end;
    int pred;
    int id;
    int comp;
};

struct graph
{
    vector<int> rowPtr;
    vector<int> dest;
    int vertices;
};

int clk = 0;

void Traverse(int vertex, vector<int> rowPtr, vector<int> dest, vector<node> &prop, int scc)
{
    prop[vertex].start = ++clk;
    prop[vertex].comp = scc;
    prop[vertex].color = 1;
    for (int j = rowPtr[vertex]; j < rowPtr[vertex + 1]; j++)
    {
        int neigh = dest[j];
        if (prop[neigh].color == 0)
        {
            prop[neigh].pred = vertex;
            Traverse(neigh, rowPtr, dest, prop, scc);
        }
    }
    prop[vertex].color = 2;
    prop[vertex].end = ++clk;
}

vector<node> DFS(vector<int> rowPtr, vector<int> dest, int n)
{
    vector<node> prop;
    for (int i = 0; i < n; ++i)
    {
        node ele;
        ele.color = 0; // 0 - white, 1 - gray, 2 - black
        ele.start = -1;
        ele.end = -1;
        ele.pred = -1;
        ele.id = i;
        ele.comp = -1;
        prop.push_back(ele);
    }

    for (int i = 0; i < n; ++i)
    {
        if (prop[i].color == 0)
            Traverse(i, rowPtr, dest, prop, -1);
    }

    if (DEBUG == true) {
        cout << "After DFS" << endl;
        for (int i = 0; i < n; ++i)
        {
            printf("For node %d \n", i);
            printf("start = %d, end = %d, pred = %d\n", prop[i].start, prop[i].end, prop[i].pred);
            printf("\n");
        }
    }

    return prop;
}

// bool comp(struct node &a, struct node &b)
// {
//     return a.end >= b.end;
// }

void SCC(struct graph g, struct graph transpose)
{
    vector<node> prop;
    prop = DFS(g.rowPtr, g.dest, g.vertices);

    if (DEBUG == true)
    {
        cout << "Before sorting according to End times" << endl;
        for (int i = 0; i < g.vertices; i++)
        {
            cout << "Vertex: " << i << " "
                 << "End time: " << prop[i].end << endl;
        }
        cout << endl;
    }

    // Sorting according to the end times, performing Kosaraju's algorithm
    sort(prop.begin(), prop.end(), [](const node &a, const node &b)
         { return a.end >= b.end; });

    vector<int> decSortedVertices;
    for (int i = 0; i < g.vertices; i++)
    {
        decSortedVertices.push_back(prop[i].id);
    }

    if (DEBUG == true)
    {
        cout << "After sorting according to End times" << endl;
        for (int i = 0; i < g.vertices; i++)
        {
            cout << "Vertex: " << prop[i].id << " "
                 << "End time: " << prop[i].end << endl;
        }
        cout << endl;
    }

    int scc = 0;
    vector<node> transposeProp;
    for (int i = 0; i < g.vertices; ++i)
    {
        node ele;
        ele.color = 0; // 0 - white, 1 - gray, 2 - black
        ele.start = -1;
        ele.end = -1;
        ele.pred = -1;
        ele.id = i;
        ele.comp = -1;
        transposeProp.push_back(ele);
    }

    for (int i = 0; i < transpose.vertices; ++i)
    {
        int nextDec = decSortedVertices[i];
        if (transposeProp[nextDec].color == 0)
        {
            Traverse(nextDec, transpose.rowPtr, transpose.dest, transposeProp, scc);
            scc += 1;
        }
    }

    for (int i = 0; i < transpose.vertices; i++)
    {
        printf("Node %d, component %d\n", i, transposeProp[i].comp);
    }
}

void swap(int &n1, int &n2)
{
    int temp;
    temp = n1;
    n1 = n2;
    n2 = temp;
}

struct graph convertToCSR(vector<vector<int>> edges, int num_vertices)
{
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

    struct graph csr;

    csr.rowPtr = row_ptr;
    csr.dest = col_ind;
    csr.vertices = num_vertices;

    return csr;
}

int main()
{
    ifstream fin("file.txt");
    int num_vertices, num_edges, directed, weighted;
    fin >> num_vertices >> num_edges >> directed >> weighted;

    if (DEBUG == true)
    {
        cout << "Reading from file" << endl;
        cout << num_vertices << " " << num_edges << " " << directed << " " << weighted << endl;
    }

    int size;
    if (directed)
    {
        size = num_edges;
    }
    else
    {
        cout << "Directed graph is required" << endl;
        exit(0);
    }

    if (weighted)
    {
        cout << "Non-weighted graph is required" << endl;
        exit(0);
    }

    // Taking input from file
    vector<vector<int>> edges(size, vector<int>(2, 0));
    for (int i = 0; i < num_edges; i++)
    {
        int u, v;
        fin >> u >> v;
        if (DEBUG == true)
        {
            cout << u << " " << v << endl;
        }
        edges[i][0] = u;
        edges[i][1] = v;
        if (!directed)
        {
            edges[num_edges + i][0] = v;
            edges[num_edges + i][1] = u;
        }
    }

    struct graph csr;
    csr = convertToCSR(edges, num_vertices);

    if (DEBUG == true)
    {
        cout << "CSR after taking input" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << csr.rowPtr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.dest[i] << " ";
        }
        cout << endl;
    }

    // Conversion of input graph to its transpose
    // here edges[i][0] = u; edges[i][1] = v;

    for (int i = 0; i < num_edges; i++)
    {
        swap(edges[i][0], edges[i][1]);
    }

    struct graph graphTranspose;
    graphTranspose = convertToCSR(edges, num_vertices);

    if (DEBUG == true)
    {
        cout << "Conversion of graph to its transpose" << endl;
        for (int i = 0; i < num_vertices + 1; i++)
        {
            cout << csr.rowPtr[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < size; i++)
        {
            cout << csr.dest[i] << " ";
        }
        cout << endl;
    }

    SCC(csr, graphTranspose);
}