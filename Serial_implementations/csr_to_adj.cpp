#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <numeric>
#include "make_csr.hpp"
#define DEBUG false

struct Node
{
    int data;
    struct Node *next;
};

struct Graph
{
    int numVertices;
    struct Node **adjLists;
};

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

    clock_t totalTime;
    totalTime = clock();

    struct Graph *graph;
    graph = (struct Graph *)malloc(sizeof(struct Graph));
    graph->numVertices = num_vertices;
    graph->adjLists = (struct Node **)malloc(num_vertices * sizeof(struct Node *));

    for (int i = 0; i < num_vertices; i++)
        graph->adjLists[i] = NULL;

    for (int u = 0; u < num_vertices + 1; u++)
    {
        for (int j = csr.row_ptr[u]; j < csr.row_ptr[u + 1]; j++)
        {
            int v = csr.col_ind[j];
            struct Node *newNode;
            newNode = (struct Node *)malloc(sizeof(struct Node));
            newNode->data = v;
            newNode->next = NULL;

            newNode->next = graph->adjLists[u];
            graph->adjLists[u] = newNode;

            newNode = (struct Node *)malloc(sizeof(struct Node));
            newNode->data = u;
            newNode->next = NULL;

            newNode->next = graph->adjLists[v];
            graph->adjLists[v] = newNode;
        }
    }

    totalTime = clock() - totalTime;
    cout << "Time taken: " << totalTime << endl;

    // int v;
    // for (v = 0; v < graph->numVertices; v++)
    // {
    //     struct Node *temp = graph->adjLists[v];
    //     printf("\n Vertex %d\n: ", v);
    //     while (temp)
    //     {
    //         printf("%d -> ", temp->data);
    //         temp = temp->next;
    //     }
    //     printf("\n");
    // }
}