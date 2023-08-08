#include <stdio.h>
#include <bits/stdc++.h>
#define inf 100000
using namespace std;

void BFS(int src, vector<int> rowPtr, vector<int> dest, int vertices)
{
    int dist[vertices], pred[vertices];
    for (int i = 0; i < vertices; ++i)
    {
        dist[i] = inf;
        pred[i] = -1;
    }

    dist[src] = 0;

    while (true)
    {
        bool changed = false;
        for (int i = 0; i < vertices; i++)
        {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++)
            {
                int neigh = dest[j];
                if (dist[neigh] > dist[i] + 1)
                {
                    dist[neigh] = dist[i] + 1;
                    pred[neigh] = i;
                    changed = true;
                }
            }
        }
        if (changed == false)
            break;
    }

    for (int i = 0; i < vertices; i++)
    {
        printf("dist %d = %d, pred %d = %d\n", i, dist[i], i, pred[i]);
    }
}

int main()
{
    int vertices = 6;
    vector<int> rowPtr{0, 1, 3, 4, 7, 10, 12};
    vector<int> dest{3, 3, 4, 5, 0, 1, 4, 3, 1, 5, 4, 2};

    BFS(0, rowPtr, dest, vertices);
}