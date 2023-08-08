#include <stdio.h>
#include <bits/stdc++.h>
using namespace std;
struct node
{
    int color;
    int start;
    int end;
    int pred;
};

int clk = 0;

void Traverse(int vertex, vector<int> rowPtr, vector<int> dest, vector<node> &prop)
{
    prop[vertex].start = ++clk;
    prop[vertex].color = 1;
    for (int j = rowPtr[vertex]; j < rowPtr[vertex + 1]; j++)
    {
        int neigh = dest[j];
        if (prop[neigh].color == 0)
        {
            prop[neigh].pred = vertex;
            Traverse(neigh, rowPtr, dest, prop);
        }
    }
    prop[vertex].color = 2;
    prop[vertex].end = ++clk;
}

void DFS(vector<int> rowPtr, vector<int> dest, int n)
{
    vector<node> prop;
    for (int i = 0; i < n; ++i)
    {
        node ele;
        ele.color = 0; // 0 - white, 1 - gray, 2 - black
        ele.start = -1;
        ele.end = -1;
        ele.pred = -1;
        prop.push_back(ele);
    }

    for (int i = 0; i < n; ++i)
    {
        if (prop[i].color == 0)
            Traverse(i, rowPtr, dest, prop);
    }

    for (int i = 0; i < n; ++i)
    {
        printf("For node %d \n", i);
        printf("start = %d, end = %d, pred = %d\n", prop[i].start, prop[i].end, prop[i].pred);
        printf("\n");
    }
}

int main()
{
    int vertices = 6;
    vector<int> rowPtr{0, 1, 2, 4, 5, 6, 7};
    vector<int> dest{3, 3, 4, 5, 4, 1, 4};

    DFS(rowPtr, dest, vertices);
}