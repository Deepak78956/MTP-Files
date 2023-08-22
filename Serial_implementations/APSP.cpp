#include <iostream>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include "make_csr.hpp"
#define inf 1000000

using namespace std;

int getWeight(vector<int> rowPtr, vector<int> col_ind, vector<int> weight)
{
}

void APSP(vector<int> rowPtr, vector<int> col_ind, vector<int> weights, int vertices)
{

    int A[vertices][vertices];
    int pred[vertices][vertices];

    for (int i = 0; i < vertices; i++)
    {
        for (int j = 0; j < vertices; j++)
        {
            A[i][j] = inf;
            pred[i][j] = -1;
        }
    }

    for (int v = 0; v < vertices; v++)
    {
        for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++)
        {
            A[v][p] = getWeight(rowPtr, col_ind, weights);
        }
        A[v][v] = 0;
        pred[v][v] = 0;
    }

    for (int k = 0; k < vertices; k++)
    {
        for (int i; i < vertices; i++)
        {
            for (int j; j < vertices; j++)
            {
                if (A[i][k] + A[k][j] < A[i][j])
                {
                    A[i][j] = A[i][k] + A[k][j];
                    pred[i][j] = k;
                }
            }
        }
    }

    for (int i = 0; i < vertices; i++)
    {
        for (int j = 0; j < vertices; j++)
        {
            cout << A[i][j] << "(" << pred[i][j] << ")"
                 << " ";
        }
        cout << endl;
    }
}