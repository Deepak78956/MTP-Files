#include <stdio.h>
#include <stdlib.h>
#define N 8

int clk = 0;

struct props
{
    int color, component, start, etime, pred;
};

void Traverse(int p, int *rowPtr, int *dest, int cn, struct props *prop)
{
    prop[p].start = ++clk;
    prop[p].component = cn;
    prop[p].color = 1;

    int ptr = rowPtr[p];
    while (ptr < rowPtr[p + 1])
    {
        int t = dest[ptr];
        if (t != -1)
        {
            if (prop[t].color == 0)
            {
                prop[t].pred = p;
                Traverse(t, rowPtr, dest, cn, prop);
            }
        }
        ptr++;
    }
    prop[p].color = 2;
    prop[p].etime = ++clk;
}

void DFS_CC(int *rowPtr, int *dest)
{
    int cn = 0;
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

int main(void)
{
    int rowPtr[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int dest[8] = {4, -1, 6, 7, 5, 1, 3, -1};

    DFS_CC(rowPtr, dest);

    return 0;
}