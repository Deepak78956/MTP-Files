#include <stdio.h>
#include <stdlib.h>
#define N 8

int clk = 0;

struct Graph
{
    struct Node* head[N];
};

struct Node
{
    int dest;
    struct Node* next;
};

struct Edge {
    int src, dest;
};

struct props {
    int color, component, start, etime, pred;
};

struct Graph* createGraph(struct Edge edges[], int n)
{
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));

    for (int i = 0; i < N; i++) {
        graph->head[i] = NULL;
    }

    for (int i = 0; i < n; i++)
    {
        int src = edges[i].src;
        int dest = edges[i].dest;

        struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
        newNode->dest = dest;

        // point new node to the current head
        newNode->next = graph->head[src];

        // point head pointer to the new node
        graph->head[src] = newNode;

        newNode = (struct Node*)malloc(sizeof(struct Node));
        newNode->dest = src;

        // point new node to the current head
        newNode->next = graph->head[dest];

        // change head pointer to point to the new node
        graph->head[dest] = newNode;
    }

    return graph;
}

void printGraph(struct Graph* graph)
{
    for (int i = 0; i < N; i++)
    {
        struct Node* ptr = graph->head[i];
        while (ptr != NULL)
        {
            printf("(%d —> %d)\t", i, ptr->dest);
            ptr = ptr->next;
        }

        printf("\n");
    }
}

void Traverse(int p, struct Graph *graph, int cn, struct props *prop) {
    prop[p].start = ++clk;
    prop[p].component = cn;
    prop[p].color = 1;
    struct Node* ptr = graph->head[p];
    while (ptr != NULL)
    {
        int t = ptr->dest;
        if(prop[t].color == 0) {
            prop[t].pred = p;
            Traverse(t, graph, cn, prop);
        }
        ptr = ptr->next;
    }
    prop[p].color = 2;
    prop[p].etime = ++clk;
}

void DFS_CC(struct Graph *graph) {
    int cn = 0;
    struct props prop[N];
    for (int i = 0; i < N; i++) {
        prop[i].pred = -1;
        prop[i].color = 0;
        prop[i].component = 0;
        prop[i].start = -1;
        prop[i].etime = -1;
    }
    for (int i = 0; i < N; i++) {
        if (prop[i].color == 0) {
            Traverse(i, graph, cn, prop);
            cn += 1;
        }
    }

    for (int i = 0; i < N; i++) {
        printf("Node %d, component %d\n", i + 1, prop[i].component);
    }
}

int main(void)
{
    // input array containing edges of the graph (as per the above diagram)
    // (x, y) pair in the array represents an edge from x to y
    struct Edge edges[] =
            {
                    {0, 4},
                    {4,5},
                    {5, 1},
                    {2, 6},
                    {6, 3},
                    {3, 7},
            };

    // calculate the total number of edges
    int n = sizeof(edges)/sizeof(edges[0]);

    // construct a graph from the given edges
    struct Graph *graph = createGraph(edges, n);

    printGraph(graph);

    DFS_CC(graph);

    return 0;
}