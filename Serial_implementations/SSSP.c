#include <stdio.h>
#include <stdlib.h>

#define N 5
#define inf 100000

// Data structure to store a graph object
struct Graph
{
    // An array of pointers to Node to represent an adjacency list
    struct Node* head[N];
};

// Data structure to store adjacency list nodes of the graph
struct Node
{
    int dest, weight;
    struct Node* next;
};

// Data structure to store a graph edge
struct Edge {
    int src, dest, weight;
};

// Function to create an adjacency list from specified edges
struct Graph* createGraph(struct Edge edges[], int n)
{
    // allocate storage for the graph data structure
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));

    // initialize head pointer for all vertices
    for (int i = 0; i < N; i++) {
        graph->head[i] = NULL;
    }

    // add edges to the directed graph one by one
    for (int i = 0; i < n; i++)
    {
        // get the source and destination vertex
        int src = edges[i].src;
        int dest = edges[i].dest;
        int weight = edges[i].weight;

        // allocate a new node of adjacency list from src to dest
        struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
        newNode->dest = dest;
        newNode->weight = weight;

        // point new node to the current head
        newNode->next = graph->head[src];

        // point head pointer to the new node
        graph->head[src] = newNode;
    }

    return graph;
}

// Function to print adjacency list representation of a graph
void printGraph(struct Graph* graph)
{
    for (int i = 0; i < N; i++)
    {
        // print current vertex and all its neighbors
        struct Node* ptr = graph->head[i];
        while (ptr != NULL)
        {
            printf("%d —> %d (%d)\t", i, ptr->dest, ptr->weight);
            ptr = ptr->next;
        }

        printf("\n");
    }
}

void SSSP(struct Graph *graph, int src) {
    int dist[N], pred[N];
    for (int i = 0; i < N; i++) {
        dist[i] = inf;
        pred[i] = -1;
    }

    dist[src] = 0;
    while(1) {
        int changed = 0;
        for (int i = 0; i < N; i++) {
            struct Node *ptr = graph->head[i];
            while (ptr != NULL) {
                int t = ptr->dest;
                int wt = ptr->weight;
                if (dist[t] > (dist[i] + wt)) {
                    dist[t] = dist[i] + wt;
                    pred[t] = i;
                    changed = 1;
                }
                ptr = ptr->next;
            }
        }
        if (changed == 0) break;
    }

    for(int i = 0; i < N; i++) {
        printf("For node %d\n", i);
        printf("Distance from src = %d, Pred = %d\n", dist[i], pred[i]);
        printf("\n");
    }
}

// Weighted Directed graph implementation in C
int main(void)
{
    // input array containing edges of the graph (as per the above diagram)
    // (x, y, w) tuple represents an edge from x to y having weight `w`
    struct Edge edges[] =
            {
                    {0, 1, 5}, {0, 4, 100}, {1, 4, 10}, {1, 2, 80}, {1, 3, 15}, {2, 3, 18}, {4, 2, 40}
            };

    // calculate the total number of edges
    int n = sizeof(edges)/sizeof(edges[0]);

    // construct a graph from the given edges
    struct Graph *graph = createGraph(edges, n);

    // Function to print adjacency list representation of a graph
    printGraph(graph);

    SSSP(graph, 0);

    return 0;
}