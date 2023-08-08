#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

struct Edge
{
    int source;
    int destination;
};

struct weighEdge
{
    int source;
    int destination;
    int weight;
};

int main()
{
    ifstream fin("file.txt");
    int n, m, directed, weighted;
    fin >> n >> m >> directed >> weighted;
    if (weighted == 0)
    {
        vector<Edge> edges;
        for (int i = 0; i < m; i++)
        {
            int u, v;
            fin >> u >> v;
            edges.push_back(Edge{u, v});
        }

        // Create a graph object.
        vector<vector<int>> graph(n);
        for (Edge edge : edges)
        {
            graph[edge.source].push_back(edge.destination);
            if (!directed)
            {
                graph[edge.destination].push_back(edge.source);
            }
        }

        // Print the graph.
        for (int i = 0; i < n; i++)
        {
            cout << i << ": ";
            for (int j : graph[i])
            {
                cout << j << " ";
            }
            cout << endl;
        }

        vector<int> rowPtr;
        vector<int> dest;
        int prev = 0;
        for (int i = 0; i < n; i++)
        {
            rowPtr.push_back(prev);
            for (int j : graph[i])
            {
                dest.push_back(j);
                prev++;
            }
        }

        for (int i : rowPtr)
        {
            cout << i << " ";
        }
        cout << endl;

        for (int i : dest)
        {
            cout << i << " ";
        }
        cout << endl;
    }
    else
    {
        vector<weighEdge> edges;
        for (int i = 0; i < m; i++)
        {
            int u, v, w;
            fin >> u >> v >> w;
            edges.push_back(weighEdge{u, v, w});
        }

        vector<vector<weighEdge>> graph(n);
        for (weighEdge edge : edges)
        {
            graph[edge.source].push_back(edge);
            if (!directed)
            {
                weighEdge e = {edge.destination, edge.source, edge.weight};
                graph[edge.destination].push_back(e);
            }
        }

        for (int i = 0; i < n; i++)
        {
            cout << i << ": ";
            for (weighEdge j : graph[i])
            {
                cout << j.destination << "(" << j.weight << ")"
                     << " ";
            }
            cout << endl;
        }

        vector<int> rowPtr;
        vector<int> dest;
        vector<int> weights;
        int prev = 0;
        for (int i = 0; i < n; i++)
        {
            rowPtr.push_back(prev);
            for (weighEdge j : graph[i])
            {
                dest.push_back(j.destination);
                weights.push_back(j.weight);
                prev++;
            }
        }

        for (int i : rowPtr)
        {
            cout << i << " ";
        }
        cout << endl;

        for (int i : dest)
        {
            cout << i << " ";
        }
        cout << endl;

        for (int i : weights)
        {
            cout << i << " ";
        }
        cout << endl;
    }

    return 0;
}
