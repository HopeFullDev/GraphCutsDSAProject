import numpy as np
import queue

def bfs(rGraph, V, s, t, parent):
    q = queue.Queue()
    visited = np.zeros(V, dtype=bool)
    q.put(s)
    visited[s] = True
    parent[s] = -1

    while not q.empty():
        u = q.get()
        for v in range(V):
            if not visited[v] and rGraph[u][v] > 0:
                q.put(v)
                parent[v] = u
                visited[v] = True
    return visited[t]

def dfs(rGraph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph[v][u] > 0])

def augmentingPath(graph, s, t):
    print("Running augmenting path algorithm")
    rGraph = graph.copy()
    V = len(graph)
    parent = np.zeros(V, dtype='int32')

    while bfs(rGraph, V, s, t, parent):
        pathFlow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            pathFlow = min(pathFlow, rGraph[u][v])
            v = parent[v]

        v = t
        while v != s:
            u = parent[v]
            rGraph[u][v] -= pathFlow
            rGraph[v][u] += pathFlow
            v = parent[v]

    visited = np.zeros(V, dtype=bool)
    dfs(rGraph, V, s, visited)

    cuts = []

    for i in range(V):
        for j in range(V):
            if visited[i] and not visited[j] and graph[i][j] > 0:
                cuts.append((i, j))
    return cuts

if __name__ == "__main__":
    graph = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]
    ]

    s = 0  # Source
    t = 5  # Sink

    print(augmentingPath(np.asarray(graph), s, t))
