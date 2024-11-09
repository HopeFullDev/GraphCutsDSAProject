import numpy as np
import queue

# BFS to construct level graph
def bfs_level_graph(rGraph, V, s, t, level):
    q = queue.Queue()
    level[:] = -1  # Initialize levels as -1
    level[s] = 0
    q.put(s)

    while not q.empty():
        u = q.get()
        for v in range(V):
            if level[v] < 0 and rGraph[u][v] > 0:  # v is unvisited and there's capacity
                level[v] = level[u] + 1
                q.put(v)
    return level[t] >= 0  # Return true if we reached the sink

# DFS to send flow in blocking flow scenario
def dfs_blocking_flow(rGraph, V, s, t, flow, level, start):
    if s == t:
        return flow
    total_flow_sent = 0
    while start[s] < V:
        v = start[s]
        if level[v] == level[s] + 1 and rGraph[s][v] > 0:
            min_cap = min(flow, rGraph[s][v])
            sent_flow = dfs_blocking_flow(rGraph, V, v, t, min_cap, level, start)
            if sent_flow > 0:
                rGraph[s][v] -= sent_flow
                rGraph[v][s] += sent_flow
                total_flow_sent += sent_flow
                flow -= sent_flow
                if flow == 0:
                    break
        start[s] += 1
    return total_flow_sent

# Dinic's maximum flow and minimum cut algorithm
def dinic_algorithm(graph, s, t):
    print("Running Dinic's Algorithm")
    rGraph = graph.copy()
    V = len(graph)
    max_flow = 0
    level = np.zeros(V, dtype=int)
    
    # While there's a valid level graph
    while bfs_level_graph(rGraph, V, s, t, level):
        start = np.zeros(V, dtype=int)  # Start indices for DFS
        while True:
            flow_sent = dfs_blocking_flow(rGraph, V, s, t, float("inf"), level, start)
            if flow_sent == 0:
                break
            max_flow += flow_sent

    # After max-flow, identify the min-cut
    visited = np.zeros(V, dtype=bool)
    dfs(rGraph, V, s, visited)
    cuts = []
    for i in range(V):
        for j in range(V):
            if visited[i] and not visited[j] and graph[i][j] > 0:
                cuts.append((i, j))

    return cuts

# DFS to mark reachable nodes in the residual graph
def dfs(rGraph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph[v][u] > 0])

# Example usage
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

    min_cut_edges = dinic_algorithm(np.asarray(graph), s, t)
    print("Min-Cut Edges:", min_cut_edges)
