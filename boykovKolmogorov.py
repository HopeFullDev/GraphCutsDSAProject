import numpy as np
import queue

# BFS function to grow the S-tree or T-tree
def bfs_growth(rGraph, V, s, t, s_parent, t_parent):
    s_queue = queue.Queue()
    t_queue = queue.Queue()

    s_visited = [-1] * V
    t_visited = [-1] * V

    s_queue.put(s)
    t_queue.put(t)
    s_visited[s] = s
    t_visited[t] = t

    while not s_queue.empty() or not t_queue.empty():
        # Expand S-tree
        if not s_queue.empty():
            u = s_queue.get()
            for v in range(V):
                if s_visited[v] == -1 and rGraph[u][v] > 0:  # Unvisited with positive capacity
                    s_visited[v] = u
                    s_queue.put(v)
                    s_parent[v] = u
                    if t_visited[v] != -1:  # Found intersection
                        return v, True
        
        # Expand T-tree
        if not t_queue.empty():
            u = t_queue.get()
            for v in range(V):
                if t_visited[v] == -1 and rGraph[v][u] > 0:  # Unvisited with positive reverse capacity
                    t_visited[v] = u
                    t_queue.put(v)
                    t_parent[v] = u
                    if s_visited[v] != -1:  # Found intersection
                        return v, True

    return -1, False

# Augment path and update the residual graph
def augment_path(rGraph, s, t, s_parent, t_parent, intersection):
    path_flow = float("inf")
    v = intersection
    
    # Trace path from intersection to source
    while v != s:
        u = s_parent[v]
        path_flow = min(path_flow, rGraph[u][v])
        v = u

    # Trace path from intersection to sink
    v = intersection
    while v != t:
        u = t_parent[v]
        path_flow = min(path_flow, rGraph[v][u])
        v = u

    # Update residual graph (source to intersection)
    v = intersection
    while v != s:
        u = s_parent[v]
        rGraph[u][v] -= path_flow
        rGraph[v][u] += path_flow
        v = u

    # Update residual graph (intersection to sink)
    v = intersection
    while v != t:
        u = t_parent[v]
        rGraph[v][u] -= path_flow
        rGraph[u][v] += path_flow
        v = u

    return path_flow

# Boykov-Kolmogorov Max Flow and Min Cut
def boykovKolmogorov(graph, s, t):
    print("Running Boykov-Kolmogorov Algorithm")
    rGraph = graph.copy()
    V = len(graph)
    max_flow = 0

    s_parent = [-1] * V
    t_parent = [-1] * V

    while True:
        s_parent = [-1] * V
        t_parent = [-1] * V
        intersection, found = bfs_growth(rGraph, V, s, t, s_parent, t_parent)
        
        if not found:
            break  # No more augmenting paths
        
        # Augment the flow along the found path
        max_flow += augment_path(rGraph, s, t, s_parent, t_parent, intersection)

    # Find the min-cut
    visited = [False] * V
    bfs_marking(rGraph, s, visited)
    cuts = []

    for i in range(V):
        for j in range(V):
            if visited[i] and not visited[j] and graph[i][j] > 0:
                cuts.append((i, j))

    return cuts

# BFS to mark reachable nodes from source
def bfs_marking(rGraph, s, visited):
    queue_ = queue.Queue()
    visited[s] = True
    queue_.put(s)

    while not queue_.empty():
        u = queue_.get()
        for v in range(len(rGraph)):
            if not visited[v] and rGraph[u][v] > 0:
                visited[v] = True
                queue_.put(v)

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

    min_cut_edges = boykovKolmogorov(np.asarray(graph), s, t)
    
    print("Min-Cut Edges:", min_cut_edges)
