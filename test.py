
import cv2
import numpy as np
import os
import argparse
from math import exp, pow
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov

# np.set_printoptions(threshold=np.inf)
graphCutAlgo = {"ap": augmentingPath, 
                "pr": pushRelabel, 
                "bk": boykovKolmogorov}
SIGMA = 30
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 10
LOADSEEDS = False
# drawing = False

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plantSeed(image):

    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print(f"Planting {pixelType} seeds")
        global drawing
        drawing = False
        windowname = f"Plant {pixelType} seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while True:
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 10
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False
    
    paintSeeds(OBJ)
    paintSeeds(BKG)
    return seeds, image


# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    K = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image)
    makeTLinks(graph, seeds, K)
    return graph, seededImage

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c: # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K

def makeTLinks(graph, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                # graph[x][source] = K
                graph[SOURCE][x] = K
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K
                # graph[sink][x] = K
            # else:
            #     graph[x][source] = LAMBDA * regionalPenalty(image[i][j], BKG)
            #     graph[x][sink]   = LAMBDA * regionalPenalty(image[i][j], OBJ)

def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image

def imageSegmentation(imagefile, size=(30, 30), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    graph, seededImage = buildGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    global SOURCE, SINK
    SOURCE += len(graph) 
    SINK   += len(graph)
    
    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print("cuts:")
    print(cuts)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print(f"Saved image as {savename}")

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            f"Algorithm should be one of the following: {', '.join(graphCutAlgo.keys())}")

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()



import numpy as np
import queue

# Constants for source and sink
SOURCE = -2
SINK = -1

# BFS for level graph
def bfs_level_graph(rGraph, V, s, t, level):
    q = queue.Queue()
    visited = np.zeros(V, dtype=bool)
    q.put(s)
    level[s] = 0
    visited[s] = True

    while not q.empty():
        u = q.get()
        for v in range(V):
            if not visited[v] and rGraph[u][v] > 0:
                level[v] = level[u] + 1
                visited[v] = True
                q.put(v)
    
    return visited[t]

# DFS to send flow through level graph
def dfs_flow(rGraph, level, u, flow, t, start):
    if u == t:
        return flow
    while start[u] < len(rGraph):
        v = start[u]
        if level[v] == level[u] + 1 and rGraph[u][v] > 0:
            current_flow = min(flow, rGraph[u][v])
            temp_flow = dfs_flow(rGraph, level, v, current_flow, t, start)
            if temp_flow > 0:
                rGraph[u][v] -= temp_flow
                rGraph[v][u] += temp_flow
                return temp_flow
        start[u] += 1
    return 0

# Dinic's algorithm
def dinic_max_flow(graph, s, t):
    V = len(graph)
    rGraph = graph.copy()
    max_flow = 0
    level = np.full(V, -1, dtype=int)
    
    while bfs_level_graph(rGraph, V, s, t, level):
        start = np.zeros(V, dtype=int)
        flow = 0
        
        while True:
            current_flow = dfs_flow(rGraph, level, s, float("inf"), t, start)
            if current_flow == 0:
                break
            flow += current_flow
        
        max_flow += flow

    return max_flow, rGraph  # Return residual graph for cut extraction

# Extract the cut points from the residual graph
def extract_cut_points(graph, rGraph, source):
    V = len(graph)
    visited = np.zeros(V, dtype=bool)

    # Perform a DFS or BFS to find all nodes reachable from the source in the residual graph
    def dfs(u):
        visited[u] = True
        for v in range(V):
            if rGraph[u][v] > 0 and not visited[v]:
                dfs(v)

    dfs(source)

    # Now, find the cut edges (edges between visited and unvisited nodes with non-zero capacity)
    cut_edges = []
    for u in range(V):
        if visited[u]:
            for v in range(V):
                if not visited[v] and graph[u][v] > 0:  # Residual capacity > 0
                    cut_edges.append((u, v))

    return cut_edges

# Function to perform image segmentation using Dinic's algorithm
def image_segmentation_dinic(image, seeds, graph):
    # Construct the graph, T-links and N-links
    V = image.size + 2  # Number of vertices (including source and sink)
    graph = np.zeros((V, V), dtype='int32')

    # Set T-links based on the seeded image
    makeTLinks(graph, seeds)
    
    # Set N-links based on boundary penalties
    K = makeNLinks(graph, image)

    # Add source and sink capacities for the object and background seeds
    source = SOURCE
    sink = SINK

    # Run Dinic's max-flow algorithm
    max_flow, residual_graph = dinic_max_flow(graph, source, sink)

    # Extract cut points
    cut_edges = extract_cut_points(graph, residual_graph, source)

    # Return the cut edges (the segmentation boundary)
    return cut_edges

# Example usage
if __name__ == "__main__":
    args = parseArgs()
    # Assuming `image` and `seeds` are prepared earlier
    image = args.imagefile
    seeds , i = plantSeed(image)
    V = image.size + 2 
    graph = np.zeros((V, V), dtype=int)
    
    # Assuming you've created a graph, use Dinic's algorithm
    cut_edges = image_segmentation_dinic(image, seeds, graph)
    
    print("Cut points (edges):")
    for u, v in cut_edges:
        print(f"Edge from {u} to {v}")