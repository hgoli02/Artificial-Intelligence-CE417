class Node:
    def __init__(self, data):
        self.data = data
        self.evidence = False
        self.children = []
        self.parents = []
        self.edges = []

    def set_true(self):
        self.evidence = True


n, m, z = map(int, input().split())
Nodes = [Node(i) for i in range(1, n + 1)]
for i in range(m):
    a, b = map(int, input().split())
    Nodes[a - 1].children.append(b - 1)
    Nodes[b - 1].parents.append(a - 1)
    Nodes[a - 1].edges.append(b - 1)
    Nodes[b - 1].edges.append(a - 1)

evidence = []
for i in range(z):
    node = int(input()) - 1
    Nodes[node].set_true()
    evidence.append(node)

root, dest = map(int, input().split())

root = Nodes[root - 1]
dest = Nodes[dest - 1]
visited = [False for i in range(n)]

z_map = set()


def dfs_evidence(node):
    z_map.add(node.data)
    visited_evidence[node.data - 1] = True
    for i in node.parents:
        if not visited_evidence[i]:
            dfs_evidence(Nodes[i])


def dfs(root, dest, visited, trail=None, dir="start"):
    if trail is None:
        trail = []
    trail.append(root)
    visited[root.data - 1] = True
    if root == dest:
        print(", ".join([str(i.data) for i in trail]))
        exit()
    if dir == "start":
        for i in root.children:
            if not visited[i]:
                dfs(Nodes[i], dest, visited, trail, "down")
        for i in root.parents:
            if not visited[i]:
                dfs(Nodes[i], dest, visited, trail, "up")
    for edge in root.edges:
        if edge in root.children:
            if dir == "down" and not visited[edge] and not Nodes[root.data - 1].evidence:  # Casual Chain
                dfs(Nodes[edge], dest, visited, trail, "down")
        if edge in root.parents:
            if dir == "up" and not visited[edge] and not Nodes[root.data - 1].evidence:  # Reverse Casual Chain
                dfs(Nodes[edge], dest, visited, trail, "up")
        if edge in root.children:
            if dir == "up" and not visited[edge] and not Nodes[root.data - 1].evidence:  # Common Effect
                dfs(Nodes[edge], dest, visited, trail, "down")
        if edge in root.parents:
            if dir == "down" and not visited[edge] and root.data in z_map:  # V Structure
                dfs(Nodes[edge], dest, visited, trail, "up")
    trail.pop()
    visited[root.data - 1] = False



for i in evidence:
    visited_evidence = [False for j in range(n)]
    dfs_evidence(Nodes[i])
dfs(root, dest, visited)
print("independent")
