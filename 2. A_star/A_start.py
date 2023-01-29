import heapq

import time


class Node:
    def __init__(self, val, level, zero_idx, prev=None, heuristic=0):
        self.val = val
        self.level = level
        self.zero_idx = zero_idx
        if prev is not None:
            self.heuristic = heuristic + (dist[prev.zero_idx][goal_map[self.val[prev.zero_idx]]] - dist[self.zero_idx][
                goal_map[self.val[prev.zero_idx]]])
        else:
            self.heuristic = heuristic

    def make_children(self):
        children = []
        for adj in graph[self.zero_idx]:
            child_val = list(self.val)
            child_val[self.zero_idx] = self.val[adj]
            child_val[adj] = '0'
            child_zero_idx = adj
            children.append(Node(child_val, self.level + 1, child_zero_idx, self, self.heuristic))
        return children

    def __lt__(self, other):
        return self.level + self.heuristic < other.level + other.heuristic

    def __gt__(self, other):
        return self.level + self.heuristic > other.level + other.heuristic

    def __str__(self):
        return self.val + " zero_idx:" + str(self.zero_idx) + " g:" + str(self.level)

    def __repr__(self):
        return str(self.val)

    def _key(self):
        return ' '.join(self.val)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return self._key() == other._key()


def bfs(root):
    dist = [0] * n
    visited = [False] * n
    queue = [root]
    visited[root] = True
    while queue:
        curr = queue.pop(0)
        for adj in graph[curr]:
            if not visited[adj]:
                queue.append(adj)
                dist[adj] = dist[curr] + 1
                visited[adj] = True
    return dist


def a_star(start):
    fringe = [start]
    closed_list = set()
    heapq.heapify(fringe)
    while fringe:
        current = heapq.heappop(fringe)
        if current.val == goal_list:
            print(current.level)
            return
        closed_list.add(current)
        children = current.make_children()
        for child in children:
            if child not in closed_list:
                heapq.heappush(fringe, child)


n, m = map(int, input().split())
graph = {}
for _ in range(m):
    u, v = map(int, input().split())
    graph[u] = graph.get(u, []) + [v]
    graph[v] = graph.get(v, []) + [u]

goal = input()
goal_list = goal.split()
goal_map = {}  #
dist = [0] * n
for i in range(n):
    dist[i] = bfs(i)
    goal_map[goal_list[i]] = i
starting_heuristic = 0
for i in range(1, n):
    starting_heuristic += dist[i][goal_map[str(i)]]
start = Node(list(map(str, list(range(n)))), 0, 0, heuristic=starting_heuristic)
a_star(start)
