"""Lab 3: Uninformed Search - BFS, DFS, DLS, IDDFS and UCS (single-file version).
State space: nodes = locations, actions = corridors, goal test = node == goal,
path cost = number of edges (BFS/DFS/DLS/IDDFS) or sum of weights (UCS).
Successor-list order is the tie-breaking order. Depth counts edges (A = depth 0).
Counts include the goal node when it is processed.
Edge weights are synthetic lab assumptions (non-negative travel-cost units).
Run: python lab3.py
"""
import heapq
from collections import deque

# ---------- Original graph (Section 3.2) ----------
graph = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"],
         "D": [], "E": ["G"], "F": [], "G": []}

# ---------- Weighted extension (Section 3.7): C -> G added ----------
wgraph = {"A": [("B", 1), ("C", 5)],
          "B": [("D", 2), ("E", 1)],
          "C": [("F", 3), ("G", 7)],
          "D": [], "E": [("G", 1)], "F": [], "G": []}


# ---------- BFS and DFS with traces ----------
def bfs(start, goal, g=graph):
    """Return (path, visit_order, trace). trace rows: (processed, path, frontier_after)."""
    q = deque([(start, [start])])
    seen = {start}
    order, trace = [], []
    while q:
        node, path = q.popleft()
        order.append(node)
        if node == goal:
            trace.append((node, path, [n for n, _ in q]))
            return path, order, trace
        for n in g[node]:
            if n not in seen:
                seen.add(n)
                q.append((n, path + [n]))
        trace.append((node, path, [n for n, _ in q]))
    return None, order, trace


def dfs(start, goal, g=graph):
    """Stack-based DFS. Neighbours are pushed reversed so the first listed is popped first."""
    stack = [(start, [start])]
    seen = set()
    order, trace = [], []
    while stack:
        node, path = stack.pop()
        if node in seen:
            continue
        order.append(node)
        if node == goal:
            trace.append((node, path, [n for n, _ in stack]))
            return path, order, trace
        seen.add(node)
        for n in reversed(g[node]):
            stack.append((n, path + [n]))
        trace.append((node, path, [n for n, _ in stack]))
    return None, order, trace


# ---------- DLS and IDDFS (path-local cycle check) ----------
def dls(node, goal, limit, g=graph, path=None, counter=None):
    """Return (status, path). status is FOUND, CUTOFF or FAILURE."""
    path = path or [node]
    if counter is not None:
        counter[0] += 1                       # node processed
    if node == goal:
        return "FOUND", path
    if limit == 0:
        return ("CUTOFF", None) if g[node] else ("FAILURE", None)
    cutoff = False
    for n in g[node]:
        if n in path:                         # path-local cycle check (no global visited set)
            continue
        status, result = dls(n, goal, limit - 1, g, path + [n], counter)
        if status == "FOUND":
            return status, result
        if status == "CUTOFF":
            cutoff = True
    return ("CUTOFF", None) if cutoff else ("FAILURE", None)


def iddfs(start, goal, max_depth, g=graph):
    """Repeat DLS with limits 0..max_depth. Return (status, path, limits_tried, total_nodes_processed)."""
    tried, counter = [], [0]
    status, path = "FAILURE", None
    for limit in range(max_depth + 1):
        tried.append(limit)
        status, path = dls(start, goal, limit, g, counter=counter)
        if status == "FOUND":
            return status, path, tried, counter[0]
    return status, path, tried, counter[0]


# ---------- UCS ----------
def ucs(start, goal, g=wgraph):
    """Return (path, cost, processing_order) where order rows are (node, g_cost)."""
    frontier = [(0, 0, start, [start])]       # (cost, tie-breaker, node, path)
    counter, done, order = 1, set(), []
    while frontier:
        cost, _, node, path = heapq.heappop(frontier)
        if node in done:
            continue
        done.add(node)
        order.append((node, cost))
        if node == goal:
            return path, cost, order
        for n, w in g[node]:
            if n not in done:
                heapq.heappush(frontier, (cost + w, counter, n, path + [n]))
                counter += 1
    return None, None, order


def path_cost(path, g=wgraph):
    """Independent verification: sum edge weights along a path."""
    total = 0
    for a, b in zip(path, path[1:]):
        total += dict(g[a])[b]
    return total


if __name__ == "__main__":
    print("=== Task 1: BFS and DFS (A -> G) ===")
    for name, fn in [("BFS", bfs), ("DFS", dfs)]:
        path, order, trace = fn("A", "G")
        print(f"\n{name}: path = {path} | visit order = {order}")
        for i, (node, p, fr) in enumerate(trace, 1):
            print(f"  step {i}: processed {node} | path {p} | frontier after {fr}")

    print("\n=== Task 2: DLS and IDDFS (limits 1-4) ===")
    for limit in [1, 2, 3, 4]:
        status, path = dls("A", "G", limit)
        print(f"DLS limit {limit}: {status} | path {path}")
    for md in [1, 2, 3, 4]:
        status, path, tried, work = iddfs("A", "G", md)
        print(f"IDDFS max depth {md}: {status} | path {path} | limits tried {tried} | nodes processed {work}")

    print("\n=== Task 3: UCS on weighted graph (C -> G added) ===")
    for node, edges in wgraph.items():
        print(f"  {node}: {edges}")
    path, cost, order = ucs("A", "G")
    print("UCS path:", path, "| cost:", cost)
    print("UCS processing order (node, g):", order)
    print("Independent sum of UCS path weights:", path_cost(path))
    bpath, border, _ = bfs("A", "G", {k: [n for n, _ in v] for k, v in wgraph.items()})
    print("BFS path:", bpath, "| hops:", len(bpath) - 1, "| cost:", path_cost(bpath))
    print("UCS hops:", len(path) - 1, "| BFS visit order:", border)

    print("\n=== Edge cases ===")
    print("Start equals goal (BFS):", bfs("A", "A")[0], "| (DLS limit 0):", dls("A", "A", 0))
    print("Unreachable goal H (BFS):", bfs("A", "H", {**graph, "H": []})[0])
    print("Unreachable goal H (DLS limit 4):", dls("A", "H", 4, {**graph, "H": []}))
    cyc = {"A": ["B"], "B": ["C"], "C": ["A", "D"], "D": []}
    print("Cycle graph BFS A->D:", bfs("A", "D", cyc)[0])
    print("Cycle graph DFS A->D:", dfs("A", "D", cyc)[0])
    print("Cycle graph DLS limit 5 A->D:", dls("A", "D", 5, cyc))
    print("Cycle graph, goal absent (DLS limit 5):", dls("A", "Z", 5, {**cyc, "Z": []}))
