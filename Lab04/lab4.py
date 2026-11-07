"""Lab 4: Informed Search - Heuristics and A* (single-file version).
f(n) = g(n) + h(n). Edge weights and h values are abstract travel-time units
(supplied by the lab manual); they are not measured campus distances.
Queue tie-breaking: smaller f, then smaller g, then node name.
Counting convention (A* and UCS): a node is EXPANDED only when its successors are
examined. Goal processing is logged but not counted as an expansion. Stale queue
entries are ignored and counted separately.
Run: python lab4.py
"""
import heapq

graph = {"S": [("A", 1), ("B", 4)], "A": [("C", 2)], "B": [("G", 5)],
         "C": [("G", 3)], "G": []}
h_given = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}


def astar(g_, h, start, goal):
    """Return dict with path, cost, trace, expansions, stale.
    trace rows: node, g, h, f, status, frontier_after (list of (node, g, h, f))."""
    pq = [(h[start], 0, start, [start])]
    best = {start: 0}
    trace, expansions, stale = [], 0, 0
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if g != best[node]:                      # stale entry: a cheaper path was found later
            stale += 1
            trace.append(dict(node=node, g=g, h=h[node], f=f, status="stale (ignored)", frontier=[]))
            continue
        if node == goal:
            front = [(p[2], p[1], h[p[2]], p[0]) for p in sorted(pq)]
            trace.append(dict(node=node, g=g, h=h[node], f=f, status="goal (not counted)", frontier=front))
            return dict(path=path, cost=g, trace=trace, expansions=expansions, stale=stale)
        expansions += 1
        for nxt, w in g_.get(node, []):
            ng = g + w
            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                heapq.heappush(pq, (ng + h[nxt], ng, nxt, path + [nxt]))
        front = [(p[2], p[1], h[p[2]], p[0]) for p in sorted(pq)]
        trace.append(dict(node=node, g=g, h=h[node], f=f, status="expanded", frontier=front))
    return dict(path=None, cost=None, trace=trace, expansions=expansions, stale=stale)


def path_cost(path, g_=graph):
    """Independent verification: sum edge weights along the path."""
    return sum(dict(g_[a])[b] for a, b in zip(path, path[1:]))


def true_remaining(g_, goal):
    """True cheapest cost from every node to goal (Dijkstra on the reversed graph)."""
    rev = {n: [] for n in g_}
    for u, edges in g_.items():
        for v, w in edges:
            rev[v].append((u, w))
    dist = {goal: 0}
    pq = [(0, goal)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist.get(n, float("inf")):
            continue
        for u, w in rev[n]:
            if d + w < dist.get(u, float("inf")):
                dist[u] = d + w
                heapq.heappush(pq, (d + w, u))
    return dist


def audit(g_, h, goal):
    """Return (admissibility rows, consistency rows)."""
    true = true_remaining(g_, goal)
    adm = [(n, h[n], true[n], h[n] <= true[n]) for n in g_ if n in true]
    cons = [(u, v, w, h[u], w + h[v], h[u] <= w + h[v]) for u, es in g_.items() for v, w in es]
    return adm, cons


def show(title, res, g_=graph):
    print(f"\n--- {title} ---")
    for i, r in enumerate(res["trace"], 1):
        fr = [f"{n}(g={g},h={hh},f={f})" for n, g, hh, f in r["frontier"]]
        print(f"  {i}. {r['node']}  g={r['g']} h={r['h']} f={r['f']}  [{r['status']}]  frontier after: {fr}")
    print("  path:", res["path"], "| cost:", res["cost"],
          "| independent sum:", path_cost(res["path"], g_) if res["path"] else None,
          "| expansions:", res["expansions"], "| stale ignored:", res["stale"])


if __name__ == "__main__":
    print("=== Heuristic audit (supplied h) ===")
    adm, cons = audit(graph, h_given, "G")
    for n, hv, tv, ok in adm:
        print(f"  admissible? {n}: h={hv} <= true={tv} -> {ok}")
    for u, v, w, hu, rhs, ok in cons:
        print(f"  consistent? {u}->{v}: h({u})={hu} <= {w}+h({v}) = {rhs} -> {ok}")

    print("\n=== Task 1 and 2: A* (supplied h) vs UCS (h = 0) ===")
    a = astar(graph, h_given, "S", "G")
    show("A* with supplied heuristic", a)
    h_zero = {n: 0 for n in graph}
    u = astar(graph, h_zero, "S", "G")
    show("UCS control (all h = 0)", u)
    print(f"\nComparison: A* expansions = {a['expansions']}, UCS expansions = {u['expansions']}, "
          f"same cost = {a['cost'] == u['cost']}")

    print("\n=== Task 3: overestimate h(C) = 10 (true remaining cost of C is 3) ===")
    h_over = dict(h_given, C=10)
    adm2, cons2 = audit(graph, h_over, "G")
    print("  admissible violations:", [(n, hv, tv) for n, hv, tv, ok in adm2 if not ok])
    print("  consistency violations:", [(x[0], x[1]) for x in cons2 if not x[5]])
    o = astar(graph, h_over, "S", "G")
    show("A* with h(C) = 10", o)
    print(f"\nOriginal: cost {a['cost']}, path {a['path']}, expansions {a['expansions']}")
    print(f"Modified: cost {o['cost']}, path {o['path']}, expansions {o['expansions']}")

    print("\n=== Supplementary: overestimate h(B) = 10 (true remaining cost of B is 5) ===")
    h_over_b = dict(h_given, B=10)
    ob = astar(graph, h_over_b, "S", "G")
    print(f"  cost {ob['cost']}, path {ob['path']}, expansions {ob['expansions']} (result unchanged here)")

    print("\n=== Edge cases ===")
    e1 = astar(graph, h_given, "S", "S")
    print("Start equals goal:", e1["path"], "cost", e1["cost"], "expansions", e1["expansions"])
    g2 = dict(graph, Z=[])
    e2 = astar(g2, dict(h_given, Z=0), "S", "Z")
    print("Unreachable goal Z:", e2["path"], "expansions", e2["expansions"])
