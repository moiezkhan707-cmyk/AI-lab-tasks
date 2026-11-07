"""Lab 5: Adversarial Search - Minimax and Alpha-Beta Pruning (single-file version).
Complete binary game tree stored as a leaf list. Depth 0 is the root (MAX), layers
alternate MAX/MIN. Both algorithms count evaluated leaves (terminal-node reads).
Pruning condition: break when alpha >= beta. Utilities are synthetic lab values.
Run: python lab5.py
"""
import math
from itertools import product


# ---------- Minimax with a leaf counter ----------
def minimax(depth, index, max_turn, scores, target, counter):
    if depth == target:
        counter[0] += 1                       # leaf evaluated
        return scores[index]
    values = [minimax(depth + 1, index * 2 + i, not max_turn, scores, target, counter)
              for i in range(2)]
    return max(values) if max_turn else min(values)


# ---------- Alpha-beta with a leaf counter and a pruning log ----------
def alphabeta(depth, index, max_turn, scores, target, alpha, beta, counter, log=None):
    if depth == target:
        counter[0] += 1                       # leaf evaluated
        return scores[index]
    value = float("-inf") if max_turn else float("inf")
    for i in range(2):
        child = alphabeta(depth + 1, index * 2 + i, not max_turn, scores, target,
                          alpha, beta, counter, log)
        if max_turn:
            value = max(value, child)
            alpha = max(alpha, value)
        else:
            value = min(value, child)
            beta = min(beta, value)
        if alpha >= beta:
            if i == 0 and log is not None:    # a sibling subtree is skipped
                skipped = 2 ** (target - depth - 1)
                log.append(f"depth {depth}, node {index} ({'MAX' if max_turn else 'MIN'}): "
                           f"alpha={alpha}, beta={beta} -> prune right child ({skipped} leaf/leaves skipped)")
            break
    return value


def tree_depth(scores):
    return int(math.log2(len(scores)))


def run_both(scores, log=None):
    """Return (value, minimax_leaves, alphabeta_value, alphabeta_leaves)."""
    d = tree_depth(scores)
    c1, c2 = [0], [0]
    v1 = minimax(0, 0, True, scores, d, c1)
    v2 = alphabeta(0, 0, True, scores, d, float("-inf"), float("inf"), c2, log)
    return v1, c1[0], v2, c2[0]


def reorder(scores, flags):
    """Swap the two child subtrees at internal node k when flags[k] is True (heap numbering)."""
    def build(lo, hi, k):
        if hi - lo == 1:
            return [scores[lo]]
        mid = (lo + hi) // 2
        left, right = build(lo, mid, 2 * k + 1), build(mid, hi, 2 * k + 2)
        return right + left if flags[k] else left + right
    return build(0, len(scores), 0)


if __name__ == "__main__":
    # ===== Task 1: supplied tree (Section 5.1), padded to a complete binary tree =====
    raw = [3, 5, 2, 9, 0, 7]
    padded = raw + [0] * (2 ** math.ceil(math.log2(len(raw))) - len(raw))
    print("=== Task 1: supplied tree (Section 5.1) ===")
    print("Leaves after padding with 0:", padded, "| depth:", tree_depth(padded))
    layer2 = [max(padded[i], padded[i + 1]) for i in range(0, 8, 2)]
    layer1 = [min(layer2[i], layer2[i + 1]) for i in range(0, 4, 2)]
    root = max(layer1)
    print("Depth 2 (MAX) values:", layer2)
    print("Depth 1 (MIN) values:", layer1)
    print("Root (MAX) value    :", root, "-> best move: left" if layer1[0] >= layer1[1] else "-> best move: right")

    # ===== Task 2: leaf counters on both algorithms =====
    print("\n=== Task 2: leaf counters ===")
    for name, sc in [("Section 5.1 tree (padded)", padded),
                     ("Section 5.2 tree", [3, 5, 2, 9, 3, 5, 2, 9])]:
        v1, n1, v2, n2 = run_both(sc)
        print(f"{name}: leaves={sc}")
        print(f"  Minimax    : value={v1}, leaves evaluated={n1}")
        print(f"  Alpha-beta : value={v2}, leaves evaluated={n2}, same value={v1 == v2}")
    log = []
    run_both([3, 5, 2, 9, 3, 5, 2, 9], log)
    print("Pruning log (Section 5.2 tree):", log if log else "no pruning")
    log = []
    run_both(padded, log)
    print("Pruning log (Section 5.1 tree):", log if log else "no pruning")

    # ===== Task 3: reorder branches =====
    print("\n=== Task 3: reordering the Section 5.2 tree [3, 5, 2, 9, 3, 5, 2, 9] ===")
    base = [3, 5, 2, 9, 3, 5, 2, 9]
    results = []
    for flags in product([False, True], repeat=7):      # all 128 child-swap orderings
        sc = reorder(base, flags)
        v1, n1, v2, n2 = run_both(sc)
        results.append((n2, sc, v1, v2))
    values_equal = all(v1 == v2 == 5 for _, _, v1, v2 in results)
    counts = [r[0] for r in results]
    best = min(results, key=lambda r: r[0])
    worst = max(results, key=lambda r: r[0])
    print("Orderings tested:", len(results), "| all root values equal 5:", values_equal)
    print("Alpha-beta leaves: min =", min(counts), ", max =", max(counts), "(Minimax always 8)")
    print("Best ordering :", best[1], "->", best[0], "leaves")
    print("Worst ordering:", worst[1], "->", worst[0], "leaves")
    print("Distribution (leaves: how many orderings):", {k: counts.count(k) for k in sorted(set(counts))})
    for label, sc in [("original", base), ("all children swapped", reorder(base, [True] * 7)),
                      ("best", best[1]), ("worst", worst[1])]:
        v1, n1, v2, n2 = run_both(sc)
        print(f"  {label:20s} {sc} | minimax {v1}/{n1} | alpha-beta {v2}/{n2}")
    log = []
    run_both(best[1], log)
    print("Pruning log (best ordering):")
    for line in log:
        print("   ", line)

    # ===== Edge cases =====
    print("\n=== Edge cases ===")
    print("Single leaf [7]            :", run_both([7]))
    print("All equal [4,4,4,4]        :", run_both([4, 4, 4, 4]))
    print("Negative utilities         :", run_both([-3, -5, -2, -9]))
    print("Padding changes the answer : pad with 0 ->", run_both(raw + [0, 0])[0],
          "| pad with 100 ->", run_both(raw + [100, 100])[0])
