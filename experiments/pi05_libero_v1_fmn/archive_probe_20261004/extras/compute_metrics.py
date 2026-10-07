#!/usr/bin/env python3
"""Recompute the three numbers of the baseline table from the matrix.json files next to this script.
    python3 compute_metrics.py            # prints a table
    python3 compute_metrics.py --json     # same numbers as JSON
matrix.json holds M[i][j] = success of task j (0-based) measured right after task i was learned (50 rollouts), for j <= i.
  learned (plasticity) = mean_j M[j][j]                         how well each task is learned
  final                = mean_j M[9][j]                         average over all 10 tasks after the last one
  NBT                  = mean_{j<9} mean_{i>j} (M[j][j]-M[i][j])  negative backward transfer: the drop from "just learned",
                                                                averaged over every later evaluation (lower is better)
  learned-final        = mean_{j<9} (M[j][j]-M[9][j])            the drop at the very end only (used in the 2026-10-05 figures)
Standard library only (works with the system python on tasl-labserver).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("sf", "sequential fine-tuning"), ("er", "experience replay")]
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]

def metrics(path):
    M = json.load(open(path)); n = len(M); assert n == 10, (path, n)
    A = [[M[str(i)][str(j)] for j in range(i + 1)] for i in range(n)]
    learned = sum(A[j][j] for j in range(n)) / n
    final = sum(A[n - 1]) / n
    nbt = sum(sum(A[j][j] - A[i][j] for i in range(j + 1, n)) / (n - 1 - j) for j in range(n - 1)) / (n - 1)
    drop = sum(A[j][j] - A[n - 1][j] for j in range(n - 1)) / (n - 1)
    return dict(learned=round(learned, 3), final=round(final, 3), nbt=round(nbt, 3), learned_minus_final=round(drop, 3), min_learned=min(A[j][j] for j in range(n)))

out = {m: {s: metrics(os.path.join(HERE, m, s, "matrix.json")) for s, _ in SUITES} for m, _ in METHODS}
if "--json" in sys.argv:
    print(json.dumps(out, indent=1))
else:
    print("%-24s %-8s %8s %8s %8s %15s" % ("method", "suite", "learned", "final", "NBT", "learned-final"))
    for m, mn in METHODS:
        for s, sn in SUITES:
            r = out[m][s]; print("%-24s %-8s %8.3f %8.3f %8.3f %15.3f" % (mn, sn, r["learned"], r["final"], r["nbt"], r["learned_minus_final"]))
