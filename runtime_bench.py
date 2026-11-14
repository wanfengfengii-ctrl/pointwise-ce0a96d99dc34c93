"""Runtime benchmark confirming the allowed O((n+m+1)(B+1) log(n+m+2)) bound.

Measures wall-clock time at fixed B while n+m grows, and at fixed size while B
grows, then prints the observed scaling ratios.  Standard library only.

Run:  python3 -I runtime_bench.py
"""

import random
import time

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align


def _channels(n, m, seed):
    rng = random.Random(seed)
    left = [rng.randint(-1000, 1000) for _ in range(n)]
    right = [rng.randint(-1000, 1000) for _ in range(m)]
    return left, right


def main():
    import math

    # Fixed B, growing total length: expect roughly linearithmic growth.
    print("fixed B=32, growing length")
    print("%-12s %-10s %-12s %-10s" % ("n+m", "seconds", "t / (N log N)", "cost"))
    rows = []
    for n in (250, 500, 1000, 2000, 4000):
        left, right = _channels(n, n - 2, seed=n)
        t0 = time.perf_counter()
        res = align.solve(left, right, 32, 5, 2)
        dt = time.perf_counter() - t0
        total = n + (n - 2)
        norm = dt / (total * math.log(total + 2))
        rows.append((total, dt, norm, res["cost"]))
        print("%-12d %-10.4f %-12.2e %-10d" %
              (total, dt, norm, res["cost"]))
    # Normalised t/(N log N) should stay in a modest band (no super-linear blow-up).
    norms = [r[2] for r in rows]
    assert max(norms) < min(norms) * 6, "runtime grows faster than N log N"

    print()
    print("fixed n+m ~ 4000, growing B")
    print("%-8s %-10s %-12s" % ("B", "seconds", "t / (B+1)"))
    brows = []
    for B in (16, 32, 64, 128):
        left, right = _channels(2000, 1998, seed=500 + B)
        t0 = time.perf_counter()
        res = align.solve(left, right, B, 5, 2)
        dt = time.perf_counter() - t0
        brows.append((B, dt, dt / (B + 1), res["cost"]))
        print("%-8d %-10.4f %-12.2e" % (B, dt, dt / (B + 1)))
    bnorms = [r[2] for r in brows]
    assert max(bnorms) < min(bnorms) * 4, "runtime grows faster than linear in B"
    print("RUNTIME BENCHMARK PASSED")


if __name__ == "__main__":
    main()
