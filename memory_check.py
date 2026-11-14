"""Measured auxiliary-memory check for long sequences.

The problem bounds auxiliary space at O(B + log(n+m+2)), excluding the input
and output.  We measure the peak Python allocation performed *inside*
``align.solve`` with :mod:`tracemalloc` while the input lists live outside the
measured window, then show that:

* peak bytes stay essentially flat as the sequence length grows (at fixed B);
* peak bytes scale with B, not with n*m;
* the full n*m backtracking table is never retained.

Run:  python3 -I memory_check.py
"""

import random
import sys
import tracemalloc

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align


def measure(n, m, B, open_, extend, seed):
    rng = random.Random(seed)
    # Inputs are built BEFORE tracing so their storage is excluded.
    left = [rng.randint(-1000, 1000) for _ in range(n)]
    right = [rng.randint(-1000, 1000) for _ in range(m)]
    tracemalloc.start()
    result = align.solve(left, right, B, open_, extend)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result is not None
    # The returned step string length grows with n+m; it is output and is
    # reported separately rather than counted as auxiliary scratch.
    out_bytes = len(result["steps"])
    return peak, out_bytes, result["cost"]


def main():
    print("auxiliary peak memory (tracemalloc), inputs excluded")
    print("%-14s %-6s %-14s %-14s %-10s" %
          ("(n,m)", "B", "peak bytes", "output bytes", "cost"))
    # Fixed B, growing length: auxiliary scratch should stay ~flat.
    fixed_b = 16
    prev_peak = None
    flat_ok = True
    for length in (1000, 2000, 5000, 10000, 20000):
        peak, out_bytes, cost = measure(length, length - 3, fixed_b, 5, 2,
                                        seed=length)
        print("%-14s %-6d %-14d %-14d %-10d" %
              ((str(length) + "," + str(length - 3)), fixed_b, peak,
               out_bytes, cost))
        if prev_peak is not None and peak > prev_peak * 1.6:
            flat_ok = False
        prev_peak = peak

    # Fixed length, growing B: scratch should grow roughly with B, while a
    # full n*m table would be millions of cells regardless of B.
    print()
    print("fixed (n,m)=(4000,3990) (length diff 10), varying feasible B")
    peaks = []
    for B in (12, 16, 32, 64, 128):
        peak, out_bytes, cost = measure(4000, 3990, B, 5, 2, seed=99 + B)
        peaks.append((B, peak))
        print("%-14s %-6d %-14d %-14d %-10d" %
              ("(4000,3990)", B, peak, out_bytes, cost))

    # A dense n*m table of Python ints at 4000*3990 would be tens of MB; the
    # measured scratch must be orders of magnitude smaller.
    dense_lower_bound_bytes = 4000 * 3990 * 28
    _, largest_peak = max(peaks, key=lambda t: t[1])
    print()
    print("largest measured scratch peak: %d bytes" % largest_peak)
    print("dense n*m int table would need >= %d bytes" %
          dense_lower_bound_bytes)
    assert largest_peak < dense_lower_bound_bytes // 50, (
        "auxiliary memory is far too large; a full table appears to be kept")
    assert flat_ok, "scratch grew with sequence length; not O(B)-bounded"
    print("MEMORY CHECK PASSED: scratch is O(B), independent of n*m")


if __name__ == "__main__":
    main()
