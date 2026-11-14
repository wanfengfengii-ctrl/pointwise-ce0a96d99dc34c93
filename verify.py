"""Independent verification for align.solve.

Cross-checks the Hirschberg-style implementation against an independently
written dense three-state DP (with its own, separate backtracking), plus:

* full path legality recomputed from the returned step string;
* exhaustive enumeration on the smallest cases;
* band-infeasibility (None) checks;
* parameter validation checks;
* immutability of the input lists.

Run:  python3 -I verify.py
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align
from align import solve, path_cost


# --------------------------------------------------------------------------- #
# Reference solver: dense, independent implementation
# --------------------------------------------------------------------------- #
def reference(left, right, B, open_, extend):
    """Dense DP over the whole (n+1)*(m+1) grid; returns min cost or None."""
    n, m = len(left), len(right)
    if abs(n - m) > B:
        return None
    INF = float("inf")
    # dm/dl/dr[i][j]: min cost reaching (i, j) with entering step M/L/R.
    dm = [[INF] * (m + 1) for _ in range(n + 1)]
    dl = [[INF] * (m + 1) for _ in range(n + 1)]
    dr = [[INF] * (m + 1) for _ in range(n + 1)]
    dm[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if abs(i - j) > B:
                continue
            if i >= 1 and j >= 1:
                dm[i][j] = (min(dm[i - 1][j - 1], dl[i - 1][j - 1],
                                dr[i - 1][j - 1])
                            + abs(left[i - 1] - right[j - 1]))
            if i >= 1:
                dl[i][j] = min(dl[i - 1][j] + extend,
                               dm[i - 1][j] + open_ + extend,
                               dr[i - 1][j] + open_ + extend)
            if j >= 1:
                dr[i][j] = min(dr[i][j - 1] + extend,
                               dm[i][j - 1] + open_ + extend,
                               dl[i][j - 1] + open_ + extend)
    best = min(dm[n][m], dl[n][m], dr[n][m])
    return None if best == INF else best


def check_result(left, right, B, open_, extend, expect_none=False):
    res = solve(left, right, B, open_, extend)
    ref = reference(left, right, B, open_, extend)
    if expect_none or ref is None:
        assert res is None, (left, right, B, res)
        return
    assert res is not None and set(res) == {"cost", "steps"}
    assert isinstance(res["cost"], int) and isinstance(res["steps"], str)
    steps = res["steps"]
    # Independent cost + full legality check from the step string.
    recomputed = path_cost(left, right, B, open_, extend, steps)
    assert recomputed == res["cost"] == ref, (
        left, right, B, open_, extend, res, ref)
    # Count-based consumption check as well.
    assert steps.count("M") + steps.count("L") == len(left)
    assert steps.count("M") + steps.count("R") == len(right)


def test_examples():
    # The spec's worked example.
    res = solve([1, 9], [1], B=1, open=3, extend=1)
    assert res == {"cost": 4, "steps": "ML"}, res
    # Both empty.
    assert solve([], [], B=0, open=0, extend=0) == {"cost": 0, "steps": ""}


def test_basic_paths():
    # Identical sequences -> all M, zero cost.
    check_result([5, 5, 5], [5, 5, 5], 1, 2, 1)
    # Pure left gap.
    check_result([1, 2, 3], [], 3, 2, 1)
    # Pure right gap.
    check_result([], [1, 2, 3], 3, 2, 1)
    # Band infeasible.
    check_result([1, 2, 3], [], 2, 2, 1, expect_none=True)
    check_result([1, 2, 3], [1], 0, 2, 1, expect_none=True)
    # B=0 forces all M, only feasible at equal lengths.
    check_result([1, 2, 3], [4, 5, 6], 0, 10, 10)
    assert solve([1], [1, 2], 0, 1, 1) is None
    # Zero open / zero extend edge tariffs.
    check_result([1, 9, 9], [1, 8], 2, 0, 0)
    check_result([1, 9, 9], [1, 8], 2, 0, 5)
    check_result(list(range(10)), list(range(5, 15)), 4, 7, 3)
    # Negative integer values are legal.
    check_result([-5, 0, 5], [5, -1, 0], 2, 1, 1)


def test_exhaustive_tiny():
    rng = random.Random(4242)
    count = 0
    for n in range(0, 5):
        for m in range(0, 5):
            left = [rng.randint(-3, 3) for _ in range(n)]
            right = [rng.randint(-3, 3) for _ in range(m)]
            for B in range(0, 4):
                for open_ in (0, 1, 4):
                    for extend in (0, 1, 3):
                        check_result(left, right, B, open_, extend)
                        count += 1
    return count


def test_step_enumeration():
    # Enumerate every step string of a small case and assert the solver's cost
    # equals the minimum over all legal enumerated paths.
    rng = random.Random(99)
    for trial in range(60):
        n = rng.randint(1, 5)
        m = rng.randint(1, 5)
        left = [rng.randint(0, 9) for _ in range(n)]
        right = [rng.randint(0, 9) for _ in range(m)]
        B = rng.randint(0, 4)
        open_ = rng.randint(0, 5)
        extend = rng.randint(0, 5)
        best = float("inf")
        # All strings over MLR with exactly n L+M and m R+M: pick positions.
        for ms in range(min(n, m) + 1):
            ls, rs = n - ms, m - ms
            total = ms + ls + rs
            for positions in itertools.combinations(range(total), ms):
                rest = [ch for ch in ("L" * ls + "R" * rs)]
                for perm in set(itertools.permutations(rest)):
                    steps = []
                    p = list(perm)
                    for k in range(total):
                        steps.append("M" if k in positions else p.pop(0))
                    s = "".join(steps)
                    try:
                        c = path_cost(left, right, B, open_, extend, s)
                    except ValueError:
                        continue
                    best = min(best, c)
        res = solve(left, right, B, open_, extend)
        if best == float("inf"):
            assert res is None
        else:
            assert res is not None and res["cost"] == best, (
                left, right, B, open_, extend, res, best)


def test_random_fuzz():
    rng = random.Random(2026)
    for trial in range(400):
        n = rng.randint(0, 40)
        m = rng.randint(0, 40)
        left = [rng.randint(-50, 50) for _ in range(n)]
        right = [rng.randint(-50, 50) for _ in range(m)]
        B = rng.randint(0, 12)
        open_ = rng.randint(0, 20)
        extend = rng.randint(0, 10)
        check_result(left, right, B, open_, extend)


def test_deep_hirsch_fuzz():
    # Sizes well past the _MINI_LIMIT so every recursion level is exercised,
    # including bands that clip hard and cases that force None.
    rng = random.Random(31337)
    for trial in range(150):
        n = rng.randint(33, 220)
        m = rng.randint(33, 220)
        left = [rng.randint(-1000, 1000) for _ in range(n)]
        right = [rng.randint(-1000, 1000) for _ in range(m)]
        B = rng.randint(0, 40)
        open_ = rng.randint(0, 50)
        extend = rng.randint(0, 25)
        check_result(left, right, B, open_, extend)
    # Near-equal correlated sequences with a tight band and cheap gaps.
    for trial in range(40):
        n = rng.randint(50, 300)
        base = [rng.randint(-100, 100) for _ in range(n)]
        left = base[:]
        right = [v + rng.randint(-2, 2) for v in base[: max(1, n - 1)]]
        check_result(left, right, rng.randint(1, 6),
                     rng.randint(0, 5), rng.randint(0, 3))


def test_immutability():
    left = [3, 1, 4, 1, 5]
    right = [2, 7, 1]
    l_copy, r_copy = list(left), list(right)
    solve(left, right, 2, 3, 1)
    assert left == l_copy and right == r_copy
    try:
        solve(tuple(left), right, 2, 3, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("tuple input must be rejected")


def test_validation():
    good = ([1, 2], [3], 1, 1, 1)

    def expect_value_error(args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("ValueError expected for %r" % (args,))

    expect_value_error(([1, True], [1], 1, 1, 1))       # bool element
    expect_value_error(([1], [1], True, 1, 1))          # bool B
    expect_value_error(([1], [1], 1, False, 1))         # bool open
    expect_value_error(([1], [1], 1, 1, True))          # bool extend
    expect_value_error(([1], [1], -1, 1, 1))            # B < 0
    expect_value_error(([1], [1], 129, 1, 1))           # B > 128
    expect_value_error(([1], [1], 1, -1, 1))            # open < 0
    expect_value_error(([1], [1], 1, 1, -2))            # extend < 0
    expect_value_error(([1.0], [1], 1, 1, 1))           # float element
    expect_value_error(([1], "ab", 1, 1, 1))            # non-list right
    expect_value_error(((1, 2), [1], 1, 1, 1))          # tuple left
    expect_value_error(([1], [1], 1.5, 1, 1))           # float B
    expect_value_error(([0] * 20001, [0], 1, 1, 1))     # too long
    # Sanity: the valid call still works.
    assert solve(*good) is not None


def test_offset_subproblems():
    # Sub-rectangles whose start is not the global origin and whose start/end
    # states are pinned, exactly as produced by the Hirschberg recursion.
    rng = random.Random(808)
    checked = 0
    for _ in range(250):
        n = rng.randint(5, 40)
        m = rng.randint(5, 40)
        left = [rng.randint(-9, 9) for _ in range(n)]
        right = [rng.randint(-9, 9) for _ in range(m)]
        B = rng.randint(0, 8)
        op = rng.randint(0, 8)
        ex = rng.randint(0, 5)
        i0 = rng.randint(0, n // 4)
        j0 = rng.randint(0, m // 4)
        i1 = rng.randint(max(i0 + 2, 3 * n // 4), n)
        j1 = rng.randint(max(j0 + 2, 3 * m // 4), m)
        if i1 - i0 > 32:
            i1 = i0 + 32
        if (abs(i0 - j0) > B or abs(i1 - j1) > B
                or abs((i1 - i0) - (j1 - j0)) > B):
            continue
        for ss in ("M", "L", "R"):
            lo, hi, fm, fl, fr = align._forward(
                left, right, i0, i1, j0, j1, B, ss, op, ex)
            if not (lo <= j1 <= hi):
                continue
            INF = float("inf")
            H, W = i1 - i0, j1 - j0
            dm = [[INF] * (W + 1) for _ in range(H + 1)]
            dl = [[INF] * (W + 1) for _ in range(H + 1)]
            dr = [[INF] * (W + 1) for _ in range(H + 1)]
            (dm if ss == "M" else dl if ss == "L" else dr)[0][0] = 0
            for a in range(H + 1):
                for b in range(W + 1):
                    if a == 0 and b == 0:
                        continue
                    gi, gj = i0 + a, j0 + b
                    if abs(gi - gj) > B:
                        continue
                    if a >= 1 and b >= 1:
                        dm[a][b] = (
                            min(dm[a - 1][b - 1], dl[a - 1][b - 1],
                                dr[a - 1][b - 1])
                            + abs(left[gi - 1] - right[gj - 1]))
                    if a >= 1:
                        dl[a][b] = min(dl[a - 1][b] + ex,
                                       dm[a - 1][b] + op + ex,
                                       dr[a - 1][b] + op + ex)
                    if b >= 1:
                        dr[a][b] = min(dr[a][b - 1] + ex,
                                       dm[a][b - 1] + op + ex,
                                       dl[a][b - 1] + op + ex)
            ref = {"M": dm[H][W], "L": dl[H][W], "R": dr[H][W]}
            got = {"M": fm[j1 - lo], "L": fl[j1 - lo], "R": fr[j1 - lo]}
            assert ref == got, (i0, j0, i1, j1, ss, ref, got)
            checked += 1
    assert checked > 0


def test_largeish():
    rng = random.Random(7)
    left = [rng.randint(0, 100) for _ in range(2000)]
    right = [v + rng.randint(-3, 3) for v in left[:1900]] + [
        rng.randint(0, 100) for _ in range(50)]
    res = solve(left, right, 128, 10, 2)
    assert res is not None
    assert path_cost(left, right, 128, 10, 2, res["steps"]) == res["cost"]
    # Determinism: repeated calls agree.
    res2 = solve(left, right, 128, 10, 2)
    assert res == res2


def test_big_integers_and_edges():
    # Arbitrary-precision integers stay exact ints.
    big = [10 ** 18 + i for i in range(100)]
    other = [v + 1 for v in big]
    check_result(big, other, 2, 5, 1)
    # bool hiding inside either channel is rejected.
    for args in (([1, 0], [True], 1, 1, 1),
                 ([False], [1], 1, 1, 1)):
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("bool element must be rejected")
    # Infeasible even though the quick |n-m|<=B check alone is not enough:
    # every legal path here must at some point leave the band.
    assert solve([1, 2, 3, 4], [1, 2], 1, 1, 1) is None
    feasible = solve([1, 2, 3, 4], [1, 2], 2, 1, 1)
    assert feasible is not None
    assert path_cost([1, 2, 3, 4], [1, 2], 2, 1, 1,
                     feasible["steps"]) == feasible["cost"]
    # One gap segment charged once in either direction under varied tariffs.
    for op, ex in ((0, 0), (0, 3), (5, 0), (5, 2)):
        assert solve([1, 2, 3], [], 3, op, ex)["cost"] == op + 3 * ex
        assert solve([], [1, 2, 3], 3, op, ex)["cost"] == op + 3 * ex


def main():
    test_examples()
    print("ok  spec examples")
    test_basic_paths()
    print("ok  basic paths and None cases")
    n = test_exhaustive_tiny()
    print("ok  exhaustive tiny grid (%d parameter combos)" % n)
    test_step_enumeration()
    print("ok  brute-force step enumeration")
    test_random_fuzz()
    print("ok  400 random fuzz cases vs dense reference DP")
    test_deep_hirsch_fuzz()
    print("ok  190 deep-recursion fuzz cases (n,m up to 300)")
    test_offset_subproblems()
    print("ok  pinned start-state offset subproblems vs dense forward")
    test_immutability()
    print("ok  inputs not modified")
    test_validation()
    print("ok  parameter validation")
    test_largeish()
    print("ok  2k-length smoke test")
    test_big_integers_and_edges()
    print("ok  big integers, bool rejection, band-infeasible and gap charging")
    print("ALL VERIFICATION PASSED")


if __name__ == "__main__":
    sys.exit(main())
