"""Banded affine-gap alignment with low auxiliary memory.

Public entry point: :func:`solve`.

Semantics
---------
Two sequences ``left`` (length n) and ``right`` (length m) are fully consumed by
a path of steps:

* ``M`` consumes one point from each side, costing ``|left[i] - right[j]|``;
* ``L`` consumes one point from the left only (a gap on the right);
* ``R`` consumes one point from the right only (a gap on the left).

Consecutive same-direction gap steps form one gap *segment*.  Each segment
costs ``open + extend * segment_length``; switching gap direction opens a new
segment.  Every visited cell ``(i, j)`` must satisfy ``|i - j| <= B``.

The function returns a globally minimum-cost alignment (one of them, on ties)
without ever retaining a full n*m (or n*band) backtracking table: a Hirschberg
recursion keeps only O(B) cells per sweep plus an O(log(n+m)) recursion stack.
"""

__all__ = ["solve"]

_INF = float("inf")

# A subproblem is solved directly with a small local DP once its left-side
# span is no larger than this.  The local DP table stays O(B) cells wide.
_MINI_LIMIT = 32


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _check_int(name, value):
    # bool is a subclass of int and is explicitly rejected.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("%s must be an integer" % name)


def _validate(left, right, B, open_, extend):
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("left and right must be lists")
    if len(left) > 20000 or len(right) > 20000:
        raise ValueError("sequence length must be <= 20000")
    for name, seq in (("left", left), ("right", right)):
        for v in seq:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError("%s must contain only integers" % name)
    _check_int("B", B)
    _check_int("open", open_)
    _check_int("extend", extend)
    if not 0 <= B <= 128:
        raise ValueError("B must be in 0..128")
    if open_ < 0 or extend < 0:
        raise ValueError("open and extend must be non-negative")


# --------------------------------------------------------------------------- #
# Band helpers
# --------------------------------------------------------------------------- #
def _j_lo(i, i0, j0, B):
    """Smallest j inside the band reachable from the subproblem's start cell."""
    return max(j0, i - B)


def _j_hi(i, j1, B):
    """Largest j inside the band that can still reach the subproblem's end."""
    return min(j1, i + B)


# --------------------------------------------------------------------------- #
# Forward sweep
# --------------------------------------------------------------------------- #
def _forward(left, right, i0, i1, j0, j1, B, start_state, open_, extend):
    """Minimum cost from (i0, j0) to every cell of row i1, split by the
    entering step type (M / L / R) at that cell.

    Only two rows (each O(B) cells) are kept.  Returns
    ``(lo, hi, f_m, f_l, f_r)`` where index ``j - lo`` holds the value for
    column ``j`` (infinity where unreachable).
    """
    m_lo = _j_lo(i0, i0, j0, B)
    m_hi = _j_hi(i0, j1, B)
    size = m_hi - m_lo + 1
    pm = [_INF] * size
    pl = [_INF] * size
    pr = [_INF] * size
    pm[j0 - m_lo] = 0 if start_state == "M" else _INF
    pl[j0 - m_lo] = 0 if start_state == "L" else _INF
    pr[j0 - m_lo] = 0 if start_state == "R" else _INF
    # The start row may extend rightward via R steps (gap on the left).
    for j in range(j0 + 1, m_hi + 1):
        k = j - m_lo
        best = _INF
        v = pr[k - 1]
        if v < _INF:
            best = v + extend
        v = pm[k - 1]
        if v < _INF:
            cand = v + open_ + extend
            if cand < best:
                best = cand
        v = pl[k - 1]
        if v < _INF:
            cand = v + open_ + extend
            if cand < best:
                best = cand
        pr[k] = best

    for i in range(i0 + 1, i1 + 1):
        lo = _j_lo(i, i0, j0, B)
        hi = _j_hi(i, j1, B)
        nsize = hi - lo + 1
        cm = [_INF] * nsize
        cl = [_INF] * nsize
        cr = [_INF] * nsize
        row_left = left[i - 1]
        for idx in range(nsize):
            j = lo + idx

            # M step (i-1, j-1) -> (i, j): any entering state.
            best = _INF
            if j - 1 >= m_lo and j - 1 <= m_hi:
                k = (j - 1) - m_lo
                match = abs(row_left - right[j - 1])
                for v in (pm[k], pl[k], pr[k]):
                    if v < _INF:
                        cand = v + match
                        if cand < best:
                            best = cand
            cm[idx] = best

            # L step (i-1, j) -> (i, j): continuation of an L segment or a
            # newly opened one.
            best = _INF
            if m_lo <= j <= m_hi:
                pidx = j - m_lo
                v = pl[pidx]
                if v < _INF:
                    best = v + extend
                v = pm[pidx]
                if v < _INF:
                    cand = v + open_ + extend
                    if cand < best:
                        best = cand
                v = pr[pidx]
                if v < _INF:
                    cand = v + open_ + extend
                    if cand < best:
                        best = cand
            cl[idx] = best

            # R step (i, j-1) -> (i, j): same row, previous column.
            best = _INF
            if j - 1 >= lo:
                k = (j - 1) - lo
                v = cr[k]
                if v < _INF:
                    best = v + extend
                v = cm[k]
                if v < _INF:
                    cand = v + open_ + extend
                    if cand < best:
                        best = cand
                v = cl[k]
                if v < _INF:
                    cand = v + open_ + extend
                    if cand < best:
                        best = cand
            cr[idx] = best

        pm, pl, pr, m_lo, m_hi = cm, cl, cr, lo, hi

    return m_lo, m_hi, pm, pl, pr


# --------------------------------------------------------------------------- #
# Backward sweep
# --------------------------------------------------------------------------- #
def _backward(left, right, i0, i1, j0, j1, B, end_state, open_, extend):
    """Minimum suffix cost from every cell of row i0 to (i1, j1).

    The three arrays are indexed by the cell's *entering* step state s:
    ``B_s(i, j)`` is the minimum cost of the steps strictly after cell
    ``(i, j)``, conditioned on that cell being entered via state ``s``.  The
    state is fixed, so a same-direction step always continues (``extend``) and
    a different-direction gap step always opens (``open + extend``):

    * B_M(i,j) = min( match + B_M(i+1,j+1),
                      open+extend + B_L(i+1,j),
                      open+extend + B_R(i,j+1) )
    * B_L(i,j) = min( match + B_M(i+1,j+1),
                      extend + B_L(i+1,j),
                      open+extend + B_R(i,j+1) )
    * B_R(i,j) = min( match + B_M(i+1,j+1),
                      open+extend + B_L(i+1,j),
                      extend + B_R(i,j+1) )

    ``end_state`` restricts the entering state accepted at the terminal cell
    (``None`` accepts any state).  Two O(B)-width rows are retained.
    """
    gap_open = open_ + extend

    # Terminal row i1.  Only the terminal cell is a sink, in accepted states.
    t_lo = _j_lo(i1, i0, j0, B)
    t_hi = _j_hi(i1, j1, B)
    size = t_hi - t_lo + 1
    bm = [_INF] * size
    bl = [_INF] * size
    br = [_INF] * size
    for col, name in ((bm, "M"), (bl, "L"), (br, "R")):
        if end_state is None or end_state == name:
            col[j1 - t_lo] = 0
    # On the terminal row only R moves remain; seed columns left of j1.
    for j in range(j1 - 1, t_lo - 1, -1):
        k = j - t_lo
        suffix_r = br[k + 1]
        if suffix_r < _INF:
            br[k] = suffix_r + extend
            bm[k] = suffix_r + gap_open
            bl[k] = suffix_r + gap_open

    for i in range(i1 - 1, i0 - 1, -1):
        lo = _j_lo(i, i0, j0, B)
        hi = _j_hi(i, j1, B)
        nsize = hi - lo + 1
        cm = [_INF] * nsize
        cl = [_INF] * nsize
        cr = [_INF] * nsize
        match_base_row = left[i]
        # R references (i, j+1) in this same row -> fill right-to-left.
        for idx in range(nsize - 1, -1, -1):
            j = lo + idx

            # M move to (i+1, j+1), entering M there.
            v_m = _INF
            nj = j + 1
            if t_lo <= nj <= t_hi:
                v = bm[nj - t_lo]
                if v < _INF:
                    v_m = v + abs(match_base_row - right[j])

            # L move to (i+1, j), entering L there.
            v_l_down = _INF
            if t_lo <= j <= t_hi:
                v_l_down = bl[j - t_lo]  # may stay INF

            # R move to (i, j+1), entering R there.
            v_r_side = _INF
            if j + 1 <= hi:
                v_r_side = cr[(j + 1) - lo]

            # B_M: M, or open an L, or open an R.
            best = v_m
            if v_l_down < _INF:
                cand = v_l_down + gap_open
                if cand < best:
                    best = cand
            if v_r_side < _INF:
                cand = v_r_side + gap_open
                if cand < best:
                    best = cand
            cm[idx] = best

            # B_L: M, or continue an L, or open an R.
            best = v_m
            if v_l_down < _INF:
                cand = v_l_down + extend
                if cand < best:
                    best = cand
            if v_r_side < _INF:
                cand = v_r_side + gap_open
                if cand < best:
                    best = cand
            cl[idx] = best

            # B_R: M, or open an L, or continue an R.
            best = v_m
            if v_l_down < _INF:
                cand = v_l_down + gap_open
                if cand < best:
                    best = cand
            if v_r_side < _INF:
                cand = v_r_side + extend
                if cand < best:
                    best = cand
            cr[idx] = best

        bm, bl, br, t_lo, t_hi = cm, cl, cr, lo, hi

    return t_lo, t_hi, bm, bl, br


# --------------------------------------------------------------------------- #
# Small subproblem: local DP retaining predecessors inside this subproblem only
# --------------------------------------------------------------------------- #
def _mini(left, right, i0, i1, j0, j1, B, start_state, end_state,
          open_, extend):
    """Exact DP for a small subproblem (i1 - i0 <= _MINI_LIMIT).

    Stores at most O((_MINI_LIMIT + 1) * (2B + 1)) cells, i.e. O(B), and
    returns the step string, or None when the requested end state cannot be
    reached.  Backtracking stays inside this subproblem's local table.
    """
    states = ("M", "L", "R")
    lo0 = _j_lo(i0, i0, j0, B)
    hi0 = _j_hi(i0, j1, B)
    prev = {}
    for s in states:
        prev[(j0, s)] = (0 if s == start_state else _INF, None)
    # The start row may extend rightward via R steps.
    for j in range(j0 + 1, hi0 + 1):
        best, src = _INF, None
        for s in states:
            v, _ = prev[(j - 1, s)]
            add = extend if s == "R" else open_ + extend
            cand = v + add
            if cand < best:
                best, src = cand, (j - 1, s)
        prev[(j, "R")] = (best, src)
        prev[(j, "M")] = (_INF, None)
        prev[(j, "L")] = (_INF, None)
    p_lo, p_hi = lo0, hi0

    rows = [(i0, lo0, hi0, prev)]
    for i in range(i0 + 1, i1 + 1):
        lo = _j_lo(i, i0, j0, B)
        hi = _j_hi(i, j1, B)
        cur = {}
        row_left = left[i - 1]
        for j in range(lo, hi + 1):
            # M
            best, src = _INF, None
            if j - 1 >= p_lo and (j - 1, "M") in prev:
                base = abs(row_left - right[j - 1])
                for s in states:
                    v, _ = prev[(j - 1, s)]
                    cand = v + base
                    if cand < best:
                        best, src = cand, (j - 1, s)
            cur[(j, "M")] = (best, src)

            # L
            best, src = _INF, None
            for s in states:
                if (j, s) in prev:
                    v, _ = prev[(j, s)]
                    add = extend if s == "L" else open_ + extend
                    cand = v + add
                    if cand < best:
                        best, src = cand, (j, s)
            cur[(j, "L")] = (best, src)

            # R
            best, src = _INF, None
            if j - 1 >= lo:
                for s in states:
                    v, _ = cur[(j - 1, s)]
                    add = extend if s == "R" else open_ + extend
                    cand = v + add
                    if cand < best:
                        best, src = cand, (j - 1, s)
            cur[(j, "R")] = (best, src)

        rows.append((i, lo, hi, cur))
        prev, p_lo, p_hi = cur, lo, hi

    end_val, end_src = _INF, None
    if end_state is None:
        for s in states:
            v, src = prev.get((j1, s), (_INF, None))
            if v < end_val:
                end_val, end_src, end_state = v, src, s
    else:
        end_val, end_src = prev.get((j1, end_state), (_INF, None))
    if end_val == _INF:
        return None

    # Backtrack locally.
    chars = []
    i = i1
    j = j1
    s = end_state
    while (i, j, s) != (i0, j0, start_state):
        cell = rows[i - i0][3]
        _, parent = cell[(j, s)]
        pj, ps = parent
        if s == "M":
            chars.append("M")
            i -= 1
            j = pj
        elif s == "L":
            chars.append("L")
            i -= 1
            # j unchanged
        else:
            chars.append("R")
            j = pj
        s = ps
    chars.reverse()
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Hirschberg-style recursion over the left-side cut
# --------------------------------------------------------------------------- #
def _hirsch(left, right, i0, i1, j0, j1, B, start_state, end_state,
            open_, extend):
    """Align the subproblem rectangle ``[i0, i1] x [j0, j1]``.

    ``start_state`` is the entering step type at (i0, j0); ``end_state`` the
    one required at (i1, j1).  Returns the step string or None.
    """
    # Pure horizontal/vertical base cases: a single same-direction gap run.
    if i0 == i1:
        if j0 == j1:
            return "" if start_state == end_state else None
        if end_state is not None and end_state != "R":
            return None
        return "R" * (j1 - j0)
    if j0 == j1:
        if end_state is not None and end_state != "L":
            return None
        return "L" * (i1 - i0)

    if i1 - i0 <= _MINI_LIMIT:
        return _mini(left, right, i0, i1, j0, j1, B, start_state,
                     end_state, open_, extend)

    cut = (i0 + i1) // 2

    flo, fhi, fm, fl, fr = _forward(left, right, i0, cut, j0, j1, B,
                                    start_state, open_, extend)
    blo, bhi, bm, bl, br = _backward(left, right, cut, i1, j0, j1, B,
                                     end_state, open_, extend)

    lo = max(flo, blo)
    hi = min(fhi, bhi)
    best = _INF
    best_j = None
    best_s = None
    fcols = (fm, fl, fr)
    bcols = (bm, bl, br)
    for s_idx in range(3):
        fcol = fcols[s_idx]
        bcol = bcols[s_idx]
        for j in range(lo, hi + 1):
            total = fcol[j - flo] + bcol[j - blo]
            if total < best:
                best = total
                best_j = j
                best_s = s_idx
    # Drop sweep arrays before recursing: only O(B) scratch exists at any time.
    del fm, fl, fr, bm, bl, br
    if best_j is None:
        return None
    state = ("M", "L", "R")[best_s]

    left_part = _hirsch(left, right, i0, cut, j0, best_j, B,
                        start_state, state, open_, extend)
    if left_part is None:
        return None
    right_part = _hirsch(left, right, cut, i1, best_j, j1, B, state,
                         end_state, open_, extend)
    if right_part is None:
        return None
    return left_part + right_part


# --------------------------------------------------------------------------- #
# Independent cost recomputation (also used by the external verifier)
# --------------------------------------------------------------------------- #
def path_cost(left, right, B, open_, extend, steps):
    """Recompute the cost of ``steps`` from scratch and check all constraints.

    Raises ValueError if the path does not fully consume both sequences inside
    the band, otherwise returns the integer cost.
    """
    i = j = 0
    n, m = len(left), len(right)
    cost = 0
    run_char = None
    run_len = 0

    def flush():
        nonlocal cost, run_len
        if run_char is not None:
            cost += open_ + extend * run_len
            run_len = 0

    for ch in steps:
        if ch == "M":
            flush()
            run_char = None
            if i >= n or j >= m:
                raise ValueError("path overruns a sequence")
            cost += abs(left[i] - right[j])
            i += 1
            j += 1
        elif ch == "L":
            if i >= n:
                raise ValueError("path overruns left sequence")
            if run_char == "L":
                run_len += 1
            else:
                flush()
                run_char = "L"
                run_len = 1
            i += 1
        elif ch == "R":
            if j >= m:
                raise ValueError("path overruns right sequence")
            if run_char == "R":
                run_len += 1
            else:
                flush()
                run_char = "R"
                run_len = 1
            j += 1
        else:
            raise ValueError("invalid step %r" % ch)
        if abs(i - j) > B:
            raise ValueError("path leaves the band")
    flush()
    if (i, j) != (n, m):
        raise ValueError("path does not fully consume both sequences")
    return cost


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def solve(left, right, B, open, extend):  # noqa: A002 - signature per spec
    _validate(left, right, B, open, extend)

    n, m = len(left), len(right)
    if n == 0 and m == 0:
        return {"cost": 0, "steps": ""}
    if abs(n - m) > B:
        return None

    # The virtual step that enters (0, 0) is treated as M (no open charge).
    steps = _hirsch(left, right, 0, n, 0, m, B, "M", None, open, extend)
    if steps is None:
        return None
    cost = path_cost(left, right, B, open, extend, steps)
    return {"cost": cost, "steps": steps}
