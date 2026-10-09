#!/usr/bin/env python
"""Pawn -> column assignment ("tenant matching").

Question answered: every pawn that is not yet home needs its own distinct
column on its home row.  A pawn may only reach columns within its allowed
reach, and every sideways step costs one gift (one opponent piece returned to
the board).  Find the assignment with the smallest total number of sideways
steps, or report that none exists.

The previous implementation enumerated every permutation (factorial time and
a large pile of intermediate dicts).  This version is an exact dynamic
program over the set of columns already used: O(pawns * 2**cols * cols),
i.e. a few thousand operations for 8 pawns on 8 columns.
"""
from itertools import permutations


def tenant_matching(mins):
    """Return the cheapest full assignment of pawns to distinct columns.

    ``mins`` maps ``pawn -> {column: cost}`` (only reachable columns listed).
    Returns ``{pawn: column}`` minimising the summed cost, or ``{}`` when some
    pawn cannot be housed (or ``mins`` is empty).  Ties are broken
    deterministically, but callers should only rely on the total cost.
    """
    if not mins:
        return {}
    pawns = sorted(mins)
    # states: bitmask of used columns -> (total cost, columns chosen so far)
    states = {0: (0, ())}
    for pawn in pawns:
        nxt = {}
        for mask, (cost, chosen) in states.items():
            for col, step in mins[pawn].items():
                bit = 1 << col
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_cost = cost + step
                best = nxt.get(new_mask)
                if best is None or new_cost < best[0]:
                    nxt[new_mask] = (new_cost, chosen + (col,))
        if not nxt:  # this pawn has nowhere left to go
            return {}
        states = nxt
    _, chosen = min(states.values(), key=lambda state: state[0])
    return dict(zip(pawns, chosen))


def assignment_cost(mins, solution):
    """Total cost of ``solution`` under the cost table ``mins``."""
    return sum(mins[pawn][col] for pawn, col in solution.items())


def _brute_force_cost(mins):
    """Reference implementation used only by the self-test below."""
    pawns = sorted(mins)
    cols = sorted({c for opts in mins.values() for c in opts})
    best = None
    for perm in permutations(cols, len(pawns)):
        if all(c in mins[p] for p, c in zip(pawns, perm)):
            cost = sum(mins[p][c] for p, c in zip(pawns, perm))
            best = cost if best is None else min(best, cost)
    return best


if __name__ == '__main__':
    # Self-test: the DP must agree with brute force on the old sample inputs.
    samples = [
        {0: {1: 1}, 1: {5: 2, 6: 1, 7: 0}, 2: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
         3: {1: 1, 2: 0, 3: 1}},
        {0: {0: 3, 4: 1}, 1: {4: 2, 0: 6}},
        {0: {3: 0, 4: 1}, 1: {0: 3, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4},
         2: {3: 3, 4: 2, 5: 1, 6: 0, 7: 1}, 3: {4: 2, 5: 1, 6: 0, 7: 1},
         4: {0: 0, 3: 3, 4: 4, 5: 5}, 5: {3: 2, 4: 1, 5: 0, 6: 1, 7: 2}},
        {0: {2: 0}, 1: {2: 0}},  # impossible: two pawns, one column
    ]
    for table in samples:
        sol = tenant_matching(table)
        expect = _brute_force_cost(table)
        got = assignment_cost(table, sol) if sol else None
        assert got == expect, (table, sol, got, expect)
        print(sol, got)
    print("tenants.py self-test passed")
