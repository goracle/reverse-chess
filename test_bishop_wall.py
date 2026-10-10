#!/usr/bin/env python
"""Regression tests for bishop_wall_check / _losing_reason (run: python test_bishop_wall.py).

A closed wall (both home pawns beside a bishop's home square locked) only strands a
bishop that is ON the board, unpromoted, not home, and of that home square's colour.
A bishop still off the board can be given back directly onto its home square, so it
must not make the team lose.  Checked on every available backend.
"""
import os

import engine
from engine import Game, Rules, back_row, square_color, pawn_home_row

TEAMS = "bw"
WALL_COLS = {2: (1, 3), 5: (4, 6)}


def setup(team, *, pawns_home=(), bishops=(), king_home=True, native=False):
    """Fresh game; ``pawns_home`` = columns with a home pawn, ``bishops`` = [(row,col)] on board."""
    g = Game(Rules(max_plies=300), native=native)
    if king_home:
        g.kings[team].update((back_row(team), 4), ignore_lock=True)
        assert g.kings[team].at_home(), "setup: king must be home"
    pawns = [p for p in g.pieces[team] if p.ptype == "p"]
    for p, col in zip(pawns, pawns_home):
        p.place((pawn_home_row(team), col), ignore_lock=True)
    bish = [p for p in g.pieces[team] if p.ptype == "b"]
    for p, (row, col) in zip(bish, bishops):
        p.place((row, col), ignore_lock=True)
    return g


def closed_walls(cols=(1, 3, 4, 6)):
    return tuple(cols)


def check(team, **kw):
    g = setup(team, **kw)
    if engine.NATIVE:                    # the Rust port must agree on the verdict
        gn = setup(team, native=True, **kw)
        assert gn._use_native, "native backend requested but not in use"
        assert gn._losing_reason(team) == g._losing_reason_py(team), (
            "native/python disagree", gn._losing_reason(team), g._losing_reason_py(team), kw)
    # stage-boundary assertions on the setup itself
    homes = [p for p in g.pieces[team] if p.ptype == "p" and p.ontheboard]
    assert all(p.at_home() for p in homes), "setup: pawns must stand on their home squares"
    assert len(homes) == len(kw.get("pawns_home", ())), "setup: wrong pawn count"
    return g


def test_offboard_bishops_are_not_walled():
    for team in TEAMS:
        g = check(team, pawns_home=(1, 3, 4, 6))            # both walls closed, no bishop on board
        assert not any(p.ontheboard for p in g.pieces[team] if p.ptype == "b")
        assert g.bishop_wall_check(team), "off-board bishops must not count as walled in"
        assert g._losing_reason_py(team) is None, g._losing_reason_py(team)


def test_all_pawns_home_with_king_is_not_lost():
    for team in TEAMS:
        g = check(team, pawns_home=tuple(range(8)))          # king + all 8 pawns home path
        assert g.bishop_wall_check(team)
        assert g._losing_reason_py(team) is None, g._losing_reason_py(team)


def test_onboard_stuck_bishop_still_loses():
    for team in TEAMS:
        for col, squares in WALL_COLS.items():
            home_sq = (back_row(team), col)
            colour = square_color(home_sq)
            # a bishop of that colour, standing on the board outside its home square
            row = back_row(team) + (1 if team == "b" else -1) * 3
            spot = next((row, c) for c in range(8) if square_color((row, c)) == colour)
            g = check(team, pawns_home=squares, bishops=[(spot[0], spot[1])])
            assert not g.bishop_wall_check(team), (team, col, spot)
            assert "bishops" in (g._losing_reason_py(team) or ""), (team, col)


def test_wall_ignored_for_other_colour_bishop():
    for team in TEAMS:
        # col-2 wall closed; the only bishop on board has col-5's colour -> it is not stranded
        colour5 = square_color((back_row(team), 5))
        row = back_row(team) + (1 if team == "b" else -1) * 3
        spot = next((row, c) for c in range(8) if square_color((row, c)) == colour5)
        g = check(team, pawns_home=WALL_COLS[2], bishops=[spot])
        assert g.bishop_wall_check(team)


def test_home_bishop_keeps_wall_irrelevant():
    for team in TEAMS:
        g = check(team, pawns_home=(1, 3, 4, 6), bishops=[(back_row(team), 2), (back_row(team), 5)])
        assert g.bishop_wall_check(team)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
