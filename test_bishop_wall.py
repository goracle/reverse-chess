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


def setup(team, *, pawns_home=(), bishops=(), king_home=True, native=False, king_at=None, extra=()):
    """Fresh game; ``pawns_home`` = columns with a home pawn, ``bishops`` = [(row,col)] on board."""
    g = Game(Rules(max_plies=300), native=native)
    if king_at is not None:
        g.kings[team].update(king_at, ignore_lock=True)
    elif king_home:
        g.kings[team].update((back_row(team), 4), ignore_lock=True)
        assert g.kings[team].at_home(), "setup: king must be home"
    for ptype, pos in extra:             # other pieces placed on the board, e.g. ("r", (3, 3))
        piece = next(p for p in g.pieces[team] if p.ptype == ptype and not p.ontheboard)
        piece.place(pos, ignore_lock=True)
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


FULL = tuple(range(8))


def outside_row(team):
    return back_row(team) + (1 if team == "b" else -1) * 3


def walled_off(g, team):
    return "pawn wall-off" in (g._losing_reason_py(team) or "")


def test_king_behind_full_wall_is_not_walled_off():
    """King + pawns only: a king already on the back row can still walk home."""
    for team in TEAMS:
        for col in (0, 2, 3, 5, 7):
            g = check(team, pawns_home=FULL, king_at=(back_row(team), col))
            assert not g.kings[team].at_home()
            assert g._losing_reason_py(team) is None, (team, col, g._losing_reason_py(team))


def test_king_outside_full_wall_is_walled_off():
    for team in TEAMS:
        g = check(team, pawns_home=FULL, king_at=(outside_row(team), 4))
        assert walled_off(g, team), g._losing_reason_py(team)


def test_other_piece_outside_full_wall_is_walled_off():
    for team in TEAMS:
        g = check(team, pawns_home=FULL, extra=[("r", (outside_row(team), 3))])
        assert walled_off(g, team), g._losing_reason_py(team)


def test_back_row_order_must_match_home_order():
    for team in TEAMS:
        b = back_row(team)
        # king left of the rooks, a rook on each side: fine
        g = check(team, pawns_home=FULL, king_at=(b, 2), extra=[("r", (b, 0)), ("r", (b, 7))])
        assert g._losing_reason_py(team) is None, g._losing_reason_py(team)
        # rooks both right of the king: one of them can never reach col 0 -> stranded
        g = check(team, pawns_home=FULL, king_at=(b, 2), extra=[("r", (b, 5)), ("r", (b, 6))])
        assert walled_off(g, team), g._losing_reason_py(team)
        # both rooks left of the king: the second can never reach col 7
        g = check(team, pawns_home=FULL, king_at=(b, 6), extra=[("r", (b, 1)), ("r", (b, 3))])
        assert walled_off(g, team), g._losing_reason_py(team)
        # a lone rook can pick either corner
        g = check(team, pawns_home=FULL, king_at=(b, 4), extra=[("r", (b, 5))])
        assert g._losing_reason_py(team) is None, g._losing_reason_py(team)


def test_queen_cannot_pass_a_home_bishop():
    for team in TEAMS:
        b = back_row(team)
        g = check(team, pawns_home=FULL, extra=[("q", (b, 1)), ("b", (b, 2))])
        assert any(p.ptype == "b" and p.at_home() for p in g.pieces[team]), "setup: bishop must be home"
        assert walled_off(g, team) or "bishops" in (g._losing_reason_py(team) or ""), g._losing_reason_py(team)


def test_back_row_bishop_off_home_is_stranded():
    for team in TEAMS:
        b = back_row(team)
        g = check(team, pawns_home=FULL, extra=[("b", (b, 3))])
        assert g._losing_reason_py(team) is not None


def test_home_win_is_not_overridden():
    """Everything on the board home => that team wins and is never scored as lost."""
    for team in TEAMS:
        b = back_row(team)
        for extra in ([], [("r", (b, 0)), ("r", (b, 7)), ("q", (b, 3))]):
            g = check(team, pawns_home=FULL, extra=extra)
            assert g.check_for_win(team)
            assert g._win_reason(team) == "%s has every piece home" % engine.full_name(team)
            assert g._losing_reason_py(team) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
