"""Shared constants and small helpers for Reverse Chess.

This module deliberately imports neither pygame nor tkinter, so the rules
engine and any neural-network code can import it on a headless machine.
"""
import os

# Order matters: it defines the piece-type index used by the observation
# encoder (env.py) and the order pieces are created in (engine.py).
PTYPES = ['p', 'k', 'r', 'b', 'q', 'n']
TEAMS = ('b', 'w')

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chessimages')
_IMAGE_NAMES = {'p': 'pawn', 'k': 'king', 'r': 'rook',
                'b': 'bishop', 'q': 'queen', 'n': 'knight'}


def image_path(team, ptype):
    """Path of the sprite for a piece, e.g. image_path('w', 'q')."""
    return os.path.join(IMAGE_DIR, '%s_%s.png' % (team, _IMAGE_NAMES[ptype]))


# Backwards-compatible lookup.  The old '..._p.png' "promoted" sprites are gone:
# the GUI now draws a small 'P' badge over the normal sprite instead, so the
# promoted flag no longer changes which file is loaded.
IMAGES = {(team, ptype, promoted): image_path(team, ptype)
          for team in TEAMS for ptype in PTYPES for promoted in (True, False)}


def _tally(pieces, on_board):
    """Count pieces by type whose ``ontheboard`` flag equals ``on_board``.

    A promoted piece is a pawn in disguise, so it is tallied under 'p'.
    Every type seen gets a key, even with a count of zero.
    """
    ret = {}
    for piece in pieces:
        ptype = 'p' if piece.promoted else piece.ptype
        ret.setdefault(ptype, 0)
        if piece.ontheboard == on_board:
            ret[ptype] += 1
    return ret


# The most of each type a team can ever have (used only as a sanity check).
_MAX_COUNT = {'q': 1, 'k': 1, 'b': 2, 'r': 2, 'n': 2}


def offboard_count(pieces):
    """Count the pieces of a team that are NOT on the board, by type."""
    ret = _tally(pieces, on_board=False)
    for ptype, limit in _MAX_COUNT.items():
        if ptype in ret:
            assert ret[ptype] <= limit, ret
    return ret


def onboard_count(pieces):
    """Count the pieces of a team that ARE on the board, by type."""
    return _tally(pieces, on_board=True)
