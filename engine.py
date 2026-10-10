#!/usr/bin/env python
"""Headless Reverse Chess rules engine.

No pygame, no tkinter, no prompts, no printing, no module-level state: every
piece of game state lives on a :class:`Game` instance, so you can run hundreds
of games in parallel processes and clone positions for tree search.

Quick tour
----------
>>> game = Game()
>>> game.turn                      # 'b' always moves first
'b'
>>> actions = game.legal_actions() # list of fully-specified Action objects
>>> game.step(actions[0])
>>> game.done, game.winner, game.end_reason

An :class:`Action` is one *complete* turn: which piece goes where, plus every
secondary choice the old GUI used to ask for in pop-ups (demote?  which piece
to give back?  reverse en passant?  promote the gifted pawn to what?).

Reverse chess in one paragraph
------------------------------
The game is chess played backwards.  Black's pawns walk *down* the board
toward row 1 and white's walk *up* toward row 6 (rows are numbered from the
top of the screen).  Instead of capturing you *give back* one of the
opponent's off-board pieces: a pawn moving diagonally must do so (that is the
reverse of a capture), any other piece may do so while moving, and the gifted
piece appears on the square the mover just left.  Instead of promoting you
un-promote: a promoted piece standing on its promotion row may step off as a
pawn.  You may not attack the enemy king (that would mean the king had moved
into check), and a king and rook alone on their back row may reverse-castle.
The goal is to get every piece home.

Coordinates are ``(row, col)`` with row 0 at the top.  Teams are ``'b'`` and
``'w'``.  Piece types are the letters in ``chessassets.PTYPES``.
"""
from __future__ import annotations

import copy
import ctypes
import os
import struct
import threading
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from chessassets import PTYPES, TEAMS
from tenants import tenant_matching

Pos = Tuple[int, int]
SIZE = 8
PROMOTION_TYPES = ("q", "n", "b", "r")  # what a gifted pawn may be promoted to
MOVE_CACHE_LIMIT = 250_000              # entries; the cache is flushed when full
_WITNESSES = 6                         # recent "this move proves the team can move" hints kept per team

_PTYPE_INDEX = {ptype: i for i, ptype in enumerate(PTYPES)}

_KNIGHT_STEPS = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
_KING_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
_RAYS = tuple((dr, dc, ("r", "q")) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))) + \
        tuple((dr, dc, ("b", "q")) for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)))

_GEOM_CACHE: Dict[tuple, tuple] = {}


# --------------------------------------------------------------------------
# Optional native (Rust) move generator
# --------------------------------------------------------------------------
# native/ holds a dependency-free Rust cdylib that re-implements Game._enumerate
# and Game._losing_reason.  Build it with ./build_native.sh; this module loads
# it through ctypes and silently falls back to the pure-Python code when it is
# missing.  RC_ENGINE=py forces pure Python, RC_ENGINE=native makes a missing
# library an error (so a benchmark can never quietly run the slow path).

def _load_native():
    mode = os.environ.get("RC_ENGINE", "auto").lower()
    if mode in ("py", "python", "pure"):
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("librc_engine.so", "librc_engine.dylib", "rc_engine.dll")
    dirs = (here, os.path.join(here, "native", "target", "release"))
    err = "no librc_engine library found (run ./build_native.sh)"
    for d in dirs:
        for name in names:
            path = os.path.join(d, name)
            if not os.path.exists(path):
                continue
            try:
                lib = ctypes.CDLL(path)
                lib.rc_abi_version.restype = ctypes.c_int32
                if lib.rc_abi_version() != 1:
                    err = "%s has the wrong ABI version" % path
                    continue
                lib.rc_enumerate.restype = ctypes.c_int32
                lib.rc_enumerate.argtypes = [ctypes.c_char_p, ctypes.c_uint32,
                                             ctypes.c_void_p, ctypes.c_uint32]
                lib.rc_losing.restype = ctypes.c_int32
                lib.rc_losing.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
                return lib
            except (OSError, AttributeError) as exc:
                err = "could not load %s: %s" % (path, exc)
    if mode == "native":
        raise ImportError("RC_ENGINE=native but " + err)
    return None


_LIB = _load_native()
NATIVE = _LIB is not None          # True when the Rust move generator is in use
_TLS = threading.local()           # per-thread action buffer for rc_enumerate
_BUF_CAP = 4096                    # actions; grown automatically on overflow
_SQ = tuple((r, c) for r in range(SIZE) for c in range(SIZE))
_GIFT_CODES = (None,) + tuple(PTYPES)                  # wire code -> gifted piece type
_PROMO_CODES = (None,) + PROMOTION_TYPES               # wire code -> promotion type


def _native_enumerate(state: bytes, team_index: int):
    """(n, raw): n > 0 playable actions (6 bytes each in raw), 0 none valid, -2 valid but none playable."""
    buf = getattr(_TLS, "buf", None)
    cap = getattr(_TLS, "cap", 0)
    if buf is None:
        cap = _BUF_CAP
        buf = _TLS.buf = ctypes.create_string_buffer(6 * cap)
        _TLS.cap = cap
    while True:
        n = _LIB.rc_enumerate(state, team_index, buf, cap)
        if n != -1:
            break
        cap *= 4
        buf = _TLS.buf = ctypes.create_string_buffer(6 * cap)
        _TLS.cap = cap
    if n < 0 and n != -2:
        raise RuntimeError("native engine failed (code %d)" % n)
    return n, (ctypes.string_at(buf, 6 * n) if n > 0 else b"")


def _geom_dsts(ptype: str, team: str, src: Tuple[int, int]) -> tuple:
    """Squares a piece of this type *could* reach from ``src`` by geometry alone
    (ignoring blockers, checks and gifts).  A superset of what check_legal_move()
    can accept, so move generation tests ~10 squares instead of all 64.
    Pure function of its arguments, so it is memoised forever."""
    key = (ptype, team, src)
    hit = _GEOM_CACHE.get(key)
    if hit is not None:
        return hit
    sr, sc = src
    out = set()
    for r in range(SIZE):
        for c in range(SIZE):
            dr, dc = r - sr, c - sc
            if dr == 0 and dc == 0:
                continue
            if ptype == "k":
                ok = max(abs(dr), abs(dc)) == 1 or (sr == r == back_row(team))  # + reverse castle
            elif ptype == "r":
                ok = dr == 0 or dc == 0
            elif ptype == "n":
                ok = (abs(dr), abs(dc)) in ((1, 2), (2, 1))
            elif ptype == "q":
                ok = dr == 0 or dc == 0 or abs(dr) == abs(dc)
            elif ptype == "b":
                ok = abs(dr) == abs(dc)
            else:  # pawn
                ok = dr == pawn_step(team) and abs(dc) <= 1
            if ok:
                out.add((r, c))
    _GEOM_CACHE[key] = res = tuple(sorted(out))
    return res
_PIECE_NAMES = {"p": "pawn", "k": "king", "r": "rook", "b": "bishop", "q": "queen", "n": "knight"}


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------

def other_team(team: str) -> str:
    """Return the opposing team letter."""
    assert team in TEAMS, team
    return "w" if team == "b" else "b"


def full_name(team: Optional[str]) -> Optional[str]:
    """'w' -> 'white', 'b' -> 'black' (None stays None)."""
    return {"w": "white", "b": "black"}.get(team)


def back_row(team: str) -> int:
    """Row holding a team's king/queen/rooks/bishops/knights at home."""
    return 0 if team == "b" else 7


def pawn_home_row(team: str) -> int:
    """Row on which a team's pawns start (and must end)."""
    return 1 if team == "b" else 6


def promotion_row(team: str) -> int:
    """Row where a team's promoted pieces may un-promote (the far back row)."""
    return 7 if team == "b" else 0


def pawn_step(team: str) -> int:
    """Row delta of one reverse pawn step: black's row number decreases (toward
    its home row 1), white's increases (toward its home row 6)."""
    return -1 if team == "b" else 1


def square_color(pos: Pos) -> int:
    """0 for a 'white' square, 1 for a 'grey' one (a bishop's identity)."""
    return (pos[0] + pos[1]) % 2


# Squares that wall a bishop off from its home square.  Both pawns of a pair
# sitting at home (and therefore locked) means the bishop can never get home.
_BISHOP_WALLS = {
    "w": {2: ((6, 1), (6, 3)), 5: ((6, 4), (6, 6))},
    "b": {2: ((1, 1), (1, 3)), 5: ((1, 4), (1, 6))},
}

# Where the king and rook stand after a (forward) castle, per side.
# (king_col, rook_col, rook_home_col)
_CASTLE_SIDES = ((2, 3, 0), (6, 5, 7))


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------

class IllegalActionError(ValueError):
    """Raised by Game.step() for an action that is not currently legal."""


@dataclass(frozen=True)
class Rules:
    """Tunable rule switches.

    win_requires_all_pieces_on_board
        False (the intended rule, and the default): a team wins when every
        piece *currently on the board* is home.  Gifts are the weapon: a piece
        given back to the opponent joins *their* to-do list (it must now get
        home too), so the opening is a tempo race.  E.g. black's king can walk
        home in three moves, and the defender's answer is to gift black a
        piece in reply.  True is a stricter experimental variant (everything,
        both teams, must be back on the board and home); it is not the
        game as designed.
    fifty_move_limit
        Draw after this many consecutive plies with no gift and no pawn move.
    repetition_limit
        Draw when the same position (with the same side to move) occurs this
        many times.
    max_plies
        Optional hard cap so self-play games always terminate.
    """
    win_requires_all_pieces_on_board: bool = False
    fifty_move_limit: int = 50
    repetition_limit: int = 3
    max_plies: Optional[int] = None


@dataclass(frozen=True)
class Action:
    """One complete, hashable turn.

    src, dst      squares of the moving piece.
    demote        a promoted piece on its promotion row steps off as a pawn.
    gift          piece type given back to the opponent (None = no gift).  The
                  gifted piece appears on ``src`` (the square just vacated).
    en_passant    reverse en passant: a gifted pawn is placed on its own home
                  row instead and the mover gets an extra turn.
    promote       type a gifted pawn is promoted to (None = stays a pawn).
    """
    src: Pos
    dst: Pos
    demote: bool = False
    gift: Optional[str] = None
    en_passant: bool = False
    promote: Optional[str] = None


class MoveOption(NamedTuple):
    """A raw (gift-less) destination for a piece, as found by move generation."""
    dst: Pos
    demote: bool
    forced_give: bool      # diagonal pawn step: a gift is mandatory
    reverse_castle: bool


class MoveCheck(NamedTuple):
    """Result of check_legal_move()."""
    legal: bool
    forced_give: bool = False
    reverse_castle: bool = False
    reason: str = ""


@dataclass
class Move:
    """An Action resolved against concrete pieces; knows how to be undone."""
    action: Action
    team: str
    mover: "Piece"
    src: Pos
    dst: Pos
    mover_ptype: str
    demote: bool
    reverse_castle: bool
    partner: Optional["Piece"]      # the rook/king dragged along by a castle
    partner_src: Optional[Pos]
    spawn: Optional["Piece"]        # the gifted piece, if any
    spawn_pos: Optional[Pos]
    promote_to: Optional[str]
    extra_turn: bool
    old_ptype: Optional[str] = None  # filled in by execute() when demoting


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------

class Piece:
    """A chess piece.  Pure state; no images and no drawing."""

    def __init__(self, game: "Game", team: str, ptype: str, home, uid: int):
        self.game = game
        self.team = team
        self.ptype = ptype
        self.home = home          # (rows, cols): the squares that count as home
        self.uid = uid            # unique and stable within a game (and its clones)
        self.promoted = False     # a pawn wearing another piece's type
        self.locked = False       # may never move again (home pawn / castled K+R)
        self.ontheboard = False
        self.pos: Optional[Pos] = None

    def __repr__(self):
        flags = ("*" if self.promoted else "") + ("L" if self.locked else "")
        return "<%s%s%s %s>" % (self.team, self.ptype, flags, self.pos)

    def code(self) -> int:
        """Small non-zero integer identifying this piece's observable state."""
        return 1 + ((((_PTYPE_INDEX[self.ptype] * 2 + self.promoted) * 2
                      + self.locked) * 2) + (self.team == "w"))

    def at_home(self) -> bool:
        """Pure check: is the piece on one of its home squares (unpromoted)?"""
        if self.pos is None or self.promoted:
            return False
        rows, cols = self.home
        return self.pos[0] in rows and self.pos[1] in cols

    # -- locking ---------------------------------------------------------
    def lock(self):
        """Freeze the piece in place."""
        assert self.ontheboard and not self.promoted
        if not self.locked:
            self.locked = True
            self.game._dirty()

    def unlock(self):
        """Allow the piece to move again."""
        assert self.ontheboard
        if self.locked:
            self.locked = False
            self.game._dirty()

    def refresh_lock(self):
        """Pawns standing on their home row can never move again."""
        if self.ptype == "p" and self.at_home():
            self.lock()
        else:
            self.unlock()

    # -- promotion -------------------------------------------------------
    def demote(self):
        """Turn a promoted piece back into the pawn it really is."""
        if self.promoted:
            self.ptype = "p"
            self.promoted = False
            self.game._dirty()

    def promote(self, ptype_new: str):
        """Give a pawn another piece's type (it stays a pawn underneath)."""
        if not self.promoted:
            assert ptype_new != "p" and ptype_new != self.ptype, ptype_new
            self.ptype = ptype_new
            self.promoted = True
            self.unlock()
            self.game._dirty()

    # -- placement -------------------------------------------------------
    def place(self, pos: Pos, ignore_lock: bool = False):
        """Put an off-board piece onto the board."""
        board = self.game.board
        assert not self.ontheboard, self
        assert board[pos] is None, (pos, board[pos])
        if not ignore_lock:
            assert not self.locked, self
        self.ontheboard = True
        self.pos = pos
        board[pos] = self
        self.game._dirty()
        if not ignore_lock:
            self.refresh_lock()

    def update(self, newpos: Pos, ignore_lock: bool = False):
        """Move an on-board piece to an empty square.

        ``ignore_lock=True`` is used for hypothetical moves: it bypasses the
        locked check and does not refresh lock flags.
        """
        assert self.ontheboard, "use place() for off-board pieces"
        if not ignore_lock:
            assert not self.locked, self
        board = self.game.board
        assert board[newpos] is None, (self, newpos, board[newpos])
        board[self.pos] = None
        board[newpos] = self
        self.pos = newpos
        self.game._dirty()
        if not ignore_lock:
            self.refresh_lock()

    def remove(self):
        """Take the piece off the board (used to undo a gift)."""
        board = self.game.board
        assert self.ontheboard and board[self.pos] is self, self
        board[self.pos] = None
        self.ontheboard = False
        self.pos = None
        self.locked = False
        self.demote()
        self.game._dirty()

    def move_home(self, col: int):
        """Send a king/rook to its home square (used by reverse castling)."""
        assert self.ptype in ("k", "r") and self.ontheboard, self
        self.update((back_row(self.team), col))


# --------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------

class Game:
    """Complete state and rules of one game of reverse chess.

    Public state
    ------------
    board        8x8 numpy object array of Piece / None.
    pieces       {'b': [Piece...], 'w': [Piece...]}.
    turn         team to move.
    actions      {Action: Move} legal for ``turn`` (empty once the game is over).
    done         True once the game has ended.
    winner       'b', 'w' or None (None with done=True means a draw).
    end_reason   human-readable reason the game ended.
    ply          number of moves played so far.
    move_log     list of every Action played, in order.
    """

    def __init__(self, rules: Optional[Rules] = None, native: Optional[bool] = None):
        """``native``: None = use the Rust move generator when it is available,
        False = force the pure-Python one, True = require the Rust one."""
        if native and _LIB is None:
            raise RuntimeError("native engine requested but librc_engine is not built")
        self._use_native = (_LIB is not None) if native is None else bool(native)
        self.rules = rules or Rules()
        self._moves_cache: dict = {}
        self._witness: dict = {}          # team -> [(piece index, dst), ...] recent proofs that it could move
        self._has_move: dict = {}         # (snapshot, team) -> bool, per game (see has_legal_move)
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def reset(self):
        """Return to the starting position (two kings, everything else off)."""
        self.board = np.full((SIZE, SIZE), None, dtype=object)
        self._snap: Optional[bytes] = None
        self._pk: Optional[bytes] = None
        self._raw: Optional[bytes] = None     # native action bytes for the current turn
        self._moves_cache.clear()
        self._witness.clear()
        self._has_move.clear()
        self.pieces: Dict[str, List[Piece]] = {}
        self.kings: Dict[str, Piece] = {}
        uid = 0
        for team in TEAMS:
            made = []
            for ptype in PTYPES:
                for _ in range({"p": 8, "r": 2, "b": 2, "n": 2}.get(ptype, 1)):
                    made.append(Piece(self, team, ptype, self._home_of(team, ptype), uid))
                    uid += 1
            self.pieces[team] = made
            self.kings[team] = next(p for p in made if p.ptype == "k")
        # Only the kings start on the board, in fixed spots.
        self.kings["b"].place((3, 5))
        self.kings["w"].place((4, 3))

        self.turn = "b"
        self.ply = 0
        self.quiet_plies = 0              # plies since the last gift / pawn move
        self.seen: Dict[bytes, int] = {}  # position -> occurrences (repetition)
        self.done = False
        self.winner: Optional[str] = None
        self.end_reason = ""
        self.last_move: Optional[Move] = None
        self.move_log: List[Action] = []
        self.actions: Dict[Action, Move] = {}
        self._begin_turn()

    def close(self):
        """Break the Game <-> Piece reference cycle so the game can be freed.

        ``self.board`` is a numpy object array holding Pieces, and every Piece
        holds ``.game``.  numpy arrays are invisible to Python's cycle
        collector, so without this the Game (and its move cache, Move/Action
        objects, ...) is never reclaimed.  Call it when a game is finished
        with; the Game must not be used afterwards.
        """
        self.board.fill(None)
        self._moves_cache.clear()
        self._has_move.clear()
        self.actions = {}
        self._raw = None

    @staticmethod
    def _home_of(team: str, ptype: str):
        """(rows, cols) of the squares where a piece of this type is home."""
        if ptype == "p":
            return (pawn_home_row(team),), tuple(range(SIZE))
        cols = {"k": (4,), "q": (3,), "r": (0, 7), "b": (2, 5), "n": (1, 6)}[ptype]
        return (back_row(team),), cols

    # ------------------------------------------------------------------
    # Cheap views
    # ------------------------------------------------------------------
    def _dirty(self):
        """Invalidate the cached snapshot after any state change."""
        self._snap = None
        self._pk = None

    def snapshot(self) -> bytes:
        """Compact, hashable fingerprint of the board (cached until it changes)."""
        if self._snap is None:
            self._snap = bytes(0 if p is None else p.code() for p in self.board.flat)
        return self._snap

    def _pack(self) -> bytes:
        """State in the 128-byte wire format of native/src/lib.rs (cached until it changes)."""
        pk = self._pk
        if pk is None:
            buf = bytearray(128)
            i = 0
            index = _PTYPE_INDEX
            for team in TEAMS:
                for p in self.pieces[team]:
                    if p.ontheboard:
                        pos = p.pos
                        buf[i] = 1 | (2 if p.promoted else 0) | (4 if p.locked else 0)
                        buf[i + 1] = pos[0] * 8 + pos[1]
                    else:
                        buf[i + 1] = 255
                    buf[i + 2] = index[p.ptype]
                    i += 4
            self._pk = pk = bytes(buf)
        return pk

    def on_board(self, team: str) -> List[Piece]:
        """The team's pieces that are currently on the board."""
        return [p for p in self.pieces[team] if p.ontheboard]

    def all_on_board(self, team: str) -> bool:
        """True when none of the team's pieces is off the board."""
        return all(p.ontheboard for p in self.pieces[team])

    def offboard_by_type(self, team: str) -> Dict[str, Piece]:
        """One representative off-board piece per type (pieces of a type are
        interchangeable, so the first one stands for all of them)."""
        found: Dict[str, Piece] = {}
        for p in self.pieces[team]:
            if not p.ontheboard:
                found.setdefault(p.ptype, p)
        return found

    def legal_actions(self) -> List[Action]:
        """All legal actions for the side to move (empty when the game is over)."""
        return list(self.actions)

    def action_array(self) -> np.ndarray:
        """Legal actions as an ``(n, 6)`` uint8 array, in ``legal_actions()`` order.

        Columns: src, dst (absolute squares, row * 8 + col), demote, gift code
        (0 = none, else 1 + index in chessassets.PTYPES), en_passant, promote code
        (0 = none, else 1 + index in PROMOTION_TYPES) -- ``env.action_components``
        before the board is flipped for black.  Treat it as read-only.
        """
        raw = self._raw
        if raw is not None and len(raw) == 6 * len(self.actions):
            return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 6)
        rows = [(a.src[0] * SIZE + a.src[1], a.dst[0] * SIZE + a.dst[1], int(a.demote),
                 0 if a.gift is None else 1 + _PTYPE_INDEX[a.gift], int(a.en_passant),
                 0 if a.promote is None else 1 + PROMOTION_TYPES.index(a.promote))
                for a in self.actions]
        return np.array(rows, dtype=np.uint8).reshape(-1, 6)

    def position_key(self) -> bytes:
        """Board fingerprint plus side to move: what repetition counts."""
        return self.snapshot() + self.turn.encode()

    # ------------------------------------------------------------------
    # Attacks and lines
    # ------------------------------------------------------------------
    def _path_blocked(self, a: Pos, b: Pos) -> bool:
        """Is any square strictly between a and b occupied?  (Only meaningful
        for straight and diagonal lines; anything else returns False.)"""
        dr, dc = b[0] - a[0], b[1] - a[1]
        if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
            return False
        sr = (dr > 0) - (dr < 0)
        sc = (dc > 0) - (dc < 0)
        for i in range(1, max(abs(dr), abs(dc))):
            if self.board[a[0] + i * sr, a[1] + i * sc] is not None:
                return True
        return False

    def check_attacking(self, piece: Piece, target: Pos) -> bool:
        """Does ``piece`` (where it stands now) attack ``target``?"""
        pos, kind = piece.pos, piece.ptype
        dr, dc = target[0] - pos[0], target[1] - pos[1]
        if dr == 0 and dc == 0:
            return False
        if kind == "n":
            return (abs(dr), abs(dc)) in ((1, 2), (2, 1))
        if kind == "k":
            return max(abs(dr), abs(dc)) == 1
        if kind == "p":
            # Pawns attack in the *forward-chess* direction: black down, white up.
            return dr == (1 if piece.team == "b" else -1) and abs(dc) == 1
        straight = dr == 0 or dc == 0
        diagonal = abs(dr) == abs(dc)
        if kind == "r":
            line = straight
        elif kind == "b":
            line = diagonal
        else:  # queen
            line = straight or diagonal
        return line and not self._path_blocked(pos, target)

    def attacking_king(self, piece: Piece) -> bool:
        """Does ``piece`` attack the enemy king?"""
        return self.check_attacking(piece, self.kings[other_team(piece.team)].pos)

    # ------------------------------------------------------------------
    # Raw (gift-less) move legality
    # ------------------------------------------------------------------
    def check_legal_move(self, src: Pos, dst: Pos, as_pawn: bool = False) -> MoveCheck:
        """Is moving the piece on ``src`` to ``dst`` legal, ignoring gifts?

        ``as_pawn`` evaluates a promoted piece as the pawn it really is
        (the un-promotion step).  The returned ``forced_give`` says a diagonal
        pawn step needs a gift; ``reverse_castle`` flags a reverse castle.
        """
        board = self.board
        piece = board[src]
        team = piece.team
        ptype = "p" if as_pawn else piece.ptype
        if src == dst or board[dst] is not None:
            return MoveCheck(False, reason="occupied")

        dr, dc = dst[0] - src[0], dst[1] - src[1]
        forced_give = False
        castle = False
        if ptype == "k":
            legal = max(abs(dr), abs(dc)) == 1
            if not legal:
                legal = (not piece.promoted) and self._reverse_castle_ok("k", team, src, dst)
                castle = legal
        elif ptype == "r":
            legal = dr == 0 or dc == 0
        elif ptype == "n":
            legal = (abs(dr), abs(dc)) in ((1, 2), (2, 1))
        elif ptype == "q":
            legal = dr == 0 or dc == 0 or abs(dr) == abs(dc)
        elif ptype == "b":
            legal = abs(dr) == abs(dc)
        else:  # pawn
            legal, forced_give = self._pawn_step_ok(src, dst, team)
        if not legal:
            return MoveCheck(False, forced_give, False, "bad movement for piece type")

        # Sliding pieces may not jump (knights do; castling is checked above).
        if ptype != "n" and not castle and self._path_blocked(src, dst):
            if ptype == "r" and self._reverse_castle_ok("r", team, src, dst):
                castle = True  # the rook "jumps" over its king
            else:
                return MoveCheck(False, forced_give, False, "piece(s) in the way")

        # Reverse chess forbids attacking the enemy king: that would mean the
        # enemy king had walked into check.  Test by pre-moving the piece.
        if self._discovered_check(piece, dst, demote=as_pawn and piece.ptype != "p"):
            return MoveCheck(False, forced_give, castle, "would attack the enemy king")

        # A home-row pawn is never allowed to be the one attacking your king.
        if not self._king_pawn_ok(team, dst if piece.ptype == "k" else None):
            return MoveCheck(False, forced_give, castle, "attacked by a pawn at home")

        # A king may not castle out of check, so it may not reverse-castle
        # into check either.
        if castle:
            home_sq = (back_row(team), 4)
            if any(self.check_attacking(p, home_sq) for p in self.on_board(other_team(team))):
                return MoveCheck(False, forced_give, castle, "reverse castle into check")

        return MoveCheck(True, forced_give, castle)

    def _pawn_step_ok(self, src: Pos, dst: Pos, team: str):
        """Reverse pawn geometry.  Returns (legal, forced_give)."""
        if dst[0] - src[0] != pawn_step(team) or dst[0] in (0, SIZE - 1):
            return False, False
        sideways = abs(dst[1] - src[1])
        forced_give = sideways == 1
        legal = forced_give or sideways == 0
        # A diagonal step un-captures, which needs something to give back.
        if forced_give and self.all_on_board(other_team(team)):
            legal = False
        return legal, forced_give

    def _reverse_castle_ok(self, kind: str, team: str, src: Pos, dst: Pos) -> bool:
        """Is src->dst a reverse castle started by the king ('k') or rook ('r')?

        Reverse castling needs the home row to hold exactly one king and one
        rook of ``team`` in their post-castle spots (king c/g file, rook d/f
        file).  The king then goes to e-file and the rook to its corner.
        """
        back = back_row(team)
        if src[0] != back or dst[0] != back:
            return False
        brow = self.board[back]
        if sum(1 for sq in brow if sq is not None) != 2:
            return False
        for king_col, rook_col, rook_home in _CASTLE_SIDES:
            king, rook = brow[king_col], brow[rook_col]
            if king is None or rook is None:
                continue
            if (king.ptype == "k" and rook.ptype == "r" and king.team == team
                    and rook.team == team and not rook.promoted):
                if kind == "k":
                    return src == king.pos and dst == (back, 4)
                return src == rook.pos and dst == (back, rook_home)
        return False

    def _team_attacks(self, team: str, target: Pos) -> bool:
        """Does any on-board piece of ``team`` attack ``target`` (an occupied
        square, normally the enemy king)?  Looks outward from the target along
        its rays instead of asking every piece, which is much cheaper.
        Same attack rules as check_attacking()."""
        board = self.board
        tr, tc = target
        for dr, dc in _KNIGHT_STEPS:
            r, c = tr + dr, tc + dc
            if 0 <= r < SIZE and 0 <= c < SIZE:
                p = board[r, c]
                if p is not None and p.team == team and p.ptype == "n":
                    return True
        for dr, dc in _KING_STEPS:
            r, c = tr + dr, tc + dc
            if 0 <= r < SIZE and 0 <= c < SIZE:
                p = board[r, c]
                if p is not None and p.team == team and p.ptype == "k":
                    return True
        # a pawn attacks in the forward-chess direction (black down, white up)
        pr = tr - (1 if team == "b" else -1)
        if 0 <= pr < SIZE:
            for pc in (tc - 1, tc + 1):
                if 0 <= pc < SIZE:
                    p = board[pr, pc]
                    if p is not None and p.team == team and p.ptype == "p":
                        return True
        for dr, dc, kinds in _RAYS:
            r, c = tr + dr, tc + dc
            while 0 <= r < SIZE and 0 <= c < SIZE:
                p = board[r, c]
                if p is not None:
                    if p.team == team and p.ptype in kinds:
                        return True
                    break
                r += dr
                c += dc
        return False

    def _discovered_check(self, piece: Piece, dst: Pos, demote: bool) -> bool:
        """After moving ``piece`` to ``dst``, would any piece of its team attack
        the enemy king?  (True means the move is illegal.)"""
        board = self.board
        src, old_ptype, old_promoted = piece.pos, piece.ptype, piece.promoted
        # Hypothetical position, edited in place (no snapshot invalidation:
        # everything is restored before returning).
        board[src] = None
        board[dst] = piece
        if demote and piece.promoted:
            piece.ptype, piece.promoted = "p", False
        try:
            return self._team_attacks(piece.team, self.kings[other_team(piece.team)].pos)
        finally:
            piece.ptype, piece.promoted = old_ptype, old_promoted
            board[dst] = None
            board[src] = piece

    def _king_pawn_ok(self, team: str, newpos: Optional[Pos]) -> bool:
        """No enemy pawn standing on its home row may attack ``team``'s king
        (at ``newpos`` if the king is the piece being moved)."""
        tr, tc = newpos if newpos is not None else self.kings[team].pos
        enemy = other_team(team)
        home = pawn_home_row(enemy)
        # a home-row pawn attacks only the row one step ahead of it (forward-chess)
        if tr != home + (1 if enemy == "b" else -1):
            return True
        board = self.board
        for pc in (tc - 1, tc + 1):
            if 0 <= pc < SIZE:
                p = board[home, pc]
                if p is not None and p.team == enemy and p.ptype == "p" and p.at_home():
                    return False
        return True

    # ------------------------------------------------------------------
    # Move generation (gift-less)
    # ------------------------------------------------------------------
    def piece_moves(self, piece: Piece, first_only: bool = False) -> List[MoveOption]:
        """Every raw destination for ``piece``.

        Locked pieces (home pawns, castled king and rook) have none.  Results
        are cached per (board fingerprint, piece).  ``first_only=True`` stops
        at the first hit and skips the cache (used for "any move at all?").
        """
        if piece.locked or not piece.ontheboard:
            return []
        key = None
        if not first_only or self._snap is not None:   # fingerprinting costs more than a peek saves
            # the square matters: two same-type pieces can swap places without
            # changing the fingerprint, and then uid alone would hit stale moves
            key = (self.snapshot(), piece.uid, piece.pos)
            cached = self._moves_cache.get(key)
            if cached is not None:
                return cached
        src = piece.pos
        can_demote = piece.promoted and src[0] == promotion_row(piece.team)
        found: List[MoveOption] = []
        # Only squares reachable by geometry (same row-major order as the old
        # 64-square scan, so the resulting action order is unchanged).
        dsts = _geom_dsts(piece.ptype, piece.team, src)
        if can_demote:
            dsts = sorted(set(dsts) | set(_geom_dsts("p", piece.team, src)))
        board = self.board
        for dst in dsts:
            if board[dst] is not None:
                continue
            chk = self.check_legal_move(src, dst)
            if chk.legal:
                found.append(MoveOption(dst, False, chk.forced_give, chk.reverse_castle))
            if can_demote:
                chk = self.check_legal_move(src, dst, as_pawn=True)
                if chk.legal:
                    found.append(MoveOption(dst, True, chk.forced_give, chk.reverse_castle))
            if first_only and found:
                return found
        if len(self._moves_cache) >= MOVE_CACHE_LIMIT:
            self._moves_cache.clear()
        self._moves_cache[key] = found
        return found

    def has_legal_move(self, team: str) -> bool:
        """Does ``team`` have at least one raw move?  (Stops at the first.)"""
        pieces = self.pieces[team]
        # Consecutive positions are nearly identical, so a move that proved the
        # team could move recently almost always still works: try those first.
        ws = self._witness.get(team)
        if ws:
            for n, (i, dst) in enumerate(ws):
                p = pieces[i]
                if p.ontheboard and not p.locked and self.check_legal_move(p.pos, dst).legal:
                    if n:                                   # most recently useful first
                        ws.insert(0, ws.pop(n))
                    return True
        # The answer depends only on the board fingerprint (which encodes every
        # piece's type, promotion and lock state; "all pieces on board" follows
        # from it), and gift checking asks about the same positions repeatedly.
        key = (self.snapshot(), team)
        hit = self._has_move.get(key)
        if hit is not None:
            return hit
        result = False
        for i, p in enumerate(pieces):
            if p.ontheboard:
                found = self.piece_moves(p, first_only=True)
                if found:
                    if not found[0].demote:
                        ws = self._witness.setdefault(team, [])
                        ws.insert(0, (i, found[0].dst))
                        del ws[_WITNESSES:]
                    result = True
                    break
        if len(self._has_move) >= MOVE_CACHE_LIMIT:
            self._has_move.clear()
        self._has_move[key] = result
        return result

    # ------------------------------------------------------------------
    # Gifts
    # ------------------------------------------------------------------
    def _reverse_ep_possible(self, src: Pos, dst: Pos, mover: Piece, spawn: Piece) -> bool:
        """Can this pawn un-capture a pawn *en passant*?

        Needs a pawn stepping diagonally onto the row from which the gifted
        pawn could have made its double step, with the two squares in the
        mover's column (the double-step path) both empty.
        """
        if not (mover.ptype == "p" and spawn.ptype == "p"):
            return False
        row, col = src
        row1 = dst[0]
        ok = ((spawn.team == "b" and row == 2 and row1 == 3)
              or (spawn.team == "w" and row == 5 and row1 == 4))
        return ok and self.board[row - 1, col] is None and self.board[row + 1, col] is None

    def _gift_legal(self, mover: Piece, dst: Pos, spawn: Piece, spawn_pos: Pos,
                    promoting: bool, memo: dict) -> bool:
        """Is giving back ``spawn`` on ``spawn_pos`` legal while ``mover`` goes to ``dst``?

        The answer does not depend on *what* a gifted pawn is promoted to, only
        on whether it is promoted at all, so it is memoised per enumeration.
        """
        src = mover.pos
        key = (src, dst, spawn.ptype, spawn_pos, promoting)
        hit = memo.get(key)
        if hit is not None:
            return hit
        ok = self._gift_legal_uncached(mover, src, dst, spawn, spawn_pos, promoting, memo)
        memo[key] = ok
        return ok

    def _gift_legal_uncached(self, mover, src, dst, spawn, spawn_pos, promoting, memo) -> bool:
        return self._gift_problem(mover, src, dst, spawn, spawn_pos, promoting, memo) is None

    def _gift_problem(self, mover, src, dst, spawn, spawn_pos, promoting, memo) -> Optional[str]:
        """Why giving back ``spawn`` on ``spawn_pos`` is illegal, or None if it is legal.

        The text is a clause meant to follow "...because"; explain_illegal()
        shows it to the player.
        """
        gteam = spawn.team  # the team receiving the gift
        gname = full_name(gteam).capitalize()
        if spawn.ontheboard:
            return "that piece is already on the board"
        if self.board[spawn_pos] is not None and spawn_pos != src:
            return "the square for the gift is occupied"
        if spawn.ptype == "b" and not self._bishop_colour_ok(gteam, spawn_pos):
            return "a team's two bishops must stand on opposite-coloured squares"

        had_moves = memo.get(("had", gteam))
        if had_moves is None:
            had_moves = memo[("had", gteam)] = self.has_legal_move(gteam)

        # Hypothetical position: mover has moved, gift has landed.
        saved = self._snap
        mover.update(dst, ignore_lock=True)
        spawn.place(spawn_pos, ignore_lock=True)
        try:
            # A gift may not leave the receiving team with nothing to play.
            if had_moves and not self.has_legal_move(gteam):
                return "it would leave %s with no legal move" % gname
            # ... nor wall its bishops off from their home squares.
            if not self.bishop_wall_check(gteam):
                return "it would wall a %s bishop off from its home square" % gname
        finally:
            spawn.remove()
            mover.update(src, ignore_lock=True)
            self._snap = saved

        # Extra rules for a gifted pawn that stays a pawn.
        if spawn.ptype == "p" and not promoting:
            if not self._pawn_spawn_home_ok(spawn, spawn_pos, dst if mover.ptype == "k" else None):
                return ("the gifted pawn would stand on its home row attacking your king, "
                        "and a home-row pawn can never give check")
            if not self.pawn_scan(gteam, spawn_pos):
                return "%s's pawns could then no longer all reach their own home squares" % gname
            # Pawns are never given back on their own team's back row.
            if spawn_pos[0] == back_row(gteam):
                return "pawns are never given back on their own back row"
        return None

    def _bishop_colour_ok(self, team: str, spawn_pos: Pos) -> bool:
        """A new bishop must not share a square colour with the team's other one."""
        colour = square_color(spawn_pos)
        return not any(p.ontheboard and p.ptype == "b" and not p.promoted
                       and square_color(p.pos) == colour for p in self.pieces[team])

    def _pawn_spawn_home_ok(self, pawn: Piece, spawn_pos: Pos, newpos: Optional[Pos]) -> bool:
        """A pawn may not be given back on its home row *attacking* the enemy king
        (a home-row pawn can never give check in ordinary chess)."""
        team = pawn.team
        if spawn_pos[0] != pawn_home_row(team):
            return True
        target = newpos if newpos is not None else self.kings[other_team(team)].pos
        dr = spawn_pos[0] + (1 if team == "b" else -1)
        attacking = target[0] == dr and abs(target[1] - spawn_pos[1]) == 1
        return not attacking

    # ------------------------------------------------------------------
    # Pawn / bishop reachability heuristics
    # ------------------------------------------------------------------
    def pawn_scan(self, team: str, spawn_pos: Optional[Pos] = None) -> bool:
        """Can all of ``team``'s remaining pawns still reach distinct home squares?

        Each sideways step of a pawn costs one gift, so the cheapest column
        assignment must also fit within the number of opposing pieces that are
        still off the board.  ``spawn_pos`` adds a hypothetical new pawn.
        """
        pawns: List[Pos] = []
        locked_cols = set()
        for p in self.pieces[team]:
            if p.ontheboard and p.ptype == "p" and not p.promoted:
                if p.at_home():
                    locked_cols.add(p.pos[1])
                else:
                    pawns.append(p.pos)
        if spawn_pos is not None:
            pawns.append(spawn_pos)
        if not pawns:
            return True
        open_cols = set(range(SIZE)) - locked_cols
        home_row = pawn_home_row(team)
        options = {}
        for idx, (row, col) in enumerate(pawns):
            reach = abs(row - home_row)  # one sideways step per row of distance
            options[idx] = {c: abs(c - col) for c in open_cols if abs(c - col) <= reach}
        solution = tenant_matching(options)
        if not solution:
            return False
        needed = sum(options[i][solution[i]] for i in solution)
        spare = sum(1 for p in self.pieces[other_team(team)] if not p.ontheboard)
        return needed <= spare

    def bishop_wall_check(self, team: str) -> bool:
        """Weak check that neither bishop is walled off from its home square.

        A bishop that is not home yet is stuck when both pawns guarding its
        home square's diagonals are locked at home.
        """
        home_cols = {p.pos[1] for p in self.pieces[team]
                     if p.ontheboard and p.ptype == "b" and p.at_home()}
        for col, squares in _BISHOP_WALLS[team].items():
            if col in home_cols:
                continue
            walled = 0
            for sq in squares:
                q = self.board[sq]
                if q is not None and q.team == team and q.ptype == "p" and q.at_home():
                    walled += 1
            if walled == 2:
                return False
        return True

    # ------------------------------------------------------------------
    # Turning actions into moves
    # ------------------------------------------------------------------
    def _candidate_actions(self, piece: Piece, opt: MoveOption,
                           offboard: Dict[str, Piece]) -> List[Action]:
        """Every Action worth testing for one raw move: with/without a gift, for each
        giftable type, ordinary/en passant spawn, and each promotion choice."""
        src = piece.pos
        opp = other_team(piece.team)
        out: List[Action] = []
        if not opt.forced_give:
            out.append(Action(src, opt.dst, opt.demote))
        # Gifts: forced for diagonal pawn steps, optional for every non-pawn
        # piece (the old "right-click"), never while reverse castling.
        if not offboard or opt.reverse_castle or not (opt.forced_give or piece.ptype != "p"):
            return out
        for gift, spawn in offboard.items():
            ep_choices = (False, True) if self._reverse_ep_possible(src, opt.dst, piece, spawn) else (False,)
            for ep in ep_choices:
                if spawn.ptype == "p" and not ep:
                    # A pawn given back on its promotion row *must* be promoted.
                    must = src[0] == promotion_row(opp)
                    promos = PROMOTION_TYPES if must else (None,) + PROMOTION_TYPES
                else:
                    promos = (None,)
                for promo in promos:
                    out.append(Action(src, opt.dst, opt.demote, gift, ep, promo))
        return out

    def _make_move(self, action: Action, opt: MoveOption,
                   offboard: Dict[str, Piece], memo: dict) -> Optional[Move]:
        """Resolve a candidate into a Move, or None if the gift is illegal."""
        mover = self.board[action.src]
        opp = other_team(mover.team)
        if action.gift is not None:
            spawn = offboard[action.gift]
            # Reverse en passant places the pawn on its home row; otherwise the
            # gift lands on the square the mover just vacated.
            spawn_pos = ((pawn_home_row(opp), action.src[1]) if action.en_passant
                         else action.src)
            promoting = spawn.ptype == "p" and action.promote is not None
            if not self._gift_legal(mover, action.dst, spawn, spawn_pos, promoting, memo):
                return None
        return self._build_move(action, opt, offboard)

    def _build_move(self, action: Action, opt: MoveOption,
                    offboard: Dict[str, Piece]) -> Move:
        """Bind an (already validated) Action to concrete pieces."""
        mover = self.board[action.src]
        opp = other_team(mover.team)
        partner = partner_src = spawn = spawn_pos = None
        if opt.reverse_castle:
            partner = next(p for p in self.board[action.src[0]]
                           if p is not None and p is not mover)
            partner_src = partner.pos
        if action.gift is not None:
            spawn = offboard[action.gift]
            spawn_pos = ((pawn_home_row(opp), action.src[1]) if action.en_passant
                         else action.src)
        return Move(action=action, team=mover.team, mover=mover, src=action.src,
                    dst=action.dst, mover_ptype=mover.ptype, demote=opt.demote,
                    reverse_castle=opt.reverse_castle, partner=partner,
                    partner_src=partner_src, spawn=spawn, spawn_pos=spawn_pos,
                    promote_to=action.promote, extra_turn=action.en_passant)

    def _resolve(self, action: Action) -> Move:
        """Move for an action the native generator already validated (built lazily,
        only for the one action that is actually played)."""
        mover = self.board[action.src]
        chk = self.check_legal_move(action.src, action.dst, as_pawn=action.demote)
        opt = MoveOption(action.dst, action.demote, chk.forced_give, chk.reverse_castle)
        return self._build_move(action, opt, self.offboard_by_type(other_team(mover.team)))

    def _enumerate(self):
        """Return (valid, playable) action dicts for the side to move.

        valid     obeys every rule.
        playable  valid *and* leaves the opponent at least one raw move (the
                  original game refused moves that stalemate the opponent).
        """
        team = self.turn
        offboard = self.offboard_by_type(other_team(team))
        memo: dict = {}
        valid: Dict[Action, Move] = {}
        for piece in self.on_board(team):
            for opt in self.piece_moves(piece):
                for action in self._candidate_actions(piece, opt, offboard):
                    move = self._make_move(action, opt, offboard, memo)
                    if move is not None:
                        valid[action] = move
        playable = {a: m for a, m in valid.items() if self._leaves_reply(m)}
        return valid, playable

    def _leaves_reply(self, move: Move) -> bool:
        """After ``move``, does the opponent still have a raw move?"""
        saved = self._snap
        self._execute(move)
        try:
            return self.has_legal_move(other_team(move.team))
        finally:
            self._undo(move)
            self._snap = saved

    # ------------------------------------------------------------------
    # Teaching aid: why is this move not available?
    # ------------------------------------------------------------------
    def lock_reason(self, piece: Piece) -> str:
        """Why a locked piece can never move."""
        if piece.ptype == "p":
            return "This pawn is on its home row, so it is locked: home pawns never move again."
        return "This piece just reverse-castled, so it is locked on its home square."

    def explain_illegal(self, src: Pos, dst: Pos) -> str:
        """Plain-English reason the side to move cannot play ``src`` -> ``dst``.

        Returns "" when some action from src to dst is in fact legal.  Slow
        (it re-derives the legality of every gift), so call it only in
        response to a user click, never inside search or training loops.
        """
        piece = self.board[src]
        if piece is None:
            return "There is no piece on that square."
        if piece.team != self.turn:
            return "That is not your piece."
        if any(a.src == src and a.dst == dst for a in self.actions):
            return ""
        if piece.locked:
            return self.lock_reason(piece)
        if self.board[dst] is not None:
            return ("That square is occupied. Nothing is ever captured in reverse chess, "
                    "so you can only move to an empty square.")

        # Step 1: is the bare movement (ignoring gifts) legal at all?
        can_demote = piece.promoted and src[0] == promotion_row(piece.team)
        options: List[MoveOption] = []
        plain = self.check_legal_move(src, dst)
        if plain.legal:
            options.append(MoveOption(dst, False, plain.forced_give, plain.reverse_castle))
        pawn = None
        if can_demote:
            pawn = self.check_legal_move(src, dst, as_pawn=True)
            if pawn.legal:
                options.append(MoveOption(dst, True, pawn.forced_give, pawn.reverse_castle))
        if not options:
            why = self._movement_reason(piece, src, dst, plain, as_pawn=False)
            if pawn is not None:
                why += " (Stepping off as a pawn instead: %s)" % self._movement_reason(
                    piece, src, dst, pawn, as_pawn=True)
            return why

        # Step 2: the movement is fine, so a gift rule or the stalemate rule must object.
        offboard = self.offboard_by_type(other_team(piece.team))
        memo: dict = {}
        problems: Dict[str, List[Optional[str]]] = {}
        for opt in options:
            for action in self._candidate_actions(piece, opt, offboard):
                label, why = self._action_problem(piece, opt, action, offboard, memo)
                if why:
                    labels = problems.setdefault(why, [])
                    if label not in labels:
                        labels.append(label)
        if not problems:
            return "That move is not available."
        parts = []
        for why, labels in problems.items():
            bits = []
            if None in labels:
                bits.append("moving without a gift")
            gifts = [lab for lab in labels if lab]
            if gifts:
                bits.append("giving back a " + "/".join(gifts))
            parts.append("%s: %s" % (" or ".join(bits), why))
        if all(opt.forced_give for opt in options):
            head = "A diagonal pawn step must give a piece back, but no gift works here. "
        else:
            head = "That move is not available. "
        return head + "; ".join(parts) + "."

    def _movement_reason(self, piece: Piece, src: Pos, dst: Pos, chk: MoveCheck,
                         as_pawn: bool) -> str:
        """Explain a failed check_legal_move() in plain English."""
        if chk.reason == "bad movement for piece type":
            return self._shape_reason(piece, "p" if as_pawn else piece.ptype, src, dst)
        return {
            "piece(s) in the way": "Another piece is in the way (only knights jump).",
            "would attack the enemy king": (
                "That would leave one of your pieces attacking the enemy king. In reverse "
                "chess that would mean their king had just walked into check, so it is illegal."),
            "attacked by a pawn at home": (
                "An enemy pawn standing on its home row attacks that square, and a home-row "
                "pawn can never give check, so your king may not stand there."),
            "reverse castle into check": (
                "A king may not reverse-castle onto a square an enemy piece attacks."),
        }.get(chk.reason, chk.reason or "That move is not available.")

    def _shape_reason(self, piece: Piece, ptype: str, src: Pos, dst: Pos) -> str:
        """Why src -> dst is the wrong geometry for ``ptype``."""
        team = piece.team
        if ptype == "n":
            return "Knights move in an L: two squares one way and one square the other."
        if ptype == "k":
            return ("Kings move one square in any direction. The only longer move is a "
                    "reverse castle: a lone king and rook on the back row, with the king "
                    "on the c/g file and the rook beside it.")
        if ptype == "r":
            return "Rooks move in straight lines along a row or column."
        if ptype == "b":
            return "Bishops move along diagonals."
        if ptype == "q":
            return "Queens move along rows, columns and diagonals."
        # pawn
        dr = dst[0] - src[0]
        if dr != pawn_step(team):
            return ("A pawn takes exactly one step per move, toward its home row: %s the "
                    "board for %s." % ("up" if pawn_step(team) < 0 else "down", full_name(team)))
        if dst[0] in (0, SIZE - 1):
            return "A pawn can never step onto a back row."
        if abs(dst[1] - src[1]) > 1:
            return "A pawn's step shifts at most one column (a diagonal step)."
        return ("A diagonal pawn step un-captures, so it must give back an opponent piece, "
                "but every %s piece is already on the board." % full_name(other_team(team)))

    def _action_problem(self, mover: Piece, opt: MoveOption, action: Action,
                        offboard: Dict[str, Piece], memo: dict):
        """(label, reason) for one candidate action; reason is None if it is playable.

        ``label`` is None for the gift-less action, else the gifted piece's name.
        """
        if action.gift is None:
            label = None
        else:
            label = _PIECE_NAMES[action.gift]
            spawn = offboard[action.gift]
            opp = other_team(mover.team)
            spawn_pos = ((pawn_home_row(opp), action.src[1]) if action.en_passant
                         else action.src)
            promoting = spawn.ptype == "p" and action.promote is not None
            why = self._gift_problem(mover, action.src, action.dst, spawn, spawn_pos,
                                     promoting, memo)
            if why:
                return label, why
        move = self._make_move(action, opt, offboard, memo)
        if move is None:
            return label, "it is not allowed"
        if not self._leaves_reply(move):
            return label, ("it would leave your opponent with no legal move at all, "
                           "which the game forbids")
        return label, None

    # ------------------------------------------------------------------
    # Execute / undo
    # ------------------------------------------------------------------
    def _execute(self, move: Move):
        """Apply a move to the board.  Assumes it has been validated."""
        mover = move.mover
        if move.partner is not None:  # reverse castle: bring the partner home first
            partner = move.partner
            partner.move_home(4 if partner.ptype == "k"
                              else (0 if partner.pos[1] < 4 else 7))
            partner.lock()
        mover.update(move.dst)
        if move.reverse_castle:
            mover.lock()
        if move.demote:
            move.old_ptype = mover.ptype
            mover.demote()
        if move.spawn is not None:
            move.spawn.place(move.spawn_pos)
            if move.promote_to is not None:
                move.spawn.promote(move.promote_to)

    def _undo(self, move: Move):
        """Exactly reverse _execute()."""
        if move.spawn is not None:
            move.spawn.remove()  # also un-promotes the gifted piece
        mover = move.mover
        if move.demote:
            mover.promote(move.old_ptype)
        mover.unlock()
        mover.update(move.src)
        if move.partner is not None:
            move.partner.unlock()
            move.partner.update(move.partner_src)

    # ------------------------------------------------------------------
    # Turn flow
    # ------------------------------------------------------------------
    def _finish(self, winner: Optional[str], reason: str):
        self.done = True
        self.winner = winner
        self.end_reason = reason
        self.actions = {}
        self._raw = None

    def _begin_turn(self):
        """Work out the side to move's legal actions, or end the game."""
        team = self.turn
        self.actions = {}
        self._raw = None
        if self._use_native:
            n, raw = _native_enumerate(self._pack(), 0 if team == "b" else 1)
            if n > 0:
                # Moves are bound to pieces lazily (see _resolve): only the one played.
                self.actions = dict.fromkeys(
                    Action(_SQ[s], _SQ[d], bool(dm), _GIFT_CODES[g], bool(ep), _PROMO_CODES[pr])
                    for s, d, dm, g, ep, pr in struct.iter_unpack("6B", raw))
                self._raw = raw
                return
            valid, playable = (n == -2), {}
        else:
            valid, playable = ({}, {})
            if self.has_legal_move(team):
                valid, playable = self._enumerate()
        if not valid:
            # No legal move: you lose, unless you are already home.
            winner = team if self.check_for_win(team) else other_team(team)
            self._finish(winner, "%s has no legal moves" % full_name(team))
        elif not playable:
            self._finish(None, "draw: every legal move leaves the opponent "
                               "without a legal move")
        else:
            self.actions = playable

    def step(self, action: Action):
        """Play ``action``.  Afterwards check ``done``/``winner``/``legal_actions()``."""
        if self.done:
            raise IllegalActionError("the game is over (%s)" % self.end_reason)
        if action not in self.actions:
            raise IllegalActionError("illegal action: %r" % (action,))
        move = self.actions[action]
        if move is None:                      # native generator: bind to pieces now
            move = self._resolve(action)
        self._execute(move)
        self.ply += 1
        self.last_move = move
        self.move_log.append(action)

        # Reverse en passant gives the mover another turn.
        next_turn = move.team if move.extra_turn else other_team(move.team)

        # Progress counter for the fifty-move rule: gifts and pawn moves
        # (including un-promotions) are progress, everything else is quiet.
        progress = (move.spawn is not None or move.mover_ptype == "p" or move.demote)
        self.quiet_plies = 0 if progress else self.quiet_plies + 1
        key = self.snapshot() + next_turn.encode()
        self.seen[key] = self.seen.get(key, 0) + 1

        # Wins and losses take priority over draws.
        b_reason = self._win_reason("b")
        w_reason = self._win_reason("w")
        self.turn = next_turn
        if b_reason and w_reason:
            self._finish(None, "draw: both teams satisfy a winning condition")
        elif b_reason or w_reason:
            self._finish("b" if b_reason else "w", b_reason or w_reason)
        elif self.quiet_plies >= self.rules.fifty_move_limit:
            self._finish(None, "draw: no gift or pawn move in %d plies" % self.quiet_plies)
        elif self.seen[key] >= self.rules.repetition_limit:
            self._finish(None, "draw: position repeated %d times" % self.seen[key])
        elif self.rules.max_plies is not None and self.ply >= self.rules.max_plies:
            self._finish(None, "draw: move limit reached")
        else:
            self._begin_turn()

    # ------------------------------------------------------------------
    # Winning and losing
    # ------------------------------------------------------------------
    def check_for_win(self, team: str) -> bool:
        """Has ``team`` brought its pieces home?

        By default only pieces already on the board count; see
        ``Rules.win_requires_all_pieces_on_board``.
        """
        if (self.rules.win_requires_all_pieces_on_board
                and not (self.all_on_board("b") and self.all_on_board("w"))):
            return False
        return all(p.at_home() for p in self.pieces[team] if p.ontheboard)

    def _win_reason(self, team: str) -> Optional[str]:
        """Why ``team`` has just won (it got home, or the other team is lost)."""
        if self.check_for_win(team):
            return "%s has every piece home" % full_name(team)
        why = self._losing_reason(other_team(team))
        if why:
            return "%s wins: %s" % (full_name(team), why)
        return None

    def check_losing(self, team: str) -> bool:
        """Has ``team`` painted itself into a corner it can never escape?"""
        return self._losing_reason(team) is not None

    def _losing_reason(self, team: str) -> Optional[str]:
        """Return why ``team`` can no longer win, or None (native when available)."""
        if not self._use_native:
            return self._losing_reason_py(team)
        code = _LIB.rc_losing(self._pack(), 0 if team == "b" else 1)
        if code == 0:
            return None
        name = full_name(team)
        if code == 1:
            return "pawn wall-off prevents %s from winning" % name
        if code == 2:
            return "no way for all of %s's pawns to get home" % name
        if code == 3:
            return "no way for all of %s's bishops to get home" % name
        raise RuntimeError("native engine failed (code %d)" % code)

    def _losing_reason_py(self, team: str) -> Optional[str]:
        """Pure-Python reference for _losing_reason: why ``team`` can no longer win, or None.

        Three kinds of dead end are detected:
          * all pawns are home (a solid wall) yet a non-knight piece is still
            outside - it can never get past the wall (approximation: a piece
            already stuck behind the wall on the back row also counts);
          * the pawns cannot be assigned distinct home columns within the
            number of gifts that are still possible;
          * a bishop is walled off from its home square.
        """
        pieces = self.pieces[team]
        name = full_name(team)
        if not all(p.ontheboard and p.at_home() for p in pieces):
            moving = False  # some pawn still has somewhere to be
            for p in pieces:
                if not p.ontheboard:
                    if p.ptype == "p" or p.promoted:
                        moving = True  # a pawn is still to be given back
                        break
                elif (p.ptype == "p" or p.promoted) and not p.at_home():
                    moving = True
                    break
            if not moving:
                outside = any(p.ontheboard and p.ptype != "n" and not p.at_home()
                              for p in pieces)
                if outside:
                    return "pawn wall-off prevents %s from winning" % name
        if not self.pawn_scan(team):
            return "no way for all of %s's pawns to get home" % name
        if not self.bishop_wall_check(team):
            return "no way for all of %s's bishops to get home" % name
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def clone(self) -> "Game":
        """Independent deep copy (for tree search).  Caches are not copied."""
        saved = self._moves_cache, self._has_move
        self._moves_cache, self._has_move = {}, {}
        try:
            other = copy.deepcopy(self)
        finally:
            self._moves_cache, self._has_move = saved
        return other

    def validate(self):
        """Assert the internal bookkeeping is consistent (use in tests)."""
        seen = 0
        for r in range(SIZE):
            for c in range(SIZE):
                p = self.board[r, c]
                if p is not None:
                    assert p.ontheboard and p.pos == (r, c), (p, r, c)
                    seen += 1
        total_on = 0
        for team in TEAMS:
            counts = {"p": 0, "k": 0, "r": 0, "b": 0, "q": 0, "n": 0}
            for p in self.pieces[team]:
                counts["p" if p.promoted else p.ptype] += 1
                if p.ontheboard:
                    total_on += 1
                    assert self.board[p.pos] is p, p
                else:
                    assert p.pos is None and not p.locked and not p.promoted, p
                if p.locked:
                    assert p.ontheboard and not p.promoted, p
            assert counts == {"p": 8, "k": 1, "r": 2, "b": 2, "q": 1, "n": 2}, (team, counts)
        assert seen == total_on, (seen, total_on)

    def ascii(self) -> str:
        """Plain-text board: white UPPER case, black lower case, '*' = promoted."""
        lines = []
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                p = self.board[r, c]
                if p is None:
                    cells.append(" . ")
                else:
                    ch = p.ptype.upper() if p.team == "w" else p.ptype
                    cells.append(" %s%s" % (ch, "*" if p.promoted else " "))
            lines.append("%d%s" % (r, "".join(cells)))
        lines.append("  " + "".join(" %d " % c for c in range(SIZE)))
        return "\n".join(lines)


if __name__ == "__main__":
    g = Game()
    print(g.ascii())
    print(len(g.legal_actions()), "legal actions for", full_name(g.turn))
