#!/usr/bin/env python
"""A light, immutable game position for tree search, stepped natively.

``engine.Game`` is built for playing one game: 32 ``Piece`` objects that refer back to the
game, a numpy object board, dicts of ``Action`` objects.  Tree search needs the opposite --
millions of throw-away positions.  Copying a ``Game`` (``mcts.fast_clone``) costs about 0.7 ms
and stepping the copy another 0.3 ms, and most of the latter is building ``Action`` objects
nobody reads.

``FastGame`` is a position as plain bytes, and ``FastGame.child(i)`` plays the i-th legal
action with ONE call into the Rust library (``rc_step``): move applied, wins and draws
decided, the next side's actions enumerated.  Nothing is copied or constructed but a few
byte strings.  Actions stay as 6-byte rows (the same rows ``Game.action_array()`` returns)
and become ``Action`` objects only if somebody asks (``legal_actions()`` is lazy).

What a FastGame offers is what the networks and the search read from a ``Game``:

    turn, done, winner, end_reason, ply, quiet_plies, rules
    legal_actions()   lazy, list-like, same order as Game.legal_actions()
    action_array()    (n, 6) uint8, same as Game.action_array()
    observation()     env.encode_observation, computed natively (env.encode_observation
                      delegates here, so advisors need no change)
    child(i)          the position after the i-th legal action
    close()           no-op, so code written for Game.close() keeps working

It is only an engine-faithful *copy*: results are checked position by position against
``Game.step`` by check_fastgame.py.  ``AVAILABLE`` is False when the library is missing or
was built before rc_step existed (rebuild with ./build_native.sh); mcts.Searcher then falls
back to ``fast_clone``.
"""
from __future__ import annotations

import ctypes
import struct
import threading
from collections.abc import Sequence
from typing import Dict, List, Optional

import numpy as np

import engine
from engine import Action, Game, IllegalActionError, Rules, full_name

_LIB = engine._LIB
_HDR = 200                      # bytes before the actions in rc_step's output (see step.rs)
_BUF_CAP = 4096                 # actions the per-thread buffer holds; grown on overflow
_SQ = engine._SQ
_GIFT_CODES = engine._GIFT_CODES
_PROMO_CODES = engine._PROMO_CODES
_HEADER = struct.Struct("<BBBBH")    # at offset 192: turn letter, status, winner, reason, quiet

# rc_step statuses (step.rs)
_LIVE, _WIN, _DRAW_BOTH, _DRAW_QUIET, _MAXPLY, _NO_MOVES, _BLOCKED = range(7)

_TEAM = "bw"
_WHY = {1: "pawn wall-off prevents %s from winning",
        2: "no way for all of %s's pawns to get home",
        3: "no way for all of %s's bishops to get home"}


def _bind():
    """Declare rc_step / rc_observe on the engine's library; None if it lacks them."""
    if _LIB is None:
        return None
    try:
        if _LIB.rc_step_abi() != 1:
            return None
        _LIB.rc_step.restype = ctypes.c_int32
        _LIB.rc_step.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p,
                                 ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                 ctypes.c_int32, ctypes.c_int32,
                                 ctypes.c_void_p, ctypes.c_uint32]
        _LIB.rc_observe.restype = ctypes.c_int32
        _LIB.rc_observe.argtypes = [ctypes.c_char_p, ctypes.c_uint32,
                                    ctypes.c_void_p, ctypes.c_void_p]
        return _LIB
    except AttributeError:
        return None


_NATIVE = _bind()
AVAILABLE = _NATIVE is not None
_TLS = threading.local()        # per-thread output buffer for rc_step


class RawActions(Sequence):
    """The legal actions of a position as a list-like view of its 6-byte rows.

    ``Action`` objects are built on access, so a search that only needs the *number* of
    actions (and, for the root, a few of them) never pays for the rest.
    """
    __slots__ = ("raw",)

    def __init__(self, raw: bytes):
        self.raw = raw

    def __len__(self) -> int:
        return len(self.raw) // 6

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(len(self)))]
        n = len(self.raw) // 6
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("action index out of range")
        s, d, dm, g, ep, pr = self.raw[6 * i:6 * i + 6]
        return Action(_SQ[s], _SQ[d], bool(dm), _GIFT_CODES[g], bool(ep), _PROMO_CODES[pr])

    def __repr__(self) -> str:
        return "RawActions(%d)" % len(self)


class _Config:
    """Rules in the form rc_step takes them; one instance is shared by a whole search tree."""
    __slots__ = ("rules", "fifty", "max_plies", "flags", "rep_limit")

    def __init__(self, rules: Rules):
        self.rules = rules
        self.fifty = int(rules.fifty_move_limit)
        self.max_plies = -1 if rules.max_plies is None else int(rules.max_plies)
        self.flags = 1 if rules.win_requires_all_pieces_on_board else 0
        self.rep_limit = int(rules.repetition_limit)


def _buffer():
    buf = getattr(_TLS, "buf", None)
    if buf is None:
        buf = _TLS.buf = ctypes.create_string_buffer(_HDR + 6 * _BUF_CAP)
        _TLS.cap = _BUF_CAP
    return buf, _TLS.cap


class FastGame:
    """One position (see the module docstring).  Never mutated: ``child`` returns a new one."""

    __slots__ = ("cfg", "turn", "ply", "quiet_plies", "done", "winner", "end_reason",
                 "state", "raw", "key", "rep", "extra_turn", "_base", "_path")

    # -- construction ----------------------------------------------------
    @classmethod
    def from_game(cls, game: Game) -> "FastGame":
        """Snapshot of a real game (which is only read, never changed)."""
        if not AVAILABLE:
            raise RuntimeError("native engine with rc_step not available (run ./build_native.sh)")
        if isinstance(game, FastGame):      # immutable: a search may own it as it is
            return game
        self = object.__new__(cls)
        self.cfg = _Config(game.rules)
        self.turn = game.turn
        self.ply = game.ply
        self.quiet_plies = game.quiet_plies
        self.done = game.done
        self.winner = game.winner
        self.end_reason = game.end_reason
        self.state = game._pack()
        self.raw = b"" if game.done else game.action_array().tobytes()
        self.key = game.position_key()
        self.rep = game.seen.get(self.key, 0)
        lm = game.last_move
        self.extra_turn = bool(lm is not None and lm.extra_turn and lm.team == game.turn)
        self._base = dict(game.seen)    # positions seen before the root; never changed
        self._path = {}                 # positions reached below the root -> total count
        return self

    @classmethod
    def from_native(cls, cfg: "_Config", state: bytes, raw: bytes, key: bytes, turn: str, ply: int,
                    quiet_plies: int, rep: int, extra_turn: bool, seen: Dict[bytes, int]) -> "FastGame":
        """The live position of a ``nativegame.NativeGame`` (see nativegame.py): a search rooted
        here counts repetitions against ``seen``, the positions the game has been through
        (the root's own included, as ``from_game`` does with ``Game.seen``)."""
        self = object.__new__(cls)
        self.cfg = cfg
        self.turn = turn
        self.ply = ply
        self.quiet_plies = quiet_plies
        self.done = not raw
        self.winner = None
        self.end_reason = ""
        self.state = state
        self.raw = raw
        self.key = key
        self.rep = rep
        self.extra_turn = bool(extra_turn)
        self._base = dict(seen)
        self._path = {}
        return self

    # -- what search and the networks read ---------------------------------
    @property
    def rules(self) -> Rules:
        return self.cfg.rules

    def legal_actions(self) -> RawActions:
        return RawActions(self.raw)

    def action_array(self) -> np.ndarray:
        """(n, 6) uint8 rows, same as ``Game.action_array``; read-only, copy before editing."""
        return np.frombuffer(self.raw, dtype=np.uint8).reshape(-1, 6)

    @property
    def num_actions(self) -> int:
        return len(self.raw) // 6

    def position_key(self) -> bytes:
        return self.key

    def all_on_board(self, team: str) -> bool:
        base = 0 if team == "b" else 16
        st = self.state
        return all(st[4 * (base + k)] & 1 for k in range(16))

    def close(self) -> None:
        """Nothing to free; present so code written for ``Game.close`` works unchanged."""

    def observation(self, max_plies: Optional[int] = None) -> Dict[str, np.ndarray]:
        """``env.encode_observation(game, max_plies)``, computed by the native library."""
        board = np.empty((18, 8, 8), dtype=np.float32)
        scal = np.empty(17, dtype=np.float32)
        rc = _NATIVE.rc_observe(self.state, 0 if self.turn == "b" else 1,
                                board.ctypes.data, scal.ctypes.data)
        if rc != 0:
            raise RuntimeError("native observation failed (code %d)" % rc)
        rules = self.cfg.rules
        cap = max_plies or rules.max_plies or 400
        scal[10] = min(self.quiet_plies / max(rules.fifty_move_limit, 1), 1.0)
        scal[11] = min(self.ply / cap, 1.0)
        scal[12] = self.rep / max(rules.repetition_limit, 1)
        scal[13] = float(self.extra_turn)
        return {"board": board, "scalars": scal}

    # -- stepping ------------------------------------------------------------
    def child(self, i: int) -> "FastGame":
        """The position after the i-th legal action (an index into ``legal_actions()``)."""
        if self.done:
            raise IllegalActionError("the game is over (%s)" % self.end_reason)
        raw = self.raw
        o = 6 * i
        if i < 0 or o + 6 > len(raw):
            raise IllegalActionError("no legal action number %d" % i)
        cfg = self.cfg
        ply = self.ply + 1
        buf, cap = _buffer()
        while True:
            n = _NATIVE.rc_step(self.state, 0 if self.turn == "b" else 1, raw[o:o + 6],
                                self.quiet_plies, ply, cfg.fifty, cfg.max_plies, cfg.flags,
                                buf, cap)
            if n != -1:
                break
            cap *= 4                                  # more actions than the buffer holds
            buf = _TLS.buf = ctypes.create_string_buffer(_HDR + 6 * cap)
            _TLS.cap = cap
        if n < 0:
            raise RuntimeError("native step failed (code %d)" % n)
        blob = ctypes.string_at(buf, _HDR + 6 * n)
        letter, status, winner, reason, quiet = _HEADER.unpack_from(blob, 192)

        new = object.__new__(FastGame)
        new.cfg = cfg
        new.ply = ply
        new.quiet_plies = quiet
        new.turn = "b" if letter == 98 else "w"       # Game.step sets the turn even at the end
        new.state = blob[:128]
        new.key = key = blob[128:193]
        new.extra_turn = raw[o + 4] != 0
        new._base = base = self._base

        # Game.step counts the position before anything else is decided.
        path = self._path
        total = path.get(key)
        if total is None:
            total = base.get(key, 0)
        total += 1
        new.rep = total
        new._path = path = dict(path)
        path[key] = total

        new.raw = b""
        new.winner = None
        new.done = True
        # Wins first, then the fifty-move rule, then repetition, then the ply cap, then the
        # next side's moves: the order of Game.step.
        if status == _WIN:
            new.winner = _TEAM[winner]
            new.end_reason = self._win_text(winner, reason)
        elif status == _DRAW_BOTH:
            new.end_reason = "draw: both teams satisfy a winning condition"
        elif status == _DRAW_QUIET:
            new.end_reason = "draw: no gift or pawn move in %d plies" % quiet
        elif total >= cfg.rep_limit:
            new.end_reason = "draw: position repeated %d times" % total
        elif status == _MAXPLY:
            new.end_reason = "draw: move limit reached"
        elif status == _NO_MOVES:
            new.winner = _TEAM[winner]
            new.end_reason = "%s has no legal moves" % full_name(new.turn)
        elif status == _BLOCKED:
            new.end_reason = "draw: every legal move leaves the opponent without a legal move"
        else:
            new.done = False
            new.end_reason = ""
            new.raw = blob[_HDR:]
        return new

    @staticmethod
    def _win_text(winner: int, reason: int) -> str:
        team = _TEAM[winner]
        if reason == 1:
            return "%s has every piece home" % full_name(team)
        return "%s wins: %s" % (full_name(team), _WHY[reason - 10] % full_name(_TEAM[1 - winner]))

    def __repr__(self) -> str:
        if self.done:
            return "<FastGame ply %d done: %s>" % (self.ply, self.end_reason)
        return "<FastGame ply %d, %s to move, %d actions>" % (self.ply, self.turn, self.num_actions)
