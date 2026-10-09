#!/usr/bin/env python
"""Neural-network interface to Reverse Chess.

Nothing here touches pygame or tkinter.  It turns a :class:`engine.Game` into
tensors and back:

* ``encode_observation(game)``  -> board planes + scalar features, always seen
  from the side to move's point of view (the board is flipped for black), so
  one network can play both colours.
* ``encode_action`` / ``decode_action`` -> a canonical integer id for every
  possible Action, plus ``action_components`` for factorised policy heads.
* ``ReverseChessEnv`` -> a small gym-style environment (``reset``/``step``)
  with legal-action masking, sparse +1/-1/0 rewards and a ply cap.

Design notes for the policy head
--------------------------------
The full action space is large (``ACTION_SPACE_SIZE``, ~570k ids) but any one
position has only tens to a few thousand legal actions.  Two sensible designs:

1. *Pointer / scoring head*: embed each legal action (use
   ``action_components`` as inputs to a few embedding tables), score it against
   the board encoding, and softmax over only the legal list.  No giant output
   layer; this is the recommended design.
2. *Factorised head*: one softmax each for src, dst, demote, gift, en-passant,
   promote, masked by the legal set.  Cheap to build, but needs care because
   the factors are not independent.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from chessassets import PTYPES
from engine import (Action, Game, Rules, SIZE, TEAMS, PROMOTION_TYPES,
                    other_team)

# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

#: Board planes, all 8x8, from the mover's perspective ("own" = side to move):
#:   0-5   own pieces by type (order of chessassets.PTYPES: p k r b q n)
#:   6-11  opponent pieces by type
#:   12/13 own / opponent promoted pieces
#:   14/15 own / opponent locked pieces (home pawns, castled king+rook)
#:   16/17 own / opponent pieces standing on a home square
NUM_BOARD_PLANES = 18

#: Giftable piece types (kings are never off the board), in a fixed order.
_COUNT_TYPES = ("p", "r", "b", "q", "n")
_MAX_OFFBOARD = {"p": 8, "r": 2, "b": 2, "q": 1, "n": 2}
#: Scalars: own off-board counts (5), opponent off-board counts (5),
#: quiet-ply fraction, game-length fraction, repetition count, extra-turn flag,
#: own-all-on-board, opponent-all-on-board.
NUM_SCALARS = 16

OBSERVATION_SHAPES = {"board": (NUM_BOARD_PLANES, SIZE, SIZE), "scalars": (NUM_SCALARS,)}

_PTYPE_INDEX = {pt: i for i, pt in enumerate(PTYPES)}


def canonical_square(team: str, pos: Tuple[int, int]) -> Tuple[int, int]:
    """Flip rows for black so that the mover's home is always at the bottom."""
    return pos if team == "w" else (SIZE - 1 - pos[0], pos[1])


def encode_observation(game: Game, max_plies: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Encode the position for the side to move.

    Returns ``{"board": float32 (18, 8, 8), "scalars": float32 (16,)}``.
    ``max_plies`` normalises the game-length feature (defaults to the rules'
    cap, or 400 when the game is uncapped).
    """
    me = game.turn
    board = np.zeros((NUM_BOARD_PLANES, SIZE, SIZE), dtype=np.float32)
    for team in TEAMS:
        side = 0 if team == me else 1
        for p in game.pieces[team]:
            if not p.ontheboard:
                continue
            r, c = canonical_square(me, p.pos)
            board[side * 6 + _PTYPE_INDEX[p.ptype], r, c] = 1.0
            if p.promoted:
                board[12 + side, r, c] = 1.0
            if p.locked:
                board[14 + side, r, c] = 1.0
            if p.at_home():
                board[16 + side, r, c] = 1.0

    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    k = 0
    for team in (me, other_team(me)):
        counts = {pt: 0 for pt in _COUNT_TYPES}
        for p in game.pieces[team]:
            if not p.ontheboard and p.ptype in counts:
                counts[p.ptype] += 1
        for pt in _COUNT_TYPES:
            scalars[k] = counts[pt] / _MAX_OFFBOARD[pt]
            k += 1
    cap = max_plies or game.rules.max_plies or 400
    scalars[10] = min(game.quiet_plies / max(game.rules.fifty_move_limit, 1), 1.0)
    scalars[11] = min(game.ply / cap, 1.0)
    scalars[12] = (game.seen.get(game.position_key(), 0)) / max(game.rules.repetition_limit, 1)
    scalars[13] = float(game.last_move is not None and game.last_move.extra_turn
                        and game.last_move.team == me)
    scalars[14] = float(game.all_on_board(me))
    scalars[15] = float(game.all_on_board(other_team(me)))
    return {"board": board, "scalars": scalars}


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

#: Gift code 0 means "no gift"; 1..6 index chessassets.PTYPES.
_NUM_GIFT = len(PTYPES) + 1
#: Promotion code 0 means "none"; 1..4 index engine.PROMOTION_TYPES.
_NUM_PROMO = len(PROMOTION_TYPES) + 1

#: Size of the canonical integer action space (most ids are never legal).
ACTION_SPACE_SIZE = (SIZE * SIZE) * (SIZE * SIZE) * 2 * _NUM_GIFT * 2 * _NUM_PROMO


def action_components(action: Action, team: str) -> Tuple[int, int, int, int, int, int]:
    """Break an action into small integers (canonical frame for ``team``):
    ``(src 0-63, dst 0-63, demote 0/1, gift 0-6, en_passant 0/1, promote 0-4)``.
    Handy as inputs to embedding tables or as targets of factorised heads."""
    sr, sc = canonical_square(team, action.src)
    dr, dc = canonical_square(team, action.dst)
    gift = 0 if action.gift is None else 1 + _PTYPE_INDEX[action.gift]
    promo = 0 if action.promote is None else 1 + PROMOTION_TYPES.index(action.promote)
    return (sr * SIZE + sc, dr * SIZE + dc, int(action.demote), gift,
            int(action.en_passant), promo)


def encode_action(action: Action, team: str) -> int:
    """Canonical integer id of ``action`` for the side ``team`` (a bijection)."""
    src, dst, demote, gift, ep, promo = action_components(action, team)
    index = src * (SIZE * SIZE) + dst
    index = index * 2 + demote
    index = index * _NUM_GIFT + gift
    index = index * 2 + ep
    index = index * _NUM_PROMO + promo
    return index


def decode_action(index: int, team: str) -> Action:
    """Inverse of :func:`encode_action` (does not check legality)."""
    index, promo = divmod(int(index), _NUM_PROMO)
    index, ep = divmod(index, 2)
    index, gift = divmod(index, _NUM_GIFT)
    index, demote = divmod(index, 2)
    src, dst = divmod(index, SIZE * SIZE)

    def square(n: int) -> Tuple[int, int]:
        return canonical_square(team, divmod(n, SIZE))  # the flip is its own inverse

    return Action(square(src), square(dst), bool(demote),
                  None if gift == 0 else PTYPES[gift - 1], bool(ep),
                  None if promo == 0 else PROMOTION_TYPES[promo - 1])


def legal_action_indices(game: Game) -> np.ndarray:
    """Canonical ids of all legal actions, as an int64 array."""
    return np.array([encode_action(a, game.turn) for a in game.legal_actions()],
                    dtype=np.int64)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

class ReverseChessEnv:
    """Two-player, alternating-turn, gym-style environment.

    Conventions
    -----------
    * Observations are from the point of view of the side *about to move*.
    * ``step`` returns the reward for the player who *just moved*:
      +1 they won, -1 they lost, 0 otherwise (including draws).
    * A reverse-en-passant move gives the same player another turn, so
      ``current_player`` does not always alternate; check ``info["player"]``.
    * ``truncated`` is True when the game was cut off by the ply cap.
    * Illegal actions raise ``engine.IllegalActionError``; use
      ``legal_action_indices()`` to mask your policy.
    """

    def __init__(self, rules: Optional[Rules] = None, max_plies: Optional[int] = 600):
        if rules is None:
            rules = Rules(max_plies=max_plies)
        self.rules = rules
        self.game = Game(rules)
        self._lookup: Dict[int, Action] = {}
        self._refresh()

    # -- gym-ish API -----------------------------------------------------
    def reset(self) -> Tuple[Dict[str, np.ndarray], dict]:
        """Start a new game; returns ``(observation, info)``."""
        self.game.reset()
        self._refresh()
        return self.observe(), self._info(player=self.game.turn)

    def step(self, action: Union[int, Action]):
        """Play an action (canonical id or Action).

        Returns ``(observation, reward, terminated, truncated, info)``.
        """
        game = self.game
        mover = game.turn
        if not isinstance(action, Action):
            action = self._lookup.get(int(action)) or decode_action(int(action), mover)
        game.step(action)
        self._refresh()

        reward = 0.0
        if game.done and game.winner is not None:
            reward = 1.0 if game.winner == mover else -1.0
        truncated = game.done and game.end_reason.startswith("draw: move limit")
        terminated = game.done and not truncated
        info = self._info(player=mover)
        return self.observe(), reward, terminated, truncated, info

    # -- helpers ---------------------------------------------------------
    @property
    def current_player(self) -> str:
        """Side to move: 'b' or 'w'."""
        return self.game.turn

    def observe(self) -> Dict[str, np.ndarray]:
        """Observation for the side to move."""
        return encode_observation(self.game, self.rules.max_plies)

    def legal_actions(self) -> List[Action]:
        """Legal Action objects for the side to move."""
        return self.game.legal_actions()

    def legal_action_indices(self) -> np.ndarray:
        """Canonical ids of the legal actions (int64)."""
        return np.fromiter(self._lookup.keys(), dtype=np.int64, count=len(self._lookup))

    def clone(self) -> "ReverseChessEnv":
        """Independent copy, e.g. for MCTS."""
        other = ReverseChessEnv.__new__(ReverseChessEnv)
        other.rules = self.rules
        other.game = self.game.clone()
        other._refresh()
        return other

    def _refresh(self):
        team = self.game.turn
        self._lookup = {encode_action(a, team): a for a in self.game.legal_actions()}

    def _info(self, player: str) -> dict:
        game = self.game
        return {"player": player, "to_move": game.turn, "winner": game.winner,
                "end_reason": game.end_reason, "ply": game.ply,
                "num_legal": len(game.actions)}
