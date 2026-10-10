#!/usr/bin/env python
"""A simple rule-based Reverse Chess player: "walk the pieces home".  (Run from reverse-chess/.)

    python heuristic.py                      # vs random, both modes, 200 games each way
    python heuristic.py --games 400 --max-plies 300

No search and no learning: every legal action gets a score from the mover's
own piece geometry, ties are broken at random.

    progress   how many squares closer the moved piece got to its nearest home
               square (king metric; pawns count rows; a promoted piece first
               walks to its un-promotion row, then back to the pawn row)
    arrival    bonus for stepping onto a home square, penalty for leaving one
    gifts      mode "nogift":  a gift costs a huge penalty, so it only gifts
                               when every legal action gifts (or the move
                               forces it, e.g. a diagonal pawn step)
               mode "gift":    always gifts when it can (what the demo race's
                               defender does), then picks by progress

Use ``choose(game, rng, mode)`` from other code; it returns an Action.  By default
a one-ply safety check through the engine is added (``safe=True``, ``--no-safe``
turns it off); it costs one clone+step per legal action.
"""
import argparse
import random
import time

from engine import SIZE, Game, Rules, pawn_home_row, promotion_row

GIFT_PENALTY = 100.0
ARRIVE = 5.0


def distance(piece, pos):
    """Rough number of moves for ``piece`` standing on ``pos`` to get home."""
    rows, cols = piece.home
    r, c = pos
    if piece.promoted:
        # must reach the un-promotion row, step off as a pawn, walk down to the pawn row
        return abs(r - promotion_row(piece.team)) + abs(promotion_row(piece.team) - pawn_home_row(piece.team))
    if piece.ptype == "p":
        return abs(r - rows[0])
    return min(max(abs(r - hr), abs(c - hc)) for hr in rows for hc in cols)


def at_home_square(piece, pos):
    rows, cols = piece.home
    return (not piece.promoted) and pos[0] in rows and pos[1] in cols


def score(game, action, mode):
    mover = game.board[action.src[0], action.src[1]]
    d0 = distance(mover, action.src)
    if action.demote:               # steps off as a pawn: now measured as a pawn
        d1 = abs(action.dst[0] - pawn_home_row(mover.team))
    else:
        d1 = distance(mover, action.dst)
    s = float(d0 - d1)
    if at_home_square(mover, action.src) and not at_home_square(mover, action.dst):
        s -= ARRIVE
    if at_home_square(mover, action.dst) and not mover.promoted and not action.demote:
        s += ARRIVE
    if action.gift is not None:
        s += GIFT_PENALTY if mode == "gift" else -GIFT_PENALTY
    return s


def choose(game, rng, mode="nogift", safe=True):
    """rng: a random.Random (or anything with .choice / .random).

    safe=True adds a one-ply check through the engine: take a move that wins
    outright, otherwise the best-scoring move that does not end the game
    against us (or in a draw), otherwise a draw, otherwise the best score.
    This is what stops the walk-home rule from blocking its own pawns.
    """
    actions = game.legal_actions()
    scores = [score(game, a, mode) for a in actions]
    order = sorted(range(len(actions)), key=lambda i: (-scores[i], rng.random()))
    if not safe:
        return actions[order[0]]
    me = game.turn
    draw = None
    first_ok = None
    for i in order:
        result = peek(game, actions[i])
        if result == "win":
            return actions[i]                  # outright win
        if result == "draw":
            draw = draw or actions[i]
        elif result == "ok" and first_ok is None:
            first_ok = actions[i]
        # "loss": keep looking
    return first_ok or draw or actions[order[0]]


def peek(game, action):
    """What the engine would say right after ``action``: "win" / "loss" / "draw"
    for the mover, or "ok" when the game goes on.  Uses the engine's own
    hypothetical-move pair (_execute / _undo), so nothing is cloned and the game is
    left exactly as it was.  Ignores repetition / fifty-move / ply-cap draws."""
    me = game.turn
    move = game.actions[action]
    if move is None:                           # native generator: bind to pieces now
        move = game._resolve(action)
    game._execute(move)
    try:
        mine = game._win_reason(me) is not None
        theirs = game._win_reason("w" if me == "b" else "b") is not None
    finally:
        game._undo(move)
    if mine and theirs:
        return "draw"
    return "win" if mine else ("loss" if theirs else "ok")


# --------------------------------------------------------------------------
# a quick test bench against a uniformly random player
# --------------------------------------------------------------------------

def play(rules, rng, black, white, safe=True):
    """black/white: None (uniform random) or a mode string.  Returns the Game."""
    game = Game(rules)
    who = {"b": black, "w": white}
    while not game.done:
        mode = who[game.turn]
        if mode is None:
            action = rng.choice(game.legal_actions())
        else:
            action = choose(game, rng, mode, safe)
        game.step(action)
    return game


def match(rules, rng, a_mode, b_mode, games, safe=True):
    """a vs b, half the games each colour.  Returns (A wins, draws, A losses, avg plies)."""
    w = d = l = plies = 0
    for i in range(games):
        a_team = "b" if i % 2 == 0 else "w"
        g = play(rules, rng, a_mode if a_team == "b" else b_mode, b_mode if a_team == "b" else a_mode, safe)
        plies += g.ply
        if g.winner is None:
            d += 1
        elif g.winner == a_team:
            w += 1
        else:
            l += 1
        g.close()
    return w, d, l, plies / games


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-safe", action="store_true", help="skip the one-ply safety check")
    args = ap.parse_args()
    rules = Rules(max_plies=args.max_plies)
    rng = random.Random(args.seed)
    for a, b in (("nogift", None), ("gift", None), ("nogift", "gift")):
        t = time.time()
        w, d, l, p = match(rules, rng, a, b, args.games, not args.no_safe)
        print("%-7s vs %-7s  +%d =%d -%d  score %.3f  avg %.0f plies  (%.0fs)" % (
            a, b or "random", w, d, l, (w + 0.5 * d) / args.games, p, time.time() - t), flush=True)
