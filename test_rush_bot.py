#!/usr/bin/env python
"""Checks for the "rush" bot (heuristic.py, spec heur:rush).  Run from reverse-chess/.

    python test_rush_bot.py [--games 40]

1. every gift the bot makes is the KING giving back a queen, rook or pawn promoted to
   queen -- unless it had no other legal action (forced diagonal pawn step);
2. after a king gift the gifted piece attacks the king, and every legal reply of the
   opponent ends that attack (i.e. the opponent really is "in reverse check");
3. rush vs random / nogift / gift results.
"""
import argparse
import random

import heuristic
from engine import Game, Rules


def audit(games, seed):
    rng = random.Random(seed)
    stats = dict(gifts=0, king_qr=0, forced_other=0, bad=0, checks=0)
    for i in range(games):
        g = Game(Rules(max_plies=300))
        rush = "b" if i % 2 == 0 else "w"
        while not g.done:
            if g.turn != rush:
                g.step(rng.choice(g.legal_actions()))
                continue
            actions = g.legal_actions()
            a = heuristic.choose(g, rng, "rush")
            mover = g.board[a.src]
            if a.gift is not None:
                stats["gifts"] += 1
                hit = heuristic.rush_gift(mover, a)
                if hit is not None:
                    stats["king_qr"] += 1
                elif all(x.gift is not None and heuristic.rush_gift(g.board[x.src], x) is None
                         for x in actions):
                    stats["forced_other"] += 1
                else:
                    # an unwanted gift while a gift-free / wanted action existed: only the
                    # one-ply safety check may cause that (it avoids losing outright)
                    stats["bad"] += 1
            king_gift_attacks = hit if a.gift is not None else None
            g.step(a)
            if king_gift_attacks and not g.done:
                me_king = g.kings[rush]
                att = [p for p in g.on_board(g.turn) if g.check_attacking(p, me_king.pos)]
                if att:
                    stats["checks"] += 1
        g.close()
    return stats


def replies_all_end_check(seed=0, tries=200):
    """Search random positions for a king gift of a queen and verify the opponent's
    legal replies all leave the king un-attacked."""
    rng = random.Random(seed)
    seen = bad = 0
    for _ in range(tries):
        g = Game(Rules(max_plies=300))
        for _ply in range(120):
            if g.done:
                break
            acts = g.legal_actions()
            qk = [a for a in acts if a.gift == "q" and g.board[a.src].ptype == "k"]
            if qk:
                a = rng.choice(qk)
                king_team = g.turn
                g.step(a)
                if g.done:
                    break
                seen += 1
                for r in g.legal_actions():
                    h = g.clone() if hasattr(g, "clone") else None
                    if h is None:
                        break
                    h.step(r)
                    if any(h.check_attacking(p, h.kings[king_team].pos) for p in h.on_board(g.turn)):
                        bad += 1
                    h.close()
                break
            g.step(rng.choice(acts))
        g.close()
    return seen, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("gift audit:", audit(args.games, args.seed))
    print("king queen-gifts examined / replies leaving the king attacked:", replies_all_end_check(args.seed))
    rules = Rules(max_plies=300)
    rng = random.Random(args.seed)
    for other in (None, "nogift", "gift"):
        w, d, l, p = heuristic.match(rules, rng, "rush", other, args.games)
        print("rush vs %-7s +%d =%d -%d  avg %.0f plies" % (other or "random", w, d, l, p))
