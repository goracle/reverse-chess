#!/usr/bin/env python
"""Headless entry point for the new engine (no pygame, no tkinter).

    python selfplay.py                  # 20 random games, the real rules
    python selfplay.py --demo-race      # black walks its king home; white
                                        #   defends by gifting when it can
    python selfplay.py --demo-race --no-defence   # white just plays randomly
    python selfplay.py --games 1 --show # print every move + the final board
    python selfplay.py --strict-win     # experimental: all 32 pieces on board

The real win rule: a team wins when every piece *currently on the board* is
home.  Gifting a piece to the opponent puts it on the board and adds it to
*their* to-do list, so gifts are how you slow the other side down.  The
opening is a tempo race: black's king needs three moves to get home.

This file does not replace the pygame GUI, which still lives in
reverse_chess.py.
"""
import argparse
import collections
import random

from engine import Action, Game, Rules, full_name


def random_game(rules, rng, show=False, validate=False):
    """Play one game with uniformly random legal moves; return the Game."""
    game = Game(rules)
    while not game.done:
        action = rng.choice(game.legal_actions())
        if show:
            print("%3d %s %s" % (game.ply + 1, full_name(game.turn), action))
        game.step(action)
        if validate:
            game.validate()
    return game


def demo_race(rules, rng, defend=True):
    """Black walks its king (3,5) -> ... -> (0,4) with no gifts.

    White either defends (gifts black a piece whenever a gifting move is
    legal, which gives black a new piece to bring home) or plays randomly.
    Without a defence black is home at ply 5.  With one, watch whether the
    gift is always available and how long black's to-do list gets.
    """
    game = Game(rules)
    home = (0, 4)
    while not game.done:
        legal = game.legal_actions()
        if game.turn == "b":
            king = game.kings["b"].pos
            quiet = [a for a in legal
                     if a.src == king and a.gift is None and not a.demote]
            action = min(quiet or legal,
                         key=lambda a: max(abs(a.dst[0] - home[0]),
                                           abs(a.dst[1] - home[1])))
        else:
            gifts = [a for a in legal if a.gift is not None]
            action = rng.choice(gifts if (defend and gifts) else legal)
        print("%3d %s %s" % (game.ply + 1, full_name(game.turn), action))
        game.step(action)
    print(game.ascii())
    print("winner:", full_name(game.winner), "|", game.end_reason, "| plies:", game.ply)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--strict-win", action="store_true",
                    help="experimental: every piece of both teams must be on the board")
    ap.add_argument("--show", action="store_true", help="print each move and final board")
    ap.add_argument("--validate", action="store_true",
                    help="run Game.validate() after every ply (slow)")
    ap.add_argument("--demo-race", action="store_true",
                    help="black king walk vs. a gifting defence")
    ap.add_argument("--no-defence", action="store_true",
                    help="with --demo-race: white plays randomly instead of gifting")
    args = ap.parse_args()

    rules = Rules(win_requires_all_pieces_on_board=args.strict_win,
                  max_plies=args.max_plies)
    rng = random.Random(args.seed)

    if args.demo_race:
        demo_race(rules, rng, defend=not args.no_defence)
        return

    results = collections.Counter()
    reasons = collections.Counter()
    plies = []
    for _ in range(args.games):
        game = random_game(rules, rng, show=args.show, validate=args.validate)
        if args.show:
            print(game.ascii())
        results[game.winner or "draw"] += 1
        reasons[game.end_reason.split(":")[0]] += 1
        plies.append(game.ply)
    print("results:", dict(results))
    print("average plies: %.1f" % (sum(plies) / len(plies)))
    for reason, count in reasons.most_common():
        print("  %3d  %s" % (count, reason))


if __name__ == "__main__":
    main()
