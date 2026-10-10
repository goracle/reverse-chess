#!/usr/bin/env python
"""Native (Rust) move generator vs the pure-Python reference, in lockstep.

    python check_native.py                    # 100 random games, compare every position
    python check_native.py --games 500 --max-plies 600 --seed0 1000
    python check_native.py --no-time          # skip the speed comparison

Two Game objects play the same random game, one with native=True and one with
native=False.  After every ply it checks that both report the identical ordered
list of legal actions (repr-compared, like check_engine.py), the same board
fingerprint, and the same end of game (winner and reason).  Then it replays the
recorded games on each backend alone and reports plies/second.
"""
import argparse
import random
import sys
import time

import engine
from engine import Game, Rules


def diff_report(seed, ply, gn, gp, an, ap):
    print("MISMATCH  seed=%d ply=%d  (turn %s)" % (seed, ply, gp.turn))
    print(gp.ascii())
    only_n = [a for a in an if a not in set(ap)]
    only_p = [a for a in ap if a not in set(an)]
    print("native: %d actions, python: %d actions" % (len(an), len(ap)))
    print("only in native (%d):" % len(only_n))
    for a in only_n[:10]:
        print("   ", a)
    print("only in python (%d):" % len(only_p))
    for a in only_p[:10]:
        print("   ", a)
    if not only_n and not only_p:
        print("same action set, different order (first difference):")
        for k, (x, y) in enumerate(zip(an, ap)):
            if x != y:
                print("   index %d: native %r | python %r" % (k, x, y))
                break
    print("moves so far:", [repr(a) for a in gp.move_log])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()

    if not engine.NATIVE:
        print("native library not loaded: run ./build_native.sh first "
              "(or unset RC_ENGINE=py)")
        sys.exit(2)

    rules = Rules(max_plies=args.max_plies)
    recorded = []
    positions = 0
    for seed in range(args.seed0, args.seed0 + args.games):
        rng = random.Random(seed)
        gn = Game(rules, native=True)
        gp = Game(rules, native=False)
        played = []
        while True:
            an = gn.legal_actions()
            ap_ = gp.legal_actions()
            same = ([repr(a) for a in an] == [repr(a) for a in ap_]
                    and gn.position_key() == gp.position_key()
                    and gn.done == gp.done and gn.winner == gp.winner
                    and gn.end_reason == gp.end_reason and gn.ply == gp.ply)
            if not same:
                diff_report(seed, gp.ply, gn, gp, an, ap_)
                sys.exit(1)
            positions += 1
            if gn.done:
                break
            action = an[rng.randrange(len(an))]
            played.append(action)
            gn.step(action)
            gp.step(action)
        recorded.append(played)
        gn.close()
        gp.close()
    plies = sum(len(g) for g in recorded)
    print("OK: native == python on %d games, %d plies, %d positions"
          % (len(recorded), plies, positions))

    if args.no_time:
        return
    for label, native in (("python", False), ("native", True)):
        t = time.perf_counter()
        for played in recorded:
            g = Game(rules, native=native)
            for action in played:
                g.step(action)
            g.close()
        took = time.perf_counter() - t
        print("%-7s %7.2fs  %9.0f plies/s" % (label, took, plies / took))


if __name__ == "__main__":
    main()
