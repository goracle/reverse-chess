#!/usr/bin/env python
"""Regression check for the rules engine.

    python check_engine.py            # replay 30 seeded random games, compare with golden_engine.txt
    python check_engine.py --record   # (re)write golden_engine.txt from the current engine

For every position it hashes the full ordered list of legal actions, so ANY
change to the rules (or to the order of moves) shows up as a mismatch.  Run it
after touching engine.py: if it prints OK, speed-ups did not change the game.
"""
import hashlib
import os
import random
import sys
import time

import engine
from engine import Game, Rules

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_engine.txt")
N_GAMES = 30


def replay():
    lines = []
    for seed in range(N_GAMES):
        rng = random.Random(seed)
        g = Game(Rules(max_plies=300))
        while not g.done:
            acts = g.legal_actions()
            h = hashlib.md5("|".join(map(repr, acts)).encode()).hexdigest()[:12]
            lines.append("%d %d %s" % (seed, g.ply, h))
            g.step(rng.choice(acts))
        lines.append("END %d %s %s %s" % (seed, g.winner, g.end_reason, g.ply))
    return lines


if __name__ == "__main__":
    print("backend: %s" % ("native (Rust)" if engine.NATIVE else "pure Python"))
    if "--record" in sys.argv and engine.NATIVE:
        sys.exit("refusing to --record with the native backend: the golden record must "
                 "come from the Python reference.  Run:  RC_ENGINE=py python check_engine.py --record")
    t = time.time()
    lines = replay()
    took = time.time() - t
    if "--record" in sys.argv:
        open(GOLDEN, "w").write("\n".join(lines) + "\n")
        print("recorded %d lines to %s (%.1fs)" % (len(lines), GOLDEN, took))
    else:
        want = open(GOLDEN).read().split("\n")[:-1]
        bad = next((i for i, (a, b) in enumerate(zip(lines, want)) if a != b), None)
        if bad is None and len(lines) == len(want):
            print("OK: engine matches the golden record (%d positions, %.1fs)" % (len(lines), took))
        else:
            print("MISMATCH at line %s: got %r, expected %r" % (
                bad, lines[bad] if bad is not None else None, want[bad] if bad is not None else None))
            sys.exit(1)
