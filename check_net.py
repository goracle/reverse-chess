#!/usr/bin/env python
"""Checks for the fast paths in net.choose_action.  Imports torch (via net) but runs no network.

    python check_net.py

1. action_components_array(game) must equal [env.action_components(a, team) for a in
   legal_actions()] at every position of random games, on both engine backends.
2. The inline sampler must pick exactly what rng.choice(n, p=probs) picks, for the same
   random stream.
"""
import random
import sys

import numpy as np

import engine
from engine import Game, Rules
from env import action_components
from net import action_components_array


def check_components(games=40, max_plies=300):
    backends = [False] + ([True] if engine.NATIVE else [])
    positions = 0
    for native in backends:
        for seed in range(games):
            rng = random.Random(seed)
            g = Game(Rules(max_plies=max_plies), native=native)
            while not g.done:
                acts = g.legal_actions()
                want = np.asarray([action_components(a, g.turn) for a in acts], dtype=np.uint8)
                got = action_components_array(g)
                if got.dtype != np.uint8 or got.shape != want.shape or not (got == want).all():
                    print("MISMATCH (native=%s) seed=%d ply=%d" % (native, seed, g.ply))
                    print(g.ascii())
                    sys.exit(1)
                positions += 1
                g.step(acts[rng.randrange(len(acts))])
            g.close()
    print("OK: action components match env.action_components (%d positions, backends: %s)"
          % (positions, "python+native" if engine.NATIVE else "python only"))


def check_sampler(trials=20000):
    gen = np.random.default_rng(123)
    for t in range(trials):
        n = int(gen.integers(1, 200))
        logp = np.log(gen.dirichlet(np.ones(n) * float(gen.choice([0.1, 1.0, 5.0])))
                      + 1e-12).astype(np.float32)
        seed = int(gen.integers(1 << 30))
        probs = np.exp(logp.astype(np.float64))
        probs /= probs.sum()
        want = int(np.random.default_rng(seed).choice(n, p=probs))
        r = np.random.default_rng(seed)
        cdf = np.cumsum(np.exp(logp.astype(np.float64)))
        cdf /= cdf[-1]
        got = min(int(cdf.searchsorted(r.random(), side="right")), n - 1)
        if got != want:
            print("SAMPLER MISMATCH trial %d: n=%d got %d want %d" % (t, n, got, want))
            sys.exit(1)
    print("OK: inline sampler == rng.choice on %d random distributions" % trials)


if __name__ == "__main__":
    check_components()
    check_sampler()
