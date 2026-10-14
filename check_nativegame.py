#!/usr/bin/env python
"""Check nativegame.py (the whole game in Rust) against the old Python self-play loop.

    ./build_native.sh && python check_nativegame.py
    python check_nativegame.py --games 100 --ckpt runs/run4/latest.pt     # a real network
    python check_nativegame.py --bench 40                                  # also time both

For each game the SAME draws are fed to both:

* reference: ``engine.Game`` + ``net.choose_action`` + ``rollout._Recorder`` -- the loop
  ``rollout.selfplay_task`` ran before nativegame.py -- with a stand-in for ``rng`` whose
  ``random()`` hands out the next draw;
* native: ``nativegame.play(..., draws=...)``.

Compared: plies, winner, end reason, and every array of the result (board planes, scalar
features, action rows, counts, chosen indices, players exactly; log-probs, values, potentials to
float tolerance).  Both plain self-play and "net vs an older net" are checked.  A mismatch of the
chosen index at some ply can in principle come from a float rounding difference in the softmax
(numpy float32 sums vs. Rust's); it needs a draw within ~1e-7 of a cumulative-probability
boundary, so it is reported but should essentially never happen.

Searched decisions (``--search-frac``) cannot be compared draw for draw (the search uses ``rng``
itself); a smoke test checks that games with search come out well formed.
"""
import argparse
import sys
import time

import numpy as np

import nativegame
import net as netlib
import rollout
import rustnet
from engine import Game, Rules
from env import potential


class Feed:
    """Stands in for numpy's Generator: ``random()`` returns the next prepared draw."""

    def __init__(self, u):
        self.u, self.j = u, 0

    def random(self):
        v = self.u[self.j]
        self.j += 1
        return float(v)


def reference(model, opp, rules, u, mask):
    """The pre-nativegame loop of rollout.selfplay_task (no search, no bots)."""
    game = Game(rules)
    rec = rollout._Recorder()
    feed = Feed(u)
    while not game.done:
        team = game.turn
        if (mask >> (0 if team == "b" else 1)) & 1:
            phi = potential(game, team)
            action, sample = netlib.choose_action(model, game, feed)
            rec.add(team, sample, phi, None)
        else:
            action, _ = netlib.choose_action(opp, game, feed)
        game.step(action)
    out = rec.finish(game, 0, 0.0)
    game.close()
    return out


EXACT = ("boards", "comps", "counts", "chosen", "player")
CLOSE = {"scalars": 1e-6, "logp": 1e-4, "value": 1e-4, "phi": 1e-7}


def compare(ref, nat):
    """List of differences (empty = identical)."""
    bad = []
    for k in ("plies", "winner", "reason"):
        if ref[k] != nat[k]:
            bad.append("%s: %r vs %r" % (k, ref[k], nat[k]))
    for k in EXACT:
        a, b = ref[k], nat[k]
        if a.shape != b.shape:
            bad.append("%s: shape %s vs %s" % (k, a.shape, b.shape))
        elif not np.array_equal(a, b):
            first = int(np.flatnonzero((a != b).reshape(len(a), -1).any(axis=1))[0]) if a.ndim else 0
            bad.append("%s differs (first at row %d)" % (k, first))
    for k, tol in CLOSE.items():
        a, b = ref[k], nat[k]
        if a.shape != b.shape:
            bad.append("%s: shape %s vs %s" % (k, a.shape, b.shape))
        elif not np.allclose(a, b, atol=tol, rtol=0):
            worst = float(np.abs(a.astype(np.float64) - b).max())
            bad.append("%s differs (max abs error %.3g, tolerance %g)" % (k, worst, tol))
    return bad


def make_net(seed, ckpt):
    if ckpt:
        model, ck = netlib.load_checkpoint(ckpt)
        return rustnet.RustNet.from_torch(model, int(ck.get("version", 0)))
    import torch
    torch.manual_seed(seed)
    model = netlib.PolicyValueNet()
    # a random net is nearly uniform; sharpen it a little so the sampling is not trivial
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.0)
    return rustnet.RustNet.from_torch(model, 0)


def smoke_search(model, rules, rng, games):
    """Games with search: well formed arrays, pi a distribution on every searched decision."""
    cfg = {"frac": 0.3, "sims": 8, "noise": 0.25, "temp": 1.0, "sweep": 0, "smooth": 0.0}
    problems, searched_total = [], 0
    for g in range(games):
        r = nativegame.play(model, None, rules, rng, 3, 0, cfg)
        n, t = int(r["counts"].sum()), len(r["counts"])
        if r["comps"].shape != (n, 6) or r["boards"].shape != (t, 18, 8, 8) or r["scalars"].shape != (t, 17):
            problems.append("game %d: bad shapes" % g)
        if (r["chosen"] >= r["counts"]).any() or (r["chosen"] < 0).any():
            problems.append("game %d: chosen index out of range" % g)
        if "searched" in r:
            searched_total += int(r["searched"].sum())
            if len(r["pi"]) != n:
                problems.append("game %d: pi has %d entries, %d actions" % (g, len(r["pi"]), n))
            else:
                start = np.concatenate([[0], np.cumsum(r["counts"])])
                for i in np.flatnonzero(r["searched"]):
                    s = r["pi"][start[i]:start[i + 1]].sum()
                    if abs(s - 1.0) > 1e-4:
                        problems.append("game %d decision %d: pi sums to %g" % (g, i, s))
                        break
    if searched_total == 0:
        problems.append("no decision was searched in %d games (frac 0.3)" % games)
    return problems, searched_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--ckpt", help="a .pt checkpoint to use instead of a random network")
    ap.add_argument("--bench", type=int, default=0, help="also time this many games each way")
    args = ap.parse_args()

    if not nativegame.AVAILABLE:
        print("native game support missing: run ./build_native.sh (and check rustnet / fastgame load)")
        return 2
    rules = Rules(max_plies=args.max_plies)
    mine, old = make_net(args.seed, args.ckpt), make_net(args.seed + 1, None)
    rng = np.random.default_rng(args.seed)
    failures = 0
    plies = 0
    for g in range(args.games):
        # alternate: self-play, then the net as black / white against the older one
        mode = g % 3
        mask, opp = (3, None) if mode == 0 else ((1, old) if mode == 1 else (2, old))
        u = rng.random(4096)
        ref = reference(mine, opp, rules, u, mask)
        nat = nativegame.play(mine, opp, rules, rng, mask, 0, None, draws=(u, np.zeros_like(u)))
        plies += ref["plies"]
        bad = compare(ref, nat)
        if bad:
            failures += 1
            print("game %d (%s) MISMATCH:" % (g, ("self-play", "net=black vs old", "net=white vs old")[mode]))
            for line in bad:
                print("    " + line)
    print("%d games, %d plies: %d identical, %d mismatched" % (args.games, plies, args.games - failures, failures))

    problems, searched = smoke_search(mine, rules, rng, 6)
    print("search smoke test: %d searched decisions, %s" % (searched, "ok" if not problems else "PROBLEMS"))
    for p in problems:
        print("    " + p)
    failures += len(problems)

    if args.bench:
        n = args.bench
        us = [rng.random(4096) for _ in range(n)]
        t0 = time.time()
        pl_ref = sum(reference(mine, None, rules, u, 3)["plies"] for u in us)
        t_ref = time.time() - t0
        t0 = time.time()
        pl_nat = sum(nativegame.play(mine, None, rules, rng, 3, 0, None, draws=(u, np.zeros_like(u)))["plies"]
                     for u in us)
        t_nat = time.time() - t0
        print("python loop: %d games %.2fs (%.0f plies/s)" % (n, t_ref, pl_ref / t_ref))
        print("native loop: %d games %.2fs (%.0f plies/s)  -> %.1fx" % (n, t_nat, pl_nat / t_nat, t_ref / t_nat))
    print("PASS" if failures == 0 else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
