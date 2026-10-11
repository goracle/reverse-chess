#!/usr/bin/env python
"""How much does batched search (virtual loss) distort the move distribution the learner trains on?

    python check_vl.py runs/run4/latest.pt                  # your network (needs torch to read the .pt)
    python check_vl.py                                      # random weights with a real policy spread
    python check_vl.py ai.npz --batch 8 --sims 32 --vl 1,0.5,0.25,0

For positions from random games it runs the same search (``--sims`` simulations, no root noise,
same root analysis) one leaf per network call (``batch`` 1, the reference), then with ``--batch``
leaves per call for each virtual-loss strength in ``--vl``, and reports against the reference:

  TV      mean total-variation distance between the visit distributions (0 = identical)
  top1    how often the most visited move is the same
  ent     mean entropy of the visit distribution (nats; the reference's is the baseline.  Strong
          virtual loss pushes it up: sims get spread over many root moves)
  moves   mean number of distinct root moves visited
  leaves  mean leaves per network call actually achieved (collisions cut batches short)

Needs the native library (rust net + rc_step).  Torch is not imported.
"""
import argparse
import random
import sys

import numpy as np

import check_rustnet as cr
import mcts
import numpy_net
import rustnet
from engine import Game, Rules
from env import encode_observation
from numpy_net import Analysis


class RustEvaluator:
    """What ``net.SearchEvaluator`` is for torch: ``analyse_fresh`` / ``analyse_many`` with RustNet."""

    def __init__(self, net):
        self.net = net
        self.calls = []                       # positions per analyse_many call

    def analyse_many(self, games):
        self.calls.append(len(games))
        obs = [encode_observation(g) for g in games]
        actions = [g.legal_actions() for g in games]
        comps = []
        for g in games:
            c = g.action_array().copy()
            if g.turn == "b":
                c[:, :2] ^= 56
            comps.append(c)
        counts = np.array([len(a) for a in actions], dtype=np.int64)
        logits, value = self.net.forward(np.stack([o["board"] for o in obs]),
                                         np.stack([o["scalars"] for o in obs]), counts, np.concatenate(comps))
        out, start = [], 0
        for k, n in enumerate(counts):
            z = logits[start:start + n].astype(np.float64)
            start += n
            p = np.exp(z - z.max())
            out.append(Analysis(float(value[k]), actions[k], p / p.sum()))
        return out

    def analyse_fresh(self, game):
        return self.analyse_many([game])[0]


def roots(n, seed0, min_actions=8):
    """Up to n positions with real choice, from random games, spread over game phases."""
    rules = Rules(max_plies=150)
    out = []
    seed = seed0
    while len(out) < n:
        rng = random.Random(seed)
        seed += 1
        g = Game(rules)
        ply = 0
        while not g.done and ply < 100:
            acts = g.legal_actions()
            if ply >= 4 and ply % 9 == 0 and len(acts) >= min_actions:
                out.append(g.clone())
            g.step(acts[rng.randrange(len(acts))])
            ply += 1
        g.close()
    return out[:n]


def visits(ev, game, sims, batch, vl, root_analysis):
    s = mcts.Searcher(ev, sims=sims, batch=batch, vl=vl)
    n_calls = len(ev.calls)
    job = s.search(game, sims=sims, ms=10 ** 9, root_analysis=root_analysis, noise=0.0, sweep_k=0)
    v = job.root.N.astype(np.float64)
    calls = ev.calls[n_calls:]
    return v, calls


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", help="checkpoint (.pt, needs torch) or ai.npz; default: random weights")
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--vl", default="1,0.75,0.5,0.25,0", help="virtual-loss strengths to compare")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    rustnet.require()
    if args.weights:
        w = numpy_net.load(args.weights).w
        print("weights:", args.weights)
    else:
        w = cr.random_weights(64, 4, 128, 0)
        print("weights: random (64ch/4/128) -- the policy is only a stand-in; use a checkpoint for real numbers")
    ev = RustEvaluator(rustnet.RustNet(w))
    pos = roots(args.positions, args.seed0)
    print("%d positions, avg %.0f legal moves; %d sims, batch %d\n"
          % (len(pos), np.mean([len(g.legal_actions()) for g in pos]), args.sims, args.batch))

    refs = []
    prior_ent = []
    for g in pos:
        a = ev.analyse_fresh(g)
        prior_ent.append(entropy(a.probs))
        v, _ = visits(ev, g, args.sims, 1, 1.0, a)
        refs.append((a, v))
    ref_p = [v / v.sum() for _, v in refs]
    print("root policy entropy %.2f nats;  reference (batch 1): visit entropy %.2f, %.1f distinct moves\n"
          % (np.mean(prior_ent), np.mean([entropy(p) for p in ref_p]),
             np.mean([(p > 0).sum() for p in ref_p])))
    print("  vl     TV    top1    ent   moves  leaves")
    for vl in [float(x) for x in args.vl.split(",")]:
        tv, top1, ent, moves, leaves = [], [], [], [], []
        for (a, _), rp, g in zip(refs, ref_p, pos):
            v, calls = visits(ev, g, args.sims, args.batch, vl, a)
            p = v / v.sum()
            tv.append(0.5 * np.abs(p - rp).sum())
            top1.append(int(np.argmax(p) == np.argmax(rp)))
            ent.append(entropy(p))
            moves.append((p > 0).sum())
            leaves.append(np.mean(calls))
        print("  %.2f  %.3f  %.2f  %5.2f  %5.1f  %5.2f"
              % (vl, np.mean(tv), np.mean(top1), np.mean(ent), np.mean(moves), np.mean(leaves)))


if __name__ == "__main__":
    sys.exit(main())
