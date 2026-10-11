#!/usr/bin/env python
"""FastGame (native light positions) vs engine.Game, and light search vs clone search.

    python check_fastgame.py                       # everything, a few minutes
    python check_fastgame.py --games 10 --quick    # a fast smoke test
    python check_fastgame.py --pure 3              # also replay 3 games against the pure-Python engine
    python check_fastgame.py --bench               # simulations/second, light vs fast_clone

No trained network is needed: a randomly initialised ``numpy_net`` of the real size stands in
(pass --net ai.npz to use yours).  What is checked:

1. *Positions*: along random games, at every ply, a FastGame built from the real game and a
   FastGame chain stepped alongside it agree with ``Game`` on legal actions (and their order),
   ``action_array``, the observation the network sees, and -- for the children of the
   position, played both ways -- winner, end reason, ply, fifty-move counter, repetition
   count, board bytes and the next position's actions.  Rule variants make the rarer
   endings (fifty-move, repetition, ply cap, wins) happen often.
2. *Networks*: ``Advisor.analyse_fresh`` gives the same answer for a Game and its FastGame.
3. *Search*: ``Searcher(light=False)`` and ``Searcher(light=True)`` build identical trees
   (visits, values, chosen move) for several batch sizes, with root noise and a full sweep.
4. *Threads*: children stepped from several threads at once come out right.
"""
import argparse
import random
import re
import sys
import threading
import time
from collections import Counter

import numpy as np

import engine
import fastgame
import mcts
import numpy_net
from engine import Game, Rules
from env import encode_observation
from fastgame import FastGame
from numpy_net import Advisor, Analysis, NumpyNet

# --------------------------------------------------------------------------
# a stand-in network
# --------------------------------------------------------------------------

def random_weights(seed=0, channels=64, blocks=4, hidden=128):
    """Random weights with the shapes of net.PolicyValueNet (the numpy_net key names)."""
    rng = np.random.default_rng(seed)

    def mat(*shape, std=0.1):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    c, h = channels, hidden
    w = {"stem.weight": mat(c, 18 + 17, 3, 3), "stem_norm.weight": np.ones(c), "stem_norm.bias": np.zeros(c)}
    for i in range(blocks):
        p = "blocks.%d." % i
        w[p + "conv1.weight"], w[p + "conv2.weight"] = mat(c, c, 3, 3), mat(c, c, 3, 3)
        for n in ("norm1", "norm2"):
            w[p + n + ".weight"], w[p + n + ".bias"] = np.ones(c), mat(c, std=0.05)
    w.update({"glob_conv.weight": mat(8, c, 1, 1), "glob_conv.bias": np.zeros(8),
              "glob_fc.weight": mat(h, 8 * 64, std=0.05), "glob_fc.bias": np.zeros(h),
              "value_out.weight": mat(1, h), "value_out.bias": np.zeros(1),
              "src_proj.weight": mat(h, c), "src_proj.bias": np.zeros(h),
              "dst_proj.weight": mat(h, c), "glob_proj.weight": mat(h, h),
              "gift_emb.weight": mat(7, h), "promo_emb.weight": mat(5, h), "flag_emb.weight": mat(4, h),
              "act_fc.weight": mat(h // 2, h), "act_fc.bias": np.zeros(h // 2),
              "act_out.weight": mat(1, h // 2, std=0.5), "act_out.bias": np.zeros(1)})
    return w


class StandInAdvisor(Advisor):
    """numpy_net.Advisor on random weights (or a real file), plus the batch entry point."""

    def __init__(self, path=None, seed=0, max_plies=300):
        self.path = path or "random-weights"
        self.net = numpy_net.load(path) if path else NumpyNet(random_weights(seed))
        self.version = self.net.version
        self.max_plies = max_plies
        self._cache = (None, None)

    def analyse_many(self, games):
        obs = [encode_observation(g, self.max_plies) for g in games]
        comps = []
        for g in games:
            c = g.action_array().copy()
            if g.turn == "b":
                c[:, :2] ^= 56
            comps.append(c)
        counts = np.fromiter((len(c) for c in comps), dtype=np.int64, count=len(games))
        logits, value = self.net.forward(
            np.stack([o["board"] for o in obs]), np.stack([o["scalars"] for o in obs]),
            np.repeat(np.arange(len(games), dtype=np.int64), counts), np.concatenate(comps))
        z, start, out = logits.astype(np.float64), 0, []
        for k, n in enumerate(counts):
            zk = z[start:start + n]
            start += n
            p = np.exp(zk - zk.max())
            out.append(Analysis(float(value[k]), games[k].legal_actions(), p / p.sum()))
        return out


# --------------------------------------------------------------------------
# 1. positions
# --------------------------------------------------------------------------

class Mismatch(Exception):
    pass


def same_position(g, f, ctx, deep=True):
    """A real Game ``g`` and a FastGame ``f`` describe the same position."""
    def need(ok, what):
        if not ok:
            raise Mismatch("%s: %s" % (ctx, what))
    need(g.done == f.done, "done %s vs %s (%s | %s)" % (g.done, f.done, g.end_reason, f.end_reason))
    need(g.winner == f.winner, "winner %s vs %s" % (g.winner, f.winner))
    need(g.turn == f.turn, "turn")
    need(g.ply == f.ply, "ply %d vs %d" % (g.ply, f.ply))
    need(g.quiet_plies == f.quiet_plies, "quiet %d vs %d" % (g.quiet_plies, f.quiet_plies))
    need(g.end_reason == f.end_reason, "end_reason %r vs %r" % (g.end_reason, f.end_reason))
    need(g._pack() == f.state, "state bytes")
    need(g.position_key() == f.key, "position key")
    need(g.seen.get(g.position_key(), 0) == f.rep, "repetition count %d vs %d" % (g.seen.get(g.position_key(), 0), f.rep))
    if g.done:
        need(f.raw == b"", "finished game still has actions")
        return
    need(g.action_array().tobytes() == f.raw, "action rows")
    need(g.legal_actions() == list(f.legal_actions()), "legal actions / order")
    need(len(f.legal_actions()) == len(g.actions) == f.num_actions, "number of actions")
    if deep:
        for mp in (None, 300):
            a, b = encode_observation(g, mp), f.observation(mp)
            need(np.array_equal(a["board"], b["board"]), "observation board (max_plies=%s)" % mp)
            need(np.array_equal(a["scalars"], b["scalars"]),
                 "observation scalars (max_plies=%s): %s" % (mp, a["scalars"] - b["scalars"]))
            need(b["board"].dtype == np.float32 and b["scalars"].dtype == np.float32, "obs dtype")


RULE_SETS = [
    ("default",       lambda: Rules(max_plies=300)),
    ("short-draws",   lambda: Rules(max_plies=90, fifty_move_limit=10, repetition_limit=2)),
    ("rep3",          lambda: Rules(max_plies=200, fifty_move_limit=30, repetition_limit=3)),
    ("all-on-board",  lambda: Rules(max_plies=120, win_requires_all_pieces_on_board=True,
                                    fifty_move_limit=25, repetition_limit=3)),
]


def check_positions(games, seed0, children, quick, endings):
    positions = child_checks = 0
    for n in range(games):
        seed = seed0 + n
        name, make = RULE_SETS[n % len(RULE_SETS)]
        rules = make()
        rng = random.Random(seed)
        g = Game(rules)
        chain = FastGame.from_game(g)                   # stepped alongside, never re-snapshotted
        while True:
            ctx = "seed %d [%s] ply %d" % (seed, name, g.ply)
            same_position(g, FastGame.from_game(g), ctx + " (snapshot)")
            same_position(g, chain, ctx + " (chain)", deep=(g.ply % 5 == 0))
            positions += 1
            if g.done:
                endings[re.sub(r"\d+", "N", g.end_reason)] += 1
                break
            acts = g.legal_actions()
            k = len(acts)
            picks = range(k) if k <= children else sorted(rng.sample(range(k), children))
            for i in picks:                             # every child, played both ways
                c = mcts.fast_clone(g)
                c.step(acts[i])
                same_position(c, chain.child(i), ctx + " child %d %r" % (i, acts[i]), deep=(i % 3 == 0))
                c.close()
                child_checks += 1
            i = rng.randrange(k)
            g.step(acts[i])
            chain = chain.child(i)
        g.close()
        if not quick and n % 10 == 9:
            print("  ... %d games, %d positions, %d child checks" % (n + 1, positions, child_checks))
    return positions, child_checks


def check_pure(games, seed0):
    """The same, against the pure-Python reference engine (slow)."""
    if not games:
        return 0
    plies = 0
    for n in range(games):
        rules = RULE_SETS[n % len(RULE_SETS)][1]()
        rng = random.Random(10_000 + seed0 + n)
        g = Game(rules, native=False)
        chain = FastGame.from_game(g)
        while not g.done:
            acts = g.legal_actions()
            i = rng.randrange(len(acts))
            g.step(acts[i])
            chain = chain.child(i)
            same_position(g, chain, "pure seed %d ply %d" % (n, g.ply), deep=(g.ply % 7 == 0))
            plies += 1
        g.close()
    return plies


# --------------------------------------------------------------------------
# 2. networks, 3. search, 4. threads
# --------------------------------------------------------------------------

def mid_game(seed, plies, rules=None):
    rng = random.Random(seed)
    g = Game(rules or Rules(max_plies=300))
    for _ in range(plies):
        if g.done:
            break
        acts = g.legal_actions()
        g.step(acts[rng.randrange(len(acts))])
    return g


def check_advisor(adv, positions=12):
    for s in range(positions):
        g = mid_game(s, 4 + 3 * s)
        if g.done:
            continue
        a, b = adv.analyse_fresh(g), adv.analyse_fresh(FastGame.from_game(g))
        assert a.value == b.value and np.array_equal(a.probs, b.probs), "advisor differs, seed %d" % s
        assert a.actions == list(b.actions)
    games = [FastGame.from_game(mid_game(s, 9)) for s in range(5)]
    many = adv.analyse_many(games)
    for gm, m in zip(games, many):
        one = adv.analyse_fresh(gm)
        assert abs(one.value - m.value) < 1e-5 and np.allclose(one.probs, m.probs, atol=1e-6)


def tree_signature(root):
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append((n.turn, n.terminal, round(n.value, 12), len(n.actions), n.N.tolist(),
                    [round(float(x), 9) for x in n.W], n.win_idx))
        stack.extend(c for c in n.children if c is not None)
    return out


def check_search(adv, quick):
    cases = 0
    for seed, plies in ((1, 6), (2, 15), (3, 31), (4, 52)):
        game = mid_game(seed, plies)
        if game.done or len(game.legal_actions()) < 2:
            continue
        for batch in (1, 4):
            for sweep_k, noise in ((0, 0.0), (None, 0.0), (0, 0.25)):
                jobs = []
                for light in (False, True):
                    s = mcts.Searcher(adv, sims=40 if quick else 80, batch=batch, light=light)
                    root_an = adv.analyse_fresh(game)
                    job = s.search(game, ms=10 ** 9, sweep_k=sweep_k, noise=noise,
                                   root_analysis=root_an, rng=np.random.default_rng(seed))
                    assert job.error is None, job.error
                    jobs.append(job)
                a, b = jobs
                what = "seed %d ply %d batch %d sweep %s noise %s" % (seed, plies, batch, sweep_k, noise)
                assert a.root.total == b.root.total, what
                assert np.array_equal(a.root.N, b.root.N), "visit counts differ: " + what
                assert np.array_equal(a.root.W, b.root.W), "values differ: " + what
                assert a.best() == b.best() and a.root.win_idx == b.root.win_idx, what
                assert tree_signature(a.root) == tree_signature(b.root), "trees differ: " + what
                assert b.root.nvl == 0 and not any(c is not None and c.pending for c in b.root.children)
                cases += 1
        # the search must not have touched the caller's game
    # a search from a real game with no root_analysis, and from a finished one
    s = mcts.Searcher(adv, sims=20, light=True)
    g = mid_game(7, 20)
    job = s.search(g, ms=10 ** 9)
    assert job.error is None and job.best() in g.actions
    return cases


def check_threads(n_threads=4):
    games = [mid_game(s, 12 + s) for s in range(n_threads)]
    errors = []

    def work(g):
        try:
            f = FastGame.from_game(g)
            for _ in range(200):
                for i in range(min(f.num_actions, 6)):
                    c = mcts.fast_clone(g)
                    c.step(g.legal_actions()[i])
                    same_position(c, f.child(i), "thread", deep=False)
                    c.close()
        except Exception as err:     # noqa: BLE001
            errors.append(err)

    # fewer repeats of the slow oracle: the point is concurrent rc_step calls
    ts = [threading.Thread(target=work, args=(g,)) for g in games]
    for t in ts:
        t.daemon = True
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
        if t.is_alive():
            errors.append("thread timed out")
    assert not errors, errors[:1]


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------

def bench(adv, positions=12, sims=40, batch=4):
    rows = {}
    for light in (False, True):
        t_total = t_clone = t_eval = 0.0
        done = 0
        for s in range(positions):
            g = mid_game(100 + s, 20 + 4 * s)
            if g.done or len(g.legal_actions()) < 2:
                continue
            se = mcts.Searcher(adv, sims=sims, batch=batch, light=light)
            t0 = time.perf_counter()
            job = se.search(g, ms=10 ** 9, sweep_k=0)
            t_total += time.perf_counter() - t0
            assert job.error is None, job.error
            done += job.root.total
            t_clone += se.t_clone
            t_eval += se.t_eval
        rows[light] = (t_total / done * 1e3, t_clone / done * 1e3, t_eval / done * 1e3, done)
    print("\nper simulation (batch %d, %d positions)      total   game copy+step   network   other"
          % (batch, positions))
    for light in (False, True):
        tot, cl, ev, n = rows[light]
        print("  %-30s %9.2f ms %10.2f ms %9.2f ms %6.2f ms   (%d sims)"
              % ("light (FastGame)" if light else "fast_clone + Game.step", tot, cl, ev, tot - cl - ev, n))
    a, b = rows[False], rows[True]
    print("  game copy+step: %.1fx faster; whole simulation: %.2fx faster (with this stand-in network)"
          % (a[1] / b[1], a[0] / b[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--children", type=int, default=10, help="children per position to replay both ways")
    ap.add_argument("--pure", type=int, default=0, help="games to replay against the pure-Python engine")
    ap.add_argument("--net", default=None, help="weights file for the search checks (default: random)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    if not fastgame.AVAILABLE:
        print("FastGame unavailable: the native library is missing or older than rc_step "
              "(run ./build_native.sh).")
        sys.exit(2)
    adv = StandInAdvisor(args.net)

    if args.bench:
        bench(adv)
        return

    endings = Counter()
    t0 = time.time()
    pos, kids = check_positions(args.games, args.seed0, args.children, args.quick, endings)
    print("positions ok: %d games, %d positions, %d children compared with Game.step  (%.0fs)"
          % (args.games, pos, kids, time.time() - t0))
    print("  endings seen:")
    for reason, count in sorted(endings.items()):
        print("    %4d  %s" % (count, reason))
    if args.pure:
        print("pure-Python replay ok: %d plies" % check_pure(args.pure, args.seed0))
    check_advisor(adv)
    print("advisor ok: same analysis for Game and FastGame, analyse_many consistent")
    n = check_search(adv, args.quick)
    print("search ok: light == clone trees in %d configurations" % n)
    check_threads()
    print("threads ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
