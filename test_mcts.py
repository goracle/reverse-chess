#!/usr/bin/env python
"""Sanity checks for mcts.py.  Run:  python test_mcts.py [ai.npz]

Uses a uniform-policy stand-in for the network unless a weights file is given, so it checks
the search machinery (bookkeeping, legality, terminal handling), not playing strength.
"""
import sys

import numpy as np

import mcts
from engine import Game, Rules
from numpy_net import Advisor, Analysis


class Uniform:
    """Stand-in advisor: equal prior on every move, value 0."""
    def analyse_fresh(self, game):
        acts = game.legal_actions()
        return Analysis(0.0, acts, np.full(len(acts), 1.0 / len(acts)))


def check(job, game):
    root = job.root
    assert job.error is None, job.error
    assert root is not None and job.finished
    assert root.total == int(root.N.sum()), (root.total, root.N.sum())
    best = job.best()
    assert best in game.actions, "best move is not legal"
    assert -1.0 <= job.value() <= 1.0
    assert all(0.0 <= share <= 1.0 for _, share, _ in job.top(3))


def main():
    advisor = Advisor(sys.argv[1]) if len(sys.argv) > 1 else Uniform()
    searcher = mcts.Searcher(advisor, sims=150, think_ms=20000)

    game = Game(Rules(max_plies=300))
    job = searcher.search(game)
    check(job, game)
    print("start position: %d sims, value %+.2f, best %s" % (job.sims, job.value(), job.best()))

    # play a few plies of search-vs-search and re-check each position
    for ply in range(8):
        if game.done:
            break
        job = searcher.search(game, sims=60)
        check(job, game)
        game.step(job.best())
    print("search-vs-search: %d plies played, done=%s" % (game.ply, game.done))

    # the original game must never be touched by a search
    g2 = Game(Rules(max_plies=300))
    before = (g2.ply, g2.turn, len(g2.legal_actions()))
    searcher.search(g2, sims=40)
    assert before == (g2.ply, g2.turn, len(g2.legal_actions()))
    print("ok")


def check_training_search():
    """net.search_action (the self-play path): needs PyTorch, uses a tiny random network."""
    try:
        import torch
        import net as netlib
    except ImportError:
        print("torch not installed: skipping the self-play search check")
        return
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = netlib.PolicyValueNet(16, 1, 32).eval()
    evaluator = netlib.SearchEvaluator(model)
    searcher = mcts.Searcher(evaluator, sims=8)
    game = Game(Rules(max_plies=300))
    sims_seen = []
    for _ in range(6):                                  # a few plies, all chosen by search
        if game.done or len(game.legal_actions()) < 2:
            break
        action, sample, pi, sims = netlib.search_action(evaluator, searcher, game, rng, 8)
        n = len(game.legal_actions())
        assert action in game.actions
        assert pi.shape == (n,) and abs(float(pi.sum()) - 1.0) < 1e-4, (pi.shape, pi.sum())
        board, scalars, comps, idx, logp, value = sample
        assert comps.shape == (n, 6) and 0 <= idx < n and game.legal_actions()[idx] == action
        assert logp <= 0.0 and -1.0 <= value <= 1.0
        sims_seen.append(sims)
        game.step(action)
    print("self-play search: %d searched plies, sims per search %s" % (len(sims_seen), sims_seen))


if __name__ == "__main__":
    main()
    check_training_search()
