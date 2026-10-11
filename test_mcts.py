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


def check_fast_clone():
    """mcts.fast_clone must behave exactly like Game.clone (and be independent of the original)."""
    rng = np.random.default_rng(1)
    game = Game(Rules(max_plies=300))
    for _ in range(25):
        if game.done:
            break
        acts = game.legal_actions()
        game.step(acts[int(rng.integers(len(acts)))])
    a, b = game.clone(), mcts.fast_clone(game)
    assert a.legal_actions() == b.legal_actions() == game.legal_actions()
    assert a.position_key() == b.position_key() == game.position_key()
    assert a.move_log == b.move_log == game.move_log and a.seen == b.seen == game.seen
    # play the same random continuation on all three: identical results, original untouched
    before = (game.ply, game.position_key())
    for _ in range(15):
        if a.done:
            break
        acts = a.legal_actions()
        pick = acts[int(rng.integers(len(acts)))]
        a.step(pick)
        b.step(pick)
        assert a.position_key() == b.position_key() and a.legal_actions() == b.legal_actions()
        assert a.done == b.done and a.winner == b.winner
    assert (game.ply, game.position_key()) == before, "fast_clone shares state with the original"
    assert b.move_log is not game.move_log and b.seen is not game.seen
    print("fast_clone ok")


class UniformMany(Uniform):
    """Same stand-in, with the batch entry point; counts the calls."""
    def __init__(self):
        self.calls = []

    def analyse_many(self, games):
        self.calls.append(len(games))
        return [self.analyse_fresh(g) for g in games]


def check_batched():
    """Batched search (virtual loss): the bookkeeping must come out exactly as in the plain one --
    every simulation counted once, no virtual loss left behind, one network call per round."""
    game = Game(Rules(max_plies=300))
    for _ in range(6):                                   # a mid-game position, not the opening
        game.step(game.legal_actions()[0])
    for batch in (1, 2, 4, 7):
        adv = UniformMany()
        searcher = mcts.Searcher(adv, sims=40, batch=batch)
        job = searcher.search(game, sims=40)
        check(job, game)
        root = job.root
        assert root.total == 40 and root.nvl == 0 and not root.vl.any(), (batch, root.total, root.nvl)
        stack = [root]
        while stack:                                     # no pending node or virtual loss anywhere
            node = stack.pop()
            assert not node.pending and node.nvl == 0 and not node.vl.any()
            assert node.total == int(node.N.sum())
            stack.extend(c for c in node.children if c is not None)
        assert sum(adv.calls) <= 40 and (batch == 1 or max(adv.calls) <= batch), adv.calls
        if batch > 1:
            assert len(adv.calls) < 39, "batching made no difference: %s" % adv.calls
        print("batch %d: %d sims, %d network calls (sizes %s)" % (batch, root.total, len(adv.calls),
                                                                   sorted(set(adv.calls))))
    # an advisor without analyse_many still works with batch > 1
    job = mcts.Searcher(Uniform(), sims=20, batch=4).search(game, sims=20)
    check(job, game)
    # batch=1 must not depend on how the advisor evaluates
    a = mcts.Searcher(Uniform(), sims=30, batch=1).search(game, sims=30)
    b = mcts.Searcher(UniformMany(), sims=30, batch=1).search(game, sims=30)
    assert (a.root.N == b.root.N).all()
    print("batched search ok")


def check_evaluate_many():
    """SearchEvaluator.analyse_many == analyse_fresh position by position (needs PyTorch)."""
    try:
        import torch
        import net as netlib
    except ImportError:
        print("torch not installed: skipping the analyse_many check")
        return
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    model = netlib.PolicyValueNet(16, 1, 32).eval()
    for p in model.parameters():                         # make the heads non-trivial
        p.data += 0.05 * torch.randn_like(p)
    ev = netlib.SearchEvaluator(model)
    games = []
    for k in range(5):
        g = Game(Rules(max_plies=300))
        for _ in range(3 + 4 * k):
            if g.done:
                break
            acts = g.legal_actions()
            g.step(acts[int(rng.integers(len(acts)))])
        games.append(g)
    many = ev.analyse_many(games)
    for g, m in zip(games, many):
        one = ev.analyse_fresh(g)
        assert m.actions == one.actions
        assert abs(m.value - one.value) < 1e-4, (m.value, one.value)
        assert np.abs(m.probs - one.probs).max() < 1e-4 and abs(m.probs.sum() - 1) < 1e-9
    print("analyse_many matches analyse_fresh on %d positions" % len(games))


def main():
    check_fast_clone()
    check_batched()
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
    searcher = mcts.Searcher(evaluator, sims=8, batch=4)
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
    check_evaluate_many()
    check_training_search()
