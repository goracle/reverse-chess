#!/usr/bin/env python
"""Tests for returns.py and env.potential (numpy only; run: python test_returns.py)."""
import random

import numpy as np

import returns
from engine import Game, Rules, other_team
from env import burden, potential


def fake_game(rng, T=40, winner="b"):
    player = np.array([rng.randint(0, 1) for _ in range(T)], dtype=np.uint8)
    return {"player": player,
            "value": np.array([rng.uniform(-1, 1) for _ in range(T)], dtype=np.float32),
            "phi": np.array([rng.uniform(-.5, .5) for _ in range(T)], dtype=np.float32),
            "winner": winner}


def test_off_equals_old_formula():
    """coef=0, gamma=1, lambda=1  ==  the original advantage = outcome - value."""
    rng = random.Random(0)
    for winner in ("b", "w", None):
        g = fake_game(rng, winner=winner)
        adv, ret = returns.game_returns(g, draw_value=-0.1, coef=0.0, gamma=1.0, lam=1.0)
        z = np.array([returns.outcome_for(winner, int(p), -0.1) for p in g["player"]])
        assert np.allclose(ret, z, atol=1e-6), "ret should be the plain outcome"
        assert np.allclose(adv, z - g["value"], atol=1e-6), "adv should be outcome - value"


def test_shaping_telescopes():
    """gamma=1, lambda=1: shaped return = outcome - coef*Phi(s_i), i.e. shaping
    only adds a state-dependent constant (policy invariance)."""
    rng = random.Random(1)
    for winner in ("b", "w", None):
        g = fake_game(rng, winner=winner)
        coef = 0.7
        adv, ret = returns.game_returns(g, 0.0, coef, 1.0, 1.0)
        z = np.array([returns.outcome_for(winner, int(p), 0.0) for p in g["player"]])
        assert np.allclose(ret, z - coef * g["phi"], atol=1e-5)


def test_gae_matches_bruteforce():
    rng = random.Random(2)
    n, coef, gamma, lam, outcome = 12, 0.5, 0.97, 0.9, 1.0
    v = np.array([rng.uniform(-1, 1) for _ in range(n)])
    phi = np.array([rng.uniform(-.5, .5) for _ in range(n)])
    adv, ret = returns.player_gae(v, phi, outcome, coef, gamma, lam)
    r = [coef * (gamma * phi[i + 1] - phi[i]) for i in range(n - 1)] + [coef * (0 - phi[-1]) + outcome]
    vn = list(v[1:]) + [0.0]
    delta = [r[i] + gamma * vn[i] - v[i] for i in range(n)]
    for i in range(n):
        want = sum((gamma * lam) ** k * delta[i + k] for k in range(n - i))
        assert abs(adv[i] - want) < 1e-9
    assert np.allclose(ret, adv + v)


def test_draw_worth_draw_value_regardless_of_progress():
    """Two draws, one ending near home and one far: identical total return."""
    coef = 1.0
    for end_phi in (-0.5, 0.0, 0.5):
        v = np.zeros(5)
        phi = np.array([0.0, 0.1, 0.2, 0.3, end_phi])
        adv, ret = returns.player_gae(v, phi, 0.0, coef, 1.0, 1.0)
        assert abs(ret[0] - (0.0 - coef * phi[0])) < 1e-9


def test_potential_on_real_games():
    g = Game(Rules(max_plies=300))
    assert potential(g, "b") == 0.0 and potential(g, "w") == 0.0     # symmetric start
    rng = random.Random(3)
    saw_home_win = False
    for seed in range(12):
        rng = random.Random(seed)
        g = Game(Rules(max_plies=300))
        while not g.done:
            assert abs(potential(g, "b") + potential(g, "w")) < 1e-9   # zero-sum
            a = rng.choice(g.legal_actions())
            mover, receiver = g.turn, other_team(g.turn)
            before = burden(g, receiver)
            g.step(a)
            delta = burden(g, receiver) - before
            if a.gift is None:
                assert delta == 0, "a non-gift move must not change the opponent's burden"
            else:
                assert delta in (0, 1), "a gift adds at most one piece of work (0 if it lands home)"
        if "has every piece home" in g.end_reason:
            assert burden(g, g.winner) == 0
            saw_home_win = True
    print("  (home-win seen in sample: %s)" % saw_home_win)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all returns/potential tests passed")
