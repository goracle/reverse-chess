#!/usr/bin/env python
"""Returns and advantages for self-play PPO (numpy only, no torch).

Two-player, zero-sum, alternating game.  Each player has its *own* trajectory:
the decisions it made, in order (a reverse-en-passant extra turn simply means
two of one player's decisions are adjacent).  For every player we compute

    r_i = [final step only] outcome            (+1 win, -1 loss, draw_value)
        + coef * (gamma * Phi(s_{i+1}) - Phi(s_i))      <- potential-based shaping
    delta_i = r_i + gamma * V(s_{i+1}) - V(s_i)
    A_i     = delta_i + gamma*lambda * A_{i+1}          (GAE)
    ret_i   = A_i + V(s_i)                              (target for the value head)

where s_i is the state the player faced at its i-th decision, V is the value
estimate recorded when the game was played, and Phi(terminal) = 0.

Why this is "principled"
------------------------
* Phi is a *potential* (env.potential: opponent's remaining burden minus ours).
  Rewarding the change in a potential cannot change which policy is optimal
  (Ng, Harada & Russell 1999): with gamma = 1 the shaping terms telescope to a
  constant per game, so a policy gains nothing by farming them.  They only
  hand out credit for progress *earlier* than the final +-1.
* The Phi(terminal) = 0 convention is what makes that telescoping exact, so a
  drawn game is worth exactly ``draw_value`` however close to home it ended.
* That also means shaping only helps when advantages are *bootstrapped*
  (gae_lambda < 1).  With lambda = 1 the advantage is the full Monte-Carlo
  return and the shaping just moves a state-dependent constant into the
  baseline: no effect on learning.  Use lambda around 0.95 with coef > 0.

With coef = 0, gamma = 1, lambda = 1 this reduces exactly to the original
``advantage = outcome - V`` (see test_returns.py).
"""
import numpy as np


def outcome_for(winner, player_index: int, draw_value: float) -> float:
    """Result for player_index (0 = black, 1 = white); ``winner`` is 'b'/'w'/None."""
    if winner is None:
        return float(draw_value)
    winner_index = 0 if winner == "b" else 1
    return 1.0 if player_index == winner_index else -1.0


def player_gae(value, phi, outcome, coef, gamma, lam):
    """Advantages and value targets for ONE player's trajectory.

    value, phi: arrays over that player's decisions, in order.  Returns
    ``(adv, ret)`` float64 arrays of the same length.
    """
    n = len(value)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    value = np.asarray(value, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    reward = np.zeros(n)
    if coef:
        reward[:-1] = coef * (gamma * phi[1:] - phi[:-1])
        reward[-1] = coef * (0.0 - phi[-1])           # Phi(terminal) = 0
    reward[-1] += outcome
    next_value = np.append(value[1:], 0.0)             # V(terminal) = 0
    delta = reward + gamma * next_value - value
    adv = np.zeros(n)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = delta[i] + gamma * lam * running
        adv[i] = running
    return adv, adv + value


def game_returns(game, draw_value=0.0, coef=0.0, gamma=1.0, lam=1.0):
    """``(adv, ret)`` float32 arrays aligned with one finished game's decisions.

    ``game`` is a rollout result dict with keys player, value, phi, winner.
    """
    player = game["player"]
    adv = np.zeros(len(player), dtype=np.float32)
    ret = np.zeros(len(player), dtype=np.float32)
    for p in (0, 1):
        idx = np.flatnonzero(player == p)
        if len(idx) == 0:
            continue
        a, r = player_gae(game["value"][idx], game["phi"][idx],
                          outcome_for(game["winner"], p, draw_value),
                          coef, gamma, lam)
        adv[idx], ret[idx] = a, r
    return adv, ret
