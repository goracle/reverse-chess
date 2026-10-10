#!/usr/bin/env python
"""Worker-process side of self-play (imported by train.py, never run directly).

Each worker is a long-lived process.  Before every game it checks whether the
learner has written a newer ``latest.pt`` and reloads it if so, so workers
always play with the most recent weights without any messaging.  A worker
plays one whole game per task and returns the trajectory as numpy arrays.

Memory: workers use one torch thread and a deliberately small engine move
cache (``MOVE_CACHE_LIMIT``); the engine's default would be ~hundreds of MB
per process.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time

import numpy as np
import torch

import engine
import net as netlib
from engine import Game, Rules
from env import potential

_STATE = {"path": None, "rules": None, "key": None, "model": None, "version": -1}


def init_worker(ckpt_path: str, rules_kwargs: dict, move_cache: int):
    """Runs once in each worker process."""
    torch.set_num_threads(1)
    engine.MOVE_CACHE_LIMIT = move_cache
    _STATE["path"] = ckpt_path
    _STATE["rules"] = Rules(**rules_kwargs)


def _current_model():
    """The newest checkpoint's model (reloaded only when the file changed)."""
    st = os.stat(_STATE["path"])
    key = (st.st_mtime_ns, st.st_ino)       # os.replace() gives a new inode
    if key != _STATE["key"]:
        model, ckpt = netlib.load_checkpoint(_STATE["path"])
        _STATE.update(key=key, model=model, version=int(ckpt["version"]))
    return _STATE["model"], _STATE["version"]


class _Recorder:
    """Collects one game's decisions."""

    def __init__(self):
        self.boards, self.scalars, self.comps = [], [], []
        self.counts, self.chosen, self.logp, self.value, self.player = [], [], [], [], []
        self.phi = []             # progress potential of each decision's mover (reward shaping)

    def add(self, team: str, sample, phi: float = 0.0):
        board, scalars, comps, idx, logp, value = sample
        self.phi.append(phi)
        self.boards.append(board)
        self.scalars.append(scalars)
        self.comps.append(comps)
        self.counts.append(len(comps))
        self.chosen.append(idx)
        self.logp.append(logp)
        self.value.append(value)
        self.player.append(0 if team == "b" else 1)

    def finish(self, game: Game, version: int, seconds: float) -> dict:
        return {
            "boards": np.stack(self.boards),                         # (T,18,8,8) uint8
            "scalars": np.stack(self.scalars).astype(np.float32),    # (T,17)
            "comps": np.concatenate(self.comps),                     # (sum n, 6) uint8
            "counts": np.asarray(self.counts, dtype=np.int32),       # legal actions per ply
            "chosen": np.asarray(self.chosen, dtype=np.int32),
            "logp": np.asarray(self.logp, dtype=np.float32),
            "value": np.asarray(self.value, dtype=np.float32),
            "phi": np.asarray(self.phi, dtype=np.float32),           # potential for the mover
            "player": np.asarray(self.player, dtype=np.uint8),       # 0 = black, 1 = white
            "winner": game.winner,                                   # 'b' / 'w' / None
            "reason": game.end_reason,
            "plies": game.ply,
            "version": version,
            "seconds": seconds,
        }


def selfplay_task(seed: int) -> dict:
    """Play one game, the current network against itself, sampling its policy."""
    model, version = _current_model()
    rng = np.random.default_rng(seed)
    game = Game(_STATE["rules"])
    rec = _Recorder()
    start = time.time()
    while not game.done:
        team = game.turn
        phi = potential(game, team)
        action, sample = netlib.choose_action(model, game, rng)
        rec.add(team, sample, phi)
        game.step(action)
    result = rec.finish(game, version, time.time() - start)
    game.close()                  # see Game.close: otherwise every game leaks
    return result


def eval_task(seed: int, net_team: str) -> dict:
    """Current network (sampling its policy) as ``net_team`` against a uniformly
    random opponent.  Returns a small result dict; nothing is recorded."""
    model, version = _current_model()
    rng = np.random.default_rng(seed)
    game = Game(_STATE["rules"])
    while not game.done:
        if game.turn == net_team:
            action, _ = netlib.choose_action(model, game, rng)
        else:
            actions = game.legal_actions()
            action = actions[int(rng.integers(len(actions)))]
        game.step(action)
    result = {"net_team": net_team, "winner": game.winner, "plies": game.ply,
              "version": version, "reason": game.end_reason}
    game.close()
    return result
