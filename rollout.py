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

import random
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


_OPP = {}      # snapshot path -> model; small LRU so a worker never holds many old nets


def _opponent_model(path):
    model = _OPP.pop(path, None)
    if model is None:
        model, _ = netlib.load_checkpoint(path)
    _OPP[path] = model                       # most recently used goes last
    while len(_OPP) > 6:
        _OPP.pop(next(iter(_OPP)))
    return model


def _bot_mode(spec):
    """'heur:gift' / 'heur:nogift' -> 'gift' / 'nogift' (the rule-based bot in heuristic.py);
    anything else (a snapshot path, None) -> None."""
    return spec.split(":", 1)[1] if spec and spec.startswith("heur:") else None


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


def selfplay_task(seed: int, opp_path: str = None, net_team: str = "b") -> dict:
    """Play one game with the current network, sampling its policy.

    With ``opp_path=None`` the current network plays itself and every decision is
    recorded.  With a snapshot path, the current network plays ``net_team`` against
    that older network; only the current network's decisions are recorded (the
    learner must not train on moves it did not make).  ``opp_path`` may also be
    ``"heur:nogift"`` or ``"heur:gift"``: the rule-based walk-home bots from heuristic.py,
    handled the same way.
    """
    model, version = _current_model()
    bot = _bot_mode(opp_path)
    opp = _opponent_model(opp_path) if opp_path and bot is None else None
    if bot is not None:
        import heuristic
        py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)
    game = Game(_STATE["rules"])
    rec = _Recorder()
    start = time.time()
    while not game.done:
        team = game.turn
        if bot is not None and team != net_team:
            action = heuristic.choose(game, py_rng, bot)              # not recorded
        elif opp is not None and team != net_team:
            action, _ = netlib.choose_action(opp, game, rng)          # not recorded
        else:
            phi = potential(game, team)
            action, sample = netlib.choose_action(model, game, rng)
            rec.add(team, sample, phi)
        game.step(action)
    result = rec.finish(game, version, time.time() - start)
    result["opp"] = opp_path                                  # None = plain self-play
    result["net_team"] = net_team if opp_path else None
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
