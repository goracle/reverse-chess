#!/usr/bin/env python
"""NumPy-only inference for the network in net.py, so the game can ship without PyTorch.

``net.PolicyValueNet`` is tiny (~0.4M parameters) and play needs one position at a time, so a
plain NumPy forward pass is fast enough (a few milliseconds) and the packaged game does not
have to carry PyTorch (hundreds of MB).

    python export_weights.py runs/run4/latest.pt ai.npz     # once, wherever PyTorch is installed
    python check_numpy_net.py runs/run4/latest.pt           # optional: NumPy vs PyTorch outputs

``Advisor`` is what play.py uses: it loads ``ai.npz`` (no PyTorch) or a PyTorch ``.pt``
checkpoint (PyTorch is then imported just to read the file), and for a position returns the
network's value and the probability of every legal action.

This module must stay in step with ``net.PolicyValueNet`` (``check_numpy_net.py`` verifies it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine import SIZE, Action, Game
from env import encode_observation

_EPS = 1e-5            # torch.nn.GroupNorm default
_GROUPS = 8


def _relu(x):
    return np.maximum(x, 0.0)


def _conv3(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """3x3 convolution, stride 1, zero padding 1, no bias.  x (B,C,8,8), w (O,C,3,3)."""
    b, c = x.shape[:2]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))   # (B,C,8,8,3,3)
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(b * SIZE * SIZE, c * 9)
    out = cols @ w.reshape(w.shape[0], c * 9).T                               # (B*64, O)
    return out.reshape(b, SIZE, SIZE, -1).transpose(0, 3, 1, 2)


def _group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    b, c, h, w = x.shape
    g = x.reshape(b, _GROUPS, -1)
    g = (g - g.mean(axis=2, keepdims=True)) / np.sqrt(g.var(axis=2, keepdims=True) + _EPS)
    return g.reshape(b, c, h, w) * weight[None, :, None, None] + bias[None, :, None, None]


class NumpyNet:
    """Forward pass of ``net.PolicyValueNet`` from a ``{state_dict key: array}`` mapping."""

    def __init__(self, weights: Dict[str, np.ndarray], version: Optional[int] = None):
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.version = version
        self.channels = self.w["stem.weight"].shape[0]
        self.hidden = self.w["glob_fc.weight"].shape[0]
        self.blocks = 0
        while "blocks.%d.conv1.weight" % self.blocks in self.w:
            self.blocks += 1

    def trunk(self, board: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """(B,18,8,8), (B,S) -> (B,C,8,8)."""
        w = self.w
        b = board.shape[0]
        planes = np.broadcast_to(scalars[:, :, None, None], (b, scalars.shape[1], SIZE, SIZE))
        x = np.concatenate([board, planes], axis=1)
        x = _relu(_group_norm(_conv3(x, w["stem.weight"]),
                              w["stem_norm.weight"], w["stem_norm.bias"]))
        for i in range(self.blocks):
            p = "blocks.%d." % i
            y = _relu(_group_norm(_conv3(x, w[p + "conv1.weight"]),
                                  w[p + "norm1.weight"], w[p + "norm1.bias"]))
            y = _group_norm(_conv3(y, w[p + "conv2.weight"]),
                            w[p + "norm2.weight"], w[p + "norm2.bias"])
            x = _relu(x + y)
        return x

    def global_features(self, x: np.ndarray):
        """(B,C,8,8) -> global vector (B,H) and value (B,)."""
        w = self.w
        b = x.shape[0]
        gc = np.einsum("bchw,oc->bohw", x, w["glob_conv.weight"][:, :, 0, 0])
        gc = _relu(gc + w["glob_conv.bias"][None, :, None, None]).reshape(b, -1)
        g = _relu(gc @ w["glob_fc.weight"].T + w["glob_fc.bias"])
        value = np.tanh(g @ w["value_out.weight"].T + w["value_out.bias"])[:, 0]
        return g, value

    def action_logits(self, x, g, owner: np.ndarray, comps: np.ndarray) -> np.ndarray:
        """owner (N,): position of each action in the batch; comps (N,6): src, dst, demote,
        gift, en_passant, promote (see env.action_components)."""
        w = self.w
        b, c = x.shape[:2]
        comps = comps.astype(np.int64)
        squares = x.reshape(b, c, SIZE * SIZE).transpose(0, 2, 1)         # (B,64,C)
        a_src = squares @ w["src_proj.weight"].T + w["src_proj.bias"]     # (B,64,H)
        a_dst = squares @ w["dst_proj.weight"].T
        a_glob = g @ w["glob_proj.weight"].T                              # (B,H)
        h = (a_src[owner, comps[:, 0]] + a_dst[owner, comps[:, 1]] + a_glob[owner]
             + w["gift_emb.weight"][comps[:, 3]] + w["promo_emb.weight"][comps[:, 5]]
             + w["flag_emb.weight"][comps[:, 2] * 2 + comps[:, 4]])
        h = _relu(_relu(h) @ w["act_fc.weight"].T + w["act_fc.bias"])
        return (h @ w["act_out.weight"].T + w["act_out.bias"])[:, 0]

    def forward(self, board, scalars, owner, comps):
        """Same contract as ``PolicyValueNet.forward``: (logits (N,), value (B,))."""
        x = self.trunk(board, scalars)
        g, value = self.global_features(x)
        return self.action_logits(x, g, owner, comps), value


# --------------------------------------------------------------------------
# Weight files
# --------------------------------------------------------------------------

def save_npz(path: str, state_dict: Dict[str, np.ndarray], version: Optional[int]) -> None:
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in state_dict.items()}
    arrays["meta_version"] = np.asarray(-1 if version is None else int(version))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path: str) -> NumpyNet:
    with np.load(path, allow_pickle=False) as data:
        weights = {k: data[k] for k in data.files if not k.startswith("meta_")}
        version = int(data["meta_version"]) if "meta_version" in data.files else -1
    return NumpyNet(weights, None if version < 0 else version)


def load_checkpoint(path: str) -> NumpyNet:
    """Read a PyTorch checkpoint written by train.py (imports torch just to unpickle it)."""
    try:
        import torch
    except ImportError as err:
        raise RuntimeError(
            "%s is a PyTorch checkpoint and PyTorch is not installed.  On a machine that has it, "
            "run:  python export_weights.py %s ai.npz   and use ai.npz instead." % (path, path)
        ) from err
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    weights = {k: v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()}
    return NumpyNet(weights, int(ckpt.get("version", -1)))


def load(path: str) -> NumpyNet:
    return load_npz(path) if path.lower().endswith(".npz") else load_checkpoint(path)


def find_ai_file(*folders: str) -> Optional[str]:
    """Newest of ai.npz / ai.pt / latest.pt in the given folders (newest, so that copying a
    fresh latest.pt in is never shadowed by an older ai.npz)."""
    found = [os.path.join(f, name) for f in folders if f
             for name in ("ai.npz", "ai.pt", "latest.pt") if os.path.isfile(os.path.join(f, name))]
    return max(found, key=os.path.getmtime) if found else None


# --------------------------------------------------------------------------
# What play.py asks the network
# --------------------------------------------------------------------------

@dataclass
class Analysis:
    value: float                 # expected result for the side to move, -1 (loss) .. +1 (win)
    actions: List[Action]        # the legal actions, in game.legal_actions() order
    probs: np.ndarray            # the policy's probability of each one (sums to 1)

    def top(self, k: int = 3) -> List[Tuple[Action, float]]:
        order = np.argsort(-self.probs)[:k]
        return [(self.actions[i], float(self.probs[i])) for i in order]


class Advisor:
    """The network as an opponent, an evaluator and a hint generator for play.py."""

    def __init__(self, path: str, max_plies: int = 300):
        self.path = path
        self.net = load(path)
        self.version = self.net.version
        # encode_observation normalises the move counter by the cap training used (train.py's
        # --max-plies, default 300), so the network sees the same numbers it learnt on.
        self.max_plies = max_plies
        self._cache = (None, None)

    @property
    def name(self) -> str:
        label = "v%d" % self.version if self.version is not None else "unknown version"
        return "%s (%s)" % (label, os.path.basename(self.path))

    def analyse(self, game: Game) -> Analysis:
        key = (id(game), game.ply, game.turn)
        if self._cache[0] == key:
            return self._cache[1]
        result = self.analyse_fresh(game)
        self._cache = (key, result)
        return result

    def analyse_fresh(self, game: Game) -> Analysis:
        """``analyse`` without the one-entry cache: safe for tree search, where games are
        short-lived copies (ids get reused) and several threads may ask at once."""
        obs = encode_observation(game, self.max_plies)
        actions = game.legal_actions()
        comps = game.action_array().copy()           # canonical frame: flip rows for black
        if game.turn == "b":
            comps[:, :2] ^= 56
        logits, value = self.net.forward(obs["board"][None], obs["scalars"][None],
                                         np.zeros(len(actions), dtype=np.int64), comps)
        z = logits.astype(np.float64)
        p = np.exp(z - z.max())
        return Analysis(float(value[0]), actions, p / p.sum())

    def white_value(self, game: Game, analysis: Optional[Analysis] = None) -> float:
        """The evaluation from white's side: +1 white is winning, -1 black is."""
        a = analysis or self.analyse(game)
        return a.value if game.turn == "w" else -a.value

    def choose(self, game: Game, rng: np.random.Generator) -> Action:
        """Sample a move from the policy, as training does."""
        a = self.analyse(game)
        idx = min(int(np.cumsum(a.probs).searchsorted(rng.random() * a.probs.sum(), side="right")),
                  len(a.actions) - 1)
        return a.actions[idx]
