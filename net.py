#!/usr/bin/env python
"""Policy/value network for Reverse Chess (PyTorch).

One network plays both colours: ``env.encode_observation`` shows it the board
from the side-to-move's point of view.

Architecture
------------
* A small residual conv tower over the 18 board planes; the 17 scalar
  features are broadcast over the board as extra input planes.
* **Value head**: global features -> tanh, the expected result for the side to
  move (+1 win, -1 loss).
* **Policy head** (a "pointer" head, see env.py): the game has ~570k possible
  action ids but only tens to thousands are legal in any position, so there is
  no giant output layer.  Each *legal* action is scored from
      features of its source square + features of its destination square
      + global features + embeddings of (gift, promotion, demote/en-passant)
  and the softmax runs over the legal actions only.  Pieces' types are already
  encoded in the tower's per-square features, so "which piece moves where" is
  learned from the board, not memorised per action id.

The first layer of the per-action scorer is computed by *gathering* per-square
projections, so scoring thousands of actions costs about as much as one small
matrix multiply.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import SIZE, Game
from env import (NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO,
                 action_components, encode_observation)

NUM_SQUARES = SIZE * SIZE


class ResBlock(nn.Module):
    """conv-norm-relu-conv-norm plus skip.  GroupNorm, so behaviour does not
    depend on batch size (batch 1 in self-play workers, hundreds in training)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """See the module docstring.  ``channels`` must be a multiple of 8."""

    def __init__(self, channels: int = 64, blocks: int = 4, hidden: int = 128):
        super().__init__()
        assert channels % 8 == 0, "channels must be a multiple of 8 (GroupNorm)"
        self.cfg = dict(channels=channels, blocks=blocks, hidden=hidden)
        self.stem = nn.Conv2d(NUM_BOARD_PLANES + NUM_SCALARS, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.GroupNorm(8, channels)
        self.blocks = nn.ModuleList(ResBlock(channels) for _ in range(blocks))

        # global features and the value head
        self.glob_conv = nn.Conv2d(channels, 8, 1)
        self.glob_fc = nn.Linear(8 * NUM_SQUARES, hidden)
        self.value_out = nn.Linear(hidden, 1)

        # policy head: first layer of the action scorer, split into gatherable parts
        self.src_proj = nn.Linear(channels, hidden)
        self.dst_proj = nn.Linear(channels, hidden, bias=False)
        self.glob_proj = nn.Linear(hidden, hidden, bias=False)
        self.gift_emb = nn.Embedding(_NUM_GIFT, hidden)
        self.promo_emb = nn.Embedding(_NUM_PROMO, hidden)
        self.flag_emb = nn.Embedding(4, hidden)      # demote * 2 + en_passant
        self.act_fc = nn.Linear(hidden, hidden // 2)
        self.act_out = nn.Linear(hidden // 2, 1)

        # Start with a (nearly) uniform policy and a value near zero, i.e.
        # random play, instead of arbitrary strong opinions.
        nn.init.normal_(self.act_out.weight, std=0.01)
        nn.init.zeros_(self.act_out.bias)
        nn.init.normal_(self.value_out.weight, std=0.01)
        nn.init.zeros_(self.value_out.bias)

    # -- pieces ---------------------------------------------------------
    def trunk(self, board: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """(B,18,8,8), (B,S) -> (B,C,8,8) per-square features."""
        b = board.shape[0]
        planes = scalars[:, :, None, None].expand(b, scalars.shape[1], SIZE, SIZE)
        x = F.relu(self.stem_norm(self.stem(torch.cat([board, planes], dim=1))))
        for block in self.blocks:
            x = block(x)
        return x

    def global_features(self, x: torch.Tensor):
        """(B,C,8,8) -> global vector (B,H) and value (B,)."""
        g = F.relu(self.glob_fc(F.relu(self.glob_conv(x)).flatten(1)))
        value = torch.tanh(self.value_out(g)).squeeze(-1)
        return g, value

    def action_logits(self, x, g, owner, comps):
        """Score every legal action.

        owner  (N,) long: which position in the batch each action belongs to.
        comps  (N,6) long: src, dst, demote, gift, en_passant, promote
               (see env.action_components; squares are row*8+col).
        """
        squares = x.flatten(2).transpose(1, 2)            # (B, 64, C)
        a_src = self.src_proj(squares)                    # (B, 64, H)
        a_dst = self.dst_proj(squares)
        a_glob = self.glob_proj(g)                        # (B, H)
        h = (a_src[owner, comps[:, 0]] + a_dst[owner, comps[:, 1]] + a_glob[owner]
             + self.gift_emb(comps[:, 3]) + self.promo_emb(comps[:, 5])
             + self.flag_emb(comps[:, 2] * 2 + comps[:, 4]))
        h = F.relu(self.act_fc(F.relu(h)))
        return self.act_out(h).squeeze(-1)                # (N,)

    def forward(self, board, scalars, owner, comps):
        """Returns (logits over the flat list of legal actions (N,), value (B,))."""
        x = self.trunk(board, scalars)
        g, value = self.global_features(x)
        return self.action_logits(x, g, owner, comps), value


# --------------------------------------------------------------------------
# Playing with a network (used by self-play workers and by play.py)
# --------------------------------------------------------------------------

def action_components_array(game: Game) -> np.ndarray:
    """``env.action_components`` for every legal action, as a fresh ``(n, 6)`` uint8
    array in ``game.legal_actions()`` order (canonical frame of the side to move).

    The engine already holds the actions as integers; the only thing left to do is
    flip rows for black.  Row r -> 7 - r is r ^ 7, so on ``row * 8 + col`` it is ``^ 56``.
    """
    comps = game.action_array().copy()
    if game.turn == "b":
        comps[:, :2] ^= 56
    return comps


def choose_action(model: PolicyValueNet, game: Game, rng: np.random.Generator,
                  greedy: bool = False):
    """Pick an action for the side to move.

    Returns ``(action, sample)`` where ``sample`` is everything the learner
    needs to retrain on this decision:
    ``(board uint8 (18,8,8), scalars f32 (17,), comps uint8 (n,6),
       chosen index, log-prob of the choice, value estimate)``.
    """
    obs = encode_observation(game)
    actions = game.legal_actions()
    n = len(actions)
    comps = action_components_array(game)
    with torch.inference_mode():
        logits, value = model(
            torch.from_numpy(obs["board"])[None],
            torch.from_numpy(obs["scalars"])[None],
            torch.zeros(n, dtype=torch.long),
            torch.from_numpy(comps.astype(np.int64)),
        )
        logp = torch.log_softmax(logits, dim=0).numpy()
    if greedy:
        idx = int(np.argmax(logp))
    else:
        # The same draw as ``rng.choice(n, p=probs)`` (one uniform, searchsorted on the
        # normalised cumulative sum) without that call's argument validation.
        cdf = np.cumsum(np.exp(logp.astype(np.float64)))
        cdf /= cdf[-1]
        idx = min(int(cdf.searchsorted(rng.random(), side="right")), n - 1)
    sample = (obs["board"].astype(np.uint8), obs["scalars"], comps, idx,
              float(logp[idx]), float(value[0]))
    return actions[idx], sample


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def save_checkpoint(path: str, model: PolicyValueNet, version: int,
                    opt_state: Optional[dict] = None):
    """Atomically write a checkpoint (workers may be reading the old one)."""
    tmp = path + ".tmp"
    torch.save({"version": version, "cfg": model.cfg,
                "state_dict": _to_cpu(model.state_dict()), "opt": _to_cpu(opt_state)}, tmp)
    os.replace(tmp, path)


def _to_cpu(obj):
    """Copy tensors in a (nested) state dict to the CPU, so checkpoints load anywhere
    (workers always run on the CPU, even when the learner trains on a GPU)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[PolicyValueNet, dict]:
    """Load ``(model in eval mode, raw checkpoint dict)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PolicyValueNet(**ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device), ckpt


def make_agent(path: str, greedy: bool = False, rng: Optional[np.random.Generator] = None):
    """A ``game -> Action`` function for play.py (``--white ckpt:runs/x/latest.pt``)."""
    torch.set_num_threads(1)
    model, _ = load_checkpoint(path)
    rng = rng or np.random.default_rng()
    return lambda game: choose_action(model, game, rng, greedy)[0]
