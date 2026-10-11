#!/usr/bin/env python
"""Self-play reinforcement learning for Reverse Chess (PPO, parallel workers).

    pip install torch            # the only new dependency (CPU build is fine)
    python train.py --smoke      # ~1 minute pipeline check, tiny network
    python train.py              # real run, writes runs/run1/
    python train.py --resume     # continue runs/run1/ where it stopped
    python play.py --white ckpt:runs/run1/latest.pt    # play against it

How it works
------------
* ``--workers`` processes (default: cores - 2) play games continuously, each
  sampling moves from the *latest* network (see rollout.py).  Finished games
  stream back to this process, which is the learner.
* After ``--plies-per-iter`` decisions have arrived, the learner runs a few
  epochs of PPO on them and writes a new ``latest.pt``.  Workers keep playing
  during the update (so cores stay busy) and pick up the new weights at their
  next game.  Games more than ``--max-stale`` updates old are discarded; the
  stored behaviour log-probs make slightly stale games fine for PPO.
* Reward: +1 win, -1 loss, ``--draw-value`` for a draw (default 0), given to
  each player at the end of the game.  Advantages are GAE per player
  (returns.py).  Defaults (--gae-lambda 1, --shape-coef 0) reproduce the plain
  "result - value estimate".
* Optional progress shaping (--shape-coef 1.0 --gae-lambda 0.95): a small
  per-move reward for the change in a *potential* (opponent's remaining
  burden minus ours; env.potential).  Potential-based shaping provably leaves
  the optimal policy unchanged, so it cannot be farmed; it just credits
  progress toward getting home earlier than the final +-1.
* Optional look-ahead in self-play (on by default, ``--search-frac 0`` turns it off): a
  fraction of the network's decisions are made by a short MCTS (``--search-sims`` simulations,
  see mcts.py / net.search_action) instead of straight from the policy.  The searched
  position is stored with the search's move distribution, and the learner adds a
  cross-entropy term pulling the policy towards it (``--search-coef``); PPO's policy-gradient
  term is skipped for those decisions, since their move was not drawn from the policy.  The
  value head keeps learning from the game result, so everything else is unchanged and an
  existing checkpoint can simply be resumed.
* Every ``--eval-every`` updates the current network plays ``--eval-games``
  games against a uniformly random player (half as each colour).  A network
  that has learned anything should score well above 0.5 there.

Memory (16 GB box): workers are small (one torch thread each, small engine
caches), a few hundred MB apiece.  If memory is tight lower ``--workers`` or
``--move-cache``; watch with htop the first few minutes.

Outputs in ``runs/<name>/``: ``latest.pt`` (read by workers and play.py),
``iter_NNNN.pt`` snapshots, ``log.csv``.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # workers inherit; the learner sets its own below

import argparse
import csv
import faulthandler
import glob
import signal
import sys
import multiprocessing as mp
import re
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

import net as netlib
import returns
import rollout
import rustnet

# ---- terminal colours --------------------------------------------------------
# Off when piped or when NO_COLOR is set; FORCE_COLOR=1 turns them on regardless.
#
# The log is built so it can be read at a glance:
#   * every line of an iteration starts with a coloured TAG (v57, train, time, learner,
#     steps, ended, vs old, phase, EVAL), so the eye can jump to the line it wants;
#   * inside a line the text is the terminal's normal colour; only numbers that carry a
#     judgement are coloured:  green = healthy,  yellow = watch,  red = a problem
#     (for the time lines: share of the learner's wall time spent there);
#   * nothing is dimmed, so nothing has to be squinted at.
# Tags are black on a mid-tone colour and the rest uses the standard bold hues, which are
# readable on both dark and light terminals.
_COLOR = ((sys.stdout.isatty() and "NO_COLOR" not in os.environ)
          or bool(os.environ.get("FORCE_COLOR")))

GOOD, WATCH, BAD = "1;32", "1;33", "1;31"
BLACK_TEAM, WHITE_TEAM = "1;34", "1;33"


def _c(code, text):
    return "\033[%sm%s\033[0m" % (code, text) if _COLOR else str(text)


def _tag(label, bg):
    """Line prefix: the label in black on a 256-colour background, padded to a fixed width."""
    return _c("1;38;5;16;48;5;%d" % bg, " %-7s " % label) + " "


def _health(text, value, good, ok, higher_better=False):
    """Green / yellow / red (all bold) by rule-of-thumb thresholds; plain if value is nan."""
    if value != value:
        return text
    if not higher_better:
        code = GOOD if value <= good else WATCH if value <= ok else BAD
    else:
        code = GOOD if value >= good else WATCH if value >= ok else BAD
    return _c(code, text)


def _share(text, seconds, total):
    """Colour a duration by its share of ``total``: >= 40% red, >= 20% yellow, else plain."""
    f = seconds / max(total, 1e-9)
    return _c(BAD if f >= 0.4 else WATCH if f >= 0.2 else "1", text)


_REASON_SHORT = {
    "draw: no gift or pawn move in 50 plies": "draw: 50 idle plies",
    "draw: position repeated 3 times": "draw: repetition",
    "draw: move limit reached": "draw: move limit",
    "draw: every legal move leaves the opponent without a legal move": "draw: no safe move",
    "draw: both teams satisfy a winning condition": "draw: both won",
}


def _short_reason(r):
    """Shorter wording for the game-ending reasons, so a whole iteration fits on a line or two."""
    if r in _REASON_SHORT:
        return _REASON_SHORT[r]
    m = re.match(r"^(black|white) wins: no way for all of (black|white)'s (\w+) to get home", r)
    if m:
        return "%s wins: %s's %s stuck" % m.groups()
    m = re.match(r"^(black|white) wins: pawn wall-off prevents (black|white) from winning", r)
    if m:
        return "%s wins: wall-off vs %s" % m.groups()
    return r[:44]


def _ended_line(reasons, limit=6):
    """'134x black wins: white's pawns stuck  .  107x ...', each reason in its own colour."""
    items = []
    for r, k in reasons.most_common(limit):
        text = _short_reason(r)
        code = "33" if text.startswith("draw") else "36" if "wins" in text else "32"
        items.append(_c("1", "%d\u00d7" % k) + " " + _c(code, text))
    rest = sum(reasons.values()) - sum(k for _, k in reasons.most_common(limit))
    if rest:
        items.append("+%d other" % rest)
    return "  \u00b7  ".join(items)


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------

def assemble(games, draw_value, shape_coef=0.0, gamma=1.0, gae_lambda=1.0):
    """Concatenate finished games into one training set of decisions.

    ``adv`` are GAE advantages and ``ret`` the matching value targets, computed
    per player (see returns.py).  With the defaults (no shaping, gamma = lambda
    = 1) this is the original ``outcome - value``.
    """
    advs, rets = [], []
    for g in games:
        a, r = returns.game_returns(g, draw_value, shape_coef, gamma, gae_lambda)
        advs.append(a)
        rets.append(r)
    counts = np.concatenate([g["counts"] for g in games]).astype(np.int64)
    # search targets (rollout.py): games played without search simply have none
    searched = np.concatenate([g["searched"] if "searched" in g
                               else np.zeros(len(g["player"]), np.uint8) for g in games])
    pi = (np.concatenate([g["pi"] if "pi" in g else np.zeros(len(g["comps"]), np.float32)
                          for g in games]) if searched.any() else None)
    return {
        "searched": searched,                        # (decisions,) 1 = the move came from the search
        "pi": pi,                                    # (actions,) search move distribution, or None
        "boards": np.concatenate([g["boards"] for g in games]),
        "scalars": np.concatenate([g["scalars"] for g in games]),
        "comps": np.concatenate([g["comps"] for g in games]),
        "counts": counts,
        "starts": np.cumsum(counts) - counts,       # offset of each decision's actions in comps
        "chosen": np.concatenate([g["chosen"] for g in games]).astype(np.int64),
        "logp": np.concatenate([g["logp"] for g in games]),
        "value": np.concatenate([g["value"] for g in games]),
        "ret": np.concatenate(rets),
        "adv": np.concatenate(advs),
        # how far through its game each decision was (0 = first move, 1 = last); diagnostics only
        "phase": np.concatenate([np.arange(len(g["player"])) / max(len(g["player"]) - 1, 1)
                                 for g in games]).astype(np.float32),
    }


def make_batches(counts, minibatch, max_actions, rng, drop_tail=False):
    """Shuffled batches of decision indices: at most ``minibatch`` decisions and
    (roughly) ``max_actions`` legal actions each, which bounds memory.

    ``drop_tail`` skips the final, smaller batch of the epoch.  Its size differs every
    iteration, and on the Intel GPU each new batch size pays a one-off setup cost.  The
    permutation is fresh every epoch, so the dropped decisions are random ones, and they
    are used in the next epoch.  (If that would leave no batch at all, it is kept.)"""
    batch, n_actions, yielded = [], 0, 0
    for i in rng.permutation(len(counts)):
        batch.append(i)
        n_actions += counts[i]
        if len(batch) >= minibatch or n_actions >= max_actions:
            yield np.asarray(batch)
            yielded += 1
            batch, n_actions = [], 0
    if batch and not (drop_tail and yielded):
        yield np.asarray(batch)


def collate(data, adv, mb, device, pad=0, tm=None):
    """Turn a batch of decision indices into tensors.

    ``owner_pad`` / ``comps_pad`` are the action list padded (with harmless zero rows) up to a
    multiple of ``pad`` (``pad`` = 0: not padded, they are the same arrays as ``owner`` /
    ``comps``).  The number of legal actions differs in every minibatch, and on the Intel GPU
    each *new* length makes the first call of the policy head cost ~0.3 s of host-side setup
    (measured with xpu_shape_probe.py).  Feeding the head a few fixed lengths avoids that; the
    caller slices the logits back to ``n``.

    If ``tm`` is a dict, ``tm["collate_np"]`` (building the arrays on the host) and
    ``tm["collate_xfer"]`` (copying them to the device) accumulate the two halves' wall time.
    A large xfer share means the copy is waiting for the device to finish the previous step.
    """
    t_np = time.perf_counter()
    counts = data["counts"][mb]
    comps = np.concatenate([data["comps"][s:s + n]
                            for s, n in zip(data["starts"][mb], counts)])
    owner = np.repeat(np.arange(len(mb)), counts)
    n = len(owner)
    n_pad = -(-n // pad) * pad if pad > 0 else n
    if n_pad == n:
        comps_pad, owner_pad = comps, owner
    else:
        comps_pad = np.zeros((n_pad, comps.shape[1]), dtype=comps.dtype)
        comps_pad[:n] = comps
        owner_pad = np.zeros(n_pad, dtype=owner.dtype)  # row 0 is a valid owner; the logits are discarded
        owner_pad[:n] = owner
    t_x = time.perf_counter()

    def t(array, dtype):
        return torch.as_tensor(array, dtype=dtype, device=device)

    searched = data["searched"][mb]
    pi_mat = None
    if searched.any():                # (B, Amax) search distributions, zero rows for the rest
        pi_mat = np.zeros((len(mb), int(counts.max())), dtype=np.float32)
        for r in np.flatnonzero(searched):
            s0, n_r = data["starts"][mb[r]], counts[r]
            pi_mat[r, :n_r] = data["pi"][s0:s0 + n_r]
    out = {
        "searched": t(searched, torch.float32),
        "pi": None if pi_mat is None else t(pi_mat, torch.float32),
        "board": t(data["boards"][mb], torch.float32),
        "scalars": t(data["scalars"][mb], torch.float32),
        "owner": t(owner, torch.long),
        "comps_pad": t(comps_pad, torch.long),
        "owner_pad": t(owner_pad, torch.long),
        "n": n,
        "n_pad": n_pad,
        "counts": t(counts, torch.long),
        "chosen": t(data["chosen"][mb], torch.long),
        "old_logp": t(data["logp"][mb], torch.float32),
        "ret": t(data["ret"][mb], torch.float32),
        "adv": t(adv[mb], torch.float32),
    }
    if tm is not None:
        tm["collate_np"] = tm.get("collate_np", 0.0) + t_x - t_np
        tm["collate_xfer"] = tm.get("collate_xfer", 0.0) + time.perf_counter() - t_x
    return out


_BUFFERS = {}     # (name, device) -> persistent device buffer, grown (never shrunk) when needed


def _to_device(name, array, dtype, device, persist):
    """``array`` as a ``dtype`` tensor on ``device``.

    With ``persist`` the data is copied into a buffer that lives across iterations (allocated with
    25% slack and only regrown if an iteration needs more) and a view of it is returned.  Without
    it every iteration allocates fresh device tensors of a slightly different size and frees them
    again, which on the Intel GPU (device memory is system memory) pays allocation / first-touch
    costs again and again.  Rows past the view's end are never read."""
    if not persist:
        return torch.as_tensor(array, dtype=dtype, device=device)
    host = torch.as_tensor(array, dtype=dtype)
    n = host.shape[0]
    key = (name, str(device))
    buf = _BUFFERS.get(key)
    if buf is None or buf.shape[0] < n or buf.dtype != dtype or buf.shape[1:] != host.shape[1:]:
        _BUFFERS.pop(key, None)                                   # free the old one first
        buf = torch.empty((int(n * 1.25) + 16,) + tuple(host.shape[1:]), dtype=dtype, device=device)
        _BUFFERS[key] = buf
    view = buf[:n]
    view.copy_(host)
    return view


def _dev_call(device, name, *a):
    """``torch.<xpu|cuda>.<name>(*a)``, or None when this backend has no such function (the log
    then says n/a; nothing is guessed)."""
    mod = getattr(torch, torch.device(device).type, None)
    fn = getattr(mod, name, None) if mod is not None else None
    return None if fn is None else fn(*a)


def _device_total_bytes(device):
    props = _dev_call(device, "get_device_properties", 0)
    return getattr(props, "total_memory", None)


class DeviceData:
    """One iteration's training set, resident on the device.

    ``collate`` copied ~13 small tensors to the device for EVERY minibatch (boards, scalars,
    actions, advantages, ...), each copy queued behind the step still running there.  Here the
    whole set goes over once (a few hundred MB at most), and a minibatch is gathered on the
    device with ``tensor[indices]``.  The indices of a whole epoch's minibatches (decision
    ids, action ids, owner ids, search-target ids) are built on the host and sent as ONE
    int64 array, so a step needs no host-to-device copy at all.

    Produces exactly what ``collate`` does (checked on the first minibatch of a run by
    ``_check_resident_equivalent``), except that padding rows of the action list repeat action
    row 0 instead of being zeros; they are scored and then sliced away like before.
    """

    def __init__(self, data, adv, device, tm, persist=False):
        t0 = time.perf_counter()
        self.device = device
        self.h_counts, self.h_starts = data["counts"], data["starts"]
        self.h_searched = data["searched"]

        def t(name, array, dtype):
            return _to_device(name, array, dtype, device, persist)

        self.boards = t("boards", data["boards"], torch.float32)
        self.scalars = t("scalars", data["scalars"], torch.float32)
        self.comps = t("comps", data["comps"], torch.long)
        self.counts = t("counts", data["counts"], torch.long)
        self.chosen = t("chosen", data["chosen"], torch.long)
        self.old_logp = t("old_logp", data["logp"], torch.float32)
        self.ret = t("ret", data["ret"], torch.float32)
        self.adv = t("adv", adv, torch.float32)
        self.searched = t("searched", data["searched"], torch.float32)
        self.pad_slot = 0
        self.pi = None
        if data["pi"] is not None:            # one extra zero entry that "no target" slots point at
            self.pad_slot = len(data["pi"])
            self.pi = t("pi", np.concatenate([data["pi"], np.zeros(1, np.float32)]), torch.float32)
        tm["collate_xfer"] = tm.get("collate_xfer", 0.0) + time.perf_counter() - t0

    def nbytes(self):
        """Bytes of each resident tensor (what this iteration's data occupies on the device)."""
        names = ("boards", "scalars", "comps", "counts", "chosen", "old_logp", "ret", "adv", "searched", "pi")
        return {k: getattr(self, k).numel() * getattr(self, k).element_size()
                for k in names if getattr(self, k) is not None}

    def plan(self, batches, pad, tm):
        """Index arrays for a list of minibatches (decision-id arrays), uploaded in one copy."""
        t_np = time.perf_counter()
        pieces, meta, off = [], [], 0
        for mb in batches:
            counts, starts = self.h_counts[mb], self.h_starts[mb]
            B, n = len(mb), int(counts.sum())
            owner = np.repeat(np.arange(B), counts)
            act_idx = starts[owner] + (np.arange(n) - (np.cumsum(counts) - counts)[owner])
            n_pad = -(-n // pad) * pad if pad > 0 else n
            if n_pad > n:                     # harmless filler rows: valid row 0, owner 0
                act_idx = np.concatenate([act_idx, np.zeros(n_pad - n, dtype=act_idx.dtype)])
                owner = np.concatenate([owner, np.zeros(n_pad - n, dtype=owner.dtype)])
            amax = int(counts.max())
            parts = [np.asarray(mb, dtype=np.int64), act_idx.astype(np.int64), owner.astype(np.int64)]
            has_pi = self.pi is not None and self.h_searched[mb].any()
            if has_pi:                        # (B, amax) gather ids; non-searched rows -> zero slot
                cols = np.arange(amax)[None, :]
                valid = (cols < counts[:, None]) & (self.h_searched[mb][:, None] != 0)
                parts.append(np.where(valid, starts[:, None] + cols, self.pad_slot).reshape(-1).astype(np.int64))
            o_act, o_own = off + B, off + B + n_pad
            o_pi = o_own + n_pad if has_pi else None
            meta.append((off, B, n, n_pad, amax, o_act, o_own, o_pi))
            off += sum(len(x) for x in parts)
            pieces.extend(parts)
        packed = np.concatenate(pieces)
        t_x = time.perf_counter()
        packed_d = torch.as_tensor(packed, dtype=torch.long, device=self.device)
        tm["collate_np"] = tm.get("collate_np", 0.0) + t_x - t_np
        tm["collate_xfer"] = tm.get("collate_xfer", 0.0) + time.perf_counter() - t_x
        return packed_d, meta

    def batch(self, plan, k):
        """Minibatch ``k`` of a plan, as the dict ``collate`` returns (all views / gathers)."""
        P, meta = plan
        o_mb, B, n, n_pad, amax, o_act, o_own, o_pi = meta[k]
        mb = P[o_mb:o_mb + B]
        owner_pad = P[o_own:o_own + n_pad]
        pi = None if o_pi is None else self.pi[P[o_pi:o_pi + B * amax]].view(B, amax)
        return {
            "searched": self.searched[mb],
            "pi": pi,
            "board": self.boards[mb],
            "scalars": self.scalars[mb],
            "owner": owner_pad[:n],
            "comps_pad": self.comps[P[o_act:o_act + n_pad]],
            "owner_pad": owner_pad,
            "n": n,
            "n_pad": n_pad,
            "counts": self.counts[mb],
            "chosen": self.chosen[mb],
            "old_logp": self.old_logp[mb],
            "ret": self.ret[mb],
            "adv": self.adv[mb],
        }


_RES_CHECKED = []


def _check_resident_equivalent(b, data, adv, mb, device, pad):
    """Once per run: the device-resident minibatch must equal the one ``collate`` builds."""
    ref = collate(data, adv, mb, device, pad)
    for key in ("searched", "board", "scalars", "owner", "owner_pad", "counts", "chosen",
                "old_logp", "ret", "adv"):
        if key == "owner_pad":
            ok = torch.equal(ref[key][:ref["n"]].cpu(), b[key][:b["n"]].cpu())
        else:
            ok = torch.equal(ref[key].cpu(), b[key].cpu())
        assert ok, "device-resident batch differs from collate() in %r (rerun with --host-batches)" % key
    assert torch.equal(ref["comps_pad"][:ref["n"]].cpu(), b["comps_pad"][:b["n"]].cpu()), "comps differ"
    assert (ref["n"], ref["n_pad"]) == (b["n"], b["n_pad"]), "action counts differ"
    assert (ref["pi"] is None) == (b["pi"] is None), "pi presence differs"
    if ref["pi"] is not None:
        assert torch.equal(ref["pi"].cpu(), b["pi"].cpu()), "search targets differ"
    _RES_CHECKED.append(True)


def per_decision_logp(logits, owner, counts, amax=None):
    """Log-softmax of the flat legal-action logits within each decision.

    Returns ``(logp (B, Amax), mask (B, Amax))``; padded slots hold a huge
    negative number and are False in the mask.
    """
    b = counts.shape[0]
    if amax is None:
        amax = int(counts.max())          # GPU->CPU sync; callers should pass amax
    offsets = torch.cumsum(counts, 0) - counts
    rank = torch.arange(owner.shape[0], device=owner.device) - offsets[owner]
    padded = torch.full((b, amax), -1e9, device=logits.device, dtype=logits.dtype)
    padded = padded.index_put((owner, rank), logits)
    mask = torch.arange(amax, device=owner.device)[None, :] < counts[:, None]
    return torch.log_softmax(padded, dim=1), mask


# --------------------------------------------------------------------------
# PPO update
# --------------------------------------------------------------------------

_PAD_CHECKED = []
_SEEN_SHAPES = set()      # (decisions, padded actions) combinations this process has already run


def _check_padding_equivalent(model, b, logits, value):
    """Once per run: padding the action list must not change the logits or the value."""
    with torch.no_grad():
        ref_logits, ref_value = model(b["board"], b["scalars"], b["owner"], b["comps_pad"][:b["n"]])
        assert ref_logits.shape == logits.shape, (ref_logits.shape, logits.shape)
        assert torch.allclose(ref_logits, logits.detach(), atol=1e-4, rtol=1e-4), \
            "padded and unpadded action logits differ: max |diff| %.3g" % (ref_logits - logits.detach()).abs().max().item()
        assert torch.allclose(ref_value, value.detach(), atol=1e-4, rtol=1e-4), "padded/unpadded values differ"
    _PAD_CHECKED.append(True)


def ppo_update(model, opt, data, args, device, rng, timing=None):
    """A few epochs of PPO on one iteration's data; returns averaged statistics.

    If ``timing`` is a dict it is filled with where the learner's wall time went:
    collate (host builds the batch), dispatch (host queues forward/backward/Adam),
    wait (host blocked until the previous step finished on the device) and steps.
    wait >> collate + dispatch means the device is the bottleneck; the reverse means
    the host is, and the device sits idle between steps.  Also filled in:
    collate_np / collate_xfer (the two halves of collate: building arrays vs copying them to
    the device), step_dispatch (per-step dispatch seconds, in order), new_shape (per step: was
    this (decisions, padded actions) shape new to this process), and kl_stop (the KL gate
    ended the iteration early).
    """
    tm = timing if timing is not None else {}
    tm.update(collate=0.0, dispatch=0.0, wait=0.0, steps=0, collate_np=0.0, collate_xfer=0.0,
              step_dispatch=[], new_shape=[], kl_stop=False, decisions=0)
    kl_hist = deque(maxlen=max(1, args.kl_window))
    adv = data["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    names = ("pg", "vf", "ent", "kl", "clipfrac", "ce")
    stats = {k: [] for k in names}
    prev = None          # previous step's stats, still on the device

    def read(t):
        """Copy one step's stats to the host (waits for that step); returns its kl."""
        t_w = time.perf_counter()
        vals = t.tolist()
        tm["wait"] += time.perf_counter() - t_w
        for k, v in zip(names, vals):
            stats[k].append(v)
        return vals[3]

    stop = False
    _dev_call(device, "reset_peak_memory_stats")
    resident = None if args.host_batches else DeviceData(data, adv, device, tm, args.persist_data)
    tm["gpu_resident"] = _dev_call(device, "memory_allocated")     # data + model + optimizer state
    tm["gpu_parts"] = {} if resident is None else resident.nbytes()
    for _ in range(args.epochs):
        batches = list(make_batches(data["counts"], args.minibatch, args.max_actions, rng, args.drop_tail))
        plan = resident.plan(batches, args.pad_actions, tm) if resident is not None else None
        for k, mb in enumerate(batches):
            amax = int(data["counts"][mb].max())      # host-side, so no GPU sync
            t_c = time.perf_counter()
            if resident is not None:                  # gathered on the device: no host copies
                b = resident.batch(plan, k)
                if not _RES_CHECKED:
                    _check_resident_equivalent(b, data, adv, mb, device, args.pad_actions)
            else:
                b = collate(data, adv, mb, device, args.pad_actions, tm)   # CPU work; overlaps the GPU still busy with the previous step
            tm["collate"] += time.perf_counter() - t_c
            if prev is not None:
                kl = read(prev)
                prev = None
                kl_hist.append(kl)
                # Policy moved too far: stop this iteration.  One minibatch's KL is noisy (and the
                # first ones also carry the staleness of the games), so the gate looks at the mean
                # of the last --kl-window steps, plus a hard cap on any single step.
                if (kl > args.kl_hard * args.target_kl
                        or (len(kl_hist) == kl_hist.maxlen
                            and sum(kl_hist) / len(kl_hist) > 1.5 * args.target_kl)):
                    stop = tm["kl_stop"] = True
                    break
            shape = (len(mb), b["n_pad"])
            tm["new_shape"].append(shape not in _SEEN_SHAPES)
            _SEEN_SHAPES.add(shape)
            t_d = time.perf_counter()
            try:
                logits, value = model(b["board"], b["scalars"], b["owner_pad"], b["comps_pad"])
                logits = logits[:b["n"]]                  # drop the padding rows (their gradient is zero)
                if args.pad_actions > 0 and not _PAD_CHECKED:
                    _check_padding_equivalent(model, b, logits, value)
                logp_all, mask = per_decision_logp(logits, b["owner"], b["counts"], amax)
                rows = torch.arange(len(mb), device=device)
                logp = logp_all[rows, b["chosen"]]

                ratio = torch.exp(logp - b["old_logp"])
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                # PPO only learns from decisions drawn from the policy itself; a searched one
                # was drawn from the search's distribution (w = 0), and teaches through ce below.
                w = 1.0 - b["searched"]
                wsum = w.sum().clamp(min=1.0)
                pg_loss = -(torch.min(ratio * b["adv"], clipped * b["adv"]) * w).sum() / wsum
                vf_loss = torch.mean((value - b["ret"]) ** 2)
                probs = logp_all.exp()                       # exactly 0 in padded slots
                entropy = -(probs * logp_all.masked_fill(~mask, 0.0)).sum(1).mean()
                if b["pi"] is not None:
                    ce_rows = -(b["pi"] * logp_all.masked_fill(~mask, 0.0)).sum(1)
                    ce_loss = (ce_rows * b["searched"]).sum() / b["searched"].sum().clamp(min=1.0)
                else:
                    ce_loss = torch.zeros((), device=device)
                loss = (pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy
                        + args.search_coef * ce_loss)

                opt.zero_grad(set_to_none=True)
                loss.backward()
            except RuntimeError as e:      # still raises; just says which shapes were involved
                print("\n!! %s on a minibatch of %d decisions, %d actions, amax %d: %s"
                      % (device, len(mb), b["owner"].shape[0], amax, str(e).splitlines()[0]), flush=True)
                raise
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            dt = time.perf_counter() - t_d
            tm["dispatch"] += dt
            tm["step_dispatch"].append(dt)
            tm["steps"] += 1
            tm["decisions"] += len(mb)

            with torch.no_grad():                     # stays on the device; read next time round
                prev = torch.stack([
                    pg_loss.detach(), vf_loss.detach(), entropy.detach(),
                    # kl / clipfrac over the policy-sampled decisions only (searched ones were
                    # not drawn from the old policy, so they would bias both)
                    ((b["old_logp"] - logp) * w).sum() / wsum,
                    # == ((ratio - 1).abs() > clip).float().mean(), but pure float arithmetic:
                    # the bool -> float cast raises UR_RESULT_ERROR_INVALID_ARGUMENT on Iris Xe.
                    (((ratio - 1).abs() - args.clip).sign().clamp(min=0) * w).sum() / wsum,
                    ce_loss.detach()])
        if stop:
            break
    if prev is not None:
        read(prev)
    tm["gpu_peak"] = _dev_call(device, "max_memory_allocated")
    tm["gpu_total"] = _device_total_bytes(device)
    return {k: float(np.mean(v)) for k, v in stats.items()}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def pick_device(choice):
    """auto -> CUDA, else an Intel GPU (torch.xpu, PyTorch >= 2.5), else CPU."""
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    return choice


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("run")
    g.add_argument("--name", default="run1")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--iters", type=int, default=100000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    g = ap.add_argument_group("parallelism / memory")
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    g.add_argument("--learner-threads", type=int, default=4)
    g.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu (learner only; workers always use the CPU)")
    g.add_argument("--move-cache", type=int, default=50_000,
                   help="engine move-cache entries per worker (memory knob)")
    g.add_argument("--net-backend", choices=("native", "torch"), default="native",
                   help="how the self-play workers evaluate the network: native = the Rust forward pass "
                        "(rustnet.py; needs ./build_native.sh, raises if the library is missing or stale), "
                        "torch = PyTorch on the CPU (the old behaviour).  The learner always uses torch.")
    g.add_argument("--persist-data", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the iteration's training data in device buffers that live across iterations "
                        "(25%% slack, regrown when needed) instead of allocating and freeing fresh device "
                        "tensors every iteration.  Uses a few hundred MB more device memory; "
                        "--no-persist-data is the old behaviour.")
    g = ap.add_argument_group("self-play")
    g.add_argument("--plies-per-iter", type=int, default=8000)
    g.add_argument("--max-plies", type=int, default=300, help="draw after this many plies")
    g.add_argument("--draw-value", type=float, default=0.0,
                   help="reward for a draw (try -0.1 if games keep drawing)")
    g.add_argument("--max-stale", type=int, default=2)
    g.add_argument("--pool-power", type=float, default=2.0,
                   help="how strongly to favour the older snapshots that beat the current net "
                        "(0 = pick uniformly, 2 = default, higher = focus harder on the toughest)")
    g.add_argument("--heur-frac", type=float, default=0.25,
                   help="fraction of games played against the rule-based walk-home bots from heuristic.py "
                        "(heur:nogift and heur:gift; the one the net scores worse against is picked more "
                        "often, see --pool-power) -- but only while games are being queued in every "
                        "--heur-every-th iteration.  0 = never.  Only the net's own moves are trained on, "
                        "and the 'vs bots' line reports only those training games (no separate eval games).")
    g.add_argument("--heur-every", type=int, default=20,
                   help="queue the bot games only in every this-many-th iteration (they are slower to play "
                        "than self-play games; the queue is ~1 iteration deep, so they show up in the "
                        "batch a little later).  A sanity check, not a real-time measure.  1 = every iteration.")
    g.add_argument("--pool-frac", type=float, default=0.25,
                   help="fraction of games played against a random older iter_*.pt snapshot "
                        "instead of the current net (stops self-play from cycling; "
                        "0 = pure self-play, the old behaviour)")
    g.add_argument("--shape-coef", type=float, default=0.0,
                   help="weight of potential-based progress shaping (0 = off; try 1.0 with "
                        "--gae-lambda 0.95).  See returns.py for why this is safe.")
    g.add_argument("--gamma", type=float, default=1.0, help="discount (1.0 keeps shaping exactly neutral)")
    g.add_argument("--gae-lambda", type=float, default=1.0,
                   help="GAE lambda; 1.0 = plain Monte-Carlo (old behaviour); 0.95 is typical "
                        "and is needed for shaping to have any effect")
    g.add_argument("--queue-factor", type=int, default=10,
                   help="games queued per worker; workers keep playing while the learner "
                        "trains, so this must be deep enough that they never run dry "
                        "(roughly learn-time x plies/s / avg-game-plies / workers).  Deeper "
                        "means staler games; those older than --max-stale are dropped")
    g = ap.add_argument_group("network")
    g.add_argument("--channels", type=int, default=64)
    g.add_argument("--blocks", type=int, default=4)
    g.add_argument("--hidden", type=int, default=128)
    g = ap.add_argument_group("search in self-play (mcts.py)")
    g.add_argument("--search-frac", type=float, default=0.15,
                   help="fraction of the network's decisions made by a short tree search (0 = "
                        "no search, the old behaviour).  Cost is roughly frac * sims extra "
                        "network calls + game copies per decision; watch the 'search' log line")
    g.add_argument("--search-sims", type=int, default=16, help="simulations per searched decision")
    g.add_argument("--search-coef", type=float, default=1.0,
                   help="weight of the cross-entropy pulling the policy towards the search's move "
                        "distribution")
    g.add_argument("--search-noise", type=float, default=0.25,
                   help="Dirichlet noise mixed into the root priors (exploration)")
    g.add_argument("--search-temp", type=float, default=1.0,
                   help="searched moves are drawn from visit counts ** (1/temp); 0 = most visited")
    g.add_argument("--search-sweep", type=int, default=0,
                   help="root moves (likeliest first) played out and checked for an immediate "
                        "win/loss before searching; 0 = off.  Finds one-move wins the policy "
                        "misses, at one game copy per move checked")
    g.add_argument("--search-cpuct", type=float, default=1.5)
    g.add_argument("--search-batch", type=int, default=1,
                   help="leaves evaluated per network call (virtual loss).  1 = the plain "
                        "one-at-a-time search (default).  Above 1 the 16 simulations are spread "
                        "over many root moves, which flattens the visit distribution the policy "
                        "is trained on (policy entropy climbs, ce explodes): see --search-vl, "
                        "and check_vl.py to measure it on your network")
    g.add_argument("--search-vl", type=float, default=0.25,
                   help="virtual-loss strength for --search-batch > 1, 0..1.  1 = every descent in "
                        "flight counts as a sure loss (the old behaviour: spreads the simulations "
                        "over many root moves, flattening the visit distribution); 0 = it only "
                        "counts as a visit (descents spread the way PUCT would spread them; batches "
                        "shrink when descents collide).  0.25 keeps most of the batching with about "
                        "half the distortion (check_vl.py measures it on your network).  No effect at "
                        "--search-batch 1")
    g.add_argument("--search-smooth", type=float, default=0.0,
                   help="blend this fraction of the network's own policy into the search's target "
                        "(the move is still drawn from the unsmoothed search).  Few simulations "
                        "give sparse targets that sharpen the policy fast; try 0.1 if entropy "
                        "falls quickly or repetition draws climb")
    g = ap.add_argument_group("PPO")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--epochs", type=int, default=2)
    g.add_argument("--minibatch", type=int, default=256, help="decisions per batch")
    g.add_argument("--max-actions", type=int, default=60_000,
                   help="legal actions per batch (memory knob)")
    g.add_argument("--clip", type=float, default=0.2)
    g.add_argument("--vf-coef", type=float, default=0.5)
    g.add_argument("--ent-coef", type=float, default=0.01)
    g.add_argument("--pad-actions", type=int, default=4096,
                   help="pad each minibatch's action list up to a multiple of this, so the policy head sees "
                        "only a few distinct shapes (a new shape costs ~0.3 s on the Intel GPU); 0 = off")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--target-kl", type=float, default=0.03)
    g.add_argument("--kl-window", type=int, default=8,
                   help="stop an iteration early when the mean KL of the last this-many minibatches exceeds "
                        "1.5 * --target-kl (1 = judge every minibatch on its own, the old behaviour)")
    g.add_argument("--kl-hard", type=float, default=4.0,
                   help="also stop at once if a single minibatch's KL exceeds this multiple of --target-kl")
    g.add_argument("--host-batches", action="store_true",
                   help="build and copy every minibatch from the host (the old behaviour) instead of "
                        "keeping the iteration's data on the device and gathering minibatches there")
    g.add_argument("--drop-tail", action=argparse.BooleanOptionalAction, default=True,
                   help="skip each epoch's last, smaller minibatch (its size is new every iteration, and a "
                        "new batch size costs a one-off setup on the Intel GPU); --no-drop-tail keeps it")
    g = ap.add_argument_group("evaluation / checkpoints")
    g.add_argument("--eval-every", type=int, default=10)
    g.add_argument("--eval-games", type=int, default=40)
    g.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()
    if args.smoke:
        args.name = "smoke"
        args.workers = min(args.workers, 4)
        args.plies_per_iter = 400
        args.max_plies = 60
        args.channels, args.blocks, args.hidden = 16, 1, 32
        args.iters = 3
        args.eval_every, args.eval_games, args.save_every = 2, 8, 1
        args.minibatch = 64
        args.search_sims = min(args.search_sims, 8)
    return args


POOL_SCORE = {}   # snapshot path -> (decayed score sum, decayed game count): current net vs that snapshot


def _pool_weights(snaps, power):
    """Sampling weights over snapshots: the ones that beat the current net get picked more.
    power=0 -> uniform.  Snapshots with few games are treated as 0.5 (an even match)."""
    w = []
    for p in snaps:
        sc, c = POOL_SCORE.get(p, (0.0, 0.0))
        s = (sc + 1.0) / (c + 2.0)
        w.append((1.0 - s) ** power + 0.05)           # the 0.05 floor keeps every snapshot in play
    w = np.asarray(w)
    return w / w.sum()


def top_up(pending, pool, args, next_seed):
    """Keep ``queue_factor * workers`` self-play games outstanding (running, queued
    or finished-but-not-yet-used), so every worker stays fed while the learner trains."""
    snaps = (sorted(glob.glob(os.path.join("runs", args.name, "iter_*.pt")))
             if args.pool_frac > 0 else [])
    weights = _pool_weights(snaps, args.pool_power) if snaps else None
    while len(pending) < args.queue_factor * args.workers:
        seed = next_seed()
        pick = np.random.default_rng(seed + 1)               # picks opponent and colour
        u = pick.random()
        if getattr(args, "bots_now", False) and u < args.heur_frac:     # a rule-based bot
            bots = ["heur:nogift", "heur:gift"]
            pending.append(pool.submit(rollout.selfplay_task, seed,
                                       bots[int(pick.choice(2, p=_pool_weights(bots, args.pool_power)))],
                                       "b" if pick.random() < 0.5 else "w"))
        elif snaps and pick.random() < args.pool_frac:       # an older snapshot of the net
            pending.append(pool.submit(rollout.selfplay_task, seed,
                                       snaps[int(pick.choice(len(snaps), p=weights))],
                                       "b" if pick.random() < 0.5 else "w"))
        else:
            pending.append(pool.submit(rollout.selfplay_task, seed))


def run_eval(pool, args, next_seed):
    """Current network vs a random player, half the games as each colour."""
    futures = [pool.submit(rollout.eval_task, next_seed(), "b" if i % 2 == 0 else "w")
               for i in range(args.eval_games)]
    wait(futures)
    results = [f.result() for f in futures]
    wins = sum(r["winner"] == r["net_team"] for r in results)
    draws = sum(r["winner"] is None for r in results)
    losses = len(results) - wins - draws
    score = (wins + 0.5 * draws) / max(len(results), 1)
    return wins, draws, losses, score, results[0]["version"]


def _mb(b):
    return "n/a" if b is None else "%.0f MB" % (b / 2 ** 20)


def _gpu_line(tm, args, data, it):
    """What the learner kept on the device this iteration, and how much room is left."""
    res, peak, total = tm.get("gpu_resident"), tm.get("gpu_peak"), tm.get("gpu_total")
    parts = tm.get("gpu_parts") or {}
    groups = {"boards": parts.get("boards", 0), "actions": parts.get("comps", 0),
              "search targets": parts.get("pi", 0)}
    groups["other"] = sum(parts.values()) - sum(groups.values())
    line = ("data " + _c("1", _mb(sum(parts.values()) if parts else None))
            + (" (" + " \u00b7 ".join("%s %s" % (k, _mb(v)) for k, v in groups.items() if v) + ")" if parts else "")
            + "    allocated after upload " + _c("1", _mb(res))
            + "    peak " + _c("1", _mb(peak)))
    if peak is not None and total:
        line += " = %.1f%% of %s" % (100.0 * peak / total, _mb(total))
        # Per-step memory (activations) grows about linearly with --minibatch; the data does not.
        # Say how far the batch could grow while the peak stays under 60% of the device.
        step_mem = max(peak - (res or 0), 1)
        grow = (0.6 * total - (res or 0)) / step_mem
        if grow > 1.5 and (it % 50 == 0 or not _gpu_line.hinted):
            _gpu_line.hinted = True
            mb = int(grow * args.minibatch) // 64 * 64
            acts = -(-int(mb * float(data["counts"].mean()) * 2) // 4096) * 4096
            line += ("    room for ~%.0fx the batch: --minibatch %d --max-actions %d "
                     "(more decisions per step, fewer steps; mind --lr)" % (grow, mb, max(acts, args.max_actions)))
    print(_tag("gpu", 183) + line, flush=True)


_gpu_line.hinted = False


def main():
    args = parse_args()
    if hasattr(signal, "SIGUSR1"):          # kill -USR1 <pid> prints where every thread is, then carries on
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    device = pick_device(args.device)
    torch.set_num_threads(args.learner_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(run_dir, "latest.pt")
    log_path = os.path.join(run_dir, "log.csv")

    version = 0
    if args.resume:
        if not os.path.exists(latest):
            raise SystemExit("nothing to resume: %s does not exist" % latest)
        model, ckpt = netlib.load_checkpoint(latest)
        model = model.to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
        if ckpt.get("opt"):
            opt.load_state_dict(ckpt["opt"])
        version = int(ckpt["version"])
    else:
        if os.path.exists(latest):
            raise SystemExit("%s already exists: use --resume, or pick another --name" % latest)
        model = netlib.PolicyValueNet(args.channels, args.blocks, args.hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    netlib.save_checkpoint(latest, model, version, opt.state_dict())

    n_params = sum(p.numel() for p in model.parameters())
    if args.net_backend == "native":
        rustnet.require()                   # fail here, not inside a worker
        rustnet.RustNet.from_torch(model)   # and make sure this network's shapes are supported
    print("device=%s  workers=%d  learner threads=%d  params=%.2fM  start version=%d  net=%s"
          % (device, args.workers, args.learner_threads, n_params / 1e6, version,
             args.net_backend + (" (avx2)" if args.net_backend == "native" and rustnet.simd_active()
                                 else " (portable kernels)" if args.net_backend == "native" else "")),
          flush=True)

    seed_counter = [args.seed * 1_000_003 + version * 7919]

    def next_seed():
        seed_counter[0] += 1
        return seed_counter[0]

    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=rollout.init_worker,
        initargs=(latest, {"max_plies": args.max_plies}, args.move_cache,
                  {"frac": args.search_frac, "sims": args.search_sims, "noise": args.search_noise,
                   "temp": args.search_temp, "sweep": args.search_sweep,
                   "c_puct": args.search_cpuct, "smooth": args.search_smooth,
                   "batch": args.search_batch, "vl": args.search_vl},
                  args.net_backend))
    pending = []          # futures in submission order (oldest first)
    t_start = time.time()
    log_file = open(log_path, "a", newline="")
    writer = None
    try:
        for it in range(args.iters):
            # ---- collect self-play games --------------------------------
            t0 = time.time()
            args.bots_now = args.heur_frac > 0 and it % max(1, args.heur_every) == 0
            games, plies, stale = [], 0, 0
            while plies < args.plies_per_iter:
                top_up(pending, pool, args, next_seed)
                ready = [f for f in pending if f.done()]    # oldest first
                if not ready:
                    wait(pending, return_when=FIRST_COMPLETED)
                    continue
                for fut in ready:
                    pending.remove(fut)
                    result = fut.result()                    # re-raises worker errors
                    if version - result["version"] > args.max_stale:
                        stale += 1
                        continue
                    games.append(result)
                    plies += len(result["player"])
                    if plies >= args.plies_per_iter:
                        break                               # leave the rest for the next update
            # Refill *before* learning: the workers keep playing during the update, and
            # without this the queue is already partly drained when learning starts.
            top_up(pending, pool, args, next_seed)
            t_collect = time.time() - t0
            print("\n" + _tag("collect", 111) + "collected %d games / %d plies in %.0fs; learning on %s ..."
                  % (len(games), plies, t_collect, device), flush=True)

            # ---- learn ---------------------------------------------------
            t1 = time.time()
            data = assemble(games, args.draw_value, args.shape_coef, args.gamma, args.gae_lambda)
            model.train()
            timing = {}
            stats = ppo_update(model, opt, data, args, device, rng, timing)
            version += 1
            netlib.save_checkpoint(latest, model, version, opt.state_dict())
            if version % args.save_every == 0:
                netlib.save_checkpoint(os.path.join(run_dir, "iter_%05d.pt" % version),
                                       model, version, None)
            t_learn = time.time() - t1

            # ---- report ---------------------------------------------------
            n = len(games)
            winners = Counter(g["winner"] or "draw" for g in games if not g["opp"])
            vs_all = [g for g in games if g["opp"]]
            vs_past = [g for g in vs_all if not g["opp"].startswith("heur:")]     # older snapshots
            vs_bots = [g for g in vs_all if g["opp"].startswith("heur:")]         # rule-based bots
            avg_plies = sum(g["plies"] for g in games) / n
            reasons = Counter(g["reason"][:64] for g in games)
            ret_var = float(np.var(data["ret"]))
            expl_var = (1 - float(np.var(data["ret"] - data["value"])) / ret_var
                        if ret_var > 1e-8 else float("nan"))
            row = {
                "version": version, "games": n, "plies": plies, "stale": stale,
                "black": winners["b"], "white": winners["w"], "draw": winners["draw"],
                "avg_plies": round(avg_plies, 1),
                "avg_legal": round(float(data["counts"].mean()), 1),
                "plies_per_s": round(plies / max(t_collect, 1e-9), 1),
                **{k: round(v, 5) for k, v in stats.items() if k != "ce"},
                "expl_var": round(expl_var, 3),
                "collect_s": round(t_collect, 1), "learn_s": round(t_learn, 1),
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }
            draw_frac = winners["draw"] / max(sum(winners.values()), 1)
            print(_tag("v%d" % version, 45)
                  + _c("1", "%d games, %d plies" % (n, plies))
                  + "  (avg %.0f plies, %.0f legal)    " % (avg_plies, data["counts"].mean())
                  + _c(BLACK_TEAM, "black %d" % winners["b"]) + "   "
                  + _c(WHITE_TEAM, "white %d" % winners["w"]) + "   "
                  + _health("draw %d" % winners["draw"], draw_frac, 0.10, 0.25)
                  + ((_c(WATCH, "    [%d stale dropped]" % stale)) if stale else ""),
                  flush=True)
            print(_tag("train", 114) + "    ".join([
                "pg " + _c("1", "%+.3f" % stats["pg"]),
                "vf " + _c("1", "%.3f" % stats["vf"]),
                "ent " + _health("%.2f" % stats["ent"], stats["ent"], 0.3, 0.1, True),
                "kl " + _health("%.4f" % stats["kl"], stats["kl"], 0.03, 0.045),
                "clip " + _health("%.2f" % stats["clipfrac"], stats["clipfrac"], 0.25, 0.4),
                "ev " + _health("%.2f" % expl_var, expl_var, 0.5, 0.2, True)]), flush=True)
            if data["searched"].any():
                n_s = int(data["searched"].sum())
                s_sims = sum(g.get("search_sims", 0) for g in games)
                s_sec = sum(g.get("search_seconds", 0.0) for g in games)
                w_sec = sum(g["seconds"] for g in games)
                print(_tag("search", 183)
                      + "searched " + _c("1", "%.0f%%" % (100.0 * n_s / len(data["counts"])))
                      + " of decisions (%d)    ce " % n_s + _c("1", "%.3f" % stats["ce"])
                      + "    %.0f sims, %.0f ms per searched move    " % (s_sims / n_s, 1000 * s_sec / n_s)
                      + "search = " + _share("%.0f%%" % (100 * s_sec / max(w_sec, 1e-9)), s_sec, w_sec)
                      + " of worker game time", flush=True)
                # where the search's own time goes: copying games, network calls, everything else
                # (root evaluation, selection / backup, freeing the tree)
                s_cl = sum(g.get("search_clone", 0.0) for g in games)
                s_nt = sum(g.get("search_net", 0.0) for g in games)
                s_ot = max(s_sec - s_cl - s_nt, 0.0)
                print(_tag("search", 183) + "search time:  game copies "
                      + _c("1", "%.0f%%" % (100 * s_cl / max(s_sec, 1e-9)))
                      + "    network (new nodes) " + _c("1", "%.0f%%" % (100 * s_nt / max(s_sec, 1e-9)))
                      + "    other " + _c("1", "%.0f%%" % (100 * s_ot / max(s_sec, 1e-9)))
                      + "    (%.1f ms per simulation)" % (1000 * s_sec / max(s_sims, 1)), flush=True)
            # the longer of the two is what limits the cycle; collect ~0 s means games were already waiting
            print(_tag("time", 111)
                  + "collect " + _c(WATCH if t_collect > t_learn else "1", "%.0fs" % t_collect)
                  + ("" if t_collect < 1 else " (%.0f plies/s)" % (plies / t_collect))
                  + "    learn " + _c(WATCH if t_learn >= t_collect else "1", "%.0fs" % t_learn)
                  + " (%d steps, %.2f of %d passes over the batch)"
                  % (timing["steps"], timing["decisions"] / max(len(data["counts"]), 1), args.epochs)
                  + (_c(WATCH, "  KL gate stopped it early") if timing["kl_stop"] else "")
                  + ("   games were already waiting" if t_collect < 1 else ""), flush=True)
            print(_tag("learner", 183)
                  + "collate " + _share("%.1fs" % timing["collate"], timing["collate"], t_learn)
                  + " (build %.1fs + to-device " % timing["collate_np"]
                  + _share("%.1fs" % timing["collate_xfer"], timing["collate_xfer"], t_learn) + ")"
                  + "    dispatch " + _share("%.1fs" % timing["dispatch"], timing["dispatch"], t_learn)
                  + "    blocked on %s " % device + _share("%.1fs" % timing["wait"], timing["wait"], t_learn)
                  + "    other %.1fs" % (t_learn - timing["collate"] - timing["dispatch"] - timing["wait"]),
                  flush=True)
            _gpu_line(timing, args, data, it)
            sd = timing["step_dispatch"]
            if len(sd) >= 4:
                # Is the dispatch cost per step, or paid once per new shape / once per iteration?
                new = [d for d, f in zip(sd, timing["new_shape"]) if f]
                old = [d for d, f in zip(sd, timing["new_shape"]) if not f]
                med = float(np.median(old)) if old else float("nan")
                print(_tag("steps", 183)
                      + "first 3: " + " ".join(_c(WATCH if d > 3 * med else "1", "%.2f" % d) for d in sd[:3]) + " s"
                      + "    typical " + _c("1", "%.3f s" % med)
                      + "    new shapes: %d, cost " % len(new) + _share("%.1fs" % sum(new), sum(new), t_learn)
                      + "    slowest %.2f s" % max(sd), flush=True)
            print(_tag("ended", 215) + _ended_line(reasons), flush=True)
            for g in vs_all:                          # remember how the current net does against each opponent
                s_ = 1.0 if g["winner"] == g["net_team"] else 0.5 if g["winner"] is None else 0.0
                sc, c = POOL_SCORE.get(g["opp"], (0.0, 0.0))
                POOL_SCORE[g["opp"]] = (sc * 0.97 + s_, c * 0.97 + 1.0)       # recent games count more
            if vs_bots:
                parts = []
                for mode in ("heur:nogift", "heur:gift"):
                    gs = [g for g in vs_bots if g["opp"] == mode]
                    if not gs:
                        continue
                    bw = sum(g["winner"] == g["net_team"] for g in gs)
                    bd = sum(g["winner"] is None for g in gs)
                    sc, c = POOL_SCORE[mode]
                    parts.append("%s: " % mode[5:] + _c(GOOD, "won %d" % bw) + " " + _c(WATCH, "drew %d" % bd)
                                 + " " + _c(BAD if len(gs) - bw - bd else "1", "lost %d" % (len(gs) - bw - bd))
                                 + " (running score " + _health("%.2f" % (sc / max(c, 1e-9)), sc / max(c, 1e-9),
                                                                0.9, 0.75, True) + ")")
                print(_tag("vs bots", 150) + "    ".join(parts), flush=True)
            if vs_past:
                pw = sum(g["winner"] == g["net_team"] for g in vs_past)
                pd = sum(g["winner"] is None for g in vs_past)
                hard = sorted(((sc + 1) / (c + 2), p) for p, (sc, c) in POOL_SCORE.items()
                              if c >= 3 and not p.startswith("heur:"))[:3]
                print(_tag("vs old", 80)
                      + "against %d older snapshots: " % len(vs_past)
                      + _c(GOOD, "won %d" % pw) + "  " + _c(WATCH, "drew %d" % pd) + "  "
                      + _c(BAD if len(vs_past) - pw - pd else "1", "lost %d" % (len(vs_past) - pw - pd))
                      + "    toughest: " + "  ".join(
                          "v%d " % int(os.path.basename(p)[5:10]) + _health("%.2f" % sc_, sc_, 0.75, 0.55, True)
                          for sc_, p in hard), flush=True)
            # ev split by how far through the game the decision was: if the value head works,
            # the last bucket (outcome nearly decided) should be much higher than the first.
            err = data["ret"] - data["value"]
            parts = []
            for lo, hi, label in ((0, .33, "first third"), (.33, .66, "middle"),
                                  (.66, .9, "late"), (.9, 1.01, "last 10%")):
                m = (data["phase"] >= lo) & (data["phase"] < hi)
                v = float(np.var(data["ret"][m])) if m.any() else 0.0
                e = 1 - float(np.var(err[m])) / v if v > 1e-8 else float("nan")
                parts.append("%s %s" % (label, _health("%.2f" % e, e, 0.5, 0.2, True)))
            print(_tag("phase", 218) + "ev by game phase:  " + "    ".join(parts), flush=True)

            if args.eval_every and version % args.eval_every == 0:
                w, d, l, score, ev = run_eval(pool, args, next_seed)
                print(_tag("EVAL", 207) + _c("1", "v%d vs random:" % ev) + " "
                      + _c(GOOD, "won %d" % w) + "  " + _c(WATCH, "drew %d" % d) + "  "
                      + _c(BAD if l else "1", "lost %d" % l) + "    score "
                      + _health("%.3f" % score, score, 0.9, 0.7, True), flush=True)
                row.update(eval_score=round(score, 3), eval_w=w, eval_d=d, eval_l=l)

            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row) + (
                    [] if "eval_score" in row else ["eval_score", "eval_w", "eval_d", "eval_l"]),
                    extrasaction="ignore", restval="")
                if log_file.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            log_file.flush()
    except KeyboardInterrupt:
        print("\ninterrupted; the last checkpoint is runs/%s/latest.pt "
              "(continue with --resume)" % args.name, flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        log_file.close()


if __name__ == "__main__":
    main()
