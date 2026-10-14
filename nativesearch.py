#!/usr/bin/env python
"""The tree search of mcts.py, run natively (native/src/mcts.rs, ``rc_search_run``), for play.py.

``mcts.Searcher`` keeps its whole interface -- ``start`` / ``search`` / ``Job`` with ``best()``,
``top()``, ``value()``, ``sims``, ``solved`` -- but when it can, the search thread hands the
position to Rust instead of looping in Python.  What changes is speed: the leaves are evaluated
by the Rust network (``rustnet.RustNet``, built from the Advisor's weights) and the tree is
stepped, selected and backed up natively, so a search gets many times more simulations in the
same think time.  The result is the search ``mcts.py`` would have run: the same PUCT, first-play
urgency, exact terminals, virtual loss, and the exact root sweep.

While it runs, Rust writes the root's visit counts, values and simulation count into the numpy
arrays of ``NativeRoot`` after every round, so the UI polls a live ``Job`` as before.  The call
blocks the search thread only; ctypes drops the GIL, so the window stays responsive.
``Job.stop()`` raises a flag Rust polls between rounds.

Used when all of these hold (otherwise the Python search runs, unchanged):

* the library has ``rc_search_run`` (``./build_native.sh``) and ``RC_NATIVESEARCH`` is not 0;
* the searcher is in light mode (``FastGame`` positions) and has not been built with
  ``native=False``;
* the advisor is a ``numpy_net.Advisor`` whose weights ``rustnet.RustNet`` accepts;
* ``sweep_k`` is None (every root move) or 0 (none).
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import numpy as np

import engine
import mcts
import rustnet

_LIB = engine._LIB


def _bind():
    if _LIB is None or not rustnet.AVAILABLE:
        return None
    try:
        if _LIB.rc_search_abi() != 1:
            return None
    except AttributeError:
        return None
    vp, i32, u32, dbl = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_double
    cp = ctypes.c_char_p
    _LIB.rc_search_run.restype = i32
    _LIB.rc_search_run.argtypes = [
        cp, u32, i32, i32, i32, i32, i32, i32, i32, i32, u32,      # state .. rep
        cp, u32, cp, vp, u32,                                      # actions, seen table
        vp, dbl, vp,                                               # net, root value, priors
        u32, u32, dbl, dbl, dbl, i32, dbl,                         # sims batch c_puct vl ms sweep share
        vp, vp, vp, vp]                                            # stop, visits, w, info
    return _LIB


_NATIVE = _bind()
AVAILABLE = _NATIVE is not None


def default() -> bool:
    """On when the library has it, unless RC_NATIVESEARCH=0."""
    if os.environ.get("RC_NATIVESEARCH", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    return AVAILABLE


def rust_net_for(advisor) -> Optional["rustnet.RustNet"]:
    """The Rust copy of a ``numpy_net.Advisor``'s network (built once, kept on the advisor);
    None for any other kind of advisor or weights the Rust net does not take."""
    cached = getattr(advisor, "_rust_net", False)
    if cached is not False:
        return cached
    net = None
    weights = getattr(getattr(advisor, "net", None), "w", None)
    if weights is not None:
        try:
            net = rustnet.RustNet(weights, getattr(advisor, "version", None))
        except Exception:                      # odd shapes, no library: the Python search it is
            net = None
    try:
        advisor._rust_net = net
    except AttributeError:
        pass
    return net


class NativeRoot(mcts.Node):
    """A search root whose simulation count and winning move live in the ``info`` array Rust
    writes (as do ``N`` and ``W``), so ``Job`` reads them live."""

    __slots__ = ("_info",)

    def __init__(self, info: np.ndarray, node: mcts.Node):
        self._info = info
        info[0], info[1] = 0, -1
        mcts.Node.__init__(self, node.game, node.turn, node.terminal, node.value, node.actions,
                           node.priors)

    @property
    def total(self) -> int:
        return int(self._info[0])

    @total.setter
    def total(self, v: int) -> None:
        self._info[0] = v

    @property
    def win_idx(self) -> Optional[int]:
        w = int(self._info[1])
        return None if w < 0 else w

    @win_idx.setter
    def win_idx(self, v: Optional[int]) -> None:
        self._info[1] = -1 if v is None else v


def eligible(searcher, job, root) -> bool:
    return (searcher.native and AVAILABLE and searcher.light and not root.terminal
            and job.sweep_k in (None, 0) and len(root.actions) > 0
            and rust_net_for(searcher.advisor) is not None)


def make_root(job, root: mcts.Node) -> NativeRoot:
    return NativeRoot(np.zeros(8, dtype=np.float64), root)


def run(searcher, job, root: NativeRoot, t0: float, end: float) -> int:
    """Run the search in Rust (blocking).  0 when it ran; a negative code when Rust refused
    before doing anything (the caller then searches in Python)."""
    g = job.game
    net = rust_net_for(searcher.advisor)
    n = len(root.actions)
    if 6 * n != len(g.raw):
        return -4
    base = g._base
    keys = b"".join(base.keys())
    counts = np.fromiter(base.values(), dtype=np.uint32, count=len(base))
    priors = np.ascontiguousarray(root.priors, dtype=np.float64)
    cfg = g.cfg
    remaining_ms = max(1.0, (end - time.monotonic()) * 1000.0)
    obs_cap = int(getattr(searcher.advisor, "max_plies", 0) or 0)
    rc = _NATIVE.rc_search_run(
        g.state, 0 if g.turn == "b" else 1, g.ply, g.quiet_plies, cfg.fifty, cfg.max_plies,
        cfg.flags, cfg.rep_limit, obs_cap, int(g.extra_turn), int(g.rep),
        g.raw, n, keys, counts.ctypes.data, len(base),
        net._h, float(root.value), priors.ctypes.data,
        int(job.max_sims), searcher.batch, float(searcher.c_puct), float(searcher.vl),
        remaining_ms, 1 if job.sweep_k is None else 0, mcts.SWEEP_SHARE,
        job.stop_flag.ctypes.data, root.N.ctypes.data, root.W.ctypes.data,
        root._info.ctypes.data)
    info = root._info
    job.sweep_done, job.sweep_total = int(info[2]), int(info[3])
    searcher.t_eval += float(info[4])
    searcher.t_clone += float(info[5])
    return rc
