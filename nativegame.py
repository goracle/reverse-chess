#!/usr/bin/env python
"""A whole self-play game in Rust (native/src/play.rs), driven from rollout.py.

``rollout.selfplay_task`` used to play every ply through ``engine.Game``: an ``Action`` object
for every legal move, a Python ``encode_observation`` with a loop over all pieces, numpy softmax
and sampling, the books in a ``_Recorder``.  Here one ``NativeGame`` plays the game and keeps
the record natively; Python is only called back for the decisions that need it:

* a **searched** decision (``--search-frac``): the tree search runs natively too
  (native/src/mcts.rs, ``rc_game_search``): PUCT with first-play urgency, exact terminals, virtual
  loss batches, the network evaluated in Rust.  Python draws the root Dirichlet noise and the move
  (the same draws, in the same order, as ``net.search_action``), then the decision is recorded and
  played natively (``record_apply``).  The root sweep (``sweep`` != 0), or a library without
  ``rc_game_search``, or ``RC_NATIVESEARCH=0``, hands the position over as a ``FastGame`` to the
  Python ``mcts.Searcher`` / ``net.search_action`` instead, exactly as before;
* an opponent that is a **rule-based bot** is not supported here (heuristic.py needs a
  ``Game``); ``usable`` says no and rollout.py takes the old path.

The result dict is the one ``rollout._Recorder.finish`` builds (same keys, dtypes, shapes).

The random draws are made here and passed in (``u`` for sampling the moves, ``su`` for "is this
decision searched"), so a game is a pure function of its inputs: ``check_nativegame.py`` replays
the same draws through the old Python loop and compares everything.  The stream is NOT the one
the old loop consumed from ``rng`` (it drew one number per decision as it went), so a given seed
plays a different game than before; the distribution is the same.

Needs the native library built from this tree (``./build_native.sh``); ``AVAILABLE`` is False
without it, and ``RC_NATIVEGAME=0`` switches it off.
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Dict, Optional

import numpy as np

import engine
import fastgame
import rustnet
from engine import Game, Rules, full_name

_LIB = engine._LIB
BLOCK = 256                    # draws handed to the library per call (it asks again when used up)
_ACT_CAP = 4096                # legal actions the position buffer holds (grown on demand)

# rc_game_info fields
(_DONE, _KIND, _WINNER, _REASON, _TURN, _PLY, _QUIET, _REP, _EXTRA, _NACT, _NDEC, _NROWS,
 _NSEEN) = range(13)
# how a game ended (play.rs K_*)
_K_WIN, _K_DRAW_BOTH, _K_DRAW_QUIET, _K_REPEAT, _K_MAXPLY, _K_NO_MOVES, _K_BLOCKED = range(1, 8)


def _bind():
    if _LIB is None:
        return None
    try:
        if _LIB.rc_game_abi() != 1:
            return None
    except AttributeError:
        return None
    vp, i32, u32 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32
    _LIB.rc_game_new.restype = vp
    _LIB.rc_game_new.argtypes = [ctypes.c_char_p, u32, i32, i32, i32, i32, i32, i32, i32,
                                 ctypes.c_char_p, u32, vp, vp, u32]
    _LIB.rc_game_free.restype = None
    _LIB.rc_game_free.argtypes = [vp]
    _LIB.rc_game_run.restype = i32
    _LIB.rc_game_run.argtypes = [vp, vp, vp, u32, vp, vp, u32, ctypes.c_double]
    _LIB.rc_game_apply.restype = i32
    _LIB.rc_game_apply.argtypes = [vp, u32]
    _LIB.rc_game_record_apply.restype = i32
    _LIB.rc_game_record_apply.argtypes = [vp, vp, vp, vp, u32, u32, ctypes.c_float, ctypes.c_float]
    _LIB.rc_game_info.restype = i32
    _LIB.rc_game_info.argtypes = [vp, vp]
    _LIB.rc_game_position.restype = i32
    _LIB.rc_game_position.argtypes = [vp, vp, vp, vp, u32]
    _LIB.rc_game_seen.restype = i32
    _LIB.rc_game_seen.argtypes = [vp, vp, vp, u32]
    _LIB.rc_game_export.restype = i32
    _LIB.rc_game_export.argtypes = [vp] * 10
    if hasattr(_LIB, "rc_game_search"):
        _LIB.rc_game_search.restype = i32
        _LIB.rc_game_search.argtypes = [vp, vp, ctypes.c_double, vp, u32, u32, ctypes.c_double,
                                        ctypes.c_double, u32, vp, vp, vp, vp, vp, vp]
    return _LIB


_NATIVE = _bind() if fastgame.AVAILABLE and rustnet.AVAILABLE else None
AVAILABLE = _NATIVE is not None
SEARCH_AVAILABLE = AVAILABLE and hasattr(_LIB, "rc_game_search")


def search_enabled() -> bool:
    """Searched decisions run natively when the library has ``rc_game_search``, unless
    RC_NATIVESEARCH=0."""
    if os.environ.get("RC_NATIVESEARCH", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    return SEARCH_AVAILABLE


def enabled() -> bool:
    """Native games are on when the library has them, unless RC_NATIVEGAME=0."""
    if os.environ.get("RC_NATIVEGAME", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    return AVAILABLE


def usable(model, opp) -> bool:
    """Can a game between these networks be played natively?  ``opp`` is the opponent network
    (None = the net plays itself); both must be ``RustNet`` (rollout's "native" backend)."""
    return (enabled() and isinstance(model, rustnet.RustNet)
            and (opp is None or isinstance(opp, rustnet.RustNet)))


_START: Dict[tuple, tuple] = {}


def _start(rules: Rules):
    """(config, packed start position, its legal action rows), cached per rules."""
    key = (rules.win_requires_all_pieces_on_board, rules.fifty_move_limit,
           rules.repetition_limit, rules.max_plies)
    hit = _START.get(key)
    if hit is None:
        game = Game(rules)
        fg = fastgame.FastGame.from_game(game)
        hit = _START[key] = (fg.cfg, fg.state, fg.raw)
        game.close()
    return hit


def _last_error() -> str:
    return rustnet._last_error()


class NativeGame:
    """One game, held by the native library.  Use as a context manager (or call ``close``)."""

    def __init__(self, rules: Rules):
        if not AVAILABLE:
            raise RuntimeError("native game support missing (rebuild with ./build_native.sh)")
        cfg, state, raw = _start(rules)
        self.cfg = cfg
        self._h = _NATIVE.rc_game_new(state, 0, 0, 0, cfg.fifty, cfg.max_plies, cfg.flags,
                                      cfg.rep_limit, 0, raw, len(raw) // 6, None, None, 0)
        if not self._h:
            raise RuntimeError("rc_game_new failed")
        self._info_buf = (ctypes.c_int32 * 16)()
        self._act_buf = ctypes.create_string_buffer(6 * _ACT_CAP)

    def close(self):
        h, self._h = getattr(self, "_h", None), None
        if h and _NATIVE is not None:
            _NATIVE.rc_game_free(h)

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- state -------------------------------------------------------------
    def info(self) -> list:
        rc = _NATIVE.rc_game_info(self._h, self._info_buf)
        if rc != 0:
            raise RuntimeError("rc_game_info failed (%d)" % rc)
        return list(self._info_buf)

    def position(self) -> "fastgame.FastGame":
        """The live position as a ``FastGame`` (what ``mcts`` and the networks read)."""
        state = ctypes.create_string_buffer(128)
        key = ctypes.create_string_buffer(65)
        buf = self._act_buf
        n = _NATIVE.rc_game_position(self._h, state, key, buf, _ACT_CAP)
        if n > _ACT_CAP:
            buf = ctypes.create_string_buffer(6 * n)
            n = _NATIVE.rc_game_position(self._h, state, key, buf, n)
        if n < 0:
            raise RuntimeError("rc_game_position failed (%d)" % n)
        m = _NATIVE.rc_game_seen(self._h, None, None, 0)
        keys = ctypes.create_string_buffer(65 * max(m, 1))
        counts = (ctypes.c_uint32 * max(m, 1))()
        _NATIVE.rc_game_seen(self._h, keys, counts, m)
        raw_keys = keys.raw
        seen = {raw_keys[65 * i:65 * i + 65]: counts[i] for i in range(m)}
        info = self.info()
        return fastgame.FastGame.from_native(
            self.cfg, state.raw, buf.raw[:6 * n], key.raw, "b" if info[_TURN] == 0 else "w",
            info[_PLY], info[_QUIET], info[_REP], bool(info[_EXTRA]), seen)

    # -- playing -----------------------------------------------------------
    def run(self, mine, opp, record_mask: int, u: np.ndarray, su: np.ndarray, frac: float) -> int:
        """Play natively until the game ends (0), a recorded decision is to be searched (1), the
        draws run out (2) or an unrecorded side with no network is to move (3)."""
        rc = _NATIVE.rc_game_run(self._h, mine._h, opp._h if opp is not None else None, record_mask,
                                 u.ctypes.data, su.ctypes.data, len(u), frac)
        if rc < 0:
            raise RuntimeError("rc_game_run failed (%d): %s" % (rc, _last_error()))
        return rc

    def apply(self, idx: int) -> None:
        rc = _NATIVE.rc_game_apply(self._h, idx)
        if rc != 0:
            raise RuntimeError("rc_game_apply failed (%d)" % rc)

    def record_apply(self, sample) -> None:
        """Record a decision made in Python and play it.  ``sample`` is what
        ``net.choose_action`` / ``net.search_action`` return: ``(board uint8 (18,8,8), scalars
        f32 (17,), comps uint8 (n,6), chosen index, log-prob, value)``."""
        board, scalars, comps, idx, logp, value = sample
        board = np.ascontiguousarray(board, dtype=np.uint8)
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        comps = np.ascontiguousarray(comps, dtype=np.uint8)
        rc = _NATIVE.rc_game_record_apply(self._h, board.ctypes.data, scalars.ctypes.data,
                                          comps.ctypes.data, len(comps), int(idx), float(logp),
                                          float(value))
        if rc != 0:
            raise RuntimeError("rc_game_record_apply failed (%d)" % rc)

    def search(self, net, sims: int, noise: float = 0.0, dirichlet: Optional[np.ndarray] = None,
               batch: int = 1, c_puct: float = 1.5, vl: float = 1.0) -> dict:
        """A native tree search of the live position (nothing is played or recorded).

        ``dirichlet`` (one draw per legal action) is mixed into the root priors with weight
        ``noise``.  Returns ``visits`` (n,) int64, ``probs`` (n,) float64 (the network's own
        policy), ``board`` / ``scalars`` / ``comps`` (the observation, for ``record_apply``),
        ``value`` (the network's value of the root), ``sims`` (simulations run), ``win_idx`` (a
        move that wins on the spot, -1 for none), ``t_eval`` / ``t_step`` (seconds in the network /
        stepping positions) and ``evals`` (positions the network evaluated)."""
        if not SEARCH_AVAILABLE:
            raise RuntimeError("native search missing (rebuild with ./build_native.sh)")
        n = self.info()[_NACT]
        if n <= 0:
            raise RuntimeError("no legal action to search")
        if dirichlet is not None:
            dirichlet = np.ascontiguousarray(dirichlet, dtype=np.float64)
            if dirichlet.shape != (n,):
                raise ValueError("dirichlet needs one value per legal action (%d)" % n)
        visits = np.zeros(n, dtype=np.int64)
        probs = np.zeros(n, dtype=np.float64)
        board = np.zeros((18, 8, 8), dtype=np.uint8)
        scal = np.zeros(17, dtype=np.float32)
        comps = np.zeros((n, 6), dtype=np.uint8)
        info = np.zeros(8, dtype=np.float64)
        rc = _NATIVE.rc_game_search(
            self._h, net._h, float(noise), dirichlet.ctypes.data if dirichlet is not None else None,
            int(sims), max(1, int(batch)), float(c_puct), float(vl), n, visits.ctypes.data,
            probs.ctypes.data, board.ctypes.data, scal.ctypes.data, comps.ctypes.data,
            info.ctypes.data)
        if rc < 0:
            raise RuntimeError("rc_game_search failed (%d): %s" % (rc, _last_error()))
        return {"visits": visits, "probs": probs, "board": board, "scalars": scal, "comps": comps,
                "value": float(info[0]), "sims": int(info[1]), "win_idx": int(info[2]),
                "t_eval": float(info[3]), "t_step": float(info[4]), "evals": int(info[5])}

    # -- results -----------------------------------------------------------
    def export(self) -> dict:
        info = self.info()
        t, rows = info[_NDEC], info[_NROWS]
        out = {
            "boards": np.empty((t, 18, 8, 8), dtype=np.uint8),
            "scalars": np.empty((t, 17), dtype=np.float32),
            "comps": np.empty((rows, 6), dtype=np.uint8),
            "counts": np.empty(t, dtype=np.int32),
            "chosen": np.empty(t, dtype=np.int32),
            "logp": np.empty(t, dtype=np.float32),
            "value": np.empty(t, dtype=np.float32),
            "phi": np.empty(t, dtype=np.float32),
            "player": np.empty(t, dtype=np.uint8),
        }
        order = ("boards", "scalars", "comps", "counts", "chosen", "logp", "value", "phi", "player")
        rc = _NATIVE.rc_game_export(self._h, *[out[k].ctypes.data for k in order])
        if rc != 0:
            raise RuntimeError("rc_game_export failed (%d)" % rc)
        return out

    def outcome(self):
        """``(winner 'b'/'w'/None, end_reason text, plies)`` in the words of ``Game``."""
        info = self.info()
        kind, w = info[_KIND], info[_WINNER]
        winner = None if w < 0 else "bw"[w]
        if kind == _K_WIN:
            text = fastgame.FastGame._win_text(w, info[_REASON])
        elif kind == _K_DRAW_BOTH:
            text = "draw: both teams satisfy a winning condition"
        elif kind == _K_DRAW_QUIET:
            text = "draw: no gift or pawn move in %d plies" % info[_QUIET]
        elif kind == _K_REPEAT:
            text = "draw: position repeated %d times" % info[_REP]
        elif kind == _K_MAXPLY:
            text = "draw: move limit reached"
        elif kind == _K_NO_MOVES:
            text = "%s has no legal moves" % full_name("b" if info[_TURN] == 0 else "w")
        elif kind == _K_BLOCKED:
            text = "draw: every legal move leaves the opponent without a legal move"
        else:
            raise RuntimeError("the game is not over")
        return winner, text, info[_PLY]


def _search_native(game: "NativeGame", model, n: int, rng: np.random.Generator, cfg: dict):
    """``net.search_action`` with the tree search in Rust: the same root noise, the same move
    distribution, the same draw (one Dirichlet vector, then one uniform), the same sample.
    Returns ``(sample, pi, simulations run, seconds stepping, seconds in the network)``."""
    import mcts
    noise = float(cfg.get("noise", 0.25))
    temp = cfg.get("temp", 1.0)
    smooth = cfg.get("smooth", 0.0)
    dirichlet = None
    if noise > 0 and n > 1:                  # mcts.Searcher._run
        alpha = min(1.0, max(0.03, 10.0 / n))
        dirichlet = rng.dirichlet([alpha] * n)
    r = game.search(model, cfg["sims"], noise, dirichlet, cfg.get("batch", 1),
                    cfg.get("c_puct", mcts.C_PUCT), cfg.get("vl", mcts.VL_DEFAULT))
    probs = r["probs"]
    if r["win_idx"] >= 0:
        pi = np.zeros(n)
        pi[r["win_idx"]] = 1.0
    else:
        visits = r["visits"].astype(np.float64)
        if visits.sum() <= 0:                # no simulations ran: trust the policy
            pi = probs.copy()
        elif temp <= 1e-3:
            pi = (visits == visits.max()).astype(np.float64)
        else:
            pi = visits ** (1.0 / temp)
        pi /= pi.sum()
    cdf = np.cumsum(pi)
    idx = min(int(cdf.searchsorted(rng.random() * cdf[-1], side="right")), n - 1)
    if smooth > 0 and r["win_idx"] < 0:      # a proven win stays sharp
        pi = (1.0 - smooth) * pi + smooth * probs
    sample = (r["board"], r["scalars"], r["comps"], idx,
              float(np.log(max(probs[idx], 1e-30))), r["value"])
    return sample, pi.astype(np.float32), r["sims"], r["t_step"], r["t_eval"]


def play(model, opp, rules: Rules, rng: np.random.Generator, record_mask: int, version: int,
         search: Optional[dict] = None, draws: Optional[tuple] = None) -> dict:
    """Play one game natively and return what ``rollout._Recorder.finish`` returns.

    ``model`` records the sides in ``record_mask`` (bit 0 black, bit 1 white; 3 = self-play);
    ``opp`` (a ``RustNet`` or None) plays the others, unrecorded.  ``search`` is rollout's
    search config (``frac``, ``sims``, ``noise``, ``temp``, ``sweep``, ``smooth``, ``c_puct``,
    ``batch``, ``vl``) or None.  ``draws=(u, su)`` replaces the random draws (testing)."""
    import net as netlib
    t_start = time.time()
    searcher = evaluator = None
    frac = 0.0
    pis: Dict[int, np.ndarray] = {}
    search_sims, search_seconds = 0, 0.0
    t_step = t_eval = 0.0
    native_search = False
    if search is not None and search.get("frac", 0) > 0:
        import mcts
        frac = float(search["frac"])
        # the native search has no root sweep (a time-boxed exact check of the root moves)
        native_search = search_enabled() and not search.get("sweep", 0)
        if not native_search:
            evaluator = netlib.SearchEvaluator(model)
            searcher = mcts.Searcher(evaluator, sims=search["sims"],
                                     c_puct=search.get("c_puct", mcts.C_PUCT),
                                     batch=search.get("batch", 1),
                                     vl=search.get("vl", mcts.VL_DEFAULT))
    with NativeGame(rules) as game:
        while True:
            if draws is not None:
                u, su = draws
            else:
                d = rng.random(2 * BLOCK)
                u, su = d[:BLOCK], d[BLOCK:]
            u = np.ascontiguousarray(u, dtype=np.float64)
            su = np.ascontiguousarray(su, dtype=np.float64)
            rc = game.run(model, opp, record_mask, u, su, frac)
            if rc == 0:
                break
            if rc == 1:                      # a decision to search
                info = game.info()
                n_dec = info[_NDEC]
                t_s = time.time()
                if native_search:
                    sample, pi, sims, ts, te = _search_native(game, model, info[_NACT], rng, search)
                    t_step += ts
                    t_eval += te
                else:                        # the Python searcher, on a FastGame
                    _, sample, pi, sims = netlib.search_action(
                        evaluator, searcher, game.position(), rng, search["sims"],
                        search.get("noise", 0.25), search.get("temp", 1.0),
                        search.get("sweep", 0), search.get("smooth", 0.0))
                search_sims += sims
                search_seconds += time.time() - t_s
                pis[n_dec] = pi
                game.record_apply(sample)
            elif rc == 2:
                if draws is not None:
                    raise RuntimeError("the supplied draws ran out before the game ended")
            else:
                raise RuntimeError("an unrecorded side has no network (bots are not played natively)")
        rec = game.export()
        winner, reason, plies = game.outcome()
    t = len(rec["counts"])
    extra = {}
    if pis:
        searched = np.zeros(t, dtype=np.uint8)
        searched[list(pis)] = 1
        extra = {
            "searched": searched,
            "pi": np.concatenate([pis[i] if i in pis else np.zeros(int(c), np.float32)
                                  for i, c in enumerate(rec["counts"])]),
            "search_sims": search_sims,
            "search_seconds": search_seconds,
            "search_clone": t_step if native_search else searcher.t_clone,
            "search_net": t_eval if native_search else searcher.t_eval,
        }
    return {
        **extra,
        **rec,
        "winner": winner,
        "reason": reason,
        "plies": plies,
        "version": version,
        "seconds": time.time() - t_start,
    }
