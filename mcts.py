#!/usr/bin/env python
"""Monte-Carlo tree search (PUCT) on top of the policy/value network, for play.py.

The network alone has two weaknesses that show up in play:

* its value head is jumpy (the eval bar flips from "white is 99% winning" to "black is
  winning" after one move), and
* its policy head does not know the "go home" race: a move that wins on the spot can have
  a tiny prior.

Search fixes both.  Values are averaged over many lines of play, and the game engine is
asked the exact question ("does this move end the game?") instead of the network:

* **Root sweep** - before the tree search proper, every legal root move is played on a
  copy of the game, best-prior first, and checked for an immediate result.  A move that
  wins is taken at once; a move that loses (or draws) enters the tree with its exact value.
* **Exact terminals** - a position where the game is over is scored +1 / -1 / 0 for the side
  to move, never by the network.  A node with a winning move available is *solved* and
  search stops descending anywhere else.
* **PUCT** - the usual AlphaZero selection, with the network's policy as the prior and its
  value as the leaf evaluation.  Reverse en passant gives the same player a second turn,
  so values are flipped between parent and child only when the side to move changes.

All of it runs in a background thread (``Searcher.start``) so the pygame window stays
responsive; the UI polls the ``Job`` for the best move so far.  Nothing here imports
pygame.  The game handed to ``start`` is copied, never touched afterwards.

Memory: ``Game.clone()`` copies are only freed once ``Game.close()`` is called on them (the
Game <-> Piece reference cycle runs through a numpy object array the garbage collector cannot
see).  This module closes every copy it makes: terminal and swept ones at once, tree nodes
when the search ends (``_release``).  After a search the tree keeps only its numbers
(visits, values), which is all ``Job`` reads.

Training (see ``net.search_action`` and rollout.py) uses the same search with root noise and
no root sweep: ``Searcher.search(game, sims=N, root_analysis=..., noise=0.25, sweep_k=0)``.

Light mode: when the native library has ``rc_step`` (see fastgame.py) the tree holds
``FastGame`` positions instead of ``Game`` copies.  A simulation then steps with one native
call (``FastGame.child(i)``) instead of ``fast_clone`` + ``Game.step``, which is most of the
non-network time of a simulation.  The search itself is unchanged and picks the same moves
(check_fastgame.py runs both and compares the trees); ``Searcher(light=False)`` or the
environment variable ``RC_FASTGAME=0`` switches it off.
"""
from __future__ import annotations

import copy
import math
import os
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

import fastgame
from engine import Action, Game

C_PUCT = 1.5          # exploration constant
FPU_REDUCTION = 0.25  # an unvisited move is assumed this much worse than its parent
SWEEP_SHARE = 0.35    # at most this fraction of the time budget goes on the root sweep
VL_DEFAULT = 1.0      # virtual-loss strength for batched search (see Searcher): 1 = a full loss


def fast_clone(game: Game) -> Game:
    """``Game.clone()`` without rebuilding immutable objects.

    ``Game.clone`` deep-copies everything, including every legal ``Action`` (the dict of legal
    actions holds dozens to thousands) and the whole ``move_log``, one slow reconstruction each.
    ``Action`` is a frozen dataclass, so copies can share them; the log and the repetition
    table are copied shallowly.  Everything mutable (pieces, board, moves) is still deep-copied.
    Keep in step with ``Game.clone`` (it also leaves the move caches out)."""
    memo = {}
    for a in game.actions:
        memo[id(a)] = a
    for a in game.move_log:
        memo[id(a)] = a
    memo[id(game.move_log)] = list(game.move_log)
    memo[id(game.seen)] = dict(game.seen)
    saved = game._moves_cache, game._has_move
    game._moves_cache, game._has_move = {}, {}
    try:
        return copy.deepcopy(game, memo)
    finally:
        game._moves_cache, game._has_move = saved


def light_default() -> bool:
    """Light mode is on when the native library supports it, unless RC_FASTGAME=0."""
    if os.environ.get("RC_FASTGAME", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    return fastgame.AVAILABLE


def terminal_value(game: Game) -> float:
    """Exact result of a finished game for the side to move (``game.turn``)."""
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner == game.turn else -1.0


class Node:
    """A position.  ``W`` and ``N`` are per *edge* (per legal action), with W summed from the
    point of view of this node's side to move."""

    __slots__ = ("game", "turn", "terminal", "value", "actions", "priors",
                 "N", "W", "children", "total", "win_idx", "vl", "nvl", "pending")

    def __init__(self, game: Optional[Game], turn: str, terminal: bool, value: float,
                 actions: Optional[List[Action]] = None, priors: Optional[np.ndarray] = None):
        self.game = game                      # kept only while the node can still be expanded
        self.turn = turn
        self.terminal = terminal
        self.value = value                    # network (or exact, if terminal) value, mover's view
        self.actions = actions or []
        self.priors = priors if priors is not None else np.zeros(0)
        n = len(self.actions)
        self.N = np.zeros(n, dtype=np.int64)
        self.W = np.zeros(n, dtype=np.float64)
        self.children: List[Optional["Node"]] = [None] * n
        self.total = 0                        # visits through this node
        self.win_idx: Optional[int] = None    # index of a move known to win on the spot
        self.vl = np.zeros(n, dtype=np.int64)  # virtual losses: batched descents in flight per edge
        self.nvl = 0                          # their sum
        self.pending = False                  # created, network evaluation not done yet

    def fill(self, a) -> None:
        """Give a pending node its network result (an ``Analysis``)."""
        self.value = a.value
        self.actions = a.actions
        self.priors = a.probs
        n = len(self.actions)
        self.N = np.zeros(n, dtype=np.int64)
        self.W = np.zeros(n, dtype=np.float64)
        self.vl = np.zeros(n, dtype=np.int64)
        self.children = [None] * n
        self.pending = False


def _edge_value(parent: Node, child: Node, v: float) -> float:
    """``v`` is from ``child``'s mover's view; return it from ``parent``'s mover's view."""
    return v if child.turn == parent.turn else -v


class Job:
    """One search of one position, run in its own thread.  Read-only for the UI."""

    def __init__(self, key, game: Game, sims: int, ms: int, root_analysis=None,
                 noise: float = 0.0, rng=None, sweep_k: Optional[int] = None):
        self.key = key
        self.root_analysis = root_analysis    # network output for the root, if the caller has it
        self.noise = noise                    # Dirichlet noise mixed into the root priors
        self.rng = rng
        self.sweep_k = sweep_k                # None: sweep every root move; 0: none; k: top k
        self.game = game                      # private copy
        self.max_sims = sims
        self.deadline_ms = ms
        self.root: Optional[Node] = None
        self.finished = False
        self.error: Optional[str] = None
        self.sweep_done = 0                   # root moves checked exactly
        self.sweep_total = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.started = time.monotonic()

    # -- control ---------------------------------------------------------
    def stop(self):
        self._stop.set()

    # -- results (safe to call any time) ----------------------------------
    @property
    def sims(self) -> int:
        root = self.root
        return 0 if root is None else int(root.total)

    @property
    def solved(self) -> bool:
        root = self.root
        return root is not None and root.win_idx is not None

    def value(self) -> Optional[float]:
        """Search value of the root for the side to move, -1..+1 (None before any result)."""
        with self._lock:
            root = self.root
            if root is None:
                return None
            if root.terminal:
                return root.value
            if root.win_idx is not None:
                return 1.0
            if root.total == 0:
                return root.value
            return float(np.clip(root.W.sum() / root.total, -1.0, 1.0))

    def white_value(self) -> Optional[float]:
        v = self.value()
        if v is None or self.root is None:
            return None
        return v if self.root.turn == "w" else -v

    def best(self) -> Optional[Action]:
        """The move to play: a sure win if one was found, else the most visited move."""
        with self._lock:
            root = self.root
            if root is None or root.terminal or not root.actions:
                return None
            if root.win_idx is not None:
                return root.actions[root.win_idx]
            if root.total == 0:
                return root.actions[int(np.argmax(root.priors))]
            q = root.W / np.maximum(root.N, 1)
            order = np.lexsort((q, root.N))          # most visits, ties by value
            return root.actions[int(order[-1])]

    def top(self, k: int = 3) -> List[Tuple[Action, float, float]]:
        """The k most visited moves as (action, share of visits, value for the mover)."""
        with self._lock:
            root = self.root
            if root is None or root.terminal or not root.actions:
                return []
            if root.win_idx is not None:
                return [(root.actions[root.win_idx], 1.0, 1.0)]
            tot = max(int(root.N.sum()), 1)
            q = root.W / np.maximum(root.N, 1)
            order = np.lexsort((q, root.N))[::-1][:k]
            return [(root.actions[int(i)], float(root.N[i]) / tot, float(q[i]))
                    for i in order if root.N[i] > 0]


class Searcher:
    """Runs searches for play.py.  One search at a time: starting a new one stops the old."""

    def __init__(self, advisor, sims: int = 600, think_ms: int = 2500, c_puct: float = C_PUCT,
                 batch: int = 1, light: Optional[bool] = None, vl: float = VL_DEFAULT):
        if light and not fastgame.AVAILABLE:
            raise RuntimeError("light search needs the native engine with rc_step "
                               "(run ./build_native.sh)")
        if not 0.0 <= vl <= 1.0:
            raise ValueError("vl (virtual-loss strength) must be in [0, 1], got %r" % (vl,))
        self.light = light_default() if light is None else bool(light)
        self.advisor = advisor
        self.batch = max(1, int(batch))       # leaves evaluated per network call (virtual loss)
        # What a descent that is still in flight counts as: one extra visit of an edge whose result
        # is  (1 - vl) * (the edge's current value) - vl.  vl = 1: a sure loss (the classic virtual
        # loss: the next descent of the batch is pushed to another move hard, which spreads a small
        # simulation budget over many root moves); vl = 0: a visit that changes nothing but the
        # exploration term (descents spread the way PUCT itself would spread them).  Irrelevant
        # when batch = 1.
        self.vl = float(vl)
        self.sims = sims
        self.think_ms = think_ms
        self.c_puct = c_puct
        self.job: Optional[Job] = None
        # where the time goes (seconds, summed over this searcher's life): copying + stepping
        # games, network evaluation of new nodes.  Read by rollout.py for the training log.
        self.t_clone = 0.0
        self.t_eval = 0.0

    # -- public ----------------------------------------------------------
    def start(self, game: Game, key, ms: Optional[int] = None,
              sims: Optional[int] = None) -> Job:
        if self.job is not None:
            self.job.stop()
        job = Job(key, self._copy(game), sims or self.sims, ms or self.think_ms)
        self.job = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def stop(self):
        if self.job is not None:
            self.job.stop()

    def search(self, game: Game, sims: Optional[int] = None, ms: Optional[int] = None,
               root_analysis=None, noise: float = 0.0, rng=None,
               sweep_k: Optional[int] = None) -> Job:
        """Blocking search (self-play, scripts, tests): returns the finished Job.

        ``root_analysis`` (an ``Analysis`` of ``game``) saves re-evaluating the root;
        ``noise`` mixes Dirichlet noise into the root priors for exploration; ``sweep_k``
        limits the exact root sweep to the k likeliest moves (0 = none, None = all)."""
        job = Job(None, self._copy(game), sims or self.sims, ms or self.think_ms,
                  root_analysis, noise, rng, sweep_k)
        self._run(job)
        return job

    # -- internals -------------------------------------------------------
    def _copy(self, game: Game):
        """Private copy of the caller's game for a search to own."""
        return fastgame.FastGame.from_game(game) if self.light else fast_clone(game)

    def _child_game(self, node: Node, i: int):
        """The position after ``node``'s i-th legal action."""
        if self.light:
            return node.game.child(i)
        g = fast_clone(node.game)
        g.step(node.actions[i])
        return g

    def _make_node(self, game: Game) -> Node:
        if game.done:
            node = Node(None, game.turn, True, terminal_value(game))
            game.close()
            return node
        a = self.advisor.analyse_fresh(game)
        return Node(game, game.turn, False, a.value, a.actions, a.probs)

    def _select(self, node: Node) -> int:
        if node.win_idx is not None:
            return node.win_idx
        # the FPU baseline uses real visits only, so a virtual loss never makes unvisited moves
        # look worse than moves already in flight
        parent_q = float(node.W.sum()) / node.total if node.total else node.value
        fpu = parent_q - FPU_REDUCTION
        if node.nvl == 0:                      # no batched descents in flight (always, if batch=1)
            n_eff, total = node.N, node.total
            q = np.where(n_eff > 0, node.W / np.maximum(n_eff, 1), fpu)
        else:                                  # each one counts as a visit worth v_virt (see vl)
            n_eff, total = node.N + node.vl, node.total + node.nvl
            q_now = np.where(node.N > 0, node.W / np.maximum(node.N, 1), fpu)
            v_virt = q_now * (1.0 - self.vl) - self.vl
            q = np.where(n_eff > 0, (node.W + node.vl * v_virt) / np.maximum(n_eff, 1), fpu)
        u = self.c_puct * node.priors * math.sqrt(total + 1.0) / (1.0 + n_eff)
        return int(np.argmax(q + u))

    @staticmethod
    def _backup(path: List[Tuple[Node, int]], v: float):
        """Add result ``v`` (from the mover's view at the end of ``path``) to every edge on the
        path and take back the virtual loss each descent put there."""
        for parent, i in reversed(path):
            child = parent.children[i]
            v = _edge_value(parent, child, v)
            parent.W[i] += v
            parent.N[i] += 1
            parent.total += 1
            parent.vl[i] -= 1
            parent.nvl -= 1

    @staticmethod
    def _undo(path: List[Tuple[Node, int]]):
        for parent, i in path:
            parent.vl[i] -= 1
            parent.nvl -= 1

    def _evaluate(self, games: List[Game]):
        """Network results for several positions at once, in one call when the advisor can."""
        many = getattr(self.advisor, "analyse_many", None)
        if many is not None:
            return many(games)
        return [self.advisor.analyse_fresh(g) for g in games]

    def _round(self, root: Node, want: int):
        """Run up to ``want`` simulations as one batch: descend ``want`` times (each descent
        leaves a virtual loss on its path so the next one prefers another line), evaluate all the
        new leaves with a single network call, then back everything up.  With ``want`` = 1 this
        is the plain one-at-a-time PUCT simulation."""
        leaves: List[Tuple[List[Tuple[Node, int]], Node]] = []
        known: List[Tuple[List[Tuple[Node, int]], float]] = []   # descents that ended on a result
        issued = 0
        while issued < want and root.win_idx is None:
            path: List[Tuple[Node, int]] = []
            node = root
            leaf = None
            while True:
                if node.terminal:
                    v = node.value
                    break
                if node.pending:               # another descent of this batch got here first
                    v = None
                    break
                i = self._select(node)
                path.append((node, i))
                node.vl[i] += 1
                node.nvl += 1
                child = node.children[i]
                if child is None:
                    t0 = time.perf_counter()
                    g = self._child_game(node, i)
                    self.t_clone += time.perf_counter() - t0
                    if g.done:
                        child = Node(None, g.turn, True, terminal_value(g))
                        g.close()
                        node.children[i] = child
                        if _edge_value(node, child, child.value) >= 1.0:
                            node.win_idx = i
                        v = child.value
                    else:
                        child = Node(g, g.turn, False, 0.0)
                        child.pending = True
                        node.children[i] = child
                        leaf = child
                        v = 0.0
                    break
                node = child
            if v is None:                      # collision: stop collecting, flush what we have
                self._undo(path)
                break
            issued += 1
            if leaf is not None:
                leaves.append((path, leaf))
            else:                              # result known: back it up with the rest, so its
                known.append((path, v))        # virtual loss steers the other descents away too
        if leaves:
            t0 = time.perf_counter()
            results = self._evaluate([leaf.game for _, leaf in leaves])
            self.t_eval += time.perf_counter() - t0
            for (path, leaf), a in zip(leaves, results):
                leaf.fill(a)
                self._backup(path, a.value)
        for path, v in known:
            self._backup(path, v)

    def _sweep(self, job: Job, root: Node, until: float):
        """Play every root move on a copy, best prior first, looking only for a result."""
        order = np.argsort(-root.priors)
        if job.sweep_k is not None:
            order = order[:job.sweep_k]
        job.sweep_total = len(order)
        for n, i in enumerate(order):
            if job._stop.is_set() or time.monotonic() > until or root.win_idx is not None:
                break
            t0 = time.perf_counter()
            g = self._child_game(root, int(i))
            self.t_clone += time.perf_counter() - t0
            job.sweep_done = n + 1
            if not g.done:
                g.close()
                continue
            child = Node(None, g.turn, True, terminal_value(g))
            g.close()
            v = _edge_value(root, child, child.value)
            with job._lock:
                root.children[i] = child
                root.W[i] += v
                root.N[i] += 1
                root.total += 1
                if v >= 1.0:
                    root.win_idx = int(i)

    def _run(self, job: Job):
        try:
            t0 = time.monotonic()
            end = t0 + job.deadline_ms / 1000.0
            a = job.root_analysis
            if a is not None and not job.game.done:
                root = Node(job.game, job.game.turn, False, a.value, a.actions, a.probs.copy())
            else:
                root = self._make_node(job.game)
            if job.noise > 0 and not root.terminal and len(root.priors) > 1:
                n = len(root.priors)
                rng = job.rng if job.rng is not None else np.random.default_rng()
                alpha = min(1.0, max(0.03, 10.0 / n))
                root.priors = ((1.0 - job.noise) * root.priors
                               + job.noise * rng.dirichlet([alpha] * n))
            with job._lock:
                job.root = root
            if root.terminal:
                return
            if job.sweep_k != 0:
                self._sweep(job, root, t0 + SWEEP_SHARE * job.deadline_ms / 1000.0)
            while (not job._stop.is_set() and root.win_idx is None
                   and root.total < job.max_sims and time.monotonic() < end):
                with job._lock:
                    self._round(root, min(self.batch, job.max_sims - root.total))
        except Exception as err:               # never kill the UI over a search bug
            job.error = "%s: %s" % (type(err).__name__, err)
        finally:
            with job._lock:
                self._release(job.root)
                job.game.close()
            job.finished = True

    @staticmethod
    def _release(root: Optional[Node]):
        """Close every game still held by the tree (see the Memory note above)."""
        stack = [root] if root is not None else []
        while stack:
            node = stack.pop()
            if node.game is not None:
                node.game.close()
                node.game = None
            stack.extend(c for c in node.children if c is not None)
