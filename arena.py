#!/usr/bin/env python
"""Play one checkpoint against another.  (Run from reverse-chess/.)

    python arena.py runs/run2/latest.pt runs/run2/iter_00700.pt
    python arena.py runs/run2/latest.pt runs/run2/iter_00700.pt --games 400 --workers 8

Either side may also be the rule-based walk-home player from heuristic.py, written
heur:nogift or heur:gift (e.g. ``python arena.py runs/run2/latest.pt heur:nogift``).
Both networks sample from their policy (same as the vs-random EVAL in train.py),
half the games with A as black and half as white.  Score is from A's point of
view: win = 1, draw = 0.5, loss = 0.  0.5 means equal strength; the +/- is a
rough 95% interval, so with few games a score like 0.55 means nothing.
Everything runs on the CPU, so it is fine to run while train.py is training
(use fewer --workers then).
"""
import argparse
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

_S = {}


def _is_heur(spec):
    return spec.startswith("heur:")


def _load(spec):
    """(model, version) for a checkpoint path, or (("heur", mode), 0)."""
    if _is_heur(spec):
        mode = spec.split(":", 1)[1]
        if mode not in ("nogift", "gift"):
            raise SystemExit("unknown heuristic %r (use heur:nogift or heur:gift)" % spec)
        return ("heur", mode), 0
    import net as netlib
    model, ckpt = netlib.load_checkpoint(spec)
    return model, int(ckpt["version"])


def _init(path_a, path_b, max_plies):
    if not (_is_heur(path_a) and _is_heur(path_b)):
        import torch
        torch.set_num_threads(1)
    from engine import Rules
    model_a, va = _load(path_a)
    model_b, vb = _load(path_b)
    _S.update(models={"a": model_a, "b": model_b}, rules=Rules(max_plies=max_plies),
              versions=(va, vb))


def _play(seed, a_team, greedy=False):
    from engine import Game
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)                 # for the heuristic player
    game = Game(_S["rules"])
    while not game.done:
        who = "a" if game.turn == a_team else "b"
        model = _S["models"][who]
        if isinstance(model, tuple):
            import heuristic
            action = heuristic.choose(game, py_rng, model[1])
        else:
            import net as netlib
            action, _ = netlib.choose_action(model, game, rng, greedy)
        game.step(action)
    result = {"a_team": a_team, "winner": game.winner, "reason": game.end_reason,
              "plies": game.ply, "versions": _S["versions"]}
    game.close()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="checkpoint A (e.g. the newest), or heur:nogift / heur:gift")
    ap.add_argument("b", help="checkpoint B (e.g. an older snapshot), or heur:nogift / heur:gift")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-plies", type=int, default=300, help="draw after this many plies (train.py default)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--greedy", action="store_true",
                    help="networks play their single most likely move instead of sampling (training "
                         "samples; a sampling net with entropy ~2 nats is much noisier than its best play)")
    args = ap.parse_args()
    for p in (args.a, args.b):
        if not _is_heur(p) and not os.path.exists(p):
            raise SystemExit("no such checkpoint: %s" % p)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init,
                             initargs=(args.a, args.b, args.max_plies)) as pool:
        futures = [pool.submit(_play, args.seed + i, "b" if i % 2 == 0 else "w", args.greedy)
                   for i in range(args.games)]
        results = [f.result() for f in futures]            # re-raises worker errors

    va, vb = results[0]["versions"]
    scores = np.array([0.5 if r["winner"] is None else float(r["winner"] == r["a_team"])
                       for r in results])
    wins, draws = int((scores == 1).sum()), int((scores == 0.5).sum())
    losses = len(scores) - wins - draws
    half = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else float("nan")
    print("A = v%d (%s)   vs   B = v%d (%s)   %d games in %.0fs"
          % (va, args.a, vb, args.b, len(results), time.time() - t0))
    print("A: +%d =%d -%d   score %.3f +/- %.3f" % (wins, draws, losses, scores.mean(), half))
    for team, name in (("b", "A as black"), ("w", "A as white")):
        s = np.array([x for x, r in zip(scores, results) if r["a_team"] == team])
        print("  %-10s score %.3f  (%d games)" % (name, s.mean(), len(s)))
    print("avg %.0f plies; ended by: %s" % (
        np.mean([r["plies"] for r in results]),
        dict(Counter(r["reason"][:48] for r in results).most_common(5))))


if __name__ == "__main__":
    main()
