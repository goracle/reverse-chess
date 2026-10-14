#!/usr/bin/env python
"""Check the native tree search (native/src/mcts.rs, ``NativeGame.search``) against ``mcts.Searcher``.

    ./build_native.sh && python check_nativesearch.py
    python check_nativesearch.py --positions 60 --sims 80 --ckpt runs/run4/latest.pt
    python check_nativesearch.py --bench 20          # also time both

Positions come from random native play (so repetition counts, extra turns and the fifty-move
counter are all live).  For each position and each (batch, virtual-loss) setting the SAME network
(a ``RustNet``) is searched twice, without root noise and without the root sweep:

* reference: ``mcts.Searcher.search`` on the position as a ``FastGame`` (what ``net.search_action``
  used before the native search);
* native: ``NativeGame.search``.

Compared: simulations run, the move proved to win (if any), the root visit counts, and the
network's root value / policy.  The two do their sums in a different order (numpy pairwise vs
sequential), so a PUCT tie can in principle break differently; that is reported as a mismatch but
should be very rare -- the run only FAILS if more than ``--tolerance`` (default 2%) of the
searches differ, or if anything other than visit counts differs.

Also checked: the games of ``nativegame.play`` with search switched on are well formed with the
native search, and with RC_NATIVESEARCH off (the Python path) -- both must give a proper pi.

Part two is the search play.py runs (nativesearch.py): the same ``numpy_net.Advisor`` searched by
``mcts.Searcher(native=False)`` (Python) and ``Searcher(native=True)`` (Rust), with the exact root
sweep, compared the same way; then a live job is polled while it runs (the simulation count must
grow before it finishes), stopped (must end promptly), and given a deadline (must be kept).
"""
import argparse
import sys
import time

import numpy as np

import mcts
import nativegame
import net as netlib
import rustnet
from engine import Rules

SETTINGS = ((1, 1.0), (8, 1.0), (8, 0.0), (16, 0.5))      # (batch, virtual loss)


def make_net(seed, ckpt):
    if ckpt:
        model, ck = netlib.load_checkpoint(ckpt)
        rn = rustnet.RustNet.from_torch(model, int(ck.get("version", 0)))
        rn._torch = model
        return rn
    import torch
    torch.manual_seed(seed)
    model = netlib.PolicyValueNet()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.0)                       # sharpen a random net so the priors are not flat
    rn = rustnet.RustNet.from_torch(model, 0)
    rn._torch = model                         # part two needs the weights again
    return rn


def random_positions(rules, rng, count, max_depth=120):
    """Live NativeGames at random depths of random play (the caller closes them)."""
    games = []
    while len(games) < count:
        g = nativegame.NativeGame(rules)
        depth = int(rng.integers(0, max_depth))
        for _ in range(depth):
            info = g.info()
            if info[0] or info[9] == 0:
                break
            g.apply(int(rng.integers(0, info[9])))
        info = g.info()
        if info[0] or info[9] < 1:
            g.close()
            continue
        games.append(g)
    return games


def python_search(model, position, sims, batch, vl):
    evaluator = netlib.SearchEvaluator(model)
    analysis = evaluator.evaluate(position)[0]
    searcher = mcts.Searcher(evaluator, sims=sims, batch=batch, vl=vl, light=True, native=False)
    job = searcher.search(position, sims=sims, ms=10 ** 9, root_analysis=analysis, noise=0.0,
                          sweep_k=0)
    if job.error:
        raise RuntimeError("python search failed: " + job.error)
    root = job.root
    return {"visits": root.N.copy(), "total": int(root.total),
            "win_idx": -1 if root.win_idx is None else int(root.win_idx),
            "value": float(analysis.value), "probs": np.asarray(analysis.probs)}


def compare(ref, nat):
    """(differences in the search itself, differences in the network's root output)."""
    search, net_out = [], []
    if ref["total"] != nat["sims"]:
        search.append("simulations: %d vs %d" % (ref["total"], nat["sims"]))
    if ref["win_idx"] != nat["win_idx"]:
        search.append("winning move: %d vs %d" % (ref["win_idx"], nat["win_idx"]))
    if not np.array_equal(ref["visits"], nat["visits"]):
        d = np.flatnonzero(ref["visits"] != nat["visits"])
        search.append("visits differ on %d of %d moves (first: move %d, %d vs %d)"
                      % (len(d), len(ref["visits"]), d[0], ref["visits"][d[0]], nat["visits"][d[0]]))
    if abs(ref["value"] - nat["value"]) > 1e-5:
        net_out.append("root value: %.6f vs %.6f" % (ref["value"], nat["value"]))
    if not np.allclose(ref["probs"], nat["probs"], atol=1e-6, rtol=0):
        net_out.append("root policy differs (max abs %.3g)"
                       % float(np.abs(ref["probs"] - nat["probs"]).max()))
    return search, net_out


def smoke_games(model, rules, rng, games, native):
    """nativegame.play with search on: shapes, indices and a proper pi on every searched move."""
    import os
    old = os.environ.get("RC_NATIVESEARCH")
    os.environ["RC_NATIVESEARCH"] = "1" if native else "0"
    cfg = {"frac": 0.3, "sims": 12, "noise": 0.25, "temp": 1.0, "sweep": 0, "smooth": 0.1,
           "batch": 4}
    problems, searched_total = [], 0
    try:
        for g in range(games):
            r = nativegame.play(model, None, rules, rng, 3, 0, cfg)
            n, t = int(r["counts"].sum()), len(r["counts"])
            if (r["chosen"] >= r["counts"]).any() or (r["chosen"] < 0).any():
                problems.append("game %d: chosen index out of range" % g)
            if "searched" in r:
                searched_total += int(r["searched"].sum())
                start = np.concatenate([[0], np.cumsum(r["counts"])])
                for i in np.flatnonzero(r["searched"]):
                    s = r["pi"][start[i]:start[i + 1]].sum()
                    if abs(s - 1.0) > 1e-4:
                        problems.append("game %d decision %d: pi sums to %g" % (g, i, s))
                        break
                if r["search_sims"] <= 0:
                    problems.append("game %d: search_sims is %d" % (g, r["search_sims"]))
    finally:
        if old is None:
            os.environ.pop("RC_NATIVESEARCH", None)
        else:
            os.environ["RC_NATIVESEARCH"] = old
    if searched_total == 0:
        problems.append("no decision was searched in %d games" % games)
    return problems, searched_total


def play_checks(model, games, args):
    """The play.py search: Python Searcher vs the native one, plus live / stop / deadline."""
    import os
    import tempfile

    import numpy_net
    weights = {k: v.detach().cpu().numpy() for k, v in model_state(model).items()}
    path = os.path.join(tempfile.mkdtemp(), "check_ai.npz")
    numpy_net.save_npz(path, weights, 0)
    advisor = numpy_net.Advisor(path, args.max_plies)
    problems, differ, total = [], 0, 0

    def run(position, native, sweep, sims=args.sims, ms=10 ** 9):
        searcher = mcts.Searcher(advisor, sims=sims, think_ms=ms, light=True, native=native)
        analysis = advisor.analyse_fresh(position)
        job = searcher.search(position, sims=sims, ms=ms, root_analysis=analysis, sweep_k=None if sweep else 0)
        if job.error:
            raise RuntimeError(job.error)
        return job

    for sweep in (True, False):
        bad = 0
        for k, g in enumerate(games):
            position = g.position()
            a, b = run(position, False, sweep), run(position, True, sweep)
            total += 1
            same = (a.root.total == b.root.total and a.root.win_idx == b.root.win_idx
                    and np.array_equal(a.root.N, b.root.N))
            if not same:
                bad += 1
                print("play search, position %d (sweep %s): sims %d vs %d, win %s vs %s, %d moves differ"
                      % (k, sweep, a.root.total, b.root.total, a.root.win_idx, b.root.win_idx,
                         int((a.root.N != b.root.N).sum())))
            elif sweep and (a.value() is None or abs(a.value() - b.value()) > 1e-3):
                bad += 1
                print("play search, position %d: value %s vs %s" % (k, a.value(), b.value()))
        differ += bad
        print("play search, sweep %-5s: %d/%d identical" % (sweep, len(games) - bad, len(games)))

    # live progress, stop, deadline
    position = games[0].position()
    searcher = mcts.Searcher(advisor, sims=10 ** 7, think_ms=60000, light=True, native=True)
    job = searcher.start(position, "check", ms=60000, sims=10 ** 7)
    seen, t0 = [], time.time()
    while not job.finished and time.time() - t0 < 0.6:
        seen.append(job.sims)
        time.sleep(0.05)
    job.stop()
    t1 = time.time()
    while not job.finished and time.time() - t1 < 3.0:
        time.sleep(0.01)
    if not job.finished:
        problems.append("a stopped native search did not finish within 3 s")
    elif job.error:
        problems.append("native search error: " + job.error)
    else:
        print("live job: sims seen while running %s ... final %d, stopped in %.0f ms"
              % (seen[:1] + seen[-1:], job.sims, 1000 * (time.time() - t1)))
        if job.sims <= 0 or not seen or max(seen) <= 0:
            problems.append("no live progress while the search ran (sims seen: %s)" % seen[-3:])
    t0 = time.time()
    job = run(position, True, True, sims=10 ** 7, ms=400)
    el = time.time() - t0
    print("deadline 400 ms: finished in %.0f ms with %d sims" % (1000 * el, job.sims))
    if el > 1.5 or job.sims <= 0:
        problems.append("deadline not kept (%.2f s, %d sims)" % (el, job.sims))
    if args.bench:
        n, sims = min(args.bench, len(games)), max(args.sims, 200)
        t0 = time.time()
        for g in games[:n]:
            run(g.position(), False, True, sims=sims)
        t_py = time.time() - t0
        t0 = time.time()
        for g in games[:n]:
            run(g.position(), True, True, sims=sims)
        t_nat = time.time() - t0
        print("play search, python: %d x %d sims %.2fs (%.0f sims/s)" % (n, sims, t_py, n * sims / t_py))
        print("play search, native: %d x %d sims %.2fs (%.0f sims/s)  -> %.1fx"
              % (n, sims, t_nat, n * sims / max(t_nat, 1e-9), t_py / max(t_nat, 1e-9)))
    return problems, differ, total


def model_state(model):
    """The state dict of the torch model behind a RustNet (kept by make_net)."""
    return model._torch.state_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--ckpt", help="a .pt checkpoint to use instead of a random network")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="fraction of searches allowed to differ in visit counts (float tie-breaks)")
    ap.add_argument("--bench", type=int, default=0, help="also time this many searches each way")
    args = ap.parse_args()

    if not nativegame.SEARCH_AVAILABLE:
        print("native search missing: run ./build_native.sh (rc_game_search not in the library)")
        return 2
    rules = Rules(max_plies=args.max_plies)
    model = make_net(args.seed, args.ckpt)
    rng = np.random.default_rng(args.seed)
    games = random_positions(rules, rng, args.positions)
    total = differ = hard = 0
    try:
        for batch, vl in SETTINGS:
            bad_here = 0
            for k, g in enumerate(games):
                position = g.position()
                ref = python_search(model, position, args.sims, batch, vl)
                nat = g.search(model, args.sims, batch=batch, vl=vl, c_puct=mcts.C_PUCT)
                search, net_out = compare(ref, nat)
                total += 1
                if search or net_out:
                    differ += 1
                    bad_here += 1
                    hard += bool(net_out)
                    print("position %d (batch %d, vl %.1f):" % (k, batch, vl))
                    for line in search + net_out:
                        print("    " + line)
            print("batch %2d vl %.1f: %d/%d searches identical" % (batch, vl, len(games) - bad_here, len(games)))

        for native in (True, False):
            problems, searched = smoke_games(model, rules, rng, 4, native)
            print("games with search (%s): %d searched decisions, %s"
                  % ("native" if native else "python", searched, "ok" if not problems else "PROBLEMS"))
            for p in problems:
                print("    " + p)
            hard += len(problems)

        problems, d2, t2 = play_checks(model, games, args)
        hard += len(problems)
        differ += d2
        total += t2
        for p in problems:
            print("    " + p)

        if args.bench:
            n = min(args.bench, len(games))
            t0 = time.time()
            for g in games[:n]:
                python_search(model, g.position(), args.sims, 8, 1.0)
            t_py = time.time() - t0
            t0 = time.time()
            for g in games[:n]:
                g.search(model, args.sims, batch=8, vl=1.0)
            t_nat = time.time() - t0
            print("python search: %d x %d sims %.2fs" % (n, args.sims, t_py))
            print("native search: %d x %d sims %.2fs  -> %.1fx" % (n, args.sims, t_nat, t_py / max(t_nat, 1e-9)))
    finally:
        for g in games:
            g.close()
    frac = differ / max(total, 1)
    print("%d searches, %d differ (%.1f%%)" % (total, differ, 100 * frac))
    ok = hard == 0 and frac <= args.tolerance
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
