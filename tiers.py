#!/usr/bin/env python
"""Difficulty tiers: an easy and a medium AI that lose a chosen share of their games to the main AI.
(Run from reverse-chess/, like train.py and arena.py.)

    python tiers.py ladder --vs-random 0.95            # suggest loss rates from how far main is above random
    python tiers.py table                              # what a loss rate means in Elo
    python tiers.py scan  --main runs/run4/latest.pt --targets easy=0.85 medium=0.70
    python tiers.py scan  --main runs/run4/latest.pt --targets easy=0.85 medium=0.70 --install
    python tiers.py train --main runs/run4/latest.pt --name tiers1 --targets easy=0.85 medium=0.70 \\
                          --workers 9 --device xpu   # everything unknown to tiers.py goes to train.py

What is measured
----------------
"Loses X%" = the tier loses X% of the games it plays against the main AI, half the games as each colour,
both sides sampling from their policy (the same way arena.py plays), no look-ahead search.  Draws are
reported separately and do not count as losses; ``--score`` counts a draw as half a loss instead (the
main AI's score).  The main AI is frozen (``--main`` is copied to ``runs/NAME/main_ref.pt``), so the
measurement does not drift while run4 keeps training.

scan   Plays every ``iter_*.pt`` snapshot of an existing run against the main AI and says which one is
       closest to each target.  Costs no training.  With ``--install`` the closest snapshots are
       re-measured with more games and installed.
train  A fresh network trained with train.py's ordinary self-play, in chunks of ``--chunk`` iterations.
       After every chunk the newest weights play ``--games`` games against the main AI.  Strength rises
       as it trains, so the first time the loss rate falls to a target that checkpoint (or the one
       before it) is re-measured with ``--verify-games`` games and, if it is within ``--tol`` of the
       target, installed; training goes on towards the next target and stops after the last one.
       Rerunning the same command resumes.
install  Installing means: ``nets/NAME.npz`` (what play.py loads) and an entry in ``nets/tiers.json``,
       which gives play.py its Game > Difficulty menu.  "Insane" is the main AI itself, with search on.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n (k may be fractional when draws count half)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def elo_gap(p):
    """Elo difference for a side that scores ``p`` (0.5 -> 0)."""
    p = min(max(p, 1e-3), 1 - 1e-3)
    return 400.0 * math.log10(p / (1.0 - p))


def p_from_elo(gap):
    return 1.0 / (1.0 + 10.0 ** (-gap / 400.0))


def ladder(vs_random, tiers):
    """Loss rates (easiest first) that put the tiers at equal Elo steps between 'random' and the main AI.
    ``vs_random`` is the main AI's score against a random player (train.py's EVAL line)."""
    floor = elo_gap(vs_random)
    step = floor / (tiers + 1)
    return [p_from_elo(step * k) for k in range(tiers, 0, -1)]


def judge(rate, target, tol):
    """'ok' within tol of the target, 'weak' (still loses too often) or 'strong' (overshot)."""
    if abs(rate - target) <= tol:
        return "ok"
    return "weak" if rate > target else "strong"


# --------------------------------------------------------------------------
# Playing the match
# --------------------------------------------------------------------------

def match(cand, main, games, workers, seed, max_plies=300, greedy=False, use_score=False):
    """``cand`` (a checkpoint) against ``main``; half the games with cand as black, half as white.
    Returns the candidate's loss rate (``use_score``: the main AI's score) with a 95% interval."""
    import arena                                         # (its _init / _play are the worker side)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=arena._init,
                             initargs=(cand, main, max_plies)) as pool:
        futures = [pool.submit(arena._play, seed + i, "b" if i % 2 == 0 else "w", greedy) for i in range(games)]
        results = [f.result() for f in futures]          # re-raises worker errors
    wins = sum(r["winner"] == r["a_team"] for r in results)
    draws = sum(r["winner"] is None for r in results)
    losses = games - wins - draws
    k = losses + (0.5 * draws if use_score else 0.0)
    lo, hi = wilson(k, games)
    colour = {}
    for team in ("b", "w"):
        rs = [r for r in results if r["a_team"] == team]
        colour[team] = (len(rs), sum(1 for r in rs if r["winner"] not in (None, team)))
    return {"games": games, "wins": wins, "draws": draws, "losses": losses, "rate": k / games,
            "lo": lo, "hi": hi, "colour": colour, "version": results[0]["versions"][0]}


def describe(m):
    b, w = m["colour"]["b"], m["colour"]["w"]
    return ("loses %5.1f%% [%4.1f-%4.1f]  (+%d =%d -%d of %d; as black lost %d/%d, as white %d/%d)  ~ Elo gap %+.0f"
            % (100 * m["rate"], 100 * m["lo"], 100 * m["hi"], m["wins"], m["draws"], m["losses"], m["games"],
               b[1], b[0], w[1], w[0], elo_gap(m["rate"])))


# --------------------------------------------------------------------------
# Installing a tier for play.py
# --------------------------------------------------------------------------

def export_npz(pt, npz):
    """The PyTorch-free weight file play.py loads (same as export_weights.py)."""
    import torch
    import numpy_net
    ckpt = torch.load(pt, map_location="cpu", weights_only=True)
    state = {k: v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()}
    numpy_net.save_npz(npz, state, int(ckpt.get("version", -1)))


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def install(name, pt, info, main_ref):
    """nets/NAME.npz from checkpoint ``pt`` plus its entry in nets/tiers.json (and 'Insane' = the main AI, once)."""
    nets = os.path.join(ROOT, "nets")
    os.makedirs(nets, exist_ok=True)
    manifest = os.path.join(nets, "tiers.json")
    data = {"tiers": []}
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as fh:
            data = json.load(fh)
    export_npz(pt, os.path.join(nets, name + ".npz"))
    entry = dict(info, name=name.capitalize(), path="nets/%s.npz" % name, search=False, source=os.path.relpath(pt, ROOT))
    tiers = [t for t in data.get("tiers", []) if t.get("name", "").lower() != name.lower()] + [entry]
    if not any(t.get("name", "").lower() == "insane" for t in tiers):
        export_npz(main_ref, os.path.join(nets, "insane.npz"))
        tiers.append({"name": "Insane", "path": "nets/insane.npz", "search": True, "loss_rate": 0.5,
                      "source": os.path.relpath(main_ref, ROOT)})
    tiers.sort(key=lambda t: -float(t.get("loss_rate") or 0.0))
    data["tiers"] = tiers
    _write_json(manifest, data)
    print("installed %s -> nets/%s.npz  (play.py: Game > Difficulty)" % (entry["name"], name), flush=True)


def tier_info(m, target, source_version):
    return {"loss_rate": round(m["rate"], 4), "target": target, "games": m["games"],
            "draws": m["draws"], "ci": [round(m["lo"], 4), round(m["hi"], 4)], "version": source_version}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def parse_targets(items):
    out = []
    for item in items:
        name, _, value = item.partition("=")
        try:
            rate = float(value)
        except ValueError:
            raise SystemExit("targets look like easy=0.85, not %r" % item)
        if not name or not 0.5 < rate < 1.0:
            raise SystemExit("target %r: the loss rate must be between 0.5 and 1 (0.5 = as strong as main)" % item)
        out.append((name.lower(), rate))
    out.sort(key=lambda t: -t[1])                         # easiest first
    if len({n for n, _ in out}) != len(out):
        raise SystemExit("two targets with the same name")
    return out


def cmd_table(args):
    print("main AI wins      Elo gap     main AI wins      Elo gap")
    rates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
    half = (len(rates) + 1) // 2
    for i in range(half):
        left = "%4.0f%%      %+8.0f" % (100 * rates[i], elo_gap(rates[i]))
        right = ("%4.0f%%      %+8.0f" % (100 * rates[i + half], elo_gap(rates[i + half]))) if i + half < len(rates) else ""
        print("   %-24s%s" % (left, right))


def cmd_ladder(args):
    floor = elo_gap(args.vs_random)
    rates = ladder(args.vs_random, args.tiers)
    print("main scores %.3f against random = %.0f Elo above it; %d tiers at equal steps of %.0f Elo:"
          % (args.vs_random, floor, args.tiers, floor / (args.tiers + 1)))
    names = ["easy", "medium", "tier3", "tier4", "tier5"]
    for name, rate in zip(names, rates):
        print("   %-7s loses %4.1f%% to main  (%.0f Elo below it)" % (name, 100 * rate, elo_gap(rate)))
    print("   args: --targets " + " ".join("%s=%.2f" % (n, r) for n, r in zip(names, rates)))


def _snapshot_version(path):
    m = re.search(r"iter_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def cmd_scan(args):
    targets = parse_targets(args.targets)
    main_real = os.path.realpath(args.main)
    snaps = sorted((p for p in glob.glob(args.snapshots) if os.path.realpath(p) != main_real), key=_snapshot_version)
    snaps = snaps[::max(1, args.every)]
    if not snaps:
        raise SystemExit("no snapshots match %r" % args.snapshots)
    print("%d snapshots vs %s, %d games each" % (len(snaps), args.main, args.games), flush=True)
    rows = []
    done = set()                                          # targets that are settled (installed, or matched without --install)

    def settle(name, target, path, m):
        """A snapshot within the tolerance of ``target``: confirm it with more games and install it."""
        if not args.install:
            print("   -> %s: %s is within %.0f points of %.0f%%" % (name, os.path.relpath(path, ROOT), 100 * args.tol, 100 * target))
            return True
        v = match(path, args.main, args.verify_games, args.match_workers, args.match_seed + 7_000_001, args.max_plies,
                  args.greedy, args.score)
        print("   re-measured: " + describe(v), flush=True)
        if abs(v["rate"] - target) > 2 * args.tol:
            print("   too far from the target after all; scanning on")
            return False
        install(name, path, tier_info(v, target, _snapshot_version(path)), args.main)
        return True

    for i, path in enumerate(snaps):
        m = match(path, args.main, args.games, args.match_workers, args.match_seed + 100_003 * i, args.max_plies,
                  args.greedy, args.score)
        rows.append((path, m))
        print("  %-34s %s" % (os.path.relpath(path, ROOT), describe(m)), flush=True)
        for name, target in targets:                      # a hit is installed at once; no need to scan the rest for it
            if name not in done and abs(m["rate"] - target) <= args.tol:
                print("\n%s (target %.0f%%): %s is on target" % (name, 100 * target, os.path.relpath(path, ROOT)))
                if settle(name, target, path, m):
                    done.add(name)
        if len(done) == len(targets):
            print("\nall targets done; %d of %d snapshots scanned" % (i + 1, len(snaps)))
            return
    for name, target in targets:                          # nothing hit within the tolerance: the closest one, if it is near
        if name in done:
            continue
        path, m = min(rows, key=lambda r: abs(r[1]["rate"] - target))
        print("\n%s (target %.0f%%): closest is %s at %.1f%% [%.1f-%.1f]"
              % (name, 100 * target, os.path.relpath(path, ROOT), 100 * m["rate"], 100 * m["lo"], 100 * m["hi"]))
        print("   (more than %.0f points from the target: the snapshots are too far apart here; "
              "use `train` for a finer fit)" % (100 * args.tol))
        if args.install and abs(m["rate"] - target) <= 2 * args.tol:
            settle(name, target, path, m)


def _load_state(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"found": {}, "history": [], "iters": 0}


def _checkpoint_version(path):
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=True)["version"])


def cmd_train(args, passthrough):
    targets = parse_targets(args.targets)
    run_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(os.path.join(run_dir, "cands"), exist_ok=True)
    ref = os.path.join(run_dir, "main_ref.pt")
    if not os.path.exists(ref):
        if not os.path.exists(args.main):
            raise SystemExit("no such checkpoint: %s" % args.main)
        shutil.copy2(args.main, ref)                      # frozen: the yardstick must not move while we train
    latest = os.path.join(run_dir, "latest.pt")
    state_path = os.path.join(run_dir, "tiers_state.json")
    state = _load_state(state_path)
    seed_base = args.match_seed + 31 * len(state["history"])
    verified = {}                                         # candidate path -> verification match
    print("tiers %s against %s (%s games per check, %s to verify); main = %s"
          % (", ".join("%s %.0f%%" % (n, 100 * r) for n, r in targets), ref, args.games, args.verify_games, args.main),
          flush=True)

    def remaining():
        return [(n, r) for n, r in targets if n not in state["found"]]

    while remaining() and state["iters"] < args.max_iters:
        cmd = [sys.executable, os.path.join(ROOT, "train.py")] + passthrough + \
              ["--name", args.name, "--iters", str(args.chunk), "--max-plies", str(args.max_plies)]
        if os.path.exists(latest):
            cmd.append("--resume")
        print("\n$ " + " ".join(cmd), flush=True)
        try:
            rc = subprocess.call(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            raise SystemExit("\ninterrupted; run the same command again to carry on")
        if rc != 0:
            raise SystemExit("train.py exited with status %d" % rc)
        state["iters"] += args.chunk
        version = _checkpoint_version(latest)
        cand = os.path.join(run_dir, "cands", "v%05d.pt" % version)
        shutil.copy2(latest, cand)
        m = match(cand, ref, args.games, args.match_workers, seed_base + 100_003 * len(state["history"]),
                  args.max_plies, args.greedy, args.score)
        prev = state["history"][-1]["cand"] if state["history"] else None
        state["history"].append({"version": version, "cand": cand, "rate": m["rate"], "lo": m["lo"], "hi": m["hi"],
                                 "games": m["games"], "draws": m["draws"]})
        _write_json(state_path, state)
        print("\n[v%d] %s   (next target: %s %.0f%%)" % (version, describe(m), remaining()[0][0],
                                                      100 * remaining()[0][1]), flush=True)

        for name, target in remaining():
            if m["rate"] > target:
                break                                     # still too weak for this one, so for the rest too
            pool = [c for c in (prev, cand) if c and os.path.exists(c)]
            results = []
            for c in pool:
                if c not in verified:
                    verified[c] = match(c, ref, args.verify_games, args.match_workers,
                                        args.match_seed + 7_000_001 + 11 * _snapshot_version_of(c), args.max_plies,
                                        args.greedy, args.score)
                results.append((c, verified[c]))
                print("   verify %-12s %s" % (os.path.basename(c), describe(verified[c])), flush=True)
            ok = [(c, v) for c, v in results if judge(v["rate"], target, args.tol) == "ok"]
            if ok:
                best = min(ok, key=lambda r: abs(r[1]["rate"] - target))
            elif all(judge(v["rate"], target, args.tol) == "weak" for _, v in results):
                print("   %s: only noise took it below %.0f%%; training on" % (name, 100 * target), flush=True)
                break
            else:
                best = min(results, key=lambda r: abs(r[1]["rate"] - target))
                print("   %s: training stepped over the target (try a smaller --chunk); taking the closest" % name,
                      flush=True)
            c, v = best
            state["found"][name] = {"cand": c, "rate": v["rate"], "target": target}
            _write_json(state_path, state)
            install(name, c, tier_info(v, target, _snapshot_version_of(c)), ref)

    if remaining():
        print("\nstopped at %d iterations (--max-iters) without reaching: %s"
              % (state["iters"], ", ".join(n for n, _ in remaining())))
        print("rerun with a higher --max-iters to continue")
    else:
        print("\nall tiers installed:")
        for name, target in targets:
            f = state["found"][name]
            print("   %-8s target %.0f%%   measured %.1f%%   %s" % (name, 100 * target, 100 * f["rate"],
                                                                 os.path.relpath(f["cand"], ROOT)))


def _snapshot_version_of(path):
    m = re.search(r"v(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else _snapshot_version(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("table", help="loss rate -> Elo", allow_abbrev=False).set_defaults(fn=cmd_table)
    p = sub.add_parser("ladder", help="suggest loss rates", allow_abbrev=False)
    p.add_argument("--vs-random", type=float, required=True,
                   help="the main AI's score against a random player (train.py's EVAL line, e.g. 0.95)")
    p.add_argument("--tiers", type=int, default=2)
    p.set_defaults(fn=cmd_ladder)

    def common(p):
        p.add_argument("--main", required=True, help="the main AI's checkpoint, e.g. runs/run4/latest.pt")
        p.add_argument("--targets", nargs="+", required=True, metavar="NAME=RATE",
                       help="e.g. easy=0.85 medium=0.70: the share of games each tier should lose to main")
        p.add_argument("--games", type=int, default=160, help="games per measurement (default 160: +/- ~6 points)")
        p.add_argument("--verify-games", type=int, default=600, help="games to confirm a pick (default 600: +/- ~4)")
        p.add_argument("--tol", type=float, default=0.03, help="accept a tier within this of its target (default 0.03)")
        p.add_argument("--match-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                       help="processes for the matches (they run between training chunks, never alongside)")
        p.add_argument("--max-plies", type=int, default=300)
        p.add_argument("--greedy", action="store_true", help="both sides play their single likeliest move")
        p.add_argument("--score", action="store_true", help="count a draw as half a loss")
        p.add_argument("--match-seed", type=int, default=4242, help="seed of the match games")

    p = sub.add_parser("scan", help="measure existing snapshots against main", allow_abbrev=False)
    common(p)
    p.add_argument("--snapshots", default=os.path.join("runs", "*", "iter_*.pt"),
                   help="glob of the snapshots to try (default runs/*/iter_*.pt; quote it)")
    p.add_argument("--every", type=int, default=1, help="only every Nth snapshot")
    p.add_argument("--install", action="store_true", help="re-measure and install the closest snapshot per target")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("train", help="train fresh networks until they sit at the targets", allow_abbrev=False)
    common(p)
    p.add_argument("--name", default="tiers1", help="the training run (runs/NAME/); also names its logs")
    p.add_argument("--chunk", type=int, default=20, help="train.py iterations between measurements (default 20)")
    p.add_argument("--max-iters", type=int, default=3000, help="give up after this many iterations in total")
    p.set_defaults(fn=cmd_train)

    args, rest = ap.parse_known_args()
    if args.cmd == "train":
        args.fn(args, rest)
    else:
        if rest:
            ap.error("unrecognised arguments: " + " ".join(rest))
        args.fn(args)


if __name__ == "__main__":
    main()
