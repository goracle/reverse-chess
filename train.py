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
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

import net as netlib
import returns
import rollout


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
    return {
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
    }


def make_batches(counts, minibatch, max_actions, rng):
    """Shuffled batches of decision indices: at most ``minibatch`` decisions and
    (roughly) ``max_actions`` legal actions each, which bounds memory."""
    batch, n_actions = [], 0
    for i in rng.permutation(len(counts)):
        batch.append(i)
        n_actions += counts[i]
        if len(batch) >= minibatch or n_actions >= max_actions:
            yield np.asarray(batch)
            batch, n_actions = [], 0
    if batch:
        yield np.asarray(batch)


def collate(data, adv, mb, device):
    """Turn a batch of decision indices into tensors."""
    counts = data["counts"][mb]
    comps = np.concatenate([data["comps"][s:s + n]
                            for s, n in zip(data["starts"][mb], counts)])
    owner = np.repeat(np.arange(len(mb)), counts)

    def t(array, dtype):
        return torch.as_tensor(array, dtype=dtype, device=device)

    return {
        "board": t(data["boards"][mb], torch.float32),
        "scalars": t(data["scalars"][mb], torch.float32),
        "comps": t(comps, torch.long),
        "owner": t(owner, torch.long),
        "counts": t(counts, torch.long),
        "chosen": t(data["chosen"][mb], torch.long),
        "old_logp": t(data["logp"][mb], torch.float32),
        "ret": t(data["ret"][mb], torch.float32),
        "adv": t(adv[mb], torch.float32),
    }


def per_decision_logp(logits, owner, counts):
    """Log-softmax of the flat legal-action logits within each decision.

    Returns ``(logp (B, Amax), mask (B, Amax))``; padded slots hold a huge
    negative number and are False in the mask.
    """
    b = counts.shape[0]
    amax = int(counts.max())
    offsets = torch.cumsum(counts, 0) - counts
    rank = torch.arange(owner.shape[0], device=owner.device) - offsets[owner]
    padded = torch.full((b, amax), -1e9, device=logits.device, dtype=logits.dtype)
    padded = padded.index_put((owner, rank), logits)
    mask = torch.arange(amax, device=owner.device)[None, :] < counts[:, None]
    return torch.log_softmax(padded, dim=1), mask


# --------------------------------------------------------------------------
# PPO update
# --------------------------------------------------------------------------

def ppo_update(model, opt, data, args, device, rng):
    """A few epochs of PPO on one iteration's data; returns averaged statistics."""
    adv = data["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {k: [] for k in ("pg", "vf", "ent", "kl", "clipfrac")}
    stop = False
    for _ in range(args.epochs):
        for mb in make_batches(data["counts"], args.minibatch, args.max_actions, rng):
            b = collate(data, adv, mb, device)
            logits, value = model(b["board"], b["scalars"], b["owner"], b["comps"])
            logp_all, mask = per_decision_logp(logits, b["owner"], b["counts"])
            rows = torch.arange(len(mb), device=device)
            logp = logp_all[rows, b["chosen"]]

            ratio = torch.exp(logp - b["old_logp"])
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
            pg_loss = -torch.min(ratio * b["adv"], clipped * b["adv"]).mean()
            vf_loss = torch.mean((value - b["ret"]) ** 2)
            probs = logp_all.exp()                       # exactly 0 in padded slots
            entropy = -(probs * logp_all.masked_fill(~mask, 0.0)).sum(1).mean()
            loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            with torch.no_grad():
                kl = float((b["old_logp"] - logp).mean())
                stats["pg"].append(float(pg_loss))
                stats["vf"].append(float(vf_loss))
                stats["ent"].append(float(entropy))
                stats["kl"].append(kl)
                stats["clipfrac"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
            if kl > 1.5 * args.target_kl:     # policy moved too far; stop this iteration
                stop = True
                break
        if stop:
            break
    return {k: float(np.mean(v)) for k, v in stats.items()}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def pick_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
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
    g.add_argument("--device", default="auto", help="auto | cpu | cuda (learner only)")
    g.add_argument("--move-cache", type=int, default=50_000,
                   help="engine move-cache entries per worker (memory knob)")
    g = ap.add_argument_group("self-play")
    g.add_argument("--plies-per-iter", type=int, default=8000)
    g.add_argument("--max-plies", type=int, default=300, help="draw after this many plies")
    g.add_argument("--draw-value", type=float, default=0.0,
                   help="reward for a draw (try -0.1 if games keep drawing)")
    g.add_argument("--max-stale", type=int, default=2)
    g.add_argument("--shape-coef", type=float, default=0.0,
                   help="weight of potential-based progress shaping (0 = off; try 1.0 with "
                        "--gae-lambda 0.95).  See returns.py for why this is safe.")
    g.add_argument("--gamma", type=float, default=1.0, help="discount (1.0 keeps shaping exactly neutral)")
    g.add_argument("--gae-lambda", type=float, default=1.0,
                   help="GAE lambda; 1.0 = plain Monte-Carlo (old behaviour); 0.95 is typical "
                        "and is needed for shaping to have any effect")
    g.add_argument("--queue-factor", type=int, default=6,
                   help="games queued per worker; workers keep playing while the learner "
                        "trains, so this must be deep enough that they never run dry")
    g = ap.add_argument_group("network")
    g.add_argument("--channels", type=int, default=64)
    g.add_argument("--blocks", type=int, default=4)
    g.add_argument("--hidden", type=int, default=128)
    g = ap.add_argument_group("PPO")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--epochs", type=int, default=2)
    g.add_argument("--minibatch", type=int, default=256, help="decisions per batch")
    g.add_argument("--max-actions", type=int, default=60_000,
                   help="legal actions per batch (memory knob)")
    g.add_argument("--clip", type=float, default=0.2)
    g.add_argument("--vf-coef", type=float, default=0.5)
    g.add_argument("--ent-coef", type=float, default=0.01)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--target-kl", type=float, default=0.03)
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
    return args


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


def main():
    args = parse_args()
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
    print("device=%s  workers=%d  learner threads=%d  params=%.2fM  start version=%d"
          % (device, args.workers, args.learner_threads, n_params / 1e6, version), flush=True)

    seed_counter = [args.seed * 1_000_003 + version * 7919]

    def next_seed():
        seed_counter[0] += 1
        return seed_counter[0]

    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=rollout.init_worker,
        initargs=(latest, {"max_plies": args.max_plies}, args.move_cache))
    pending = set()
    t_start = time.time()
    log_file = open(log_path, "a", newline="")
    writer = None
    try:
        for it in range(args.iters):
            # ---- collect self-play games --------------------------------
            t0 = time.time()
            games, plies, stale = [], 0, 0
            while plies < args.plies_per_iter:
                while len(pending) < args.queue_factor * args.workers:   # keep every worker fed, even while learning
                    pending.add(pool.submit(rollout.selfplay_task, next_seed()))
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    result = fut.result()                    # re-raises worker errors
                    if version - result["version"] > args.max_stale:
                        stale += 1
                        continue
                    games.append(result)
                    plies += len(result["player"])
            t_collect = time.time() - t0

            # ---- learn ---------------------------------------------------
            t1 = time.time()
            data = assemble(games, args.draw_value, args.shape_coef, args.gamma, args.gae_lambda)
            model.train()
            stats = ppo_update(model, opt, data, args, device, rng)
            version += 1
            netlib.save_checkpoint(latest, model, version, opt.state_dict())
            if version % args.save_every == 0:
                netlib.save_checkpoint(os.path.join(run_dir, "iter_%05d.pt" % version),
                                       model, version, None)
            t_learn = time.time() - t1

            # ---- report ---------------------------------------------------
            n = len(games)
            winners = Counter(g["winner"] or "draw" for g in games)
            reasons = Counter(g["reason"][:64] for g in games)
            ret_var = float(np.var(data["ret"]))
            expl_var = (1 - float(np.var(data["ret"] - data["value"])) / ret_var
                        if ret_var > 1e-8 else float("nan"))
            row = {
                "version": version, "games": n, "plies": plies, "stale": stale,
                "black": winners["b"], "white": winners["w"], "draw": winners["draw"],
                "avg_plies": round(plies / n, 1),
                "avg_legal": round(float(data["counts"].mean()), 1),
                "plies_per_s": round(plies / max(t_collect, 1e-9), 1),
                **{k: round(v, 5) for k, v in stats.items()},
                "expl_var": round(expl_var, 3),
                "collect_s": round(t_collect, 1), "learn_s": round(t_learn, 1),
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }
            print("v%-4d %3d games %5d plies (b %d w %d d %d, avg %.0f plies, %.0f legal) | "
                  "%.0f plies/s | pg %+.3f vf %.3f ent %.2f kl %.4f clip %.2f ev %.2f | "
                  "collect %.0fs learn %.0fs%s"
                  % (version, n, plies, winners["b"], winners["w"], winners["draw"],
                     plies / n, data["counts"].mean(), plies / max(t_collect, 1e-9),
                     stats["pg"], stats["vf"], stats["ent"], stats["kl"],
                     stats["clipfrac"], expl_var, t_collect, t_learn,
                     ("  [%d stale dropped]" % stale) if stale else ""), flush=True)
            print("      ended by:", dict(reasons.most_common(8)), flush=True)

            if args.eval_every and version % args.eval_every == 0:
                w, d, l, score, ev = run_eval(pool, args, next_seed)
                print("      EVAL v%d vs random: +%d =%d -%d  score %.3f"
                      % (ev, w, d, l, score), flush=True)
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
