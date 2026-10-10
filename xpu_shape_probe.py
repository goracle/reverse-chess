#!/usr/bin/env python
"""Does a changing batch SHAPE make the learner slow?  (Run from reverse-chess/.)

    python xpu_shape_probe.py                  # xpu if available
    python xpu_shape_probe.py --device cpu
    python xpu_shape_probe.py --legal 129      # average legal moves per decision (your current logs)

Why: in train.py every minibatch has a different number of legal actions (the
length of owner / comps / logits) and a different amax (width of the padded
softmax).  xpu_bench.py repeats ONE batch, so it cannot see any cost that is paid
per new shape (e.g. oneDNN primitive creation / kernel JIT on the host).  The
training log shows the learner is ~100% host time ("dispatch" ~0.45 s/step,
"blocked on xpu" ~0) so the GPU is idle, waiting for the host.

Three variants, each run for several passes over the SAME set of n-shapes batches:

  fixed    one batch repeated                       (what xpu_bench measures)
  varying  n-shapes different batches, as in train.py
  padded   same batches, but the action dimension is padded to a multiple of
           --pad and amax to a multiple of 32, so only a few distinct shapes exist
           (logits are sliced back, so the maths is identical)

How to read it (ms per step, one number per pass):

  varying: pass 1 slow, passes 2+ fast      => the cost is per NEW shape.  Padding
                                               (the "padded" row) should remove it.
  varying slow in every pass, fixed fast    => shapes are not the cache key; look at
                                               the stage breakdown below instead.
  everything slow incl. fixed               => a slow op regardless of shape; the
                                               stage breakdown says which stage.

Then a per-stage breakdown (host waits for the device after every stage, so stage
times add up to more than a normal step): trunk fwd | head fwd | loss | backward |
clip | Adam, once on shapes never seen before ("cold") and once on seen ones ("warm").
"""
import argparse
import statistics
import time

import numpy as np
import torch

import net as netlib
import train
from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO


def approx_flops(model, b):
    """Rough FLOPs of one training step (fwd + bwd = 3x fwd; 1 MAC = 2 FLOPs); batch b."""
    bs = b["board"].shape[0]
    macs = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            macs += m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * 64 * bs
    c, h = model.cfg["channels"], model.cfg["hidden"]
    macs += model.glob_fc.in_features * model.glob_fc.out_features * bs      # global features
    macs += 2 * 64 * c * h * bs + h * h * bs                                 # src/dst/glob projections
    macs += b["n"] * (h * (h // 2) + h // 2)                                 # per-action scorer
    return 6.0 * macs


def sync(dev):
    if dev == "xpu":
        torch.xpu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def make_batch(rng, dev, batch, legal, pad, pad_amax=32, realistic=False):
    """A random minibatch shaped like train.collate's output (counts vary per decision)."""
    counts = rng.integers(max(1, legal // 4), 2 * legal - legal // 4, size=batch)
    n = int(counts.sum())
    amax = int(counts.max())
    owner = np.repeat(np.arange(batch), counts)
    src = rng.integers(0, 64, n)
    demote, ep = rng.integers(0, 2, n), rng.integers(0, 2, n)
    gift, promo = rng.integers(0, _NUM_GIFT, n), rng.integers(0, _NUM_PROMO, n)
    if realistic:
        # Real batches are collision-heavy: the actions of one decision come from a handful
        # of source squares, and nearly all of them have no gift / promotion / flags.  That
        # matters because the backward of gathers and embeddings is a scatter-add.
        pieces = rng.integers(0, 64, (batch, 8))
        src = pieces[owner, rng.integers(0, 8, n)]
        rare = lambda a: np.where(rng.random(n) < 0.9, 0, a)      # noqa: E731
        demote, ep, gift, promo = rare(demote), rare(ep), rare(gift), rare(promo)
    comps = np.stack([src, rng.integers(0, 64, n), demote, gift, ep, promo], axis=1)

    def t(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=dev)

    b = {
        "board": t(rng.integers(0, 2, (batch, NUM_BOARD_PLANES, 8, 8)), torch.float32),
        "scalars": t(rng.random((batch, NUM_SCALARS)), torch.float32),
        "owner": t(owner, torch.long), "comps": t(comps, torch.long),
        "counts": t(counts, torch.long), "chosen": t(np.zeros(batch), torch.long),
        "adv": t(rng.standard_normal(batch), torch.float32),
        "n": n, "amax": amax,
    }
    n_pad = -(-n // pad) * pad
    b["n_pad"] = n_pad
    b["amax_pad"] = -(-amax // pad_amax) * pad_amax
    b["owner_pad"] = t(np.concatenate([owner, np.zeros(n_pad - n, dtype=np.int64)]), torch.long)
    b["comps_pad"] = t(np.concatenate([comps, np.zeros((n_pad - n, 6), dtype=np.int64)]), torch.long)
    return b


def loss_fn(model, b, padded, stages=None, dev="cpu"):
    """Same structure as train.ppo_update's loss; optionally timed per stage."""
    def stage(name, fn):
        if stages is None:
            return fn()
        sync(dev)
        t0 = time.perf_counter()
        out = fn()
        sync(dev)
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - t0
        return out

    owner = b["owner_pad"] if padded else b["owner"]
    comps = b["comps_pad"] if padded else b["comps"]
    amax = b["amax_pad"] if padded else b["amax"]
    x = stage("trunk fwd", lambda: model.trunk(b["board"], b["scalars"]))

    def head():
        g, value = model.global_features(x)
        logits = model.action_logits(x, g, owner, comps)
        return (logits[:b["n"]] if padded else logits), value
    logits, value = stage("head fwd", head)

    def loss():
        logp_all, mask = train.per_decision_logp(logits, b["owner"], b["counts"], amax)
        rows = torch.arange(b["board"].shape[0], device=logits.device)
        logp = logp_all[rows, b["chosen"]]
        probs = logp_all.exp()
        ent = -(probs * logp_all.masked_fill(~mask, 0.0)).sum(1).mean()
        return -(logp * b["adv"]).mean() + value.pow(2).mean() - 0.01 * ent
    return stage("loss", loss)


def step(model, opt, b, padded, stages=None, dev="cpu"):
    def st(name, fn):
        if stages is None:
            return fn()
        sync(dev)
        t0 = time.perf_counter()
        out = fn()
        sync(dev)
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - t0
        return out
    loss = loss_fn(model, b, padded, stages, dev)
    opt.zero_grad(set_to_none=True)
    st("backward", loss.backward)
    st("clip", lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
    st("Adam", opt.step)


def run_variant(name, model, opt, batches, passes, padded, dev):
    print("  %-8s" % name, end="", flush=True)
    per_pass = []
    for _ in range(passes):
        sync(dev)
        t0 = time.perf_counter()
        for b in batches:
            step(model, opt, b, padded)
        sync(dev)
        per_pass.append(1000 * (time.perf_counter() - t0) / len(batches))
        print("  pass %d: %7.0f ms/step" % (len(per_pass), per_pass[-1]), end="", flush=True)
    print(flush=True)
    return per_pass


def breakdown(label, model, opt, batches, padded, dev):
    stages = {}
    for b in batches:
        step(model, opt, b, padded, stages, dev)
    total = sum(stages.values())
    print("  %-26s" % label + " | ".join("%s %5.0f" % (k, 1000 * v / len(batches)) for k, v in stages.items())
          + "   (ms/step, total %.0f)" % (1000 * total / len(batches)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=256, help="decisions per minibatch (train.py --minibatch)")
    ap.add_argument("--legal", type=int, default=129, help="average legal actions per decision")
    ap.add_argument("--n-shapes", type=int, default=8, help="distinct batches cycled through")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--channels-last", action="store_true",
                    help="run the convs in channels_last memory format (often faster on Intel GPUs; may not be here)")
    ap.add_argument("--realistic", action="store_true",
                    help="collision-heavy indices like real batches (few source squares per decision, "
                         "mostly 'no gift'); uniform random indices otherwise")
    ap.add_argument("--pad", type=int, default=4096, help="padded variant: round the action count up to a multiple of this")
    args = ap.parse_args()

    dev = train.pick_device(args.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("torch", torch.__version__, " device", dev, flush=True)
    model = netlib.PolicyValueNet(64, 4, 128).to(dev).train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        model.stem.register_forward_pre_hook(
            lambda m, inp: (inp[0].contiguous(memory_format=torch.channels_last),))
        print("  memory format: channels_last", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    batches = [make_batch(rng, dev, args.batch, args.legal, args.pad, realistic=args.realistic) for _ in range(args.n_shapes)]
    print("  action counts per batch:", [b["n"] for b in batches], " amax:", [b["amax"] for b in batches])
    print("  padded to:              ", sorted({b["n_pad"] for b in batches}),
          " amax:", sorted({b["amax_pad"] for b in batches}), flush=True)

    print("\nwarm-up (first steps can be very slow: kernels / primitives are compiled lazily) ...", flush=True)
    t0 = time.perf_counter()
    step(model, opt, batches[0], False)
    sync(dev)
    print("  first step: %.1fs" % (time.perf_counter() - t0), flush=True)

    print("\nms per training step, by pass over the same %d batches:" % args.n_shapes, flush=True)
    fixed = run_variant("fixed", model, opt, [batches[0]] * args.n_shapes, args.passes, False, dev)
    fl = approx_flops(model, batches[0])
    print("           ~%.0f GFLOP per step => ~%.0f GFLOP/s achieved (Iris Xe fp32 peak is roughly 2000)"
          % (fl / 1e9, fl / 1e9 / (min(fixed) / 1000)), flush=True)
    run_variant("varying", model, opt, batches, args.passes, False, dev)
    run_variant("padded", model, opt, batches, args.passes, True, dev)

    print("\nper-stage breakdown (varying shapes, unpadded):", flush=True)
    fresh = [make_batch(rng, dev, args.batch, args.legal, args.pad, realistic=args.realistic) for _ in range(args.n_shapes)]
    breakdown("cold (unseen shapes)", model, opt, fresh, False, dev)
    breakdown("warm (same shapes again)", model, opt, fresh, False, dev)


if __name__ == "__main__":
    main()
