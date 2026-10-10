#!/usr/bin/env python
"""Is the Intel GPU any use for training this network?  (Run from reverse-chess/.)

    python xpu_bench.py                  # xpu, then cpu
    python xpu_bench.py --cpu-threads 8  # CPU threads to compare against
    python xpu_bench.py --fp16 --skip-cpu   # A/B: xpu fp32 vs xpu fp16 autocast

--fp16 runs the same training step a second time with the forward pass under
torch.autocast("xpu", dtype=torch.float16) (weights, Adam state and the loss
stay fp32), prints the median step time of both, and runs a numerics check
(fp32 vs fp16 outputs and gradients on the same batch).  --loss-scale picks
how fp16 gradient underflow is handled:

    static   (default) multiply the loss by a constant, divide the grads back.
             Needs no fp64 on the GPU.
    dynamic  torch.amp.GradScaler("xpu").  PyTorch's docs say this needs fp64
             hardware support and should be disabled on Arc A-series, so it is
             opt-in.
    none     no scaling.

It does NOT touch the training loop.  Each stage prints its time as soon as it
finishes, so if something hangs, the last line printed shows where.

The first xpu step may take a long time: on GPUs PyTorch doesn't ship
pre-built kernels for, they are compiled on first use.  Compiled kernels are
cached on disk (SYCL_CACHE_PERSISTENT=1 below), so run the script twice and
compare the "first step" line.
"""
import argparse
import contextlib
import os
import statistics
import time

os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

import torch  # noqa: E402

import net as netlib  # noqa: E402
from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO  # noqa: E402

BATCH = 256        # decisions per minibatch (train.py --minibatch)
LEGAL = 62         # average legal actions per decision (from your logs)


def sync(dev):
    if dev == "xpu":
        torch.xpu.synchronize()


def stage(label, dev, fn):
    print("  %-38s" % label, end="", flush=True)
    t = time.time()
    out = fn()
    sync(dev)
    print("%8.2fs" % (time.time() - t), flush=True)
    return out


def make_batch(dev):
    """Random data shaped like a real training minibatch."""
    n = BATCH * LEGAL
    owner = torch.repeat_interleave(torch.arange(BATCH), LEGAL)
    comps = torch.stack([
        torch.randint(0, 64, (n,)), torch.randint(0, 64, (n,)),
        torch.randint(0, 2, (n,)), torch.randint(0, _NUM_GIFT, (n,)),
        torch.randint(0, 2, (n,)), torch.randint(0, _NUM_PROMO, (n,))], dim=1)
    board = torch.rand(BATCH, NUM_BOARD_PLANES, 8, 8).round()
    scalars = torch.rand(BATCH, NUM_SCALARS)
    return [t.to(dev) for t in (board, scalars, owner, comps)]


STATIC_SCALE = 1024.0


def amp_ctx(dev, fp16):
    if not fp16:
        return contextlib.nullcontext()
    return torch.autocast(device_type=dev, dtype=torch.float16)


def forward(model, batch, dev, fp16, scope):
    """scope "all": the whole network under autocast.  scope "trunk": only the
    conv trunk is fp16; the pointer head (gathers, embeddings, scorer) runs in
    fp32, so gather/embedding backward never accumulates in half precision."""
    board, scalars, owner, comps = batch
    if fp16 and scope == "trunk":
        with amp_ctx(dev, True):
            x = model.trunk(board, scalars)
        x = x.float()
        g, value = model.global_features(x)
        return model.action_logits(x, g, owner, comps), value
    with amp_ctx(dev, fp16):
        return model(*batch)


def make_stepper(model, opt, dev, fp16, loss_scale, scope="trunk"):
    """Returns step(batch).  Only the forward pass is under autocast; the loss,
    backward and optimizer step are outside it (as the PyTorch AMP recipe says)."""
    scaler = None
    if fp16 and loss_scale == "dynamic":
        scaler = torch.amp.GradScaler(device=dev)
    static = fp16 and loss_scale == "static"
    params = [p for p in model.parameters() if p.requires_grad]

    def step(batch):
        logits, value = forward(model, batch, dev, fp16, scope)
        loss = value.float().pow(2).mean() + logits.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        elif static:
            (loss * STATIC_SCALE).backward()
            grads = [p.grad for p in params if p.grad is not None]
            torch._foreach_mul_(grads, 1.0 / STATIC_SCALE)
            opt.step()
        else:
            loss.backward()
            opt.step()
    return step


def bench(dev, label=None, fp16=False, loss_scale="static", steps=8, epochs=2, warmup=3, scope="trunk"):
    label = label or dev
    print("\n== %s ==" % label, flush=True)
    if dev == "xpu":
        stage("tiny op (device start-up)", dev,
              lambda: (torch.ones(4, 4, device="xpu") + 1).cpu())
        print("  device:", torch.xpu.get_device_name(0), flush=True)
    model = netlib.PolicyValueNet(64, 4, 128).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    batch = stage("make a batch", dev, lambda: make_batch(dev))
    step = make_stepper(model, opt, dev, fp16, loss_scale, scope)

    def forward_only():
        with torch.no_grad():
            forward(model, batch, dev, fp16, scope)
    stage("forward only (first call)", dev, forward_only)
    stage("training step: first step", dev, lambda: step(batch))

    for _ in range(warmup):      # lazily-compiled kernels and GPU clock ramp-up
        step(batch)
    sync(dev)
    times = []
    for _ in range(steps):
        t = time.time()
        step(batch)
        sync(dev)
        times.append(time.time() - t)
    med = statistics.median(times)
    print("  %-38s%8.3fs   (per minibatch of %d decisions)" % ("training step: median of %d" % steps, med, BATCH))
    print("  %-38s %.3fs min, %.3fs max" % ("", min(times), max(times)))
    # one training iteration is ~8000 plies = ~31 minibatches x epochs
    print("  => estimated learn time per iteration: ~%.0fs from the median, ~%.0fs from the min  (%d epochs)" % (
        med * 31 * epochs, min(times) * 31 * epochs, epochs), flush=True)
    return med


def flat_grads(model):
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters()
                      if p.grad is not None])


def check_numerics(dev, loss_scale, scope="trunk"):
    """Same weights, same batch: fp32 vs fp16-autocast outputs and gradients.

    A smoke test only: random init weights and random data say nothing about how
    training behaves, they just show whether fp16 is wildly off."""
    print("\n== numerics: %s fp32 vs fp16 autocast (loss-scale %s) ==" % (dev, loss_scale), flush=True)
    torch.manual_seed(1)
    model = netlib.PolicyValueNet(64, 4, 128).to(dev).train()
    # the real init makes the policy head ~0 (uniform policy); widen it so the
    # logits are not all ~1e-2 and the comparison is not trivially small
    torch.nn.init.normal_(model.act_out.weight, std=0.3)
    torch.nn.init.normal_(model.value_out.weight, std=0.3)
    batch = make_batch(dev)
    scale = STATIC_SCALE if loss_scale in ("static", "dynamic") else 1.0

    def run(fp16):
        model.zero_grad(set_to_none=True)
        logits, value = forward(model, batch, dev, fp16, scope)
        loss = value.float().pow(2).mean() + logits.float().pow(2).mean()
        (loss * (scale if fp16 else 1.0)).backward()
        g = flat_grads(model) / (scale if fp16 else 1.0)
        return logits.detach().float(), value.detach().float(), g

    l32, v32, g32 = run(False)
    l16, v16, g16 = run(True)
    sync(dev)
    cos = torch.nn.functional.cosine_similarity(g32, g16, dim=0).item()
    print("  logits: max|diff| %.2e   (fp32 logits std %.2e)" % ((l32 - l16).abs().max().item(), l32.std().item()))
    print("  value : max|diff| %.2e" % (v32 - v16).abs().max().item())
    print("  grads : cosine %.6f   rel. error %.2e   finite: %s   (fp32 norm %.3e, fp16 norm %.3e)" % (
        cos, ((g32 - g16).norm() / g32.norm()).item(), bool(torch.isfinite(g16).all().item()),
        g32.norm().item(), g16.norm().item()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--skip-xpu", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--fp16", action="store_true",
                    help="also benchmark the xpu with fp16 autocast and compare to fp32")
    ap.add_argument("--loss-scale", choices=["static", "dynamic", "none"], default="static",
                    help="fp16 gradient scaling (see the module docstring)")
    ap.add_argument("--fp16-scope", choices=["trunk", "all"], default="trunk",
                    help="trunk: autocast only the conv trunk, policy head stays fp32 (default); "
                         "all: autocast the whole network")
    ap.add_argument("--steps", type=int, default=12, help="timed steps per variant")
    ap.add_argument("--warmup", type=int, default=3, help="untimed steps after the first one")
    ap.add_argument("--epochs", type=int, default=2, help="PPO epochs, for the learn-time estimate only")
    args = ap.parse_args()
    torch.manual_seed(0)
    print("torch", torch.__version__, flush=True)
    results = {}
    on_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not args.skip_xpu:
        if on_xpu:
            results["xpu"] = bench("xpu", steps=args.steps, epochs=args.epochs, warmup=args.warmup)
            if args.fp16:
                try:
                    results["xpu fp16"] = bench("xpu", label="xpu fp16 (%s, loss-scale %s)" % (args.fp16_scope, args.loss_scale),
                                                fp16=True, loss_scale=args.loss_scale, scope=args.fp16_scope,
                                                steps=args.steps, epochs=args.epochs, warmup=args.warmup)
                    check_numerics("xpu", args.loss_scale, args.fp16_scope)
                except Exception as e:  # e.g. UR_RESULT_ERROR_* from an unsupported kernel
                    print("\nfp16 on the xpu FAILED: %s: %s" % (type(e).__name__, e), flush=True)
        else:
            print("torch.xpu is not available here")
    if not args.skip_cpu:
        torch.set_num_threads(args.cpu_threads)
        print("\n(cpu with %d threads)" % args.cpu_threads)
        results["cpu"] = bench("cpu", steps=args.steps, epochs=args.epochs, warmup=args.warmup)
    print()
    if "xpu" in results and "cpu" in results:
        print("xpu fp32 is %.1fx %s than cpu per step" % (
            max(results["cpu"] / results["xpu"], results["xpu"] / results["cpu"]),
            "faster" if results["xpu"] < results["cpu"] else "SLOWER"))
    if "xpu" in results and "xpu fp16" in results:
        r = results["xpu"] / results["xpu fp16"]
        print("xpu fp16 is %.2fx %s than xpu fp32 per step (%.3fs -> %.3fs)" % (
            r if r >= 1 else 1 / r, "faster" if r >= 1 else "SLOWER", results["xpu"], results["xpu fp16"]))
