#!/usr/bin/env python
"""Is the Intel GPU any use for training this network?  (Run from reverse-chess/.)

    python xpu_bench.py                  # xpu, then cpu
    python xpu_bench.py --cpu-threads 8  # CPU threads to compare against

It does NOT touch the training loop.  Each stage prints its time as soon as it
finishes, so if something hangs, the last line printed shows where.

The first xpu step may take a long time: on GPUs PyTorch doesn't ship
pre-built kernels for, they are compiled on first use.  Compiled kernels are
cached on disk (SYCL_CACHE_PERSISTENT=1 below), so run the script twice and
compare the "first step" line.
"""
import argparse
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


def train_step(model, opt, batch):
    logits, value = model(*batch)
    loss = value.pow(2).mean() + logits.pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def bench(dev, steps=8):
    print("\n== %s ==" % dev, flush=True)
    if dev == "xpu":
        stage("tiny op (device start-up)", dev,
              lambda: (torch.ones(4, 4, device="xpu") + 1).cpu())
        print("  device:", torch.xpu.get_device_name(0), flush=True)
    model = netlib.PolicyValueNet(64, 4, 128).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    batch = stage("make a batch", dev, lambda: make_batch(dev))

    def forward_only():
        with torch.no_grad():
            model(*batch)
    stage("forward only (first call)", dev, forward_only)
    stage("training step: first step", dev, lambda: train_step(model, opt, batch))

    times = []
    for _ in range(steps):
        t = time.time()
        train_step(model, opt, batch)
        sync(dev)
        times.append(time.time() - t)
    med = statistics.median(times)
    print("  %-38s%8.3fs   (per minibatch of %d decisions)" % ("training step: median of %d" % steps, med, BATCH))
    # one training iteration is ~8000 plies = ~31 minibatches x 2 epochs
    print("  => estimated learn time per iteration: ~%.0fs" % (med * 31 * 2), flush=True)
    return med


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--skip-xpu", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    print("torch", torch.__version__, flush=True)
    results = {}
    if not args.skip_xpu:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            results["xpu"] = bench("xpu")
        else:
            print("torch.xpu is not available here")
    torch.set_num_threads(args.cpu_threads)
    print("\n(cpu with %d threads)" % args.cpu_threads)
    results["cpu"] = bench("cpu")
    if "xpu" in results:
        print("\nxpu is %.1fx %s than cpu per step" % (
            max(results["cpu"] / results["xpu"], results["xpu"] / results["cpu"]),
            "faster" if results["xpu"] < results["cpu"] else "SLOWER"))
