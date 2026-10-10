#!/usr/bin/env python
"""Run train.py's real PPO update on made-up data, on the Intel GPU.  (Run from reverse-chess/.)

    python xpu_ppo_probe.py                 # xpu, 3 updates, then the same on cpu
    python xpu_ppo_probe.py --device cpu    # only the cpu

While it runs, a stack trace is printed every --watch seconds (it does NOT stop
the run).  If the same line shows up every time, that line is where it is
stuck; if the lines keep changing, it is just slowly compiling GPU kernels.
Update 1 includes all first-time compiles; later updates should be much faster.
"""
import argparse
import faulthandler
import os
import sys
import time

os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import net as netlib  # noqa: E402
import train  # noqa: E402
from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO  # noqa: E402


def synthetic_data(n_dec, rng):
    """Same layout as train.assemble(): ~8000 decisions, 20-110 legal moves each."""
    counts = rng.integers(20, 110, n_dec).astype(np.int64)
    total = int(counts.sum())
    comps = np.stack([rng.integers(0, 64, total), rng.integers(0, 64, total),
                      rng.integers(0, 2, total), rng.integers(0, _NUM_GIFT, total),
                      rng.integers(0, 2, total), rng.integers(0, _NUM_PROMO, total)], 1).astype(np.uint8)
    return {
        "boards": rng.integers(0, 2, (n_dec, NUM_BOARD_PLANES, 8, 8)).astype(np.uint8),
        "scalars": rng.random((n_dec, NUM_SCALARS)).astype(np.float32),
        "comps": comps, "counts": counts, "starts": np.cumsum(counts) - counts,
        "chosen": (rng.random(n_dec) * counts).astype(np.int64),
        "logp": (-np.log(counts)).astype(np.float32),
        "value": rng.normal(0, 0.3, n_dec).astype(np.float32),
        "ret": rng.normal(0, 0.5, n_dec).astype(np.float32),
        "adv": rng.normal(0, 1.0, n_dec).astype(np.float32),
    }


def run(device, args, targs):
    print("\n== %s ==" % device, flush=True)
    rng = np.random.default_rng(0)
    data = synthetic_data(args.decisions, rng)
    model = netlib.PolicyValueNet(64, 4, 128).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    for i in range(1, args.updates + 1):
        faulthandler.dump_traceback_later(args.watch, repeat=True)
        t = time.time()
        stats = train.ppo_update(model, opt, data, targs, device, rng)
        if device == "xpu":
            torch.xpu.synchronize()
        faulthandler.cancel_dump_traceback_later()
        print("  update %d: %6.1fs   (%s)" % (i, time.time() - t,
              " ".join("%s %.3f" % kv for kv in stats.items())), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="both", choices=["xpu", "cpu", "both"])
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--decisions", type=int, default=8000)
    ap.add_argument("--watch", type=int, default=45, help="print a stack trace every N seconds")
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]                 # let train.parse_args() give us the real defaults
    targs = train.parse_args()
    torch.set_num_threads(8)
    devices = ["xpu", "cpu"] if args.device == "both" else [args.device]
    for d in devices:
        run(d, args, targs)
