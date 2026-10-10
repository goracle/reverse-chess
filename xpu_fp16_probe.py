#!/usr/bin/env python
"""Which half-precision operations work on the Intel GPU?  (Run from reverse-chess/.)

    python xpu_fp16_probe.py                      # float16
    python xpu_fp16_probe.py --dtype bfloat16     # same, bf16
    python xpu_fp16_probe.py --timeout 180        # first runs compile kernels

Companion to xpu_probe.py: every stage runs in its own subprocess, so a hang or
a crash shows up as HUNG / CRASHED and the next stage still runs.  Built to find
which op raised UR_RESULT_ERROR_INVALID_ARGUMENT in `xpu_bench.py --fp16`.
Stages 1-8 use plain half tensors, 9-12 use the real network under autocast.
Stage 7 is what autocast does to the policy head (fp16 gather backward); stage 8 is not,
because autocast keeps embedding weights in fp32.  Stage 12 is the --fp16-scope trunk
variant of xpu_bench.py: fp16 conv trunk, fp32 head.
"""
import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

B, LEGAL, C, H = 256, 62, 64, 128
N = B * LEGAL
NAMES = [
    "elementwise add + relu",
    "linear (C -> H) on (B*64, C)",
    "conv2d 3x3 C->C, forward",
    "conv2d 3x3 C->C, forward + backward",
    "group_norm, forward",
    "group_norm, forward + backward",
    "two-index gather a[owner, sq], fwd + bwd",
    "nn.Embedding with half weights, fwd + bwd",
    "network trunk under autocast, forward only",
    "FULL network under autocast, forward only (no_grad)",
    "FULL network under autocast, forward + backward",
    "trunk under autocast + fp32 head, fwd + bwd  (--fp16-scope trunk)",
]


def stages(dt):
    import torch
    import torch.nn.functional as F
    import net as netlib
    from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO
    dev = "xpu"
    rnd = lambda *s, grad=False: torch.rand(*s, device=dev, dtype=dt, requires_grad=grad)
    owner = torch.repeat_interleave(torch.arange(B), LEGAL).to(dev)
    sq = torch.randint(0, 64, (N,)).to(dev)
    model = netlib.PolicyValueNet(C, 4, H).to(dev).train()

    def batch():
        comps = torch.stack([sq, sq.flip(0), torch.randint(0, 2, (N,), device=dev),
                             torch.randint(0, _NUM_GIFT, (N,), device=dev),
                             torch.randint(0, 2, (N,), device=dev),
                             torch.randint(0, _NUM_PROMO, (N,), device=dev)], 1)
        return (torch.rand(B, NUM_BOARD_PLANES, 8, 8, device=dev).round(),
                torch.rand(B, NUM_SCALARS, device=dev), owner, comps)

    def auto():
        return torch.autocast(device_type=dev, dtype=dt)

    def elementwise():
        F.relu(rnd(B, C, 8, 8) + rnd(B, C, 8, 8))

    def linear():
        torch.nn.Linear(C, H).to(dev, dt)(rnd(B * 64, C))

    def conv_fwd():
        torch.nn.Conv2d(C, C, 3, padding=1, bias=False).to(dev, dt)(rnd(B, C, 8, 8))

    def conv_bwd():
        c = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).to(dev, dt)
        c(rnd(B, C, 8, 8, grad=True)).float().sum().backward()

    def gn_fwd():
        torch.nn.GroupNorm(8, C).to(dev, dt)(rnd(B, C, 8, 8))

    def gn_bwd():
        g = torch.nn.GroupNorm(8, C).to(dev, dt)
        g(rnd(B, C, 8, 8, grad=True)).float().sum().backward()

    def gather():
        rnd(B, 64, H, grad=True)[owner, sq].float().sum().backward()

    def embedding():
        torch.nn.Embedding(_NUM_GIFT, H).to(dev, dt)(
            torch.randint(0, _NUM_GIFT, (N,), device=dev)).float().sum().backward()

    def trunk_fwd():
        b, s, _, _ = batch()
        with torch.no_grad(), auto():
            model.trunk(b, s)

    def full_fwd():
        with torch.no_grad(), auto():
            model(*batch())

    def full_bwd():
        with auto():
            logits, value = model(*batch())
        (value.float().pow(2).mean() + logits.float().pow(2).mean()).backward()

    def trunk_scope():
        board, scalars, owner, comps = batch()
        with auto():
            x = model.trunk(board, scalars)
        x = x.float()
        g, value = model.global_features(x)
        logits = model.action_logits(x, g, owner, comps)
        (value.pow(2).mean() + logits.pow(2).mean()).backward()

    return [elementwise, linear, conv_fwd, conv_bwd, gn_fwd, gn_bwd,
            gather, embedding, trunk_fwd, full_fwd, full_bwd, trunk_scope]


def run_child(number, dtype_name):
    import torch
    dt = getattr(torch, dtype_name)
    fn = stages(dt)[number - 1]
    t = time.time()
    try:
        fn()
        torch.xpu.synchronize()
        print("OK %.2fs" % (time.time() - t), flush=True)
    except Exception as e:
        print("ERROR %s: %s" % (type(e).__name__, str(e)[:140]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--only", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--from-stage", type=int, default=1, help="skip the stages before this one")
    args = ap.parse_args()
    if args.only:
        run_child(args.only, args.dtype)
        sys.exit(0)

    print("%s on the xpu, each stage killed after %ds\n" % (args.dtype, args.timeout), flush=True)
    for i, name in enumerate(NAMES, start=1):
        if i < args.from_stage:
            continue
        print("%2d  %-54s" % (i, name), end="", flush=True)
        try:
            r = subprocess.run([sys.executable, __file__, "--only", str(i), "--dtype", args.dtype],
                               capture_output=True, text=True, timeout=args.timeout)
            lines = [l for l in r.stdout.splitlines() if l.startswith(("OK", "ERROR"))]
            if lines:
                print(lines[-1], flush=True)
            else:
                tail = (r.stderr.strip().splitlines() or ["no output"])[-1][:100]
                print("CRASHED (exit %s): %s" % (r.returncode, tail), flush=True)
        except subprocess.TimeoutExpired:
            print("HUNG (killed)", flush=True)
