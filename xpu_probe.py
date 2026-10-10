#!/usr/bin/env python
"""Which backward-pass operations work on the Intel GPU?  (Run from reverse-chess/.)

    python xpu_probe.py                  # test everything; a hang is killed and reported
    python xpu_probe.py --timeout 120    # allow more time per stage (first runs compile kernels)

Every stage runs in its own subprocess, so a hang just shows up as HUNG and the
next stage still runs.  Stages 1-6 are different ways to write the same
"pick one row per legal action" gather that net.py uses; whichever of them are
OK can replace the one that hangs.
"""
import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

B, LEGAL, C, H = 256, 62, 64, 128
N = B * LEGAL


def stages():
    import torch
    import torch.nn.functional as F
    import net as netlib
    from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO
    dev = "xpu"
    rnd = lambda *s, grad=False: torch.rand(*s, device=dev, requires_grad=grad)
    owner = torch.repeat_interleave(torch.arange(B), LEGAL).to(dev)
    sq = torch.randint(0, 64, (N,)).to(dev)
    flat = owner * 64 + sq
    model = netlib.PolicyValueNet(64, 4, 128).to(dev).train()

    def batch():
        comps = torch.stack([sq, sq.flip(0), torch.randint(0, 2, (N,), device=dev),
                             torch.randint(0, _NUM_GIFT, (N,), device=dev),
                             torch.randint(0, 2, (N,), device=dev),
                             torch.randint(0, _NUM_PROMO, (N,), device=dev)], 1)
        return rnd(B, NUM_BOARD_PLANES, 8, 8).round(), rnd(B, NUM_SCALARS), owner, comps

    def two_index():
        rnd(B, 64, H, grad=True)[owner, sq].sum().backward()

    def one_index():
        rnd(B, H, grad=True)[owner].sum().backward()

    def index_select():
        torch.index_select(rnd(B * 64, H, grad=True), 0, flat).sum().backward()

    def embedding_gather():
        F.embedding(flat, rnd(B * 64, H, grad=True)).sum().backward()

    def gather():
        src = rnd(B * 64, H, grad=True)
        torch.gather(src, 0, flat[:, None].expand(N, H)).sum().backward()

    def onehot_bmm():
        oh = F.one_hot(sq, 64).to(torch.float32).view(B, LEGAL, 64)
        torch.bmm(oh, rnd(B, 64, H, grad=True)).sum().backward()

    def embedding_layer():
        model.gift_emb(torch.randint(0, _NUM_GIFT, (N,), device=dev)).sum().backward()

    def trunk():
        b, s, _, _ = batch()
        model.trunk(b, s).sum().backward()

    def index_put():
        rank = torch.arange(N, device=dev) % LEGAL
        logits = rnd(N, grad=True)
        torch.full((B, LEGAL), -1e9, device=dev).index_put((owner, rank), logits).sum().backward()

    def train_step():
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        logits, value = model(*batch())
        (value.pow(2).mean() + logits.pow(2).mean()).backward()
        opt.step()

    return [
        ("two-index gather a[owner, sq]   (what net.py does now)", two_index),
        ("one-index gather g[owner]", one_index),
        ("flattened torch.index_select", index_select),
        ("flattened F.embedding used as a gather", embedding_gather),
        ("flattened torch.gather", gather),
        ("one-hot + torch.bmm (no scatter at all)", onehot_bmm),
        ("nn.Embedding layer (gift/promo/flag)", embedding_layer),
        ("conv trunk", trunk),
        ("index_put into padded logits", index_put),
        ("FULL training step with the current net.py", train_step),
    ]


def run_child(number):
    import torch
    name, fn = stages()[number - 1]
    t = time.time()
    try:
        fn()
        torch.xpu.synchronize()
        print("OK %.2fs" % (time.time() - t), flush=True)
    except Exception as e:
        print("ERROR %s: %s" % (type(e).__name__, str(e)[:120]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--only", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.only:
        run_child(args.only)
        sys.exit(0)

    names = ["two-index gather a[owner, sq]   (what net.py does now)",
             "one-index gather g[owner]",
             "flattened torch.index_select",
             "flattened F.embedding used as a gather",
             "flattened torch.gather",
             "one-hot + torch.bmm (no scatter at all)",
             "nn.Embedding layer (gift/promo/flag)",
             "conv trunk",
             "index_put into padded logits",
             "FULL training step with the current net.py"]
    print("each stage: backward pass on the xpu, killed after %ds\n" % args.timeout, flush=True)
    for i, name in enumerate(names, start=1):
        print("%2d  %-56s" % (i, name), end="", flush=True)
        try:
            r = subprocess.run([sys.executable, __file__, "--only", str(i)],
                               capture_output=True, text=True, timeout=args.timeout)
            lines = [l for l in r.stdout.splitlines() if l.startswith(("OK", "ERROR"))]
            if lines:
                print(lines[-1], flush=True)
            else:
                tail = (r.stderr.strip().splitlines() or ["no output"])[-1][:100]
                print("CRASHED (exit %s): %s" % (r.returncode, tail), flush=True)
        except subprocess.TimeoutExpired:
            print("HUNG (killed)", flush=True)
