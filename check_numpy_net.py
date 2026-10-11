#!/usr/bin/env python
"""Check that numpy_net.py computes the same thing as net.py (needs PyTorch).

    python check_numpy_net.py                       # random weights
    python check_numpy_net.py runs/run4/latest.pt   # a real checkpoint

Plays random games, and on a sample of the positions compares logits and value from
PyTorch and NumPy, then round-trips the weights through an .npz file.  Exits non-zero on a
mismatch.
"""
import argparse
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch

import net as netlib
import numpy_net
from engine import Game, Rules
from env import encode_observation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="a train.py checkpoint (default: random weights)")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.checkpoint:
        model, ckpt = netlib.load_checkpoint(args.checkpoint)
        version = int(ckpt.get("version", -1))
    else:
        model, version = netlib.PolicyValueNet(), 0
        with torch.no_grad():                      # the fresh init is almost flat: make it opinionated
            for name, p in model.named_parameters():
                is_norm_scale = "norm" in name and name.endswith(".weight")   # GroupNorm scale: around 1
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if is_norm_scale else 0.0))
        model.eval()
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:     # also tests the .npz round trip
        path = os.path.join(tmp, "ai.npz")
        numpy_net.save_npz(path, state, version)
        npnet = numpy_net.load_npz(path)
    assert npnet.version == (None if version < 0 else version), npnet.version

    rng = random.Random(args.seed)
    worst_logit = worst_value = 0.0
    checked = 0
    t_np = 0.0
    while checked < args.positions:
        game = Game(Rules(max_plies=200))
        while not game.done and checked < args.positions:
            actions = game.legal_actions()
            if rng.random() < 0.3:
                obs = encode_observation(game)
                comps = game.action_array().copy()
                if game.turn == "b":
                    comps[:, :2] ^= 56
                owner = np.zeros(len(actions), dtype=np.int64)
                with torch.inference_mode():
                    lt, vt = model(torch.from_numpy(obs["board"])[None], torch.from_numpy(obs["scalars"])[None],
                                   torch.from_numpy(owner), torch.from_numpy(comps.astype(np.int64)))
                t0 = time.perf_counter()
                ln, vn = npnet.forward(obs["board"][None], obs["scalars"][None], owner, comps)
                t_np += time.perf_counter() - t0
                worst_logit = max(worst_logit, float(np.abs(lt.numpy() - ln).max()))
                worst_value = max(worst_value, float(np.abs(vt.numpy() - vn).max()))
                checked += 1
            game.step(rng.choice(actions))
    print("%d positions: max |logit diff| %.2e, max |value diff| %.2e, numpy %.1f ms per position"
          % (checked, worst_logit, worst_value, 1000 * t_np / checked))
    if worst_logit > 1e-3 or worst_value > 1e-3:
        print("MISMATCH")
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
