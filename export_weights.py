#!/usr/bin/env python
"""Turn a PyTorch checkpoint from train.py into a small PyTorch-free weight file.

    python export_weights.py runs/run4/latest.pt            # writes ai.npz next to play.py
    python export_weights.py runs/run4/latest.pt my_ai.npz

Needs PyTorch (only for reading the checkpoint).  play.py and numpy_net.py then run from the
.npz with just NumPy, which is what you want in a packaged game.
"""
import os
import sys

import numpy as np
import torch

import numpy_net


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai.npz")
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    state = {k: v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()}
    numpy_net.save_npz(dst, state, int(ckpt.get("version", -1)))
    n = sum(int(np.prod(v.shape)) for v in state.values())
    print("wrote %s: version %s, %d parameters, %.1f MB"
          % (dst, ckpt.get("version", "?"), n, os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    main()
