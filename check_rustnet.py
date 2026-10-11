#!/usr/bin/env python
"""Rust forward pass (rustnet.py) vs the reference networks, on positions from random games.

    python check_rustnet.py                         # random weights, 64ch/4 blocks/128 hidden and a tiny net
    python check_rustnet.py runs/run4/latest.pt     # a real checkpoint (needs torch to read it)
    python check_rustnet.py --games 40 --no-time
    RC_NET_SIMD=0 python check_rustnet.py           # the portable (non-AVX2) kernels

References: ``numpy_net.NumpyNet`` always; ``net.PolicyValueNet`` too when torch is importable
(then the checks are Rust vs PyTorch directly).  Also checks that bad input raises, and times
Rust against NumPy (and PyTorch) for one position and for a batch of 4.
"""
import argparse
import random
import sys
import time

import numpy as np

import engine
import numpy_net
import rustnet
from engine import Game, Rules
from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO, encode_observation


def random_weights(channels, blocks, hidden, seed):
    """A state dict with non-trivial values everywhere (norm scales/shifts, heads), so every
    term of the forward pass matters; logits get a real spread."""
    rng = np.random.default_rng(seed)
    c, h, cin = channels, hidden, NUM_BOARD_PLANES + NUM_SCALARS

    def conv(o, i, k=3):
        return (rng.standard_normal((o, i, k, k)) * np.sqrt(2.0 / (i * k * k))).astype(np.float32)

    def lin(o, i, s=1.0):
        return (rng.standard_normal((o, i)) * s * np.sqrt(1.0 / i)).astype(np.float32)

    def vec(n, base=0.0, s=0.2):
        return (base + s * rng.standard_normal(n)).astype(np.float32)

    w = {"stem.weight": conv(c, cin), "stem_norm.weight": vec(c, 1.0), "stem_norm.bias": vec(c)}
    for i in range(blocks):
        p = "blocks.%d." % i
        w[p + "conv1.weight"] = conv(c, c)
        w[p + "norm1.weight"], w[p + "norm1.bias"] = vec(c, 1.0), vec(c)
        w[p + "conv2.weight"] = conv(c, c)
        w[p + "norm2.weight"], w[p + "norm2.bias"] = vec(c, 1.0), vec(c)
    w["glob_conv.weight"], w["glob_conv.bias"] = conv(8, c, 1), vec(8)
    w["glob_fc.weight"], w["glob_fc.bias"] = lin(h, 512), vec(h)
    w["value_out.weight"], w["value_out.bias"] = lin(1, h, 3.0), vec(1)
    w["src_proj.weight"], w["src_proj.bias"] = lin(h, c), vec(h)
    w["dst_proj.weight"], w["glob_proj.weight"] = lin(h, c), lin(h, h)
    w["gift_emb.weight"] = vec(_NUM_GIFT * h, 0, 0.5).reshape(_NUM_GIFT, h)
    w["promo_emb.weight"] = vec(_NUM_PROMO * h, 0, 0.5).reshape(_NUM_PROMO, h)
    w["flag_emb.weight"] = vec(4 * h, 0, 0.5).reshape(4, h)
    w["act_fc.weight"], w["act_fc.bias"] = lin(h // 2, h), vec(h // 2)
    w["act_out.weight"], w["act_out.bias"] = lin(1, h // 2, 4.0), vec(1)
    return w


def positions(n_games, seed0, max_plies=120):
    """(board, scalars, comps) for every position of some random games (canonical frame)."""
    out = []
    rules = Rules(max_plies=max_plies)
    for seed in range(seed0, seed0 + n_games):
        rng = random.Random(seed)
        g = Game(rules)
        while not g.done:
            obs = encode_observation(g)
            comps = g.action_array().copy()
            if g.turn == "b":
                comps[:, :2] ^= 56
            out.append((np.asarray(obs["board"], np.float32), np.asarray(obs["scalars"], np.float32), comps))
            acts = g.legal_actions()
            g.step(acts[rng.randrange(len(acts))])
        g.close()
    return out


def batch_of(items):
    counts = np.array([len(c) for _, _, c in items], dtype=np.int32)
    return (np.stack([b for b, _, _ in items]), np.stack([s for _, s, _ in items]), counts,
            np.concatenate([c for _, _, c in items]))


def reference(weights, kind):
    """A function (board, scalars, counts, comps) -> (logits, value) from the reference net."""
    if kind == "numpy":
        net = numpy_net.NumpyNet(weights)

        def run(board, scalars, counts, comps):
            owner = np.repeat(np.arange(len(counts)), counts)
            lo, va = net.forward(board, scalars, owner, comps)
            return lo.astype(np.float32), va.astype(np.float32)
        return run
    import torch
    import net as netlib
    model = netlib.PolicyValueNet(**dict(channels=weights["stem.weight"].shape[0],
                                         blocks=sum(k.endswith("conv1.weight") and k.startswith("blocks.") for k in weights),
                                         hidden=weights["glob_fc.weight"].shape[0]))
    model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in weights.items()})
    model.eval()

    def run(board, scalars, counts, comps):
        owner = np.repeat(np.arange(len(counts)), counts)
        with torch.inference_mode():
            lo, va = model(torch.from_numpy(board), torch.from_numpy(scalars), torch.from_numpy(owner),
                           torch.from_numpy(comps.astype(np.int64)))
        return lo.numpy(), va.numpy()
    return run


def compare(name, ref, ru, pos, batch_sizes=(1, 3, 8), tol=2e-4):
    worst_l = worst_v = 0.0
    n = 0
    rng = random.Random(1)
    idx = list(range(len(pos)))
    rng.shuffle(idx)
    k = 0
    while k + max(batch_sizes) <= len(idx):
        bs = batch_sizes[(k // 7) % len(batch_sizes)]
        items = [pos[i] for i in idx[k:k + bs]]
        k += bs
        board, sc, counts, comps = batch_of(items)
        l_ref, v_ref = ref(board, sc, counts, comps)
        l_rs, v_rs = ru.forward(board, sc, counts, comps)
        scale_l = max(1.0, float(np.abs(l_ref).max()))
        worst_l = max(worst_l, float(np.abs(l_ref - l_rs).max()) / scale_l)
        worst_v = max(worst_v, float(np.abs(v_ref - v_rs).max()))
        n += bs
    ok = worst_l < tol and worst_v < tol
    print("  %-28s %5d positions   max logit err %.2e (relative)   max value err %.2e   %s"
          % (name, n, worst_l, worst_v, "ok" if ok else "FAIL"))
    return ok


def expect_raises(label, fn, exc=Exception):
    try:
        fn()
    except exc as err:
        print("  raises on %-32s -> %s" % (label, str(err).splitlines()[0][:70]))
        return True
    print("  FAIL: %s did not raise" % label)
    return False


def timeit(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="a train.py .pt checkpoint (needs torch) instead of random weights")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-torch", action="store_true", help="compare with NumPy only even if torch is installed")
    args = ap.parse_args()

    rustnet.require()
    print("native engine: %s    net kernels: %s"
          % (engine.NATIVE, "AVX2+FMA" if rustnet.simd_active() else "portable"))
    try:
        if args.no_torch:
            raise ImportError
        import torch  # noqa: F401
        import net  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    pos = positions(args.games, args.seed0)
    print("%d positions from %d random games (avg %.0f legal actions)"
          % (len(pos), args.games, np.mean([len(c) for _, _, c in pos])))

    if args.checkpoint:
        wsets = [("checkpoint", numpy_net.load(args.checkpoint).w)]
    else:
        wsets = [("random 64ch/4/128", random_weights(64, 4, 128, 0)),
                 ("random 16ch/1/32 (smoke)", random_weights(16, 1, 32, 1)),
                 ("random 24ch/2/48 (odd tiles)", random_weights(24, 2, 48, 2))]
    ok = True
    for name, w in wsets:
        print(name)
        ru = rustnet.RustNet(w)
        ok &= compare("vs numpy_net", reference(w, "numpy"), ru, pos)
        if have_torch:
            ok &= compare("vs torch (net.py)", reference(w, "torch"), ru, pos)

    print("input validation")
    w = random_weights(16, 1, 32, 1)
    ru = rustnet.RustNet(w)
    board, sc, counts, comps = batch_of(pos[:2])
    bad = comps.copy()
    bad[0, 0] = 64
    ok &= expect_raises("square 64", lambda: ru.forward(board, sc, counts, bad), RuntimeError)
    bad = comps.copy()
    bad[0, 3] = _NUM_GIFT
    ok &= expect_raises("gift code out of range", lambda: ru.forward(board, sc, counts, bad), RuntimeError)
    ok &= expect_raises("counts not matching comps", lambda: ru.forward(board, sc, counts + 1, comps), RuntimeError)
    ok &= expect_raises("wrong board shape", lambda: ru.forward(board[:, :17], sc, counts, comps), ValueError)
    short = dict(w)
    del short["act_out.bias"]
    ok &= expect_raises("missing weight", lambda: rustnet.RustNet(short), KeyError)
    wrong = dict(w)
    wrong["src_proj.bias"] = np.zeros(7, np.float32)
    ok &= expect_raises("wrong weight shape", lambda: rustnet.RustNet(wrong), ValueError)

    if not args.no_time and not args.checkpoint:
        print("timing (this machine, one thread; 64ch/4 blocks/128 hidden)")
        w = random_weights(64, 4, 128, 0)
        ru = rustnet.RustNet(w)
        refs = [("numpy_net", reference(w, "numpy"))] + ([("torch (1 thread)", reference(w, "torch"))] if have_torch else [])
        if have_torch:
            import torch
            torch.set_num_threads(1)
        for bs in (1, 4):
            items = pos[100:100 + bs]
            board, sc, counts, comps = batch_of(items)
            t_rs = timeit(lambda: ru.forward(board, sc, counts, comps), 200)
            line = "  batch %d (%d actions):  rust %.3f ms" % (bs, len(comps), 1000 * t_rs)
            for name, fn in refs:
                t = timeit(lambda: fn(board, sc, counts, comps), 20 if name == "numpy_net" else 100)
                line += "    %s %.3f ms (%.1fx)" % (name, 1000 * t, t / t_rs)
            print(line)

    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
