#!/usr/bin/env python
"""The policy/value network's forward pass in Rust (native/src/nnet.rs), for self-play workers.

``net.PolicyValueNet`` is tiny (~0.4M parameters) and workers evaluate it one position (or a
handful of search leaves) at a time on one CPU thread, where PyTorch's per-call overhead
dominates.  ``RustNet`` runs the same maths natively (AVX2+FMA when the CPU has them) from the
weights in a checkpoint; the learner still trains the PyTorch model.

    python build_native.sh              # builds librc_engine.so with the net inside
    python check_rustnet.py             # RustNet vs numpy_net.NumpyNet (and vs torch, if importable)

    net = RustNet.from_torch(model)     # or RustNet(state_dict_as_numpy_arrays)
    logits, value = net.forward(board, scalars, counts, comps)

Nothing falls back silently: a missing or stale library, a shape mismatch or a bad action row
raises.  Keep in step with ``net.PolicyValueNet`` (check_rustnet.py verifies it).
"""
from __future__ import annotations

import ctypes
from typing import Dict, Optional

import numpy as np

import engine
from env import NUM_BOARD_PLANES, NUM_SCALARS, _NUM_GIFT, _NUM_PROMO

_LIB = engine._LIB
_GLOB_CH = 8
_NUM_SQUARES = 64


def _bind():
    """Declare the rc_net_* functions on the engine's library; None if it lacks them."""
    if _LIB is None:
        return None
    try:
        if _LIB.rc_net_abi() != 1:
            return None
    except AttributeError:
        return None
    _LIB.rc_net_new.restype = ctypes.c_void_p
    _LIB.rc_net_new.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_uint32] * 7
    _LIB.rc_net_free.restype = None
    _LIB.rc_net_free.argtypes = [ctypes.c_void_p]
    _LIB.rc_net_forward.restype = ctypes.c_int32
    _LIB.rc_net_forward.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]
    _LIB.rc_net_error.restype = ctypes.c_size_t
    _LIB.rc_net_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _LIB.rc_net_simd.restype = ctypes.c_int32
    _LIB.rc_net_simd.argtypes = []
    return _LIB


_NATIVE = _bind()
AVAILABLE = _NATIVE is not None


def require() -> None:
    """Raise unless the native library has the network (rebuild with ./build_native.sh)."""
    if not AVAILABLE:
        raise RuntimeError("the native library has no rc_net_* functions (missing or built before "
                           "nnet.rs existed): run ./build_native.sh")


def simd_active() -> bool:
    """True when the AVX2+FMA kernels run (RC_NET_SIMD=0 forces the portable ones)."""
    require()
    return bool(_NATIVE.rc_net_simd())


def _last_error() -> str:
    buf = ctypes.create_string_buffer(512)
    n = _NATIVE.rc_net_error(buf, len(buf))
    msg = buf.value.decode("utf-8", "replace")
    return msg if n <= len(buf) - 1 else msg + "..."


def _expected_shapes(channels: int, blocks: int, hidden: int, n_gift: int, n_promo: int):
    """(state_dict key, shape) in the order nnet.rs reads them."""
    c, h, cin = channels, hidden, NUM_BOARD_PLANES + NUM_SCALARS
    out = [("stem.weight", (c, cin, 3, 3)), ("stem_norm.weight", (c,)), ("stem_norm.bias", (c,))]
    for i in range(blocks):
        p = "blocks.%d." % i
        out += [(p + "conv1.weight", (c, c, 3, 3)), (p + "norm1.weight", (c,)), (p + "norm1.bias", (c,)),
                (p + "conv2.weight", (c, c, 3, 3)), (p + "norm2.weight", (c,)), (p + "norm2.bias", (c,))]
    out += [("glob_conv.weight", (_GLOB_CH, c, 1, 1)), ("glob_conv.bias", (_GLOB_CH,)),
            ("glob_fc.weight", (h, _GLOB_CH * _NUM_SQUARES)), ("glob_fc.bias", (h,)),
            ("value_out.weight", (1, h)), ("value_out.bias", (1,)),
            ("src_proj.weight", (h, c)), ("src_proj.bias", (h,)),
            ("dst_proj.weight", (h, c)), ("glob_proj.weight", (h, h)),
            ("gift_emb.weight", (n_gift, h)), ("promo_emb.weight", (n_promo, h)),
            ("flag_emb.weight", (4, h)),
            ("act_fc.weight", (h // 2, h)), ("act_fc.bias", (h // 2,)),
            ("act_out.weight", (1, h // 2)), ("act_out.bias", (1,))]
    return out


class RustNet:
    """Forward pass of ``net.PolicyValueNet`` in Rust.  Immutable once built; thread-safe."""

    def __init__(self, weights: Dict[str, np.ndarray], version: Optional[int] = None):
        require()
        self._h = None
        missing = [k for k in ("stem.weight", "glob_fc.weight", "gift_emb.weight", "promo_emb.weight")
                   if k not in weights]
        if missing:
            raise KeyError("weights lack %s: not a PolicyValueNet state dict?" % ", ".join(missing))
        self.channels = int(weights["stem.weight"].shape[0])
        self.hidden = int(weights["glob_fc.weight"].shape[0])
        self.blocks = 0
        while "blocks.%d.conv1.weight" % self.blocks in weights:
            self.blocks += 1
        n_gift = int(weights["gift_emb.weight"].shape[0])
        n_promo = int(weights["promo_emb.weight"].shape[0])
        if (n_gift, n_promo) != (_NUM_GIFT, _NUM_PROMO):
            raise ValueError("embedding tables are %d x %d, env.py expects %d x %d"
                             % (n_gift, n_promo, _NUM_GIFT, _NUM_PROMO))
        parts = []
        for key, shape in _expected_shapes(self.channels, self.blocks, self.hidden, n_gift, n_promo):
            if key not in weights:
                raise KeyError("weights lack %r" % key)
            arr = np.asarray(weights[key], dtype=np.float32)
            if arr.shape != shape:
                raise ValueError("%s has shape %s, expected %s" % (key, arr.shape, shape))
            parts.append(np.ascontiguousarray(arr).reshape(-1))
        blob = np.concatenate(parts)
        h = _NATIVE.rc_net_new(blob.ctypes.data, blob.size, self.channels, self.blocks, self.hidden,
                               n_gift, n_promo, NUM_BOARD_PLANES, NUM_SCALARS)
        if not h:
            raise RuntimeError("rc_net_new failed: " + _last_error())
        self._h = h
        self.version = version

    @classmethod
    def from_torch(cls, model, version: Optional[int] = None) -> "RustNet":
        """From a ``net.PolicyValueNet`` (any device); copies the weights."""
        weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        return cls(weights, version)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h and _NATIVE is not None:
            _NATIVE.rc_net_free(h)

    def forward(self, board: np.ndarray, scalars: np.ndarray, counts: np.ndarray, comps: np.ndarray):
        """board (B,18,8,8) f32, scalars (B,17) f32, counts (B,) legal actions per position,
        comps (sum(counts), 6) uint8 with the positions' actions back to back.
        Returns (logits (sum(counts),) f32, value (B,) f32)."""
        board = np.ascontiguousarray(board, dtype=np.float32)
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        counts = np.ascontiguousarray(counts, dtype=np.int32)
        comps = np.ascontiguousarray(comps, dtype=np.uint8)
        b = counts.shape[0]
        if board.shape != (b, NUM_BOARD_PLANES, 8, 8) or scalars.shape != (b, NUM_SCALARS):
            raise ValueError("board %s / scalars %s do not match a batch of %d"
                             % (board.shape, scalars.shape, b))
        n = comps.shape[0]
        if comps.ndim != 2 or comps.shape[1] != 6:
            raise ValueError("comps must be (n, 6), got %s" % (comps.shape,))
        logits = np.empty(n, dtype=np.float32)
        value = np.empty(b, dtype=np.float32)
        rc = _NATIVE.rc_net_forward(self._h, b, board.ctypes.data, scalars.ctypes.data, counts.ctypes.data,
                                    n, comps.ctypes.data, logits.ctypes.data, value.ctypes.data)
        if rc != 0:
            raise RuntimeError("rc_net_forward failed (%d): %s" % (rc, _last_error()))
        return logits, value

    def forward_owner(self, board, scalars, owner, comps):
        """``PolicyValueNet.forward``'s signature: ``owner`` (N,) must list the positions in
        order (each position's actions contiguous), as every caller here builds it."""
        owner = np.asarray(owner)
        b = len(board)
        counts = np.bincount(owner, minlength=b)
        if len(owner) and np.any(np.diff(owner) < 0):
            raise ValueError("owner must be non-decreasing (actions grouped by position)")
        return self.forward(board, scalars, counts, comps)
