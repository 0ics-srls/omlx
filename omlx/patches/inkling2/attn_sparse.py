# SPDX-License-Identifier: Apache-2.0
"""0ics: the global layers' decode and verify read a fixed number of keys, whatever the context (training-free).

Inkling 2 has 8 global layers with full attention over the whole cache, 32 KiB a token of context: a decode step reads
4.3 GB at 128K, 17 GB at 512K, 34 GB at 1M (~38 ms at the M5 Ultra's bandwidth, more than the weights). Qwen3.8
Flash-Next stays flat because its 12 full-attention layers carry a trained indexer that picks 2,048 cells; Inkling has
none, so this one is built from the keys, as Quest does: for every block of ``block`` keys and every key/value head the
per-dimension min and max of the keys give an upper bound of q·k, sum_d max(q_d min_d, q_d max_d) = q·c + |q|·r with
c = (max + min) / 2, r = (max - min) / 2. Each key/value head reads the blocks with the highest bound (the rows of the
GQA group and of the verify each against their own best block, the first block always) and the last ``TAIL`` keys,
where the relative bias (1,024 keys) and the verify's own keys are; ``attn_decode.attention`` reads them from the cache
in place (``sel``), with their true distances.

Measured (28/11, Small oQ2e, M5 Ultra, 1,024 tokens of real code after a 128K prefix, verify blocks of 16,
``sperimenti/inkling/prova_sparsa.py``): NLL 0.5367 full; blocks of 16 and 8,192 keys 0.5355, argmax equal on 98.2%,
KL mean 0.006, max 0.16; 4,096 keys 0.5521 (KL max 8.3: a few tokens lose what they needed). After a 512K prefix:
full 0.7542; 8,192 keys +0.0072 (argmax equal 96.1%, KL max 0.31), 16,384 keys +0.0053 (97.0%); blocks of 32 or 64
lose more (+0.009, +0.010).

Speed, one global layer on synthetic keys (``sperimenti/inkling/prova_sparsa_tempi.py``), 8,192 keys, full against
sparse: one query 1.05 → 0.47 ms at 128K, 3.33 → 0.76 at 512K, 6.19 → 1.16 at 1M; a verify of 9 2.19 → 0.76, 7.90 →
1.30, 15.56 → 2.34. Most of what is left is the selection (1.03 of the 1.16 ms at 1M). The model at 128K: a decode
token 22.8 → 19.4 ms, a verify of 9 80.6 → 70.5 ms (61.4 at 4K).

The min/max are kept on the cache object (``_omlx_sparse``) and extended lazily to the blocks the next step needs;
``note_append`` (called on every append of a global layer) drops the blocks an append overwrites, so a trim and a new
append never leave a stale bound behind.

On with ``OMLX_0ICS_INKLING_SPARSE=<keys read>`` (for example 8192), blocks of ``OMLX_0ICS_INKLING_SPARSE_BLOCK``
(default 16) keys; only with the fused layer (``OMLX_0ICS_INKLING_FUSED=1``), batch 1, a bfloat16 cache. Off, or for a
cache shorter than the budget, the attention is the full one.
"""

from __future__ import annotations

import os

import mlx.core as mx

# relative bias over the last 1,024 keys (rel_extent) + a verify of up to 16 + margin, a multiple of 16
TAIL = 1024 + 64
_GROW = 4096
_FIRST = 1e30


def budget() -> int:
    return int(os.environ.get("OMLX_0ICS_INKLING_SPARSE", "0") or 0)


def block_size() -> int:
    return int(os.environ.get("OMLX_0ICS_INKLING_SPARSE_BLOCK", "16"))


def enabled() -> bool:
    return budget() > TAIL


def note_append(kv, start: int) -> None:
    """A global layer appended keys from position ``start``: the bounds of the blocks from there on are stale."""
    state = getattr(kv, "_omlx_sparse", None)
    if state is not None and state["blocks"] * state["block"] > start:
        state["blocks"] = start // state["block"]


def _bounds(kv, k, blocks: int, block: int):
    """The (centre, radius) of the first ``blocks`` blocks of keys, [1, Hkv, blocks, D] bfloat16 each."""
    state = getattr(kv, "_omlx_sparse", None)
    if state is None or state["block"] != block:
        state = {"block": block, "blocks": 0, "c": None, "r": None}
        kv._omlx_sparse = state
    done = state["blocks"]
    if done < blocks:
        _, hkv, _, d = k.shape
        kf = k[:, :, done * block:blocks * block].reshape(1, hkv, blocks - done, block, d)
        lo = kf.min(axis=3).astype(mx.float32)
        hi = kf.max(axis=3).astype(mx.float32)
        c = ((hi + lo) * 0.5).astype(mx.bfloat16)
        r = ((hi - lo) * 0.5).astype(mx.bfloat16)
        if state["c"] is None or state["c"].shape[2] < blocks:
            cap = (blocks + _GROW - 1) // _GROW * _GROW
            grown_c = mx.zeros((1, hkv, cap, d), mx.bfloat16)
            grown_r = mx.zeros((1, hkv, cap, d), mx.bfloat16)
            if state["c"] is not None and done:
                grown_c[:, :, :done] = state["c"][:, :, :done]
                grown_r[:, :, :done] = state["r"][:, :, :done]
            state["c"], state["r"] = grown_c, grown_r
        state["c"][:, :, done:blocks] = c
        state["r"][:, :, done:blocks] = r
        state["blocks"] = blocks
    return state["c"][:, :, :blocks], state["r"][:, :, :blocks]


def select(kv, q, k, keys: int, block: int):
    """The blocks each key/value head reads, int32 [Hkv, n] ascending, and where the tail starts; None when the
    cache holds no more than ``keys`` keys."""
    _, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    blocks = (s - TAIL) // block
    picked = (keys - TAIL) // block
    if blocks <= picked:
        return None
    c, r = _bounds(kv, k, blocks, block)
    qg = q.reshape(1, hkv, (hq // hkv) * length, d).astype(mx.bfloat16)
    bound = (qg @ c.swapaxes(-1, -2)).astype(mx.float32) + (mx.abs(qg) @ r.swapaxes(-1, -2)).astype(mx.float32)
    vote = (bound - bound.max(axis=-1, keepdims=True)).max(axis=2)
    vote = mx.where(mx.arange(blocks) == 0, mx.array(_FIRST, dtype=mx.float32), vote)
    sel = mx.sort(mx.argpartition(-vote, picked - 1, axis=-1)[..., :picked], axis=-1)
    return sel.reshape(hkv, picked).astype(mx.int32), blocks * block


def attention(kv, q, k, v, rel, sink, *, scale: float, gate, min_chunk: int):
    """The fused layer's global attention on the selected keys, or None for the full one."""
    from . import attn_decode

    if kv is None or q.shape[0] != 1 or getattr(kv, "left_padding", None) is not None:
        return None
    block = block_size()
    picked = select(kv, q, k, budget(), block)
    if picked is None:
        return None
    sel, far = picked
    return attn_decode.attention(q, k, v, rel, sink, scale=scale, window=0, gate=gate, min_chunk=min_chunk,
                                 sel=sel, block=block, far=far)
