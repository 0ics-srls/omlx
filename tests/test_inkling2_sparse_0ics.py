# SPDX-License-Identifier: Apache-2.0
"""0ics: Inkling 2 global layers on selected key blocks (omlx/patches/inkling2/attn_sparse.py)."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

HQ, HKV, D = 6, 2, 128
REL = 1024


def _unit(shape, seed):
    mx.random.seed(seed)
    x = mx.random.normal(shape)
    return (x / mx.linalg.norm(x, axis=-1, keepdims=True)).astype(mx.bfloat16)


def _inputs(length, keys, seed):
    q = _unit((1, HQ, length, D), seed) * 4
    k = _unit((1, HKV, keys + 256, D), seed + 1)[:, :, :keys]
    mx.random.seed(seed + 2)
    v = mx.random.normal((1, HKV, keys + 256, D)).astype(mx.bfloat16)[:, :, :keys]
    rel = (mx.random.normal((1, length, HQ, REL)) * 0.5).astype(mx.bfloat16)
    sink = mx.random.normal((1, length, HQ)).astype(mx.bfloat16)
    gate = mx.random.normal((1, length, HQ * D)).astype(mx.bfloat16)
    return q, k, v, rel, sink, gate


def _gathered(x, sel, block, far):
    """The keys the selection reads, in its order: each head's blocks, then the tail."""
    heads = [mx.concatenate([x[0, h, int(b) * block:(int(b) + 1) * block] for b in sel[h].tolist()]) for h in
             range(x.shape[1])]
    return mx.concatenate([mx.stack(heads)[None], x[:, :, far:]], axis=2)


@pytest.mark.parametrize("length", [1, 9, 16])
@pytest.mark.parametrize("gated", [False, True])
def test_selected_blocks_equal_the_attention_on_the_gathered_keys(length, gated):
    """The kernels read the selected blocks in place with their true distances: the result equals, bit for bit, the
    same kernels on a copy of those keys placed so that the tail keeps its distances (far keys are past the bias)."""
    from omlx.patches.inkling2 import attn_decode

    keys, block = 6000, 16
    q, k, v, rel, sink, gate = _inputs(length, keys, 7 + length)
    far = (keys - 1088) // block * block
    mx.random.seed(3)
    sel = mx.sort(mx.random.permutation(far // block)[:40].reshape(1, 40).astype(mx.int32), axis=-1)
    sel = mx.concatenate([sel, mx.sort(mx.random.permutation(far // block)[:40].reshape(1, 40), axis=-1)])
    sel = sel.astype(mx.int32)
    g = gate if gated else None
    out = attn_decode.attention(q, k, v, rel, sink, scale=D**-0.5, gate=g, sel=sel, block=block, far=far)
    ref = attn_decode.attention(q, _gathered(k, sel, block, far), _gathered(v, sel, block, far), rel, sink,
                                scale=D**-0.5, gate=g)
    assert out.shape == ref.shape
    assert mx.array_equal(out, ref)


def test_bound_is_an_upper_bound_of_every_key_in_its_block():
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import attn_sparse

    keys, block = 4096, 16
    q, k, _, _, _, _ = _inputs(1, keys, 21)
    kv = KVCache()
    c, r = attn_sparse._bounds(kv, k, keys // block, block)
    qf = q.reshape(1, HKV, HQ // HKV, D).astype(mx.float32)
    bound = qf @ c.astype(mx.float32).swapaxes(-1, -2) + mx.abs(qf) @ r.astype(mx.float32).swapaxes(-1, -2)
    true = (qf @ k.astype(mx.float32).swapaxes(-1, -2)).reshape(1, HKV, HQ // HKV, keys // block, block).max(-1)
    # the bounds are stored in bfloat16: 1% of their size
    assert float((true - bound).max()) <= 0.01 * float(mx.abs(bound).max())


def test_select_takes_the_block_that_matches_and_the_first():
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import attn_sparse

    keys, block, budget = 20000, 16, attn_sparse.TAIL + 4 * 16
    q, _, _, _, _, _ = _inputs(1, keys, 31)
    # keys of a block share a direction (with random keys in 128 dimensions the min/max bound says nothing)
    blocks = keys // block
    dirs = _unit((1, HKV, blocks, 1, D), 32).astype(mx.float32)
    noise = _unit((1, HKV, blocks, block, D), 33).astype(mx.float32) * 0.05
    hot = 700
    k = mx.array(mx.broadcast_to(dirs, noise.shape) + noise)
    k[:, 1, hot] = q[0, 4, 0].astype(mx.float32) / 4 + noise[:, 1, hot]  # along a query of head 4 (key/value head 1)
    k = k.reshape(1, HKV, keys, D).astype(mx.bfloat16)
    sel, far = attn_sparse.select(KVCache(), q, k, budget, block)
    assert far == (keys - attn_sparse.TAIL) // block * block
    assert sel.shape == (HKV, 4) and sel.dtype == mx.int32
    assert 0 in sel[0].tolist() and 0 in sel[1].tolist()
    assert hot in sel[1].tolist()
    assert sel[1].tolist() == sorted(sel[1].tolist())


def test_short_cache_keeps_the_full_attention():
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import attn_sparse

    q, k, _, _, _, _ = _inputs(1, 8192, 41)
    assert attn_sparse.select(KVCache(), q, k, 8192, 16) is None


def test_an_append_over_known_blocks_drops_their_bounds():
    """A rollback and a new append rewrite keys the bounds were computed from: they are computed again."""
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import attn_sparse

    q, k, _, _, _, _ = _inputs(1, 4096, 51)
    kv = KVCache()
    attn_sparse._bounds(kv, k, 200, 16)
    assert kv._omlx_sparse["blocks"] == 200
    attn_sparse.note_append(kv, 4000)
    assert kv._omlx_sparse["blocks"] == 200
    attn_sparse.note_append(kv, 1000)
    assert kv._omlx_sparse["blocks"] == 1000 // 16
    k2 = mx.array(k)
    k2[:, :, 1000:] = -k2[:, :, 1000:]
    c, _ = attn_sparse._bounds(kv, k2, 200, 16)
    c_new, _ = attn_sparse._bounds(KVCache(), k2, 200, 16)
    assert mx.array_equal(c, c_new)


def test_switch(monkeypatch):
    from omlx.patches.inkling2 import attn_sparse

    monkeypatch.delenv("OMLX_0ICS_INKLING_SPARSE", raising=False)
    assert not attn_sparse.enabled()
    monkeypatch.setenv("OMLX_0ICS_INKLING_SPARSE", "8192")
    assert attn_sparse.enabled() and attn_sparse.budget() == 8192
    monkeypatch.setenv("OMLX_0ICS_INKLING_SPARSE", "1000")
    assert not attn_sparse.enabled()  # no room past the tail
