# SPDX-License-Identifier: Apache-2.0
"""0ics: Inkling 2 sliding-window cache and global-layer attention paths (omlx/patches/inkling2)."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

HQ, HKV, D = 6, 2, 128
WINDOW = 1024


def _unit(shape, seed):
    mx.random.seed(seed)
    x = mx.random.normal(shape)
    return (x / mx.linalg.norm(x, axis=-1, keepdims=True)).astype(mx.bfloat16)


def _chunk(length, seed, rel_extent):
    mx.random.seed(seed)
    q = _unit((1, HQ, length, D), seed) * 4
    k = _unit((1, HKV, length, D), seed + 1)
    v = mx.random.normal((1, HKV, length, D)).astype(mx.bfloat16)
    rel = (mx.random.normal((1, length, HQ, rel_extent)) * 0.5).astype(mx.bfloat16)
    sink = mx.random.normal((1, length, HQ)).astype(mx.bfloat16)
    return q, k, v, rel, sink


@pytest.fixture()
def env(monkeypatch):
    return monkeypatch


def test_window_cache_attention_matches_full_cache_bit_for_bit(env):
    """Chunks of 512 and 300, decode steps and verify rollbacks: the sliding layer's attention over the window cache
    equals the one over the full cache exactly (drops are whole 16-key tiles of the flash kernel)."""
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import kernels, window_cache

    full, win = KVCache(), window_cache.WindowKVCache()
    seed = 0
    steps = [512] * 4 + [300] * 3 + [1] * 20 + [5, 1, 5, 1] + [512] * 2
    for i, length in enumerate(steps):
        seed += 7
        q, k, v, rel, sink = _chunk(length, seed, WINDOW)
        outs = []
        for cache in (full, win):
            kk, vv = cache.update_and_fetch(k, v)
            outs.append(kernels.banded_sink_attention(q, kk, vv, rel, sink, scale=D**-0.5, window=WINDOW))
        assert mx.array_equal(outs[0], outs[1]), f"step {i} ({length} queries) differs"
        if length == 5:  # a verify of 5 that keeps 2: both caches roll back 3 keys
            assert full.trim(3) == 3 and win.trim(3) == 3
    assert full.offset == sum(steps) - 6
    assert win.offset <= window_cache.KEEP + window_cache.SLACK + 512
    assert win.keys.shape[2] < full.keys.shape[2]


def test_window_cache_drops_whole_tiles_and_keeps_the_window():
    from omlx.patches.inkling2 import window_cache

    assert window_cache._drop(window_cache.KEEP + window_cache.SLACK) == 0
    for length in (window_cache.KEEP + window_cache.SLACK + 1, 2047, 4096, 100_003):
        drop = window_cache._drop(length)
        assert drop % window_cache.ALIGN == 0
        assert length - drop >= window_cache.KEEP


def test_window_cache_is_counted_as_sliding_by_the_memory_guard(env):
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    from omlx.memory_monitor import collect_kv_layer_specs
    from omlx.patches.inkling2 import window_cache

    caches = [CacheList(window_cache.WindowKVCache(), ArraysCache(4)) for _ in range(5)]
    caches += [CacheList(KVCache(), ArraysCache(4)) for _ in range(2)]
    full, rotating, arrays = collect_kv_layer_specs(caches)
    assert (full, rotating, arrays) == (2, [(5, WINDOW)], 7)


def test_window_cache_switch(env):
    from omlx.patches.inkling2 import window_cache

    env.delenv("OMLX_0ICS_INKLING_WINDOW", raising=False)
    assert not window_cache.window_enabled(WINDOW)
    env.setenv("OMLX_0ICS_INKLING_WINDOW", "1")
    assert window_cache.window_enabled(WINDOW)
    assert not window_cache.window_enabled(2048)


@pytest.mark.parametrize("keys", [300, 2000, 9000])
@pytest.mark.parametrize("length", [1, 2, 4, 8])
def test_split_attention_matches_the_stock_path(keys, length):
    """Global layer (no window): the split kernel against _decode_attention (1 query) or _masked_attention, on a
    cache view of a longer buffer as the KVCache gives it."""
    from omlx.patches.inkling2 import attn_decode, kernels

    rel_extent = 1024
    q, _, _, rel, sink = _chunk(length, 11 + keys, rel_extent)
    k = _unit((1, HKV, keys + 256, D), 3 + keys)[:, :, :keys]
    mx.random.seed(5 + keys)
    v = mx.random.normal((1, HKV, keys + 256, D)).astype(mx.bfloat16)[:, :, :keys]
    scale = D**-0.5
    if length == 1:
        ref = kernels._decode_attention(q, k, v, rel, sink, scale, 0)
    else:
        ref = kernels._masked_attention(q, k, v, rel, sink, scale, 0, None)
    out = attn_decode.attention(q, k, v, rel, sink, scale=scale)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 2e-3


def test_banded_sink_attention_takes_the_split_kernel_on_a_cache_view(env):
    """Through the dispatcher, on bfloat16 views of a longer buffer and on a transposed (non-contiguous) head
    dimension: the split kernel's result equals the stock path's within the tolerance."""
    from omlx.patches.inkling2 import kernels

    keys, length = 3000, 4
    q, _, _, rel, sink = _chunk(length, 41, 1024)
    k = _unit((1, HKV, keys + 64, D), 42)[:, :, :keys]
    mx.random.seed(43)
    v = mx.random.normal((1, HKV, keys + 64, D)).astype(mx.bfloat16)[:, :, :keys]
    scale = D**-0.5
    env.delenv("OMLX_0ICS_INKLING_ATTN", raising=False)
    ref = kernels.banded_sink_attention(q, k, v, rel, sink, scale=scale, window=0)
    env.setenv("OMLX_0ICS_INKLING_ATTN", "1")
    out = kernels.banded_sink_attention(q, k, v, rel, sink, scale=scale, window=0)
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 2e-3
    kt = mx.swapaxes(mx.swapaxes(k, 2, 3) * 1, 2, 3)  # same values, head dimension strided
    vt = mx.swapaxes(mx.swapaxes(v, 2, 3) * 1, 2, 3)
    out_t = kernels.banded_sink_attention(q, kt, vt, rel, sink, scale=scale, window=0)
    assert float(mx.abs(out_t.astype(mx.float32) - ref.astype(mx.float32)).max()) < 2e-3


def test_split_attention_dispatch(env):
    from omlx.patches.inkling2 import attn_decode, kernels

    q1 = mx.zeros((1, HQ, 1, D), dtype=mx.bfloat16)
    q4 = mx.zeros((1, HQ, 4, D), dtype=mx.bfloat16)
    assert attn_decode.supports(q4, 0, None, 100)
    assert not attn_decode.supports(q4, WINDOW, None, 100)  # sliding layers keep their path
    assert not attn_decode.supports(mx.zeros((1, HQ, 9, D), dtype=mx.bfloat16), 0, None, 100)
    assert not attn_decode.supports(q1, 0, None, attn_decode.MIN_KEYS_ONE_QUERY - 1)
    assert attn_decode.supports(q1, 0, None, attn_decode.MIN_KEYS_ONE_QUERY)
    env.delenv("OMLX_0ICS_INKLING_ATTN", raising=False)
    assert not attn_decode.enabled()
    long_keys = attn_decode.FLASH_MIN_KEYS
    stock_short = kernels._use_flash(512, 0, long_keys - 1)
    env.setenv("OMLX_0ICS_INKLING_ATTN", "1")
    assert attn_decode.enabled()
    if mx.default_device() == mx.gpu:
        assert kernels._use_flash(512, 0, long_keys)
    assert kernels._use_flash(512, 0, long_keys - 1) == stock_short  # short caches keep the stock choice
    assert not kernels._use_flash(16, 0, long_keys)  # verify-sized blocks never take the flash kernel


def test_flash_kernel_on_a_global_layer_matches_the_masked_path():
    """The kernel the global layers' long prefill switches to, on a short cache where both paths are cheap."""
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("the flash kernel runs on the GPU")
    length, keys = 64, 1500
    q, _, _, rel, sink = _chunk(length, 31, 1024)
    k = _unit((1, HKV, keys, D), 32)
    mx.random.seed(33)
    v = mx.random.normal((1, HKV, keys, D)).astype(mx.bfloat16)
    scale = D**-0.5
    ref = kernels._masked_attention(q, k, v, rel, sink, scale, 0, None)
    out = kernels.flash_banded_attention(q, k, v, rel, sink, scale=scale, window=0)
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 2e-3


def _feed(caches, steps, seed):
    """Append the same chunks to every cache; return the sliding-layer attention outputs of the last chunk."""
    from omlx.patches.inkling2 import kernels

    outs = None
    for length in steps:
        seed += 13
        q, k, v, rel, sink = _chunk(length, seed, WINDOW)
        outs = []
        for cache in caches:
            kk, vv = cache.update_and_fetch(k, v)
            outs.append(kernels.banded_sink_attention(q, kk, vv, rel, sink, scale=D**-0.5, window=WINDOW))
    return outs, seed


@pytest.mark.parametrize("prefix", [[512] * 3, [512] * 5 + [300], [512, 77, 1, 1, 1, 512, 512, 333]])
def test_window_cache_prefix_round_trip_is_exact(prefix):
    """Stored through its prefix-cache handler and rebuilt, a window cache continues exactly as the full cache."""
    from mlx_lm.models.cache import KVCache

    from omlx.cache.type_registry import CacheTypeRegistry
    from omlx.patches.inkling2 import cache_handlers, window_cache

    cache_handlers.register()
    full, win = KVCache(), window_cache.WindowKVCache()
    _, seed = _feed([full, win], prefix, 100)
    handler = CacheTypeRegistry.get_handler_by_class_name(type(win).__name__)
    assert not handler.supports_block_slicing
    elements = handler.serialize_state(win)
    assert elements[0].shape[2] <= window_cache.KEEP + window_cache.ALIGN
    restored = handler.deserialize_state(tuple(elements), handler.serialize_meta_state(win))
    assert type(restored) is window_cache.WindowKVCache
    outs, _ = _feed([full, win, restored], [512, 1, 1, 5, 300, 512], seed)
    assert mx.array_equal(outs[0], outs[1]) and mx.array_equal(outs[0], outs[2])


def test_window_cache_handlers_are_registered():
    from mlx_lm.models.cache import ArraysCache, CacheList

    from omlx.cache.type_registry import CacheTypeRegistry
    from omlx.patches.inkling2 import cache_handlers, window_cache

    cache_handlers.register()
    names = CacheTypeRegistry.list_known_class_names()
    assert "WindowKVCache" in names and "BatchWindowKVCache" in names
    layer = CacheList(window_cache.WindowKVCache(), ArraysCache(4))
    assert not CacheTypeRegistry.is_sliceable(layer.caches[0])
    batch = window_cache.WindowKVCache.merge([window_cache.WindowKVCache(), window_cache.WindowKVCache()])
    assert type(batch).__name__ == "BatchWindowKVCache"
    assert not CacheTypeRegistry.get_handler_by_class_name(type(batch).__name__).supports_block_slicing
