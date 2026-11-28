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
    assert full.offset == win.offset == sum(steps) - 6  # offset counts tokens, as oMLX reads it
    assert win._len <= window_cache.KEEP + window_cache.SLACK + 512
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
@pytest.mark.parametrize("length", [1, 2, 4, 8, 9, 12, 16])
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
    assert attn_decode.supports(mx.zeros((1, HQ, 9, D), dtype=mx.bfloat16), 0, None, 100)  # DSpark's verify
    assert not attn_decode.supports(mx.zeros((1, HQ, attn_decode.MAX_QUERIES + 1, D), dtype=mx.bfloat16), 0, None, 100)
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
    assert elements[0].shape[2] <= window_cache.KEEP + window_cache.TAIL_EXTRA + window_cache.ALIGN
    meta = tuple(str(x) for x in handler.serialize_meta_state(win))  # the SSD store keeps metadata as strings
    # prefix_cache's CacheList restore rewrites a sub's first metadata element to its tensor length
    meta = (elements[0].shape[2], *meta[1:])
    restored = handler.deserialize_state(tuple(elements), meta)
    assert type(restored) is window_cache.WindowKVCache
    assert restored.offset == win.offset == full.offset
    assert handler.deserialize_state(tuple(elements), None) is None  # no token count, no restore
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


def test_batched_window_cache_keeps_token_counts_through_merge_and_extract():
    """Two requests of different lengths: merged, stepped together, extracted, each row equals its own cache."""
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import kernels, window_cache

    singles, fulls, seed = [], [], 300
    for steps in ([512] * 4, [512, 300]):
        win, full = window_cache.WindowKVCache(), KVCache()
        _, seed = _feed([win, full], steps, seed)
        singles.append(win)
        fulls.append(full)
    batch = window_cache.WindowKVCache.merge(singles)
    assert [int(x) for x in batch.offset.tolist()] == [c.offset for c in singles]
    for _ in range(3):
        seed += 5
        q, k, v, rel, sink = _chunk(1, seed, WINDOW)
        kk, vv = batch.update_and_fetch(mx.concatenate([k, k]), mx.concatenate([v, v]))
        for row, full in enumerate(fulls):
            fk, fv = full.update_and_fetch(k, v)
            ref = kernels.banded_sink_attention(q, fk, fv, rel, sink, scale=D**-0.5, window=WINDOW)
            lp = int(batch.left_padding[row].item())
            out = kernels.banded_sink_attention(
                q, kk[row : row + 1, :, lp:], vv[row : row + 1, :, lp:], rel, sink, scale=D**-0.5, window=WINDOW
            )
            assert mx.array_equal(out, ref)
    for row, full in enumerate(fulls):
        single = batch.extract(row)
        assert type(single) is window_cache.WindowKVCache and single.offset == full.offset


@pytest.mark.parametrize("window", [0, WINDOW])
def test_gqa_flash_equals_the_upstream_flash_kernel_bit_for_bit(window):
    from omlx.patches.inkling2 import attn_prefill, kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("the flash kernels run on the GPU")
    length, keys = 100, 2600
    q, _, _, rel, sink = _chunk(length, 51, 1024)
    k = _unit((1, HKV, keys, D), 52)
    mx.random.seed(53)
    v = mx.random.normal((1, HKV, keys, D)).astype(mx.bfloat16)
    scale = D**-0.5
    ref = kernels.flash_banded_attention(q, k, v, rel, sink, scale=scale, window=window)
    for bq in (16, 32):
        out = attn_prefill.flash_gqa(q, k, v, rel, sink, scale=scale, window=window, bq=bq)
        assert mx.array_equal(out, ref)


@pytest.mark.parametrize("extra", [1, 44, 200])
def test_restored_window_realigns_to_the_global_position(extra):
    """A restore that lands ``extra`` tokens before the window snapshot: after realign the window continues as the
    full cache that saw only the restored tokens."""
    from mlx_lm.models.cache import KVCache

    from omlx.cache.type_registry import CacheTypeRegistry
    from omlx.patches.inkling2 import cache_handlers, window_cache

    cache_handlers.register()
    prefix = [512] * 4 + [300]
    full, ahead = KVCache(), window_cache.WindowKVCache()
    _, seed = _feed([full, ahead], prefix, 700)
    tokens = full.offset
    # the snapshot was taken ``extra`` tokens later
    q, k, v, rel, sink = _chunk(extra, seed + 1, WINDOW)
    ahead.update_and_fetch(k, v)
    handler = CacheTypeRegistry.get_handler_by_class_name("WindowKVCache")
    restored = handler.deserialize_state(tuple(handler.serialize_state(ahead)), handler.serialize_meta_state(ahead))
    assert restored._restored and restored.offset == tokens + extra
    window_cache.realign([[restored]] + [[full]], [True, False])
    assert not restored._restored and restored.offset == tokens
    outs, _ = _feed([full, restored], [1, 5, 512, 1], seed + 2)
    assert mx.array_equal(outs[0], outs[1])


def test_batched_restored_windows_realign_per_row():
    from mlx_lm.models.cache import KVCache

    from omlx.patches.inkling2 import kernels, window_cache

    rows, fulls, seed = [], [], 900
    for steps, extra in (([512] * 3, 30), ([512, 300], 0)):
        full, win = KVCache(), window_cache.WindowKVCache()
        _, seed = _feed([full, win], steps, seed)
        if extra:
            _, k, v, _, _ = _chunk(extra, seed + 1, WINDOW)
            win.update_and_fetch(k, v)
        win._restored = True
        rows.append(win)
        fulls.append(full)
    batch = window_cache.WindowKVCache.merge(rows)
    position = mx.array([f.offset for f in fulls])
    batch.realign(position)
    assert [int(x) for x in batch.offset.tolist()] == [f.offset for f in fulls]
    seed += 9
    q, k, v, rel, sink = _chunk(1, seed, WINDOW)
    kk, vv = batch.update_and_fetch(mx.concatenate([k, k]), mx.concatenate([v, v]))
    for row, full in enumerate(fulls):
        fk, fv = full.update_and_fetch(k, v)
        ref = kernels.banded_sink_attention(q, fk, fv, rel, sink, scale=D**-0.5, window=WINDOW)
        lp = int(batch.left_padding[row].item())
        out = kernels.banded_sink_attention(
            q, kk[row : row + 1, :, lp:], vv[row : row + 1, :, lp:], rel, sink, scale=D**-0.5, window=WINDOW
        )
        assert mx.array_equal(out, ref)


# --- prefix-cache round trip through the real store (harness of test_prefix_cache_cachelist_mixed.py) ---

_PC_BLOCK = 512


def _position_chunk(start, length, heads=1, dim=8):
    pos = mx.arange(start, start + length, dtype=mx.float32).reshape(1, 1, length, 1)
    keys = mx.contiguous(mx.broadcast_to(pos, (1, heads, length, dim)))
    return keys, keys + 1_000_000.0


def _pc_layers(seq_len):
    """Layer 0: a window layer, layer 1: a global one, both with Inkling's 4 conv slots; fed in 512-token chunks."""
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    from omlx.patches.inkling2 import window_cache

    layers = []
    for kv in (window_cache.WindowKVCache(), KVCache()):
        for start in range(0, seq_len, 512):
            kv.update_and_fetch(*_position_chunk(start, min(512, seq_len - start)))
        arrays = ArraysCache(size=4)
        for i in range(4):
            arrays[i] = mx.full((1, 3, 16), seq_len + i / 10.0, dtype=mx.float32)
        layers.append(CacheList(kv, arrays))
    mx.eval([t for c in layers for t in (c[0].keys, c[0].values)])
    return layers


def _pc_data(seq_len):
    from omlx.cache.type_registry import CacheTypeRegistry

    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    out = []
    for layer in _pc_layers(seq_len):
        d = handler.extract_state(layer)
        out.append({"state": list(d["sub_states"]), "meta_state": (list(d["sub_class_names"]), list(d["sub_meta_states"])),
                    "class_name": "CacheList", "cache_type": "CacheList"})
    return out


def _pc_store(tmp_path, num_blocks):
    from unittest.mock import MagicMock

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.patches.inkling2 import cache_handlers

    cache_handlers.register()
    model = MagicMock()
    model.layers = [MagicMock(), MagicMock()]
    model.args.num_hidden_layers = 2
    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(block_size=_PC_BLOCK, max_blocks=100, model_name="m", initial_blocks=100),
        paged_ssd_cache_manager=PagedSSDCacheManager(cache_dir=tmp_path / "ssd", max_size_bytes=1024**3,
                                                     hot_cache_max_bytes=512 * 1024**2, hot_cache_only=True,
                                                     expected_model_name="m"),
    )
    tokens = list(range(num_blocks * _PC_BLOCK))
    snapshots = {_PC_BLOCK * (i + 1): _pc_data(_PC_BLOCK * (i + 1)) for i in range(num_blocks)}
    table = prefix.store_cache("req", tokens, _pc_data(len(tokens)), boundary_snapshots=snapshots)
    assert table is not None and len(table.block_ids) == num_blocks
    return prefix, table


def _assert_window_at(restored, tokens):
    from omlx.patches.inkling2 import window_cache

    win, glob = restored[0].caches[0], restored[1].caches[0]
    assert type(win) is window_cache.WindowKVCache
    assert win.offset == glob.offset == tokens
    keys, values = win.keys_and_values()
    n = keys.shape[2]
    assert n >= min(tokens, window_cache.KEEP)
    expect, expect_v = _position_chunk(tokens - n, n)
    assert mx.array_equal(keys, expect) and mx.array_equal(values, expect_v), "window holds other positions"
    for i, slot in enumerate(restored[0].caches[1].cache):
        assert mx.max(mx.abs(slot - (tokens + i / 10.0))).item() == 0.0, f"conv slot {i} from another boundary"


@pytest.mark.parametrize("num_blocks", [1, 3, 6])
def test_window_layers_round_trip_through_the_prefix_cache(tmp_path, num_blocks):
    prefix, table = _pc_store(tmp_path, num_blocks)
    _assert_window_at(prefix.reconstruct_cache(table), num_blocks * _PC_BLOCK)


@pytest.mark.parametrize("keep_blocks", [2, 4])
def test_window_layers_restore_a_shorter_prefix_at_its_boundary(tmp_path, keep_blocks):
    from omlx.cache.paged_cache import BlockTable

    prefix, table = _pc_store(tmp_path, 6)
    for bid in table.block_ids[:keep_blocks]:
        prefix.paged_cache.allocated_blocks[bid].ref_count += 1
    partial = BlockTable(request_id="r2", block_ids=list(table.block_ids[:keep_blocks]),
                         num_tokens=keep_blocks * _PC_BLOCK)
    _assert_window_at(prefix.reconstruct_cache(partial), keep_blocks * _PC_BLOCK)


@pytest.mark.parametrize("diverge", [1800, 1790, 1700, 1600, 1500])
def test_window_layers_restore_a_prompt_that_leaves_the_stored_tail(tmp_path, diverge):
    """A tail stored to 1,800 tokens (3 blocks + a terminal tail), then a prompt equal to it up to ``diverge``:
    whatever the prefix cache restores, the window holds the positions just before the restored length."""
    from unittest.mock import MagicMock

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.patches.inkling2 import cache_handlers

    cache_handlers.register()
    model = MagicMock()
    model.layers = [MagicMock(), MagicMock()]
    model.args.num_hidden_layers = 2
    prefix = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(block_size=_PC_BLOCK, max_blocks=100, model_name="m", initial_blocks=100),
        paged_ssd_cache_manager=PagedSSDCacheManager(cache_dir=tmp_path / "ssd", max_size_bytes=1024**3,
                                                     hot_cache_max_bytes=512 * 1024**2, hot_cache_only=True,
                                                     expected_model_name="m"),
    )
    stored = list(range(1800))
    snapshots = {t: _pc_data(t) for t in (512, 1024, 1536, 1800)}
    assert prefix.store_cache("a", stored, _pc_data(1800), boundary_snapshots=snapshots,
                              _store_tail_terminal=True) is not None
    prompt = stored[:diverge] + [10_000 + i for i in range(300)]
    table, _remaining = prefix.fetch_cache("b", prompt)
    if table is None or table.num_tokens == 0:
        pytest.skip(f"no prefix restored for a divergence at {diverge}")
    restored = prefix.reconstruct_cache(table)
    if restored is None:
        pytest.skip(f"restore of {table.num_tokens} tokens rejected")
    _assert_window_at(restored, table.num_tokens)


@pytest.mark.parametrize("diverge", [2040, 1990, 1700, 1537])
def test_window_layers_restore_a_prompt_that_leaves_a_stored_full_block(tmp_path, diverge):
    """Four full blocks stored with their boundary snapshots, then a prompt equal to them up to ``diverge``
    (inside the last block): the window must hold the positions just before the restored length."""
    prefix, _ = _pc_store(tmp_path, 4)
    prompt = list(range(diverge)) + [10_000 + i for i in range(300)]
    table, _remaining = prefix.fetch_cache("b", prompt)
    if table is None or table.num_tokens == 0:
        pytest.skip(f"no prefix restored for a divergence at {diverge}")
    restored = prefix.reconstruct_cache(table)
    if restored is None:
        pytest.skip(f"restore of {table.num_tokens} tokens rejected")
    _assert_window_at(restored, table.num_tokens)


def test_restored_window_extended_in_a_batch_and_stored_again_is_exact():
    """The server's chain: restore a window, merge it into a batch, prefill 44 tokens there, extract the row,
    store and restore it again, continue; every step equals the full cache."""
    from mlx_lm.models.cache import KVCache

    from omlx.cache.type_registry import CacheTypeRegistry
    from omlx.patches.inkling2 import cache_handlers, kernels, window_cache

    cache_handlers.register()
    handler = CacheTypeRegistry.get_handler_by_class_name("WindowKVCache")
    full, win = KVCache(), window_cache.WindowKVCache()
    _, seed = _feed([full, win], [512] * 5 + [200], 1100)

    def round_trip(cache):
        return handler.deserialize_state(tuple(handler.serialize_state(cache)), handler.serialize_meta_state(cache))

    restored = round_trip(win)
    batch = window_cache.WindowKVCache.merge([restored])
    batch.realign(mx.array([full.offset]))
    for length in (44, 1, 1):
        seed += 3
        q, k, v, rel, sink = _chunk(length, seed, WINDOW)
        kk, vv = batch.update_and_fetch(k, v)
        fk, fv = full.update_and_fetch(k, v)
        ref = kernels.banded_sink_attention(q, fk, fv, rel, sink, scale=D**-0.5, window=WINDOW)
        lp = int(batch.left_padding[0].item())
        out = kernels.banded_sink_attention(q, kk[:, :, lp:], vv[:, :, lp:], rel, sink, scale=D**-0.5, window=WINDOW)
        assert mx.array_equal(out, ref), f"batched step of {length} differs"
    again = round_trip(batch.extract(0))
    assert again.offset == full.offset
    outs, _ = _feed([full, again], [1, 300, 1], seed + 7)
    assert mx.array_equal(outs[0], outs[1])


def _token_layers(token_ids):
    """Like _pc_layers, keys = the token id at each position: a stored block shows which request it came from."""
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    from omlx.patches.inkling2 import window_cache

    ids = mx.array(token_ids, dtype=mx.float32)
    layers = []
    for kv in (window_cache.WindowKVCache(), KVCache()):
        for start in range(0, len(token_ids), 512):
            part = ids[start : start + 512].reshape(1, 1, -1, 1)
            keys = mx.contiguous(mx.broadcast_to(part, (1, 1, part.shape[2], 8)))
            kv.update_and_fetch(keys, keys + 1_000_000.0)
        arrays = ArraysCache(size=4)
        for i in range(4):
            arrays[i] = mx.full((1, 3, 16), len(token_ids) + i / 10.0, dtype=mx.float32)
        layers.append(CacheList(kv, arrays))
    return layers


def _token_data(token_ids):
    from omlx.cache.type_registry import CacheTypeRegistry

    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    out = []
    for layer in _token_layers(token_ids):
        d = handler.extract_state(layer)
        out.append({"state": list(d["sub_states"]), "meta_state": (list(d["sub_class_names"]), list(d["sub_meta_states"])),
                    "class_name": "CacheList", "cache_type": "CacheList"})
    return out


def test_blocks_stored_after_a_restore_hold_the_new_request_tokens(tmp_path):
    """The server's chain: A stored cold; B restores A's first blocks, prefills its own tokens, stores them; B again
    restores everything. The global layer must hold B's tokens: with the window layer first, the store measured the
    sequence on the window (1,350 keys) and sliced B's new blocks with relative indices (27/11)."""
    prefix, _ = _pc_store(tmp_path, 4)  # A: tokens 0..2047, value = position = token id
    b_tokens = list(range(1536)) + [50_000 + i for i in range(700)]
    table, _remaining = prefix.fetch_cache("b", b_tokens)
    assert table is not None and table.num_tokens == 1536
    restored = prefix.reconstruct_cache(table)
    assert restored is not None
    boundary = 2048  # B's own snapshot at its first new block boundary
    stored = prefix.store_cache("b", b_tokens[:2048], _token_data(b_tokens[:2048]),
                                boundary_snapshots={boundary: _token_data(b_tokens[:boundary])})
    assert stored is not None
    table2, _ = prefix.fetch_cache("c", b_tokens[:2048] + [99_999])
    assert table2 is not None and table2.num_tokens == 2048
    again = prefix.reconstruct_cache(table2)
    glob_keys, _ = again[1].caches[0].keys_and_values()
    got = glob_keys[0, 0, :, 0]
    assert mx.array_equal(got, mx.array(b_tokens[:2048], dtype=mx.float32)), "global KV of B's new block is not B's"
    win_keys, _ = again[0].caches[0].keys_and_values()
    n = win_keys.shape[2]
    assert mx.array_equal(win_keys[0, 0, :, 0], mx.array(b_tokens[2048 - n : 2048], dtype=mx.float32))
