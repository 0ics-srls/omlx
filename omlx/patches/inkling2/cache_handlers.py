# SPDX-License-Identifier: Apache-2.0
"""0ics: prefix-cache handlers for the Inkling 2 sliding-window caches (window_cache.py).

Without a handler oMLX refuses prefix-cache storage and reuse for the whole model ("unsupported CacheList layers",
scheduler._model_has_unreconstructible_cache), so every request of a conversation reads its whole prompt again.

A window cache is not sliceable by token: it holds only the last keys. Like PoolingCache (patches/deepseek_v4), it
takes the boundary-snapshot / last-block path: the state stored is the last ``KEEP + TAIL_EXTRA`` (+ < 16) keys and values in time order,
and the cache rebuilt from it is a ``WindowKVCache`` whose buffer starts with them, which is all a sliding layer
reads (window_cache.py).
"""

from __future__ import annotations

import logging
from typing import Any

from omlx.cache.type_handlers import CacheStateAxisInfo, CacheType, CacheTypeHandler

from . import window_cache

logger = logging.getLogger(__name__)


def _tail(keys, values, tokens: int):
    """The last keys a sliding layer still reads, ``KEEP`` plus ``TAIL_EXTRA`` for a restore that lands before the
    snapshot (``window_cache.realign``), starting where the buffer's 16-key tiles start.

    A window cache's buffer starts at an absolute position (``tokens`` minus its length) that is a multiple of
    ``ALIGN``, its drops are; keeping ``n = KEEP + TAIL_EXTRA + (tokens - KEEP - TAIL_EXTRA) % ALIGN`` keys makes the
    rebuilt buffer start on a tile too, and the flash kernel then groups the same keys as before (bit for bit).
    """
    if keys is None or values is None:
        return None, None
    length = keys.shape[2]
    keep = window_cache.KEEP + window_cache.TAIL_EXTRA
    n = keep + (tokens - keep) % window_cache.ALIGN
    if length <= n:
        return keys, values
    return keys[..., -n:, :], values[..., -n:, :]


class WindowKVCacheHandler(CacheTypeHandler):
    """``window_cache.WindowKVCache`` (one request)."""

    @property
    def cache_type(self) -> CacheType:
        return CacheType.INKLING2_WINDOW_KVCACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return (
            CacheStateAxisInfo(name="keys", sequence_axis=2, sliceable=False),
            CacheStateAxisInfo(name="values", sequence_axis=2, sliceable=False),
        )

    def serialize_state(self, cache_obj: Any) -> tuple[Any, ...]:
        if cache_obj.keys is None:
            return None, None
        return _tail(*cache_obj.keys_and_values(), int(cache_obj.offset))

    def serialize_meta_state(self, cache_obj: Any) -> Any:
        """(keys stored, token count). The CacheList restore of prefix_cache rewrites the first element of a sub's
        metadata to its tensor length, as for a KV cache (27/11: a token count first came back as 1,092 instead of
        26,596); the count rides second, where that rewrite keeps it."""
        keys, _ = self.serialize_state(cache_obj)
        return (0 if keys is None else int(keys.shape[2]), int(cache_obj.offset))

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        keys, values = self.serialize_state(cache_obj)
        return {
            "keys": keys,
            "values": values,
            "offset": int(cache_obj.offset),
            "meta_state": self.serialize_meta_state(cache_obj),
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        return int(state.get("offset") or 0)

    def slice_state(self, state: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, Any] | None:
        return {**state, "is_full_state": True}

    def concatenate_states(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        return states[-1] if states else {}

    def reconstruct_cache(self, state: dict[str, Any], meta_state: tuple | None = None) -> Any:
        tokens = state.get("offset")
        if isinstance(meta_state, list | tuple) and len(meta_state) >= 2:
            tokens = meta_state[1]
        if tokens is None or state.get("keys") is None:
            # without the token count the window cannot be placed: the caller prefills again
            logger.warning("WindowKVCache restore without its token count; rejecting the cached state")
            return None
        keys, values = _tail(state.get("keys"), state.get("values"), int(tokens))
        cache = window_cache.WindowKVCache()
        cache.state = (keys, values, int(tokens))
        cache._restored = True
        return cache

    def deserialize_state(self, elements: tuple[Any, ...], meta_state: Any | None = None) -> Any:
        if not isinstance(elements, list | tuple) or len(elements) < 2:
            logger.error("WindowKVCache deserialize: expected (keys, values), got %r", type(elements).__name__)
            return None
        return self.reconstruct_cache({"keys": elements[0], "values": elements[1]}, meta_state)


class BatchWindowKVCacheHandler(WindowKVCacheHandler):
    """``window_cache.BatchWindowKVCache``: stored per request as the single-request form (``extract``)."""

    @property
    def cache_type(self) -> CacheType:
        return CacheType.INKLING2_BATCH_WINDOW_KVCACHE

    def serialize_state(self, cache_obj: Any) -> tuple[Any, ...]:
        if cache_obj.keys is None:
            return None, None
        if cache_obj.keys.shape[0] != 1:
            raise ValueError("a batched window cache is stored one request at a time (extract first)")
        return WindowKVCacheHandler.serialize_state(self, cache_obj.extract(0))

    def serialize_meta_state(self, cache_obj: Any) -> Any:
        return WindowKVCacheHandler.serialize_meta_state(self, cache_obj.extract(0))

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        return WindowKVCacheHandler.extract_state(self, cache_obj.extract(0))


_WINDOW_CLASSES = frozenset({"WindowKVCache", "BatchWindowKVCache"})


def _hide_window_members(layer_state: Any) -> Any:
    """A CacheList layer's extracted state with its window members blanked to ``()``."""
    if not isinstance(layer_state, dict) or layer_state.get("class_name") != "CacheList":
        return layer_state
    names = layer_state.get("sub_class_names")
    meta = layer_state.get("meta_state")
    if not names and isinstance(meta, list | tuple) and meta and isinstance(meta[0], list | tuple):
        names = meta[0]
    names = [str(n) for n in names or ()]
    if not any(n in _WINDOW_CLASSES for n in names):
        return layer_state
    state = layer_state.get("state") or []
    hidden = [() if j < len(names) and names[j] in _WINDOW_CLASSES else s for j, s in enumerate(state)]
    return {**layer_state, "state": hidden}


def _patch_prefix_cache_seq_len() -> None:
    """``BlockAwarePrefixCache._get_cache_seq_len`` takes, for an all-CacheList model, the first 4D member it finds
    as the sequence length. Inkling's layer 0 is a window layer, so a store measured the sequence on the window
    (~1,350 keys), took the cache for a suffix-only one and sliced the new blocks of a request that had restored a
    prefix with relative indices: the global layers' KV of those blocks came from the wrong tokens, and every later
    restore of them drifted (27/11: an argmax flipped against the cold run). With the window members hidden, the
    length comes from a global layer's KV, the full sequence."""
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    original = BlockAwarePrefixCache._get_cache_seq_len
    if getattr(original, "_0ics_windows_hidden", False):
        return

    def _get_cache_seq_len(self, cache_data):
        return original(self, [_hide_window_members(layer) for layer in cache_data] if cache_data else cache_data)

    _get_cache_seq_len._0ics_windows_hidden = True
    BlockAwarePrefixCache._get_cache_seq_len = _get_cache_seq_len


def register() -> None:
    """Register the handlers and their class names, and have the prefix cache measure sequences on the global
    layers (called by ``apply_inkling2_patch``)."""
    from omlx.cache.type_registry import CacheTypeRegistry

    CacheTypeRegistry.register(WindowKVCacheHandler())
    CacheTypeRegistry.register(BatchWindowKVCacheHandler())
    CacheTypeRegistry._class_name_map["WindowKVCache"] = CacheType.INKLING2_WINDOW_KVCACHE
    CacheTypeRegistry._class_name_map["BatchWindowKVCache"] = CacheType.INKLING2_BATCH_WINDOW_KVCACHE
    _patch_prefix_cache_seq_len()
