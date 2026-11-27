# SPDX-License-Identifier: Apache-2.0
"""0ics: prefix-cache handlers for the Inkling 2 sliding-window caches (window_cache.py).

Without a handler oMLX refuses prefix-cache storage and reuse for the whole model ("unsupported CacheList layers",
scheduler._model_has_unreconstructible_cache), so every request of a conversation reads its whole prompt again.

A window cache is not sliceable by token: it holds only the last keys. Like PoolingCache (patches/deepseek_v4), it
takes the boundary-snapshot / last-block path: the state stored is the last ``KEEP`` (+ < 16) keys and values in time order,
and the cache rebuilt from it is a ``WindowKVCache`` whose buffer starts with them, which is all a sliding layer
reads (window_cache.py).
"""

from __future__ import annotations

import logging
from typing import Any

from omlx.cache.type_handlers import CacheStateAxisInfo, CacheType, CacheTypeHandler

from . import window_cache

logger = logging.getLogger(__name__)


def _tail(keys, values):
    """The last keys a sliding layer still reads, at least ``KEEP``, starting where the buffer's 16-key tiles start.

    A window cache's buffer starts at an absolute position that is a multiple of ``ALIGN`` (its drops are), so its
    length and the token count agree modulo ``ALIGN``; keeping ``KEEP + (length - KEEP) % ALIGN`` keys makes the
    rebuilt buffer start on a tile too, and the flash kernel then groups the same keys as before (bit for bit).
    """
    if keys is None or values is None:
        return None, None
    length = keys.shape[2]
    if length <= window_cache.KEEP:
        return keys, values
    n = window_cache.KEEP + (length - window_cache.KEEP) % window_cache.ALIGN
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
        return _tail(*cache_obj.keys_and_values())

    def serialize_meta_state(self, cache_obj: Any) -> Any:
        return ()

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        keys, values = self.serialize_state(cache_obj)
        return {
            "keys": keys,
            "values": values,
            "offset": 0 if keys is None else int(keys.shape[2]),
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        keys = state.get("keys")
        return int(keys.shape[2]) if keys is not None and hasattr(keys, "shape") else 0

    def slice_state(self, state: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, Any] | None:
        return {**state, "is_full_state": True}

    def concatenate_states(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        return states[-1] if states else {}

    def reconstruct_cache(self, state: dict[str, Any], meta_state: tuple | None = None) -> Any:
        keys, values = _tail(state.get("keys"), state.get("values"))
        if keys is None:
            return None
        cache = window_cache.WindowKVCache()
        cache.keys, cache.values, cache.offset = keys, values, int(keys.shape[2])
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


def register() -> None:
    """Register the handlers and their class names (called by ``apply_inkling2_patch``)."""
    from omlx.cache.type_registry import CacheTypeRegistry

    CacheTypeRegistry.register(WindowKVCacheHandler())
    CacheTypeRegistry.register(BatchWindowKVCacheHandler())
    CacheTypeRegistry._class_name_map["WindowKVCache"] = CacheType.INKLING2_WINDOW_KVCACHE
    CacheTypeRegistry._class_name_map["BatchWindowKVCache"] = CacheType.INKLING2_BATCH_WINDOW_KVCACHE
