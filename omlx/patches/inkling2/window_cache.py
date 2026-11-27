# SPDX-License-Identifier: Apache-2.0
"""0ics: a key/value cache that keeps only what a sliding-window layer reads.

Inkling 2 Small has 40 of 48 layers with a 1,024-token window; the upstream ``make_cache`` gives every layer a full
``KVCache``, so the sliding layers keep every position and the cache grows at 192 KiB a token instead of the 32 KiB
of the 8 global layers. At 1M tokens that is ~192 GiB, which no 256 GB Mac holds beside the weights (oMLX's prefill
memory guard refused 128K+ with the 195 GB oQ2e, 27/11).

A sliding layer's attention (``kernels.banded_sink_attention``) reads keys at backward distance ``d < window`` from
each query, in buffer order, with ``left_padding`` as a buffer index. These caches drop the oldest keys before an
append, keeping at least ``KEEP = WINDOW - 1 + MARGIN``: every query of the next chunk still sees its full window, a rollback
of up to ``MARGIN`` drafted tokens (``trim``) stays inside the kept keys, and the copy runs once per ``SLACK`` tokens.
The positions they report (``offset``) are buffer lengths; a sliding layer uses no absolute position (``log_tau`` is
for the global layers) and ``CacheList.size()`` takes the global layers' full length.

Turned on by ``OMLX_0ICS_INKLING_WINDOW=1``; off, ``make_cache`` is upstream's.
"""

from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

# Inkling 2 Mini and Small: sliding_window_size 1024
WINDOW = 1024
# keys kept past the window: a speculative verify rolls back at most a block of drafts (DSpark block 8, verify <= 16)
MARGIN = 64
KEEP = WINDOW - 1 + MARGIN
# compact once the buffer holds this many keys past KEEP: one copy of the window per SLACK appended tokens
SLACK = 256
# drops are whole tiles of the flash kernel (kernels._FLASH_SOURCE, BK = 16): its key tiles start at buffer indices
# that are multiples of 16, so a drop of any other size regroups the online softmax and rounds differently (27/11:
# 1.4 logits apart after a drop of 449); a multiple of 16 keeps the same keys in the same tiles, bit for bit
ALIGN = 16


def _drop(length: int) -> int:
    """Keys to drop from a buffer of ``length`` before an append; 0 until it holds SLACK keys past KEEP."""
    if length <= KEEP + SLACK:
        return 0
    return (length - KEEP) // ALIGN * ALIGN


def window_enabled(window: int) -> bool:
    return os.environ.get("OMLX_0ICS_INKLING_WINDOW", "0") == "1" and window == WINDOW


class WindowKVCache(KVCache):
    """One request's cache of a sliding layer; ``offset`` is the buffer length.

    ``max_size``/``keep`` let the prefill memory guard (``memory_monitor.collect_kv_layer_specs``) count the layer as
    a sliding one, resident at ``window + chunk - 1`` tokens, instead of a full layer growing with the prompt.
    """

    max_size = WINDOW
    keep = 0

    def update_and_fetch(self, keys, values):
        drop = _drop(self.offset) if self.keys is not None else 0
        if drop:
            self.keys = self.keys[..., drop : self.offset, :]
            self.values = self.values[..., drop : self.offset, :]
            self.offset -= drop
        return super().update_and_fetch(keys, values)

    @classmethod
    def merge(cls, caches):
        return BatchWindowKVCache.merge(caches)


class BatchWindowKVCache(BatchKVCache):
    """The batched form: ``_idx`` and ``left_padding`` are buffer indices, shifted together on a drop."""

    max_size = WINDOW
    keep = 0

    def update_and_fetch(self, keys, values):
        drop = _drop(self._idx) if self.keys is not None else 0
        if drop:
            self.keys = self.keys[..., drop : self._idx, :]
            self.values = self.values[..., drop : self._idx, :]
            self._idx -= drop
            self.left_padding = mx.maximum(self.left_padding - drop, 0)
        return super().update_and_fetch(keys, values)

    def extract(self, idx):
        base = super().extract(idx)
        single = WindowKVCache()
        single.keys, single.values, single.offset = base.keys, base.values, base.offset
        return single
