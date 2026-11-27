# SPDX-License-Identifier: Apache-2.0
"""0ics: a key/value cache that keeps only what a sliding-window layer reads.

Inkling 2 Small has 40 of 48 layers with a 1,024-token window; the upstream ``make_cache`` gives every layer a full
``KVCache``, so the sliding layers keep every position and the cache grows at 192 KiB a token instead of the 32 KiB
of the 8 global layers. At 1M tokens that is ~192 GiB, which no 256 GB Mac holds beside the weights (oMLX's prefill
memory guard refused 128K+ with the 195 GB oQ2e, 27/11).

A sliding layer's attention (``kernels.banded_sink_attention``) reads keys at backward distance ``d < window`` from
each query, in buffer order, with ``left_padding`` as a buffer index. These caches drop the oldest keys before an
append, keeping at least ``KEEP = WINDOW - 1 + MARGIN``: every query of the next chunk still sees its full window, a
rollback of up to ``MARGIN`` drafted tokens (``trim``) stays inside the kept keys, and the copy runs once per
``SLACK`` tokens.

As in mlx-lm's rotating caches, ``offset`` is the number of tokens the cache has seen (oMLX reads it as the position:
boundary snapshots are checked against it, ``scheduler._first_leaf_cache_offset``) and the buffer has its own length,
``_len`` here, ``_idx`` in the batched form. A first version reported the buffer length as ``offset``: the prefix
cache then paired the window with another position and the restored model drifted (27/11, first-token log-probs
several nats off where upstream's caches stayed within 0.2).

Turned on by ``OMLX_0ICS_INKLING_WINDOW=1``; off, ``make_cache`` is upstream's.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

logger = logging.getLogger(__name__)

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
# a stored window carries this many keys more: a prefix-cache restore may land up to a few hundred tokens before the
# snapshot it takes the window from (realign), and the extra keys keep the window whole there
TAIL_EXTRA = 256
_STEP = 256


def _drop(length: int) -> int:
    """Keys to drop from a buffer of ``length`` before an append; 0 until it holds SLACK keys past KEEP."""
    if length <= KEEP + SLACK:
        return 0
    return (length - KEEP) // ALIGN * ALIGN


def window_enabled(window: int) -> bool:
    return os.environ.get("OMLX_0ICS_INKLING_WINDOW", "0") == "1" and window == WINDOW


class WindowKVCache(KVCache):
    """One request's cache of a sliding layer: ``offset`` tokens seen, the last ``_len`` of them in the buffer.

    ``max_size``/``keep`` let the prefill memory guard (``memory_monitor.collect_kv_layer_specs``) count the layer as
    a sliding one, resident at ``window + chunk - 1`` tokens, instead of a full layer growing with the prompt.
    """

    max_size = WINDOW
    keep = 0

    def __init__(self):
        super().__init__()
        self._len = 0
        # set by the prefix-cache handler on a rebuilt cache: the next forward realigns it to the global layers
        self._restored = False

    def update_and_fetch(self, keys, values):
        drop = _drop(self._len) if self.keys is not None else 0
        if drop:
            self.keys = self.keys[..., drop : self._len, :]
            self.values = self.values[..., drop : self._len, :]
            self._len -= drop
        prev, n = self._len, keys.shape[2]
        if self.keys is None or prev + n > self.keys.shape[2]:
            b, heads, _, k_dim = keys.shape
            grow = (n + _STEP - 1) // _STEP * _STEP
            new_k = mx.zeros((b, heads, grow, k_dim), keys.dtype)
            new_v = mx.zeros((b, heads, grow, values.shape[3]), values.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                self.keys = mx.concatenate([self.keys[..., :prev, :], new_k], axis=2)
                self.values = mx.concatenate([self.values[..., :prev, :], new_v], axis=2)
        self.keys[..., prev : prev + n, :] = keys
        self.values[..., prev : prev + n, :] = values
        self._len += n
        self.offset += n
        return self.keys[..., : self._len, :], self.values[..., : self._len, :]

    def keys_and_values(self):
        return self.keys[..., : self._len, :], self.values[..., : self._len, :]

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset
        return (*self.keys_and_values(), self.offset)

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset = v
        self._len = 0 if self.keys is None else int(self.keys.shape[2])

    def trim(self, n):
        n = min(self._len, n)
        self._len -= n
        self.offset -= n
        return n

    def realign(self, tokens: int) -> None:
        """Drop the keys past ``tokens``, the global layers' position after a prefix-cache restore (window states
        come from the snapshot at or after the restored length; see ``realign``)."""
        self._restored = False
        extra = self.offset - int(tokens)
        if extra < 0:
            raise RuntimeError(f"window cache restored at {self.offset} tokens, behind the global layers' {tokens}")
        if extra:
            self.trim(extra)
            logger.info("window cache realigned to the global layers: %d tokens dropped, %d kept", extra, self._len)
            if self._len < WINDOW - 1 and self.offset > self._len:
                logger.warning("window cache realigned by %d tokens holds %d keys, under its window", extra, self._len)

    @classmethod
    def merge(cls, caches):
        return BatchWindowKVCache.merge(caches)


class BatchWindowKVCache(BatchKVCache):
    """The batched form: ``_idx`` and ``left_padding`` are buffer indices, shifted together on a drop; ``offset``
    keeps each row's token count."""

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

    @classmethod
    def merge(cls, caches):
        """Right-justified buffers as ``BatchKVCache.merge`` builds them, from each cache's buffer (``_len``), not
        its token count; the rows keep their token counts as ``offset``."""
        lengths = [c._len for c in caches]
        longest = max(lengths)
        if longest == 0:
            return cls([0] * len(caches))
        padding = [longest - n for n in lengths]
        first = next(c for c in caches if c.keys is not None)
        b, heads, k_dim, v_dim = len(caches), first.keys.shape[1], first.keys.shape[3], first.values.shape[3]
        keys = mx.zeros((b, heads, longest, k_dim), dtype=first.keys.dtype)
        values = mx.zeros((b, heads, longest, v_dim), dtype=first.values.dtype)
        for i, (p, c) in enumerate(zip(padding, caches, strict=True)):
            if c._len:
                keys[i : i + 1, :, p:] = c.keys[..., : c._len, :]
                values[i : i + 1, :, p:] = c.values[..., : c._len, :]
        batch = cls(padding)
        batch.keys, batch.values, batch._idx = keys, values, longest
        batch.offset = mx.array([c.offset for c in caches])
        batch._restored = any(getattr(c, "_restored", False) for c in caches)
        return batch

    def realign(self, tokens) -> None:
        """Per row, ``WindowKVCache.realign``: the extra keys of a row leave at its right end, which
        ``BatchKVCache.finalize`` does for a right padding (the row rolls right, its left padding grows)."""
        self._restored = False
        extra = self.offset - tokens
        if bool(mx.any(extra < 0).item()):
            raise RuntimeError("batched window cache restored behind the global layers")
        if bool(mx.any(extra > 0).item()):
            logger.info("batched window cache realigned to the global layers: %s tokens dropped", extra.tolist())
            self._right_padding = extra
            self.finalize()

    _restored = False

    def extract(self, idx):
        base = super().extract(idx)
        single = WindowKVCache()
        single.keys, single.values = base.keys, base.values
        single._len = int(base.keys.shape[2])
        single.offset = int(self.offset[idx].item())
        return single


def realign(caches, sliding) -> None:
    """Before a forward: if a prefix-cache restore rebuilt window caches, cut them to the global layers' position.

    oMLX restores a match that ends inside its last stored block with the KV of the global layers sliced to the
    match and the non-sliceable states (window, convolutions) from the snapshot at the block's end, a few tokens
    further: harmless for a 3-tap convolution, not for a window, which would read those tokens twice for the next
    1,024 (27/11: first-token log-probs several nats off). ``sliding[i]`` says whether layer ``i`` is a window layer;
    the first other layer's cache gives the position.
    """
    if not caches or not any(s and getattr(c[0], "_restored", False) for c, s in zip(caches, sliding, strict=True)):
        return
    position = next(c[0].offset for c, s in zip(caches, sliding, strict=True) if not s)
    for c, s in zip(caches, sliding, strict=True):
        if s and getattr(c[0], "_restored", False):
            c[0].realign(position)
