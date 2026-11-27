# SPDX-License-Identifier: Apache-2.0
"""0ics: attention of the global layers for decode and speculative verify (1-8 queries), split over the keys.

Measured on Inkling 2 Small, M5 Ultra (27/11): generation fell from 57.7 t/s at 30K to 33.8 at 128K with DSpark on.
A verify of L > 1 queries goes through ``kernels._masked_attention``, which prepends a zero key to the whole cache
(``mx.concatenate([zero, k])`` and the same for v): every cycle copies the 8 global layers' cache, 4 GB at 128K, and
the SDPA behind it reads K/V once per query head and per query row. The decode of one query uses MLX's vector SDPA,
which reads the cache at ~400 GB/s.

Here one threadgroup takes one key/value head and one slice of the keys: the 6 query heads of the GQA group times the
L queries (``NR = 6 L`` rows) read each key and value once, from a threadgroup tile. One lane per key computes the
scores, the online softmax runs per row, the sink and the relative bias are applied as the masked path applies them
(the sink is a zero-value logit, the bias is added for backward distances below ``R``). A second step combines the
slices with the sinks.

On with ``OMLX_0ICS_INKLING_ATTN=1``, for the global layers (no window), batch 1, no left padding, 1-8 queries,
a bfloat16 cache. The same switch sends the global layers' prefill chunks to the flash kernel on long caches
(``FLASH_MIN_KEYS``, see ``kernels._use_flash``).
"""

from __future__ import annotations

import math
import os

import mlx.core as mx

MAX_QUERIES = 8
MIN_KEYS_ONE_QUERY = 65536
# global layers' prefill chunks take kernels.flash_banded_attention from this many keys (kernels._use_flash)
FLASH_MIN_KEYS = 65536
_BK = 32
_THREADS = 256
_MAX_SPLITS = 128
_MIN_CHUNK = 256
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453

_SOURCE = """
    constexpr int GQ = HQ / HKV;
    constexpr int NR = GQ * L;
    constexpr int PER_SG = (NR + 7) / 8;
    constexpr int LDK = D + 2;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int split = threadgroup_position_in_grid.x;
    const int hk = threadgroup_position_in_grid.y;
    const int S = dims[0];
    const int CH = dims[1];
    const int NS = dims[2];
    const int k_lo = split * CH;
    const int k_hi = min(S, k_lo + CH);
    const float qscale = scale[0] * LOG2E;

    threadgroup half Qs[NR * D];
    threadgroup half Ks[BK * LDK];
    threadgroup half Vs[BK * D];

    // query rows r = g * L + l: head hk * GQ + g, query l
    for (int i = tid; i < NR * D; i += 256) {
        const int r = i / D;
        const int c = i % D;
        const int qh = hk * GQ + r / L;
        const int l = r % L;
        Qs[i] = half(float(q[qh * q_strides[1] + l * q_strides[2] + c * q_strides[3]]));
    }

    float m[PER_SG];
    float lsum[PER_SG];
    float o[PER_SG][D / 32];
    for (int i = 0; i < PER_SG; ++i) {
        m[i] = -INFINITY;
        lsum[i] = 0.0f;
        for (int j = 0; j < D / 32; ++j) {
            o[i][j] = 0.0f;
        }
    }
    const device T* kb = k + hk * k_strides[1];
    const device T* vb = v + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int t0 = k_lo; t0 < k_hi; t0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!vec_kv) {
            for (int i = tid; i < BK * D; i += 256) {
                const int kk = i / D;
                const int c = i % D;
                const int key = t0 + kk;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < k_hi) {
                    kv = float(kb[key * k_strides[2] + c * k_strides[3]]);
                    vv = float(vb[key * v_strides[2] + c * v_strides[3]]);
                }
                Ks[kk * LDK + c] = half(kv);
                Vs[kk * D + c] = half(vv);
            }
        }
        // 8 bfloat16 values (16 bytes) per load; a bfloat16 is the top half of a float
        for (int i = tid; vec_kv && i < BK * D / 8; i += 256) {
            const int kk = i / (D / 8);
            const int c = (i % (D / 8)) * 8;
            const int key = t0 + kk;
            uint4 kw = uint4(0);
            uint4 vw = uint4(0);
            if (key < k_hi) {
                kw = *(const device uint4*)(kb + key * k_strides[2] + c);
                vw = *(const device uint4*)(vb + key * v_strides[2] + c);
            }
            threadgroup half* kd = Ks + kk * LDK + c;
            threadgroup half* vd = Vs + kk * D + c;
            for (int j = 0; j < 4; ++j) {
                kd[2 * j] = half(as_type<float>(kw[j] << 16));
                kd[2 * j + 1] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                vd[2 * j] = half(as_type<float>(vw[j] << 16));
                vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int key = t0 + int(lane);
        for (int i = 0; i < PER_SG; ++i) {
            const int r = int(sg) + 8 * i;
            if (r >= NR) {
                break;
            }
            const int qh = hk * GQ + r / L;
            const int l = r % L;
            const int d = (S - L + l) - key;
            const bool vis = key < k_hi && d >= 0;
            float s = 0.0f;
            const threadgroup half* qr = Qs + r * D;
            const threadgroup half* kr = Ks + int(lane) * LDK;
            for (int c = 0; c < D; c += 2) {
                const float2 qq = float2(qr[c], qr[c + 1]);
                const float2 kk2 = float2(kr[c], kr[c + 1]);
                s += qq.x * kk2.x + qq.y * kk2.y;
            }
            s *= qscale;
            if (vis && d < R) {
                s += float(rel[(l * HQ + qh) * R + d]) * LOG2E;
            }
            s = vis ? s : -INFINITY;
            const float tmax = simd_max(s);
            const float m_new = max(m[i], tmax);
            if (m_new == -INFINITY) {
                continue;
            }
            const float p = vis ? fast::exp2(s - m_new) : 0.0f;
            const float factor = fast::exp2(m[i] - m_new);
            lsum[i] = lsum[i] * factor + simd_sum(p);
            for (int j = 0; j < D / 32; ++j) {
                o[i][j] *= factor;
            }
            for (int kk = 0; kk < BK; ++kk) {
                const float pk = simd_shuffle(p, ushort(kk));
                const threadgroup half* vr = Vs + kk * D + int(lane) * (D / 32);
                for (int j = 0; j < D / 32; ++j) {
                    o[i][j] += pk * float(vr[j]);
                }
            }
            m[i] = m_new;
        }
    }

    for (int i = 0; i < PER_SG; ++i) {
        const int r = int(sg) + 8 * i;
        if (r >= NR) {
            break;
        }
        const size_t at = ((size_t)hk * NR + r) * NS + split;
        if (lane == 0) {
            part_m[at] = m[i];
            part_l[at] = lsum[i];
        }
        for (int j = 0; j < D / 32; ++j) {
            part_o[at * D + int(lane) * (D / 32) + j] = o[i][j];
        }
    }
"""

_kernel = None


def enabled() -> bool:
    return os.environ.get("OMLX_0ICS_INKLING_ATTN", "0") == "1"


def supports(q, window: int, left_padding, keys: int) -> bool:
    """Global layers, batch 1, 2-8 queries; one query only from ``MIN_KEYS_ONE_QUERY`` keys, below which MLX's
    vector SDPA is faster (27/11, M5 Ultra: 0.31 against 0.41 ms at 4K, 1.42 against 1.07 ms at 128K)."""
    b, _, length, d = q.shape
    if window != 0 or left_padding is not None or b != 1 or d % 32 != 0 or not 1 <= length <= MAX_QUERIES:
        return False
    return length > 1 or keys >= MIN_KEYS_ONE_QUERY


def supports_cache(k, v) -> bool:
    """A bfloat16 cache; a head dimension that is not contiguous takes the kernel's scalar loads."""
    return k.dtype == mx.bfloat16 and v.dtype == mx.bfloat16


def _chunk(s: int) -> int:
    """Keys per slice: at least _MIN_CHUNK, at most _MAX_SPLITS slices, a multiple of the key tile."""
    ch = max(_MIN_CHUNK, math.ceil(s / _MAX_SPLITS))
    return (ch + _BK - 1) // _BK * _BK


def attention(q, k, v, rel, sink, *, scale: float) -> mx.array:
    """``banded_sink_attention`` of a global layer: q [1, Hq, L, D], k/v [1, Hkv, S, D] (views of the cache),
    rel [1, L, Hq, R], sink [1, L, Hq]; returns [1, Hq, L, D] in q's dtype."""
    global _kernel
    _, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="omlx_0ics_inkling2_split_attn",
            input_names=["q", "k", "v", "rel", "dims", "scale"],
            output_names=["part_m", "part_l", "part_o"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
    ch = _chunk(s)
    ns = math.ceil(s / ch)
    nr = hq // hkv * length
    part_m, part_l, part_o = _kernel(
        inputs=[q, k, v, mx.contiguous(rel), mx.array([s, ch, ns], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", k.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("L", length), ("R", rel.shape[-1]),
                  ("BK", _BK)],
        grid=(ns * _THREADS, hkv, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(hkv, nr, ns), (hkv, nr, ns), (hkv, nr, ns, d)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    # sinks: [1, L, Hq] → rows (hk, g, l) as the kernel orders them; part_m is in the log2 domain
    sk = sink[0].astype(mx.float32).transpose(1, 0).reshape(hkv, nr) * _LOG2E
    top = mx.maximum(part_m.max(axis=-1), sk)
    w = mx.exp((part_m - top[..., None]) * _LN2)
    total = mx.exp((sk - top) * _LN2) + (part_l * w).sum(axis=-1)
    out = (part_o * w[..., None]).sum(axis=-2) / total[..., None]
    return out.reshape(1, hq, length, d).astype(q.dtype)
