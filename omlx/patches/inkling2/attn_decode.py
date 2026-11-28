# SPDX-License-Identifier: Apache-2.0
"""0ics: attention of the global layers for decode and speculative verify (1-16 queries), split over the keys.

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

On with ``OMLX_0ICS_INKLING_ATTN=1``, for the global layers (no window), batch 1, no left padding, 1-16 queries,
a bfloat16 cache. The same switch sends the global layers' prefill chunks to the flash kernel on long caches
(``FLASH_MIN_KEYS``, see ``kernels._use_flash``).
"""

from __future__ import annotations

import math
import os

import mlx.core as mx

# DSpark verifies 9 tokens (anchor + 8 drafts); the fused layer takes up to fused_decode.MAX_TOKENS = 16
MAX_QUERIES = 16
MIN_KEYS_ONE_QUERY = 65536
# global layers' prefill chunks take kernels.flash_banded_attention from this many keys (kernels._use_flash)
FLASH_MIN_KEYS = 65536
_BK = 32
_THREADS = 256
_MAX_SPLITS = 128
_MIN_CHUNK = 256

# The slices run over positions of a key list: with SEL, the first dims[4] are the dims[6] blocks of BLK keys of
# ``sel`` (one row per key/value head, see attn_sparse.py), the rest the keys from dims[5] to the end of the cache;
# without SEL, the keys from dims[3]. Loads and backward distances take the key's index in the cache, so the
# relative bias and the causal mask are the full attention's on the keys read.
_PHYS = """
#define PHYS(x) (SEL ? ((x) < dims[4] ? sel[hk * dims[6] + (x) / BLK] * BLK + (x) % BLK : dims[5] + (x) - dims[4]) : (x))
"""

_SOURCE = _PHYS + """
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
    const int k_lo = dims[3] + split * CH;
    const int k_hi = min(dims[7], k_lo + CH);
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
                    const int pk = PHYS(key);
                    kv = float(kb[pk * k_strides[2] + c * k_strides[3]]);
                    vv = float(vb[pk * v_strides[2] + c * v_strides[3]]);
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
                const int pk = PHYS(key);
                kw = *(const device uint4*)(kb + pk * k_strides[2] + c);
                vw = *(const device uint4*)(vb + pk * v_strides[2] + c);
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
            const int d = (S - L + l) - (key < k_hi ? PHYS(key) : key);
            const bool vis = key < k_hi && d >= 0 && (W == 0 || d < W);
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

# The same partials for 2-16 queries on 8x8 simdgroup matrices, as kernels._FLASH_SOURCE computes a prefill block: one
# threadgroup of NSG simdgroups, 8 rows each, takes all the GQ L rows of one key/value head (up to _MMA_MAX_SG
# simdgroups) and one slice of the keys, in 16-key tiles, so each slice of the cache is read once for the whole verify.
# With a fixed 32-row block DSpark's verify of 9 (54 rows on Inkling 2 Small) read the cache twice. The scalar kernel
# above stays for one query (6 rows would fill a quarter of a 32-row block).
_MMA_SOURCE = _PHYS + """
    constexpr int BQ = 8 * NSG;
    constexpr int THREADS = 32 * NSG;
    constexpr int BKM = 16;
    constexpr int GQ = HQ / HKV;
    constexpr int NR = GQ * L;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int LDK = BKM + 8;
    constexpr int LDV = D + 8;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int split = threadgroup_position_in_grid.x;
    const int hk = threadgroup_position_in_grid.y;
    const int q0 = threadgroup_position_in_grid.z * BQ;
    const int S = dims[0];
    const int CH = dims[1];
    const int NS = dims[2];
    const int k_lo = dims[3] + split * CH;
    const int k_hi = min(dims[7], k_lo + CH);
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < NR;
    const int qh = hk * GQ + (row_ok ? row / L : 0);
    const int lq = row_ok ? row % L : 0;
    const int a_row = S - L + lq;
    const float qscale = scale[0] * LOG2E;

    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BKM * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + qh * q_strides[1] + lq * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -INFINITY;
    float l = 0.0f;
    const device T* relp = rel + ((size_t)lq * HQ + qh) * R;
    const device T* kb_ptr = k + hk * k_strides[1];
    const device T* vb_ptr = v + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = k_lo; k0 < k_hi; k0 += BKM) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            for (int c = tid; c < BKM * D / 8; c += THREADS) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < k_hi) {
                    const int pk = PHYS(key);
                    kw = *(const device uint4*)(kb_ptr + pk * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + pk * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = tid; c < BKM * D; c += THREADS) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < k_hi) {
                    const int pk = PHYS(key);
                    kv = float(kb_ptr[pk * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[pk * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= NR) {
            continue;
        }

        simdgroup_float8x8 Sf[BKM / 8];
        for (int kb = 0; kb < BKM / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }
        float rmax = -INFINITY;
        for (int kb = 0; kb < BKM / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = a_row - (key < k_hi ? PHYS(key) : key);
                const bool vis = row_ok && key < k_hi && d >= 0 && (W == 0 || d < W);
                float sc = Sf[kb].thread_elements()[e] * qscale;
                if (vis) {
                    if (d < R) {
                        sc += float(relp[d]) * LOG2E;
                    }
                } else {
                    sc = -INFINITY;
                }
                Sf[kb].thread_elements()[e] = sc;
                rmax = max(rmax, sc);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        // a row with no visible key yet keeps m = -inf, l = 0 and a zero accumulator
        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
        const float factor = fast::exp2(m - m_use);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BKM / 8];
        for (int kb = 0; kb < BKM / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float pr = fast::exp2(Sf[kb].thread_elements()[e] - m_use);
                Pf[kb].thread_elements()[e] = half(pr);
                rsum += pr;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BKM / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const size_t at = ((size_t)hk * NR + row) * NS + split;
        if (fn == 0) {
            part_m[at] = m;
            part_l[at] = l;
        }
        for (int db = 0; db < D / 8; ++db) {
            part_o[at * D + db * 8 + fn] = Of[db].thread_elements()[0];
            part_o[at * D + db * 8 + fn + 1] = Of[db].thread_elements()[1];
        }
    }
"""

# One threadgroup of D threads per row (key/value head hk, row r = g * L + l): the slices' partials combined with the
# sink, as a zero-value logit, into the row's output; with a gate, the output projection's input
# ``out * sigmoid(gate)`` [1, L, HQ * D] (``Inkling2Attention.output``), else the attention [1, HQ, L, D].
_COMBINE_SOURCE = """
    constexpr int GQ = HQ / HKV;
    constexpr int NR = GQ * L;
    constexpr float LOG2E = 1.4426950408889634f;
    const int t = threadgroup_position_in_grid.x;
    const int d = thread_position_in_threadgroup.x;
    const int hk = t / NR;
    const int r = t % NR;
    const int l = r % L;
    const int qh = hk * GQ + r / L;
    const int NS = dims[2];
    const float sk = float(sink[l * sink_strides[1] + qh * sink_strides[2]]) * LOG2E;
    const device float* pm = part_m + (size_t)t * NS;
    const device float* pl = part_l + (size_t)t * NS;
    const device float* po = part_o + (size_t)t * NS * D;
    float top = sk;
    for (int i = 0; i < NS; ++i) top = max(top, pm[i]);
    float acc = 0.0f;
    float lsum = 0.0f;
    for (int i = 0; i < NS; ++i) {
        const float w = metal::exp2(pm[i] - top);
        lsum += pl[i] * w;
        acc += po[i * D + d] * w;
    }
    const T ob = T(acc / (metal::exp2(sk - top) + lsum));
    if (GATE) {
        const float g = float(gate[l * gate_strides[1] + (qh * D + d) * gate_strides[2]]);
        const float e = 1.0f / (1.0f + metal::exp(-metal::abs(g)));
        out[((size_t)l * HQ + qh) * D + d] = T(float(ob) * (g < 0.0f ? 1.0f - e : e));
    } else {
        out[((size_t)qh * L + l) * D + d] = ob;
    }
"""

_kernel = None
_mma_kernel = None
_combine_kernel = None
_MMA_MIN_SG = 4
_MMA_MAX_SG = 16


def _mma_simdgroups(rows: int) -> int:
    """Simdgroups of 8 rows per threadgroup: every row of the head's group, at least 4 for the tile loads."""
    return min(_MMA_MAX_SG, max(_MMA_MIN_SG, math.ceil(rows / 8)))


def enabled() -> bool:
    return os.environ.get("OMLX_0ICS_INKLING_ATTN", "0") == "1"


def supports(q, window: int, left_padding, keys: int) -> bool:
    """Global layers, batch 1, 2-16 queries; one query only from ``MIN_KEYS_ONE_QUERY`` keys, below which MLX's
    vector SDPA is faster (27/11, M5 Ultra: 0.31 against 0.41 ms at 4K, 1.42 against 1.07 ms at 128K)."""
    b, _, length, d = q.shape
    if window != 0 or left_padding is not None or b != 1 or d % 32 != 0 or not 1 <= length <= MAX_QUERIES:
        return False
    return length > 1 or keys >= MIN_KEYS_ONE_QUERY


def supports_fused(q, left_padding, k, v) -> bool:
    """The fused layer's attention (``fused_decode``): global and sliding layers, any number of keys, batch 1, 1-16
    queries, a bfloat16 cache; the combine is one kernel there, so the split wins from the shortest caches."""
    b, _, length, d = q.shape
    return (left_padding is None and b == 1 and d % 32 == 0 and 1 <= length <= MAX_QUERIES
            and supports_cache(k, v))


def supports_cache(k, v) -> bool:
    """A bfloat16 cache; a head dimension that is not contiguous takes the kernel's scalar loads."""
    return k.dtype == mx.bfloat16 and v.dtype == mx.bfloat16


def _chunk(s: int, min_chunk: int = _MIN_CHUNK) -> int:
    """Keys per slice: at least ``min_chunk``, at most _MAX_SPLITS slices, a multiple of the key tile."""
    ch = max(min_chunk, math.ceil(s / _MAX_SPLITS))
    return (ch + _BK - 1) // _BK * _BK


def attention(q, k, v, rel, sink, *, scale: float, window: int = 0, gate=None, min_chunk: int = _MIN_CHUNK,
              sel=None, block: int = 0, far: int = 0):
    """``banded_sink_attention`` for batch 1: q [1, Hq, L, D], k/v [1, Hkv, S, D] (views of the cache),
    rel [1, L, Hq, R], sink [1, L, Hq]. ``window`` > 0 limits keys to backward distances below it. Returns
    [1, Hq, L, D] in q's dtype, or with ``gate`` [1, L, Hq * D] the output projection's input.

    With ``sel`` (int32 [Hkv, n], ascending block indices, global layers only) each key/value head reads its n blocks
    of ``block`` keys and the keys from ``far`` to the end, nothing else (attn_sparse.py)."""
    global _kernel, _mma_kernel, _combine_kernel
    _, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    lo = max(0, s - length - window + 1) if window else 0
    picked = 0 if sel is None else sel.shape[1] * block
    keys = picked + s - far if sel is not None else s
    ch = _chunk(keys - lo, min_chunk)
    ns = math.ceil((keys - lo) / ch)
    nr = hq // hkv * length
    dims = mx.array([s, ch, ns, lo, picked, far, 0 if sel is None else sel.shape[1], keys], dtype=mx.int32)
    inputs = [q, k, v, mx.contiguous(rel), dims, mx.array([scale], dtype=mx.float32),
              dims if sel is None else sel]
    template = [("T", k.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("L", length), ("R", rel.shape[-1]),
                ("W", window), ("SEL", sel is not None), ("BLK", block if sel is not None else 1)]
    outputs = {"output_shapes": [(hkv, nr, ns), (hkv, nr, ns), (hkv, nr, ns, d)],
               "output_dtypes": [mx.float32, mx.float32, mx.float32]}
    if length == 1:
        if _kernel is None:
            _kernel = mx.fast.metal_kernel(
                name="omlx_0ics_inkling2_split_attn",
                input_names=["q", "k", "v", "rel", "dims", "scale", "sel"],
                output_names=["part_m", "part_l", "part_o"],
                source=_SOURCE,
                ensure_row_contiguous=False,
            )
        part_m, part_l, part_o = _kernel(inputs=inputs, template=[*template, ("BK", _BK)],
                                         grid=(ns * _THREADS, hkv, 1), threadgroup=(_THREADS, 1, 1), **outputs)
    else:
        if _mma_kernel is None:
            _mma_kernel = mx.fast.metal_kernel(
                name="omlx_0ics_inkling2_split_attn_mma",
                input_names=["q", "k", "v", "rel", "dims", "scale", "sel"],
                output_names=["part_m", "part_l", "part_o"],
                source=_MMA_SOURCE,
                ensure_row_contiguous=False,
            )
        nsg = _mma_simdgroups(nr)
        part_m, part_l, part_o = _mma_kernel(
            inputs=inputs, template=[*template, ("NSG", nsg)],
            grid=(ns * 32 * nsg, hkv, math.ceil(nr / (8 * nsg))), threadgroup=(32 * nsg, 1, 1),
            **outputs)
    if _combine_kernel is None:
        _combine_kernel = mx.fast.metal_kernel(
            name="omlx_0ics_inkling2_split_attn_combine",
            input_names=["part_m", "part_l", "part_o", "sink", "gate", "dims"],
            output_names=["out"],
            source=_COMBINE_SOURCE,
            ensure_row_contiguous=False,
        )
    gated = gate is not None
    shape = (1, length, hq * d) if gated else (1, hq, length, d)
    return _combine_kernel(
        inputs=[part_m, part_l, part_o, sink, gate if gated else sink, dims],
        template=[("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("L", length), ("GATE", gated)],
        grid=(hkv * nr * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[shape],
        output_dtypes=[q.dtype],
    )[0]
