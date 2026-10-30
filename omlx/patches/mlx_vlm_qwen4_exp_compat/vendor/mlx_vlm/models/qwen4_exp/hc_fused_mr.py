# SPDX-License-Identifier: Apache-2.0
"""Multi-row stream norm + down/inject launch for short Qwen4 verify blocks.

``hc_fused._kernel_norm_down`` launches one threadgroup per (row, stream,
row group): with four verify rows each threadgroup recomputes its stream norm
and then runs a short down slice, and every row reads the down weights again.
Measured on the model (M5 Ultra, 4 rows) that launch costs ~64 us per call,
~6 ms of a ~18.5 ms verify cycle.

Here one threadgroup per (stream, row group) takes all rows: it computes the
norm of each row's stream, keeps the normed rows in threadgroup memory, and
applies every down weight it reads to all of them. Per row the arithmetic is
``_NDN_SOURCE_TEMPLATE``'s (same loads, same simd_sum and partial order, same
helpers), so every output keeps its bits.

Enabled with OMLX_0ICS_HC_MR=1 (off by default), for 2..MAX_ROWS rows.
"""

from __future__ import annotations

import os

import mlx.core as mx

from . import hc_fused as F

ENABLED = os.environ.get("OMLX_0ICS_HC_MR", "0") == "1"
MAX_ROWS = 6  # NR * H bf16 normed rows must fit the 32 KB of threadgroup memory
RPS = int(os.environ.get("OMLX_0ICS_HC_MR_RPS", "1"))

_SOURCE_TEMPLATE = r"""
    const uint gy = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint s = gy & 3;
    const uint g = gy >> 2;
    constexpr int PER = (H + 255) / 256;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / 64;
    constexpr int PR = R + HC;
    constexpr int ROWS_TG = 8 * RPS;
    constexpr int NDG = R / ROWS_TG;
    static_assert(HC == 4 && K == 4 * H && R % ROWS_TG == 0, "down K slice ks is stream ks");
    static_assert(PER <= NDG + INJ, "one stored strip per threadgroup");
    threadgroup T xs[NR][H];
    threadgroup float npart[NR][8];
    float v[NR][PER];
    T y_mine[NR];
    T n_mine[NR];
    const device T* wp = w + (size_t)s * H;
    float wv[PER];
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        wv[i] = (k < H) ? float(wp[k]) : 0.0f;
    }
    const float e = eps[0];
    for (int r = 0; r < NR; ++r) {
        y_mine[r] = T(0.0f);
        n_mine[r] = T(0.0f);
        const size_t base = (size_t)r * K + (size_t)s * H;
        float ss = 0.0f;
        // LOAD
        ss = simd_sum(ss);
        if (lane == 0) npart[r][sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < NR; ++r) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += npart[r][i];
        const float inv = metal::rsqrt(tot / float(H) + e);
        for (int i = 0; i < PER; ++i) {
            const int k = t + i * 256;
            if (k < H) {
                const T y = T(v[r][i] * inv * (1.0f + wv[i]));
                xs[r][k] = y;
                if (i == int(g)) n_mine[r] = y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool down_tg = g < NDG;
    const int ks = int(s);
    if (down_tg) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(g) * ROWS_TG + int(sg) * RPS;
        const device uint8_t* wq = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        int xo = int(lane) * VPT;
        float result[NR][RPS];
        for (int r = 0; r < NR; ++r)
            for (int q = 0; q < RPS; ++q) result[r][q] = 0.0f;
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            for (int r = 0; r < NR; ++r) {
                float sum = hc_load_vector_tg<T, VPT, BITS_D>(&xs[r][xo], xv);
                for (int q = 0; q < RPS; ++q) {
                    result[r][q] += hc_qdot<VPT, BITS_D>(
                        wq + q * ROW_BYTES, xv,
                        float(sp[q * GROUPS]), float(bp[q * GROUPS]), sum);
                }
            }
            wq += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xo += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            for (int r = 0; r < NR; ++r) {
                float sum = hc_load_vector_tg<T, VPT, BITS_D>(&xs[r][xo], xv);
                for (int q = 0; q < RPS; ++q) {
                    result[r][q] += hc_qdot<VPT, BITS_D>(
                        wq + q * ROW_BYTES, xv,
                        float(sp[q * GROUPS]), float(bp[q * GROUPS]), sum);
                }
            }
        }
        for (int r = 0; r < NR; ++r) {
            device float* pp = parts + ((size_t)r * KS + ks) * PR;
            for (int q = 0; q < RPS; ++q) {
                float v2 = simd_sum(result[r][q]);
                if (lane == 0) pp[out_row + q] = v2;
            }
        }
    } else if (INJ != 0 && int(sg) < HC) {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int q = int(sg);
        const device uint8_t* wq = (const device uint8_t*)inject_w
            + q * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + q * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + q * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        int xo = int(lane) * VPT;
        float result[NR];
        for (int r = 0; r < NR; ++r) result[r] = 0.0f;
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            for (int r = 0; r < NR; ++r) {
                float sum = hc_load_vector_tg<T, VPT, BITS_I>(&xs[r][xo], xv);
                result[r] += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
            }
            wq += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xo += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            for (int r = 0; r < NR; ++r) {
                float sum = hc_load_vector_tg<T, VPT, BITS_I>(&xs[r][xo], xv);
                result[r] += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
            }
        }
        for (int r = 0; r < NR; ++r) {
            float v2 = simd_sum(result[r]);
            if (lane == 0) parts[((size_t)r * KS + ks) * PR + R + q] = v2;
        }
    }
    if (int(g) < PER && int(t) + int(g) * 256 < H) {
        for (int r = 0; r < NR; ++r) {
            const size_t at = (size_t)r * K + (size_t)s * H + int(t) + int(g) * 256;
            xn[at] = n_mine[r];
            // WRITE
        }
    }
"""

_LOAD = r"""
        const device T* xp = x + base;
        for (int i = 0; i < PER; ++i) {
            const int k = t + i * 256;
            v[r][i] = (k < H) ? float(xp[k]) : 0.0f;
            ss += v[r][i] * v[r][i];
        }
"""

_WRITE_LOAD = r"""
        const device T* bq = branch + (size_t)r * H;
        const float g_s = float(gate[(size_t)r * HC + s]);
        for (int i = 0; i < PER; ++i) {
            const int k = t + i * 256;
            v[r][i] = 0.0f;
            if (k < H) {
                const T y = T(float(x[base + k]) + float(T(float(bq[k]) * g_s)));
                if (i == int(g)) y_mine[r] = y;
                v[r][i] = float(y);
            }
            ss += v[r][i] * v[r][i];
        }
"""


def _source(load: str, write: bool) -> str:
    return _SOURCE_TEMPLATE.replace("        // LOAD\n", load).replace(
        "            // WRITE\n", "            y_out[at] = y_mine[r];\n" if write else ""
    )


_SRC = _source(_LOAD, write=False)
_SRCW = _source(_WRITE_LOAD, write=True)


def usable(rows: int, hc: int, hidden: int, lowrank: int, inject: bool) -> bool:
    groups = F._down_row_groups(lowrank, inject, RPS)
    return (
        ENABLED
        and 2 <= rows <= MAX_ROWS
        and hc == 4
        and lowrank % (8 * RPS) == 0
        and -(-hidden // 256) <= groups
        and rows * (hidden * 2 + 32) <= F._TG_MEMORY_BYTES
    )


def norm_down(module, flat, operands, rows, hc, hidden, lowrank, dtype, down, inject):
    """Same contract and outputs as ``hc_fused._kernel_norm_down``."""
    width = hc * hidden
    tensors = inject if inject is not None else down
    inputs = [
        module["hc_norm"]["weight"],
        F._eps_array(module),
        down["weight"],
        down["scales"],
        down["biases"],
        tensors["weight"],
        tensors["scales"],
        tensors["biases"],
    ]
    names = ["w", "eps", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"]
    launch = dict(
        template=[
            ("T", dtype),
            ("BITS_D", down.bits),
            ("BITS_I", tensors.bits),
            ("K", width),
            ("H", hidden),
            ("R", lowrank),
            ("HC", hc),
            ("INJ", 1 if inject is not None else 0),
            ("RPS", RPS),
            ("NR", rows),
        ],
        grid=(256, hc * F._down_row_groups(lowrank, inject is not None, RPS), 1),
        threadgroup=(256, 1, 1),
    )
    shapes = [(rows, width), (rows, hc, lowrank + hc)]
    types = [dtype, mx.float32]
    if operands is None:
        normed, parts = F._kernel(
            "omlx_0ics_hc_mr_norm_down", ["x", *names], ["xn", "parts"], _SRC, header=F._V2_HEADER
        )(inputs=[flat, *inputs], output_shapes=shapes, output_dtypes=types, **launch)
        return None, normed, parts
    written, normed, parts = F._kernel(
        "omlx_0ics_hc_mr_write_norm_down",
        ["x", "branch", "gate", *names],
        ["y_out", "xn", "parts"],
        _SRCW,
        header=F._V2_HEADER,
    )(
        inputs=[flat, *operands, *inputs],
        output_shapes=[(rows, width), *shapes],
        output_dtypes=[dtype, *types],
        **launch,
    )
    return written, normed, parts
