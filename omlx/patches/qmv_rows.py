# SPDX-License-Identifier: Apache-2.0
"""Quantized projection for 2..8 rows that reads each weight once (0ics).

Measured on an M5 Ultra (micro-precarica.py, 2 GB of 6/8-bit projections):
MLX's quantized_matmul reaches 887 GB/s at one row but 594 at four, 293 at
eight and ~400 at 12-32 (its small-M kernels decode the weights per row or
fall back to tiles), and the row-exact verify kernels 458 at four rows. A
verify pass is four rows, so its dense projections run at ~600 GB/s.

Here a simdgroup owns CPS output columns; each lane takes 16 consecutive
inputs per step (one 12- or 16-byte weight load per column), unpacks them
once and applies them to every row, whose inputs stay in registers. The bias
term uses the per-lane input sum (16 inputs never straddle a group). The
arithmetic order differs from MLX's (one-row qmv): outputs agree to bf16
rounding, not bit for bit.
"""

from __future__ import annotations

from functools import cache

import os

import mlx.core as mx

_SOURCE = r"""
    constexpr int VPT = 16;
    constexpr int BLOCK = 32 * VPT;
    constexpr int ROW_BYTES = K * BITS / 8;
    constexpr int GROUPS = K / GS;
    constexpr int THREADS = 32 * SGS;
    threadgroup T xt[R * BLOCK];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const int col0 = (int(threadgroup_position_in_grid.y) * SGS + int(sg)) * CPS;
    const bool active = col0 < N;
    const device uint8_t* wb = (const device uint8_t*)w;
    float acc[CPS][R];
    for (int c = 0; c < CPS; ++c)
        for (int r = 0; r < R; ++r) acc[c][r] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BLOCK) {
        // the rows' inputs of this block, once per threadgroup
        for (int i = int(tid); i < R * BLOCK; i += THREADS) {
            const int r = i / BLOCK;
            xt[i] = x[(size_t)r * K + k0 + (i - r * BLOCK)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            const int k = k0 + int(lane) * VPT;
            float xv[R][VPT];
            float xs[R];
            for (int r = 0; r < R; ++r) {
                xs[r] = 0.0f;
                const threadgroup T* xp = xt + r * BLOCK + int(lane) * VPT;
                for (int i = 0; i < VPT; ++i) {
                    const float v = float(xp[i]);
                    xv[r][i] = v;
                    xs[r] += v;
                }
            }
            const int g = k / GS;
            for (int c = 0; c < CPS; ++c) {
                const int n = col0 + c;
                const device uint8_t* p = wb + (size_t)n * ROW_BYTES + (size_t)k * BITS / 8;
                float q[VPT];
                if (BITS == 8) {
                    const uint4 u = *(const device uint4*)p;
                    const uint ws[4] = {u.x, u.y, u.z, u.w};
                    for (int j = 0; j < 16; ++j) q[j] = float((ws[j >> 2] >> (8 * (j & 3))) & 0xffu);
                } else {
                    const device uint* pu = (const device uint*)p;
                    const uint ws[3] = {pu[0], pu[1], pu[2]};
                    for (int t = 0; t < 4; ++t) {
                        const int j = 3 * t;
                        const uint b0 = (ws[j >> 2] >> (8 * (j & 3))) & 0xffu;
                        const uint b1 = (ws[(j + 1) >> 2] >> (8 * ((j + 1) & 3))) & 0xffu;
                        const uint b2 = (ws[(j + 2) >> 2] >> (8 * ((j + 2) & 3))) & 0xffu;
                        q[4 * t + 0] = float(b0 & 0x3fu);
                        q[4 * t + 1] = float((b0 >> 6) | ((b1 & 0x0fu) << 2));
                        q[4 * t + 2] = float((b1 >> 4) | ((b2 & 0x03u) << 4));
                        q[4 * t + 3] = float(b2 >> 2);
                    }
                }
                const float s = float(scales[(size_t)n * GROUPS + g]);
                const float b = float(biases[(size_t)n * GROUPS + g]);
                for (int r = 0; r < R; ++r) {
                    float d = 0.0f;
                    for (int i = 0; i < VPT; ++i) d += q[i] * xv[r][i];
                    acc[c][r] += s * d + b * xs[r];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!active) return;
    for (int c = 0; c < CPS; ++c) {
        for (int r = 0; r < R; ++r) {
            const float v = simd_sum(acc[c][r]);
            if (lane == 0) y[(size_t)r * N + col0 + c] = T(v);
        }
    }
"""

SGS = int(os.environ.get("OMLX_0ICS_QMV_SGS", 8))


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_0ics_qmv_rows",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_SOURCE,
    )


def supported(rows: int, k: int, n: int, bits: int, group_size: int) -> bool:
    cps = cps_for(rows)
    return (
        2 <= rows <= 8
        and bits in (6, 8)
        and group_size in (64, 128)
        and k % 512 == 0
        and n % cps == 0
    )


def cps_for(rows: int) -> int:
    return int(os.environ.get("OMLX_0ICS_QMV_CPS", 4 if rows <= 4 else 2))


def qmv_rows(x: mx.array, weight, scales, biases, bits: int, group_size: int) -> mx.array:
    """``x`` [R, K] times the quantized ``weight`` [N, K] transposed: [R, N]."""
    rows, k = x.shape
    n = scales.shape[0]
    cps = cps_for(rows)
    return _kernel()(
        inputs=[x, weight, scales, biases],
        template=[
            ("T", x.dtype),
            ("BITS", bits),
            ("GS", group_size),
            ("K", k),
            ("N", n),
            ("R", rows),
            ("CPS", cps),
            ("SGS", SGS),
        ],
        grid=(32, SGS * (-(-n // (cps * SGS))), 1),
        threadgroup=(32, SGS, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
