# SPDX-License-Identifier: Apache-2.0
"""Quantized projection for 2..8 rows with one output column per lane (0ics).

With several rows the inputs, not the weights, dominate the traffic of a
qmv-style kernel: each simdgroup owns a few columns and re-reads every row of
``x`` for them (4 rows: ~2.6x the weight bytes). Here a lane owns a column and
walks its weights along K; the 32 lanes of a simdgroup read the same inputs at
the same time (one broadcast load), so the input traffic per weight byte drops
by ~8x. SPLIT simdgroups of a threadgroup take slices of K for the same 32
columns and add their partial sums once, in threadgroup memory, at the end.

Outputs agree with quantized_matmul to bf16 rounding, not bit for bit.
"""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

_SOURCE = r"""
    constexpr int VPT = 16;
    constexpr int ROW_BYTES = K * BITS / 8;
    constexpr int GROUPS = K / GS;
    constexpr int SLICE = K / SPLIT;
    threadgroup float red[SPLIT][R][32];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int n = int(threadgroup_position_in_grid.y) * 32 + int(lane);
    const bool ok = n < N;
    const int kbeg = int(sg) * SLICE;
    const device uint8_t* wrow = (const device uint8_t*)w + (size_t)(ok ? n : 0) * ROW_BYTES;
    const device uint* xu = (const device uint*)x;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int k = kbeg; k < kbeg + SLICE; k += VPT) {
        float q[VPT];
        const device uint8_t* p = wrow + (size_t)k * BITS / 8;
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
        const float s = float(scales[(size_t)(ok ? n : 0) * GROUPS + k / GS]);
        const float b = float(biases[(size_t)(ok ? n : 0) * GROUPS + k / GS]);
        for (int r = 0; r < R; ++r) {
            const device uint4* src = (const device uint4*)(xu + ((size_t)r * K + k) / 2);
            const uint4 lo = src[0], hi = src[1];
            const uint v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            float d = 0.0f, xs = 0.0f;
            for (int i = 0; i < 8; ++i) {
                const float x0 = as_type<float>(v[i] << 16);
                const float x1 = as_type<float>(v[i] & 0xffff0000u);
                d += q[2 * i] * x0 + q[2 * i + 1] * x1;
                xs += x0 + x1;
            }
            acc[r] += s * d + b * xs;
        }
    }
    for (int r = 0; r < R; ++r) red[sg][r][lane] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && ok) {
        for (int r = 0; r < R; ++r) {
            float v = 0.0f;
            for (int t = 0; t < SPLIT; ++t) v += red[t][r][lane];
            y[(size_t)r * N + n] = T(v);
        }
    }
"""

SPLIT = int(os.environ.get("OMLX_0ICS_QMV_SPLIT", 8))


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_0ics_qmv_cols",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_SOURCE,
    )


def supported(rows: int, k: int, n: int, bits: int, group_size: int) -> bool:
    return 1 <= rows <= 8 and bits in (6, 8) and group_size in (64, 128) and k % (16 * SPLIT) == 0


def qmv_cols(x: mx.array, weight, scales, biases, bits: int, group_size: int) -> mx.array:
    """``x`` [R, K] times the quantized ``weight`` [N, K] transposed: [R, N]."""
    rows, k = x.shape
    n = scales.shape[0]
    return _kernel()(
        inputs=[x, weight, scales, biases],
        template=[("T", x.dtype), ("BITS", bits), ("GS", group_size), ("K", k), ("N", n), ("R", rows), ("SPLIT", SPLIT)],
        grid=(32, SPLIT * (-(-n // 32)), 1),
        threadgroup=(32, SPLIT, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
