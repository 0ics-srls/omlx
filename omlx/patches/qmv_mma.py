# SPDX-License-Identifier: Apache-2.0
"""Quantized projection for 2..8 rows on the simdgroup matrix units (0ics).

With four rows a quantized GEMV is bound by the ALUs, not by memory: every
weight is unpacked and then multiplied into each row (MLX ~450-600 GB/s,
Strata's hand kernels ~580-660, against ~887 at one row). Here the ALUs only
unpack: each lane dequantizes 16 consecutive weights of one column per step
into a per-simdgroup tile, and 8x8x8 simdgroup multiply-accumulates apply the
tile to all rows at once (rows padded to 8, inputs staged as half).

A simdgroup owns 8 output columns; a threadgroup SGS simdgroups. Weights are
dequantized to half (as MLX's qmm does to the input type), accumulation is
FP32. Outputs agree with quantized_matmul to bf16 rounding, not bit for bit.
"""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

_SOURCE = r"""
    constexpr int KB = 256;                  // inputs per step
    constexpr int ROW_BYTES = K * BITS / 8;
    constexpr int GROUPS = K / GS;
    constexpr int THREADS = 32 * SGS;
    threadgroup half xt[8 * KB];
    threadgroup half wt[SGS][KB * 8];
    threadgroup float ot[SGS][64];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_index_in_threadgroup;
    const int col0 = (int(threadgroup_position_in_grid.y) * SGS + int(sg)) * 8;
    const bool active = col0 < N;
    const int col = col0 + int(lane >> 2);
    const device uint8_t* wb = (const device uint8_t*)w;
    // padded rows stay zero
    for (int i = int(tid); i < 8 * KB; i += THREADS) xt[i] = half(0.0f);
    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    threadgroup half* my = wt[sg];
    for (int k0 = 0; k0 < K; k0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < R * KB; i += THREADS) {
            const int r = i / KB;
            const int kk = i - r * KB;
            xt[r * KB + kk] = half(float(x[(size_t)r * K + k0 + kk]));
        }
        if (active) {
            for (int j = 0; j < KB / 64; ++j) {
                const int kl = j * 64 + int(lane & 3) * 16;   // within the step
                const int k = k0 + kl;
                const device uint8_t* p = wb + (size_t)col * ROW_BYTES + (size_t)k * BITS / 8;
                const float s = float(scales[(size_t)col * GROUPS + k / GS]);
                const float b = float(biases[(size_t)col * GROUPS + k / GS]);
                float q[16];
                if (BITS == 8) {
                    const uint4 u = *(const device uint4*)p;
                    const uint ws[4] = {u.x, u.y, u.z, u.w};
                    for (int t = 0; t < 16; ++t) q[t] = float((ws[t >> 2] >> (8 * (t & 3))) & 0xffu);
                } else {
                    const device uint* pu = (const device uint*)p;
                    const uint ws[3] = {pu[0], pu[1], pu[2]};
                    for (int t = 0; t < 4; ++t) {
                        const int jj = 3 * t;
                        const uint b0 = (ws[jj >> 2] >> (8 * (jj & 3))) & 0xffu;
                        const uint b1 = (ws[(jj + 1) >> 2] >> (8 * ((jj + 1) & 3))) & 0xffu;
                        const uint b2 = (ws[(jj + 2) >> 2] >> (8 * ((jj + 2) & 3))) & 0xffu;
                        q[4 * t + 0] = float(b0 & 0x3fu);
                        q[4 * t + 1] = float((b0 >> 6) | ((b1 & 0x0fu) << 2));
                        q[4 * t + 2] = float((b1 >> 4) | ((b2 & 0x03u) << 4));
                        q[4 * t + 3] = float(b2 >> 2);
                    }
                }
                for (int t = 0; t < 16; ++t) my[(kl + t) * 8 + int(lane >> 2)] = half(s * q[t] + b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int kk = 0; kk < KB; kk += 8) {
                simdgroup_half8x8 a, bm;
                simdgroup_load(a, xt + kk, KB);
                simdgroup_load(bm, my + kk * 8, 8);
                simdgroup_multiply_accumulate(acc, a, bm, acc);
            }
        }
    }
    if (!active) return;
    simdgroup_store(acc, ot[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(lane); i < R * 8; i += 32) {
        const int r = i >> 3;
        const int c = i & 7;
        y[(size_t)r * N + col0 + c] = T(ot[sg][r * 8 + c]);
    }
"""

SGS = int(os.environ.get("OMLX_0ICS_MMA_SGS", 4))


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_0ics_qmv_mma",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_SOURCE,
    )


def supported(rows: int, k: int, n: int, bits: int, group_size: int) -> bool:
    return 1 <= rows <= 8 and bits in (6, 8) and group_size in (64, 128) and k % 256 == 0 and n % 8 == 0


def qmv_mma(x: mx.array, weight, scales, biases, bits: int, group_size: int) -> mx.array:
    """``x`` [R, K] times the quantized ``weight`` [N, K] transposed: [R, N]."""
    rows, k = x.shape
    n = scales.shape[0]
    return _kernel()(
        inputs=[x, weight, scales, biases],
        template=[("T", x.dtype), ("BITS", bits), ("GS", group_size), ("K", k), ("N", n), ("R", rows), ("SGS", SGS)],
        grid=(32, SGS * (-(-n // (8 * SGS))), 1),
        threadgroup=(32, SGS, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
