# SPDX-License-Identifier: Apache-2.0
"""Routed-expert projection for short verify blocks, one read per expert.

``moe_verify_gather`` runs one threadgroup per (row, expert) pair and output
tile and relies on the cache when rows share an expert; measured on an M5
Ultra the time follows the pair count, not the distinct experts (4 rows,
24 distinct experts of 40 pairs: 6.95 ms over 48 layers against a 3.8 ms
bandwidth floor). Here a small plan kernel groups the pairs by expert once
per layer (shared by gate/up and down), and each threadgroup takes one
distinct expert and output tile: it walks the expert tile once and, per
block, applies it to every pair of that expert.

Per pair the arithmetic is the one of ``moe_verify_gather`` (MLX 0.32.2
``qmv_fast`` / ``qmv`` traversal, same helpers, same accumulation order), so
each output equals ``gather_qmm`` bit for bit; only the loop over pairs moved
inside the block loop.

Enabled with OMLX_0ICS_MOE_GROUPED=1 (off by default).
"""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

from .moe_verify_gather import _HEADER, MAX_ROWS, supported

ENABLED = os.environ.get("OMLX_0ICS_MOE_GROUPED", "0") == "1"

# plan[0] = slots (distinct experts); plan[1 + s] = first member of slot s
# (plan[1 + slots] = pairs); plan[2 + pairs + i] = members, grouped by slot in
# order of first occurrence, each slot in pair order.
_PLAN_SOURCE = r"""
    const int t = int(thread_index_in_threadgroup);
    threadgroup uint e_s[N_PAIRS];
    threadgroup int first_s[N_PAIRS];
    if (t < N_PAIRS) e_s[t] = rhs[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int first = t;
    if (t < N_PAIRS) {
      for (int q = 0; q < t; ++q) {
        if (e_s[q] == e_s[t]) { first = q; break; }
      }
      first_s[t] = first;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < N_PAIRS) {
      int start = 0, pos = 0, slot = 0, slots = 0;
      for (int q = 0; q < N_PAIRS; ++q) {
        const int fq = first_s[q];
        start += (fq < first) ? 1 : 0;
        pos += (q < t && fq == first) ? 1 : 0;
        slot += (fq == q && q < first) ? 1 : 0;
        slots += (fq == q) ? 1 : 0;
      }
      plan[2 + N_PAIRS + start + pos] = t;
      if (first == t) plan[1 + slot] = start;
      if (t == 0) {
        plan[0] = slots;
        plan[1 + slots] = N_PAIRS;
      }
    }
"""

_SOURCE = r"""
    const int slot = int(threadgroup_position_in_grid.z);
    if (slot >= plan[0]) return;
    const int s0 = plan[1 + slot];
    const int cnt = plan[2 + slot] - s0;
    const device int* members = plan + 2 + N_PAIRS + s0;
    const int expert = int(rhs[members[0]]);
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K_SIZE / GS;
    const int out_row = int(threadgroup_position_in_grid.y) * BN +
        int(simd_gid) * RESULTS_PER_SIMDGROUP;

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * (N_SIZE * in_vec_size_w) + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * (N_SIZE * in_vec_size_g) +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * (N_SIZE * in_vec_size_g) +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

    int pair[MAXM];
    const device T* xp[MAXM];
    for (int m = 0; m < MAXM; ++m) {
      if (m < cnt) {
        pair[m] = members[m];
        xp[m] = x + (pair[m] / LHS_DIV) * K_SIZE +
            int(simd_lid) * VALUES_PER_THREAD;
      }
    }

    float x_thread[VALUES_PER_THREAD];
    float result[MAXM][RESULTS_PER_SIMDGROUP];
    for (int m = 0; m < MAXM; ++m)
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) result[m][row] = 0;

    int k = 0;
    const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
    for (; k < full_limit; k += BLOCK_SIZE) {
      if (BITS == 6) {
        // The masked weight values, converted once and shared by every pair;
        // per pair the products and sums run in qdot_n's order.
        float wv[RESULTS_PER_SIMDGROUP][W6];
        float s[RESULTS_PER_SIMDGROUP], b[RESULTS_PER_SIMDGROUP];
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          unpack6(ws + row * in_vec_size_w, wv[row]);
          s[row] = sc[row * in_vec_size_g];
          b[row] = bs[row * in_vec_size_g];
        }
        for (int m = 0; m < MAXM; ++m) {
          if (m < cnt) {
            float sum = load_vector<T>(xp[m], x_thread);
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
              result[m][row] += qdot6_unpacked(
                  wv[row], x_thread, s[row], b[row], sum, VALUES_PER_THREAD);
            }
            xp[m] += BLOCK_SIZE;
          }
        }
      } else {
        for (int m = 0; m < MAXM; ++m) {
          if (m < cnt) {
            float sum = load_vector<T>(xp[m], x_thread);
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              float s = sc[row * in_vec_size_g];
              float b = bs[row * in_vec_size_g];
              result[m][row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
            }
            xp[m] += BLOCK_SIZE;
          }
        }
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
    }
    if (!FAST) {
      const int remaining = clamp(
          int(K_SIZE - k - int(simd_lid) * VALUES_PER_THREAD),
          0,
          VALUES_PER_THREAD);
      if (remaining > 0) {
        for (int m = 0; m < MAXM; ++m) {
          if (m < cnt) {
            float sum = load_vector_safe<T>(xp[m], x_thread, remaining);
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              float s = sc[row * in_vec_size_g];
              float b = bs[row * in_vec_size_g];
              result[m][row] += qdot_n(wl, x_thread, s, b, sum, remaining);
            }
          }
        }
      }
    }

    for (int m = 0; m < MAXM; ++m) {
      if (m < cnt) {
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          float v = simd_sum(result[m][row]);
          if (simd_lid == 0) {
            y[pair[m] * N_SIZE + out_row + row] = static_cast<T>(v);
          }
        }
      }
    }
"""


# 6-bit only: the masked values of qdot_n's packs as floats, and qdot_n's
# 6-bit loop (pointer steps included) over them.
_UNPACKED = r"""
constant constexpr int W6 = VALUES_PER_THREAD / 4 * 6;

inline void unpack6(const device uint8_t* w, thread float* wv) {
  for (int j = 0; j < VALUES_PER_THREAD / 4; j++) {
    wv[6 * j + 0] = float(w[3 * j] & 0x3f);
    wv[6 * j + 1] = float(w[3 * j] & 0xc0);
    wv[6 * j + 2] = float(w[3 * j + 1] & 0x0f);
    wv[6 * j + 3] = float(w[3 * j + 1] & 0xf0);
    wv[6 * j + 4] = float(w[3 * j + 2] & 0x03);
    wv[6 * j + 5] = float(w[3 * j + 2] & 0xfc);
  }
}

inline float qdot6_unpacked(
    const thread float* wv,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int N) {
  float accum = 0;
  for (int i = 0; i < (N / 4); i++) {
    x_thread += 4 * i;
    wv += 6 * i;
    accum += wv[0] * x_thread[0];
    accum += wv[1] * x_thread[1];
    accum += wv[2] * (x_thread[1] * 256.0f);
    accum += wv[3] * x_thread[2];
    accum += wv[4] * (x_thread[2] * 256.0f);
    accum += wv[5] * x_thread[3];
  }
  return scale * accum + sum * bias;
}
"""


@cache
def _plan_kernel():
    return mx.fast.metal_kernel(
        name="omlx_0ics_moe_verify_plan",
        input_names=["rhs"],
        output_names=["plan"],
        source=_PLAN_SOURCE,
    )


@cache
def _kernel(bits: int, group_size: int, fast: bool):
    header = (
        _HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    ) + _UNPACKED
    return mx.fast.metal_kernel(
        name=f"omlx_0ics_moe_verify_grouped_b{bits}_gs{group_size}_{int(fast)}",
        input_names=["x", "w", "scales", "biases", "rhs", "plan"],
        output_names=["y"],
        header=header,
        source=_SOURCE,
    )


def plan(rhs: mx.array) -> mx.array:
    """Pairs grouped by expert (layout in ``_PLAN_SOURCE``); ``rhs`` is [P] uint32."""
    pairs = int(rhs.shape[0])
    threads = max(32, -(-pairs // 32) * 32)
    return _plan_kernel()(
        inputs=[rhs],
        template=[("N_PAIRS", pairs)],
        grid=(threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(2 + 2 * pairs,)],
        output_dtypes=[mx.int32],
    )[0]


def gather_qmv(
    linear, x: mx.array, rhs: mx.array, grouping: mx.array, pairs_per_row: int, max_members: int
) -> mx.array:
    """``moe_verify_gather.gather_qmv`` with one threadgroup per distinct expert and tile."""
    weight, scales, biases = linear["weight"], linear["scales"], linear["biases"]
    k = int(x.shape[-1])
    n = int(weight.shape[-2])
    pairs = int(rhs.shape[0])
    kernel = _kernel(int(linear.bits), int(linear.group_size), k % 512 == 0)
    return kernel(
        inputs=[x, weight, scales, biases, rhs, grouping],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("N_PAIRS", pairs),
            ("LHS_DIV", int(pairs_per_row)),
            ("MAXM", int(max_members)),
        ],
        grid=(32, 2 * (n // 8), pairs),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )[0]


def fused_switch(switch_mlp, x: mx.array, indices: mx.array):
    """Drop-in for ``moe_verify_gather.fused_switch``: same checks, same output."""
    batch, length, width = x.shape
    rows = batch * length
    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    down = getattr(switch_mlp, "down_proj", None)
    if not (
        1 < rows <= MAX_ROWS
        and gate_up is not None
        and down is not None
        and supported(gate_up, x.dtype)
        and supported(down, x.dtype)
    ):
        return None
    top_k = int(indices.shape[-1])
    pairs = indices.reshape(rows * top_k).astype(mx.uint32)
    grouping = plan(pairs)
    # A row never repeats an expert, so a slot holds at most one pair per row.
    gate_up_out = gather_qmv(gate_up, x.reshape(rows, width), pairs, grouping, top_k, rows)
    gate, up = mx.split(gate_up_out, 2, axis=-1)
    out = gather_qmv(down, switch_mlp.activation(up, gate), pairs, grouping, 1, rows)
    return out.reshape(batch, length, top_k, -1)
