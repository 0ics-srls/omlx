# SPDX-License-Identifier: Apache-2.0
"""0ics: an Inkling 2 layer for decode and speculative verify in ~17 kernels instead of ~64.

A decode token of Inkling 2 Small reads ~8.5 GB of weights, ~10 ms at the M5 Ultra's bandwidth, and took 21 ms: the
GPU ran ~3,000 kernels a token (28/11, ``sperimenti/inkling/prova_decodifica.py``), 6 of them per layer doing the
work (projections, router, experts, attention) and ~58 of glue (casts, norms, short convolutions, clips, the
attention-residual mixtures, concatenations, state copies), each ~2 µs of a GPU that waits on the previous one; the
FP32 router GEMV ran at ~100 GB/s (3.4 ms a token).

Here the glue between two matmuls is one Metal kernel, a token row per threadgroup, with the reference's arithmetic
in the same order and the same bf16 roundings (``model.py``: ``_pre``, ``Inkling2Attention.project``, ``_post``,
``Inkling2Experts``); the router is a GEMV that reads its FP32 rows once for every token of a verify; the attention
takes ``attn_decode``'s split kernel in every layer, with the sink and the output gate in its combine; the experts'
down projection is ``expert_down``. One token: the stages equal the reference bit for bit (the router within 1e-7).

Measured (28/11, M5 Ultra, oQ2e, 4K cache): a decode token 21.3 → 16.9 ms, a verify of 5 tokens 48.6 → 36.5 ms, of 9
74.8 → 63.8 ms; the server with DSpark 62.7 → 70-80 t/s at 30K, 37.6-42 → 60-67 at 128K. On 512 tokens of real text
the NLL moves by as much as a change of the router's order of sums alone (0.4467, fused 0.4494, fused with the stock
router 0.4466), no choice changes.

Turned on by ``OMLX_0ICS_INKLING_FUSED=1``, for inputs of up to ``MAX_TOKENS`` tokens a row (decode and verify);
off, or for prefill chunks, the layer is upstream's.
"""

from __future__ import annotations

import os

import mlx.core as mx

MAX_TOKENS = 16
# keys per slice of the split attention (attn_decode): a sliding layer's 1,024 keys in 16 slices a key/value head
MIN_CHUNK = 64
_TG = 768
_EPS = 1e-6
_kernels: dict = {}

_HEADER = """
// one FP32 sum over a threadgroup of TG threads, in a fixed order (simdgroups, then their partials)
template <uint TG>
inline float tg_sum(float x, threadgroup float* red, uint lane, uint sg) {
    float s = simd_sum(x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < TG / 32; ++j) t += red[j];
    return t;
}
"""


def enabled() -> bool:
    return os.environ.get("OMLX_0ICS_INKLING_FUSED", "0") == "1"


def supports(layer, length: int) -> bool:
    """Layers after the first, affine experts (oQ), up to ``MAX_TOKENS`` tokens a row."""
    experts = layer.mlp.experts
    return (
        layer.layer_idx > 0
        and 1 <= length <= MAX_TOKENS
        and experts.w13.mode == "affine"
        and experts.w2.mode == "affine"
    )


def _f(x: float) -> str:
    return f"{float(x)!r}f"


def _kernel(key, name, inputs, outputs, source):
    k = _kernels.get(key)
    if k is None:
        k = _kernels[key] = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=outputs, source=source(), header=_HEADER
        )
    return k


def _consts(attn) -> tuple[float, float]:
    """The attention's clip thresholds as Python floats, read once (``.item()`` would sync every step)."""
    c = getattr(attn, "_0ics_clips", None)
    if c is None:
        c = attn._0ics_clips = (float(attn._qk_clip.item()), float(attn._r_clip.item()))
    return c


# ---------------------------------------------------------------- router


_ROUTER_SOURCE = """
    // one expert row per threadgroup, its 4 simdgroups split the hidden size; the row is read once for all N tokens
    uint e = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    constexpr uint PART = H / 4;
    float acc[N];
    for (uint n = 0; n < N; ++n) acc[n] = 0.0f;
    const device float4* wr = (const device float4*)(w + (size_t)e * H + sg * PART);
    for (uint i = lane; i < PART / 4; i += 32) {
        float4 wv = wr[i];
        uint h = sg * PART + i * 4;
        for (uint n = 0; n < N; ++n) {
            const device bfloat16_t* xr = x + (size_t)n * H + h;
            acc[n] += wv.x * float(xr[0]) + wv.y * float(xr[1]) + wv.z * float(xr[2]) + wv.w * float(xr[3]);
        }
    }
    threadgroup float part[4][N];
    for (uint n = 0; n < N; ++n) {
        float s = simd_sum(acc[n]);
        if (lane == 0) part[sg][n] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < N) {
        logits[(size_t)tid * E + e] = part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid];
    }
"""


def router_logits(x: mx.array, weight: mx.array) -> mx.array:
    """``x.astype(float32) @ weight.T`` for [N, H] bf16 ``x`` (N <= MAX_TOKENS) and [E, H] FP32 ``weight``."""
    n, hidden = x.shape
    experts = weight.shape[0]
    kernel = _kernel("router", "omlx_0ics_inkling2_router", ["x", "w"], ["logits"], lambda: _ROUTER_SOURCE)
    return kernel(
        inputs=[x, weight],
        template=[("H", hidden), ("E", experts), ("N", n)],
        grid=(experts * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, experts)],
        output_dtypes=[mx.float32],
    )[0]


# ---------------------------------------------------------------- the attention-residual mixture


def _mix_weights(ns: int, src) -> str:
    """Dot products of NS sources with Q queries and the softmax over sources into ``wmix[NS * Q]``, as
    ``kernels._attn_res_kernel`` computes them (same loops, same order of the sums). ``src(i)`` reads source ``i``
    at channel ``h``, whose slot in the thread's registers is ``j``."""
    loads = "\n".join(f"            float s{i} = {src(i)};" for i in range(ns))
    dots = "\n".join(f"                acc[{i} * Q + qi] += s{i} * qv;" for i in range(ns))
    return f"""
    constexpr uint NS = {ns};
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    {{
        uint j = 0;
        for (uint h = tid; h < H; h += TG, ++j) {{
{loads}
            for (uint qi = 0; qi < Q; ++qi) {{
                float qv = query[qi * H + h];
{dots}
            }}
        }}
    }}
    threadgroup float mred[NS * Q][TG / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) mred[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float wmix[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint k = 0; k < TG / 32; ++k) t += mred[i][k];
        wmix[i] = t * MIX_SCALE;
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, wmix[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            wmix[n * Q + qi] = metal::exp(wmix[n * Q + qi] - m);
            z += wmix[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) wmix[n * Q + qi] /= z;
    }}
"""


def _mix_norm(ns: int, src, out: str) -> str:
    """Each query's mixture (sources summed in the reference's order) through its RMSNorm, rounded to bf16 into
    ``out(qi, h)``."""
    loads = "\n".join(f"            float s{i} = {src(i)};" for i in range(ns))
    mix = " + ".join(f"wmix[{i} * Q + qi] * s{i}" for i in range(ns))
    return f"""
    for (uint qi = 0; qi < Q; ++qi) {{
        float mv[PER];
        float msq = 0.0f;
        uint j = 0;
        for (uint h = tid; h < H; h += TG, ++j) {{
{loads}
            mv[j] = {mix};
            msq += mv[j] * mv[j];
        }}
        float inv = metal::precise::rsqrt(tg_sum<TG>(msq, red, lane, sg) / float(H) + {_f(_EPS)});
        j = 0;
        for (uint h = tid; h < H; h += TG, ++j) {out} = bfloat16_t(float(nw[qi * H + h]) * (mv[j] * inv));
    }}
"""


_ROW_PRELUDE = """
    constexpr uint PER = (H + TG - 1) / TG;
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    threadgroup float red[TG / 32];
"""


# ---------------------------------------------------------------- before the projections


def _pre_source(nb: int, q: int, scale: float) -> str:
    outs = ", ".join(f"x{i}" for i in range(q))

    def src(i):
        return f"b{i}[base + h]" if i < nb else "hv[j]"

    return f"""
    constexpr float MIX_SCALE = {_f(scale)};
{_ROW_PRELUDE}
    // h_n: the partial sum at unit RMS (FP32), kept in registers
    float hv[PER];
    float ss = 0.0f;
    {{
        uint j = 0;
        for (uint h = tid; h < H; h += TG, ++j) {{ hv[j] = float(hin[base + h]); ss += hv[j] * hv[j]; }}
    }}
    float inv0 = metal::precise::rsqrt(tg_sum<TG>(ss, red, lane, sg) / float(H) + {_f(_EPS)});
    {{
        uint j = 0;
        for (uint h = tid; h < H; h += TG, ++j) {{ hv[j] *= inv0; hn[base + h] = hv[j]; }}
    }}
{_mix_weights(nb + 1, src)}
    device bfloat16_t* xo[Q] = {{{outs}}};
{_mix_norm(nb + 1, src, "xo[qi][base + h]")}
"""


def pre(layer, h: mx.array, blocks: list):
    """``Inkling2Layer._pre`` up to the projections' inputs: (h_n [B, L, H] FP32, Q arrays [B, L, H] bf16)."""
    b, length, hidden = h.shape
    query = layer.attn_input.query
    q = query.shape[0]
    nb = len(blocks)
    nw = getattr(layer, "_0ics_nw", None)
    if nw is None:
        norms = [layer.attn_norm] if layer.sliding else [layer.q_input_norm, layer.k_input_norm, layer.v_input_norm]
        nw = layer._0ics_nw = mx.stack([n.weight for n in norms])
    scale = layer.attn_input.scale
    kernel = _kernel(
        ("pre", nb, q, scale),
        f"omlx_0ics_inkling2_pre_{nb}_{q}",
        [*(f"b{i}" for i in range(nb)), "hin", "query", "nw"],
        ["hn", *(f"x{i}" for i in range(q))],
        lambda: _pre_source(nb, q, scale),
    )
    rows = b * length
    hn, *xs = kernel(
        inputs=[*blocks, h, query, nw],
        template=[("H", hidden), ("Q", q), ("TG", _TG)],
        grid=(rows * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(b, length, hidden)] * (q + 1),
        output_dtypes=[mx.float32] + [mx.bfloat16] * q,
    )
    return hn, xs


# ---------------------------------------------------------------- after the projections


def _project_source(sliding: bool, qk_clip: float, r_clip: float, conv_scale: float, wround: bool,
                    tau: tuple | None) -> str:
    """One threadgroup of D threads per (token row, unit): units 0..HQ-1 are the query heads (q clip, r clip, the
    relative bias row), then HKV key heads (conv, clip) and HKV value heads (conv, unit RMS)."""
    xp = ("((l + (J)) < (C - 1) ? float(st[((size_t)bi * (C - 1) + l + (J)) * DKV + c])"
          " : float(src[((size_t)bi * L + l + (J) - (C - 1)) * SW + koff + c]))")
    if sliding:
        conv = """
        float acc = float(cw[c * C + 0]) * XP(0);
        for (uint t = 1; t < C; ++t) acc += float(cw[c * C + t]) * XP(t);
"""
        if wround:
            conv += "        acc = float(bfloat16_t(acc));\n"
    else:
        conv = """
        float acc = 0.0f;
        for (uint lag = 0; lag < C; ++lag) {
            float tap = float(cw[lag * DKV + c]) + float(src[(size_t)row * SW + DKV + head * C + lag]);
            float term = XP(C - 1 - lag) * tap;
            acc = lag == 0 ? term : acc + term;
        }
"""
    # log scaling of the global layers (``Inkling2Attention.log_tau``): position = tokens in the cache + l
    tau_v = tau_q = tau_r = ""
    if tau is not None:
        alpha, floor = tau
        tau_v = (f"const float tau = 1.0f + {_f(alpha)} * metal::precise::log("
                 f"metal::max(float(offs[OFFS_B ? bi : 0] + int(l) + 1) / {_f(floor)}, 1.0f));")
        tau_q = "qv = float(bfloat16_t(qv * tau));"
        tau_r = "rv = float(bfloat16_t(rv * tau));"
    return f"""
    constexpr uint TG = D;
    constexpr uint UNITS = HQ + 2 * HKV;
    uint t0 = threadgroup_position_in_grid.x;
    uint unit = t0 % UNITS;
    uint row = t0 / UNITS;
    uint bi = row / L;
    uint l = row % L;
    uint d = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[TG / 32];
    threadgroup float rs[DR];
    if (unit < HQ) {{
        uint hq = unit;
        {tau_v}
        float xf = float(qsrc[(size_t)row * QW + hq * D + d]) * float(qscale[hq * D + d]);
        float norm = metal::sqrt(tg_sum<TG>(xf * xf, red, lane, sg));
        float qv = float(bfloat16_t(xf * metal::min({_f(qk_clip)} / (norm + 1e-6f), 1.0f)));
        {tau_q}
        q[((size_t)(bi * HQ + hq) * L + l) * D + d] = bfloat16_t(qv);
        // relative bias: r clipped over its DR channels, then r @ rel_proj (a bf16 matmul, FP32 sums)
        float rf = d < DR ? float(qsrc[(size_t)row * QW + R_OFF + hq * DR + d]) * float(rscale[hq * DR + d]) : 0.0f;
        float rn = metal::sqrt(tg_sum<TG>(rf * rf, red, lane, sg));
        if (d < DR) {{
            float rv = float(bfloat16_t(rf * metal::min({_f(r_clip)} / (rn + 1e-6f), 1.0f)));
            {tau_r}
            rs[d] = rv;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = d; j < R; j += TG) {{
            float a = 0.0f;
            for (uint i = 0; i < DR; ++i) a += rs[i] * float(relp[i * R + j]);
            rel[((size_t)row * HQ + hq) * R + j] = bfloat16_t(a);
        }}
    }} else {{
        bool is_k = unit < HQ + HKV;
        uint head = is_k ? unit - HQ : unit - HQ - HKV;
        uint c = head * D + d;
        const device bfloat16_t* src = is_k ? ksrc : vsrc;
        const device float* st = is_k ? kst : vst;
        const device auto* cw = is_k ? kcw : vcw;
        device float* xpo = is_k ? xpk : xpv;
        uint koff = KV_OFF + (is_k ? 0 : V_SHIFT);
        // the conv window [state, x]: rows 0..C-2 the state, then the chunk's tokens
#define XP(J) {xp}
{conv}
        float y = float(bfloat16_t((acc + XP(C - 1)) * {_f(conv_scale)}));
        xpo[((size_t)bi * (C - 1 + L) + C - 1 + l) * DKV + c] = XP(C - 1);
        if (l == 0) {{
            for (uint i = 0; i < C - 1; ++i) {{
                xpo[((size_t)bi * (C - 1 + L) + i) * DKV + c] = st[((size_t)bi * (C - 1) + i) * DKV + c];
            }}
        }}
#undef XP
        size_t o = ((size_t)(bi * HKV + head) * L + l) * D + d;
        if (is_k) {{
            float norm = metal::sqrt(tg_sum<TG>(y * y, red, lane, sg));
            k[o] = bfloat16_t(y * metal::min({_f(qk_clip)} / (norm + 1e-6f), 1.0f));
        }} else {{
            float inv = metal::precise::rsqrt(tg_sum<TG>(y * y, red, lane, sg) / float(D) + {_f(_EPS)});
            v[o] = bfloat16_t(y * inv);
        }}
    }}
"""


def project(attn, xs: mx.array, ks: mx.array, vs: mx.array, offset):
    """``Inkling2Attention.project`` from the normed inputs ``xs`` (Q [B, L, H] bf16, Q = 1 sliding, 3 global);
    ``offset`` is the tokens already in the cache (an int, or one per batch row), for the global layers' log scaling.

    Returns q [B, HQ, L, D], k and v [B, HKV, L, D], rel [B, L, HQ, R], gate, sink and the two conv windows."""
    b, length = xs[0].shape[0], xs[0].shape[1]
    hq, hkv, dh, dr = attn.n_heads, attn.n_kv, attn.head_dim, attn.d_rel
    dkv = attn.dkv
    c = attn.conv_k
    if attn.sliding:
        qsrc = ksrc = vsrc = attn.in_proj(xs[0])
        q_off_r = hq * dh + 2 * dkv
        kv_off, v_shift = hq * dh, dkv
        kcw, vcw = attn.k_conv, attn.v_conv
        conv_scale = attn._sconv_scale
    else:
        qsrc = attn.q_proj(xs[0])
        ksrc = attn.k_proj(xs[1])
        vsrc = attn.v_proj(xs[2])
        q_off_r = hq * dh
        kv_off, v_shift = 0, 0
        kcw, vcw = attn.k_static, attn.v_static
        conv_scale = attn._dconv_scale
    gate_off = q_off_r + hq * dr
    wround = attn.sliding and kcw.dtype == mx.bfloat16
    tau = None if attn.log_floor is None else (attn.log_alpha, attn.log_floor)
    qk_clip, r_clip = _consts(attn)
    kernel = _kernel(
        ("project", attn.sliding, qk_clip, r_clip, conv_scale, wround, tau),
        f"omlx_0ics_inkling2_project_{'swa' if attn.sliding else 'full'}",
        ["qsrc", "ksrc", "vsrc", "kst", "vst", "kcw", "vcw", "qscale", "rscale", "relp", "offs"],
        ["q", "k", "v", "rel", "xpk", "xpv"],
        lambda: _project_source(attn.sliding, qk_clip, r_clip, conv_scale, wround, tau),
    )
    # tokens already in the cache, per batch row (for one request a constant upload, no kernel)
    offs = offset.astype(mx.int32) if isinstance(offset, mx.array) else mx.array([int(offset)], dtype=mx.int32)
    r_extent = attn.rel_extent
    q, k, v, rel, xpk, xpv = kernel(
        inputs=[qsrc, ksrc, vsrc, ks, vs, kcw, vcw, attn.q_scale, attn.r_scale, attn.rel_proj, offs],
        template=[("HQ", hq), ("HKV", hkv), ("D", dh), ("DR", dr), ("R", r_extent), ("C", c), ("L", length),
                  ("DKV", dkv), ("QW", qsrc.shape[-1]), ("SW", ksrc.shape[-1]), ("R_OFF", q_off_r),
                  ("KV_OFF", kv_off), ("V_SHIFT", v_shift), ("OFFS_B", offs.size > 1)],
        grid=(b * length * (hq + 2 * hkv) * dh, 1, 1),
        threadgroup=(dh, 1, 1),
        output_shapes=[(b, hq, length, dh), (b, hkv, length, dh), (b, hkv, length, dh), (b, length, hq, r_extent),
                       (b, c - 1 + length, dkv), (b, c - 1 + length, dkv)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )
    gate = qsrc[..., gate_off:gate_off + hq * dh]
    sink = qsrc[..., gate_off + hq * dh:gate_off + hq * dh + hq]
    return q, k, v, rel, gate, sink, xpk, xpv


# ---------------------------------------------------------------- after the attention's output projection


def _post_attn_source(nb: int, scale: float, conv_scale: float, wround: bool, boundary: bool) -> str:
    def src(i):
        return f"b{i}[base + h]" if i < nb else "res[j] * inv0"

    rnd = "acc = float(bfloat16_t(acc));" if wround else ""
    resid = "float(a2)" if boundary else "float(hin[base + h]) + float(a2)"
    xp = ("((l + (J)) < (C - 1) ? st[((size_t)bi * (C - 1) + l + (J)) * H + h]"
          " : float(a[((size_t)bi * L + l + (J) - (C - 1)) * H + h]))")
    return f"""
    constexpr float MIX_SCALE = {_f(scale)};
    constexpr uint Q = 1;
{_ROW_PRELUDE}
    uint bi = row / L;
    uint l = row % L;
#define XP(J) {xp}
    // short conv of the attention output, then the residual (FP32), kept in registers
    float res[PER];
    float ss = 0.0f;
    {{
        uint j = 0;
        for (uint h = tid; h < H; h += TG, ++j) {{
            float acc = float(cw[h * C + 0]) * XP(0);
            for (uint t = 1; t < C; ++t) acc += float(cw[h * C + t]) * XP(t);
            {rnd}
            bfloat16_t a2 = bfloat16_t((acc + XP(C - 1)) * {_f(conv_scale)});
            res[j] = {resid};
            residual[base + h] = res[j];
            ss += res[j] * res[j];
            xpo[((size_t)bi * (C - 1 + L) + C - 1 + l) * H + h] = XP(C - 1);
            if (l == 0) {{
                for (uint i = 0; i < C - 1; ++i) {{
                    xpo[((size_t)bi * (C - 1 + L) + i) * H + h] = st[((size_t)bi * (C - 1) + i) * H + h];
                }}
            }}
        }}
    }}
#undef XP
    float inv0 = metal::precise::rsqrt(tg_sum<TG>(ss, red, lane, sg) / float(H) + {_f(_EPS)});
{_mix_weights(nb + 1, src)}
{_mix_norm(nb + 1, src, "x[base + h]")}
"""


def post_attn(layer, a: mx.array, s2: mx.array, h: mx.array, blocks: list):
    """``Inkling2Layer._post`` from the output projection to the MoE input: (residual FP32, x bf16, conv window)."""
    b, length, hidden = a.shape
    nb = len(blocks)
    cw = layer.attn_conv
    wround = cw.dtype == mx.bfloat16
    scale = layer.mlp_input.scale
    boundary = layer.is_boundary
    kernel = _kernel(
        ("post_attn", nb, scale, layer._conv_scale, wround, boundary),
        f"omlx_0ics_inkling2_post_attn_{nb}{'_b' if boundary else ''}",
        [*(f"b{i}" for i in range(nb)), "a", "st", "cw", "hin", "query", "nw"],
        ["residual", "x", "xpo"],
        lambda: _post_attn_source(nb, scale, layer._conv_scale, wround, boundary),
    )
    residual, x, xpo = kernel(
        inputs=[*blocks, a, s2, cw, h, layer.mlp_input.query, layer.mlp_norm.weight],
        template=[("H", hidden), ("C", layer._conv_k), ("L", length), ("TG", _TG)],
        grid=(b * length * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(b, length, hidden), (b, length, hidden), (b, layer._conv_k - 1 + length, hidden)],
        output_dtypes=[mx.float32, mx.bfloat16, mx.float32],
    )
    return residual, x, xpo


# ---------------------------------------------------------------- after the experts


def _post_moe_source(conv_scale: float, wround: bool) -> str:
    rnd = "acc = float(bfloat16_t(acc));" if wround else ""
    return f"""
    uint gid = thread_position_in_grid.x;
    uint bi = gid / H;
    uint h = gid % H;
    float win[C];
    for (uint i = 0; i < C - 1; ++i) {{
        win[i] = st[((size_t)bi * (C - 1) + i) * H + h];
        xpo[((size_t)bi * (C - 1 + L) + i) * H + h] = win[i];
    }}
    for (uint l = 0; l < L; ++l) {{
        size_t row = (size_t)bi * L + l;
        // the experts' outputs weighted in FP32, rounded to bf16 as the reference's sum over the K experts
        float m = float(z[(row * K + 0) * H + h]) * wts[row * K + 0];
        for (uint e = 1; e < K; ++e) m += float(z[(row * K + e) * H + h]) * wts[row * K + e];
        win[C - 1] = float(bfloat16_t(m));
        xpo[((size_t)bi * (C - 1 + L) + C - 1 + l) * H + h] = win[C - 1];
        float acc = float(cw[h * C + 0]) * win[0];
        for (uint t = 1; t < C; ++t) acc += float(cw[h * C + t]) * win[t];
        {rnd}
        bfloat16_t mc = bfloat16_t((acc + win[C - 1]) * {_f(conv_scale)});
        hout[row * H + h] = bfloat16_t(residual[row * H + h] + float(mc));
        for (uint i = 0; i + 1 < C; ++i) win[i] = win[i + 1];
    }}
"""


def post_moe(layer, z: mx.array, weights: mx.array, s3: mx.array, residual: mx.array):
    """The experts' weighted sum, the MLP's short conv and the residual: (h bf16, conv window)."""
    b, length, hidden = residual.shape
    cw = layer.mlp_conv
    wround = cw.dtype == mx.bfloat16
    kernel = _kernel(
        ("post_moe", layer._conv_scale, wround),
        "omlx_0ics_inkling2_post_moe",
        ["z", "wts", "st", "cw", "residual"],
        ["hout", "xpo"],
        lambda: _post_moe_source(layer._conv_scale, wround),
    )
    hout, xpo = kernel(
        inputs=[z, weights, s3, cw, residual],
        template=[("H", hidden), ("C", layer._conv_k), ("L", length), ("K", z.shape[-2])],
        grid=(b * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, length, hidden), (b, layer._conv_k - 1 + length, hidden)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return hout, xpo


# ---------------------------------------------------------------- the experts' down projection


# MLX's qmv reads 4 bytes a lane at 2 bits, and the down projection's K = 2304 is not a multiple of its 512-value
# block, which sends it to the general path (28/11, M5 Ultra: 549 GB/s against 750 for the gate/up projection, K =
# 4608). Here a quad of 4 lanes takes R rows of one (token, expert) pair, each lane reading 16 bytes a row (a group
# of 64 two-bit weights, with its scale and bias) and reusing its slice of x across the R rows. A token pair per
# threadgroup: the experts two tokens of a verify share are read twice, at the same time, from the cache.
# Measured against MLX's gather (fused layer, 4K cache): decode 16.88 → 16.81 ms, verify of 5 tokens 45.1 → 36.7 ms,
# of 9 tokens 70.2 → 64.1 ms; wider quads (R = 4, 8) or reading a shared expert once for all its tokens were slower.
_DOWN_SOURCE = """
    constexpr uint G = K / 64;
    constexpr uint ROWS = NSG * 8 * R;
    const uint pair = threadgroup_position_in_grid.y;
    const uint rb = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ql = lane % 4;
    const uint e = inds[pair];
    // a quad of lanes takes R consecutive rows; its lanes split the row's groups
    const uint row0 = rb * ROWS + sg * (8 * R) + (lane / 4) * R;
    const device uint4* wr = (const device uint4*)(w + ((size_t)e * O + row0) * (K / 16));
    const device bfloat16_t* sr = scales + ((size_t)e * O + row0) * G;
    const device bfloat16_t* br = biases + ((size_t)e * O + row0) * G;
    const device uint4* xp = (const device uint4*)(x + (size_t)pair * K);
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint g = ql; g < G; g += 4) {
        uint4 wv[R];
        float dot[R];
        for (uint r = 0; r < R; ++r) {
            wv[r] = wr[r * (K / 64) + g];
            dot[r] = 0.0f;
        }
        float xs = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint4 xa = xp[g * 8 + 2 * i];
            const uint4 xb = xp[g * 8 + 2 * i + 1];
            const uint xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            for (uint j = 0; j < 8; ++j) {
                const float x0 = as_type<float>(xw[j] << 16);
                const float x1 = as_type<float>(xw[j] & 0xFFFF0000u);
                xs += x0 + x1;
                // a weight masked in place is q * 4^v, exact in a float; x carries 4^-v instead (exact too)
                const float s0 = x0 * as_type<float>((127u - 4u * j) << 23);
                const float s1 = x1 * as_type<float>((125u - 4u * j) << 23);
                for (uint r = 0; r < R; ++r) {
                    const uint wi = wv[r][i];
                    dot[r] += s0 * float(wi & (3u << (4 * j))) + s1 * float(wi & (3u << (4 * j + 2)));
                }
            }
        }
        for (uint r = 0; r < R; ++r) acc[r] += float(sr[r * G + g]) * dot[r] + float(br[r * G + g]) * xs;
    }
    for (uint r = 0; r < R; ++r) {
        float a = acc[r];
        a += simd_shuffle_xor(a, 1);
        a += simd_shuffle_xor(a, 2);
        if (ql == 0) y[(size_t)pair * O + row0 + r] = bfloat16_t(a);
    }
"""
_DOWN_R = 2
_DOWN_SG = 2


def down_supported(proj) -> bool:
    """A 2-bit affine carrier in groups of 64 (oQ2e) whose rows fill whole threadgroups."""
    return (proj.mode == "affine" and proj.bits == 2 and proj.group_size == 64
            and proj.weight.shape[1] % (_DOWN_SG * 8 * _DOWN_R) == 0)


def expert_down(proj, act: mx.array, inds: mx.array) -> mx.array:
    """The chosen experts' down projection: act [N, KS, K] bf16 (one row per token pair), inds [N, KS]. Returns
    [N, KS, O] bf16, as ``ExpertProj.__call__`` then ``squeeze(-2)``."""
    n, ks = inds.shape
    o, k = proj.weight.shape[1], proj.weight.shape[2] * 16
    kernel = _kernel("down", "omlx_0ics_inkling2_expert_down", ["x", "w", "scales", "biases", "inds"], ["y"],
                     lambda: _DOWN_SOURCE)
    return kernel(
        inputs=[act, proj.weight, proj.scales, proj.biases, inds],
        template=[("K", k), ("O", o), ("R", _DOWN_R), ("NSG", _DOWN_SG)],
        grid=(o // (_DOWN_SG * 8 * _DOWN_R) * 32 * _DOWN_SG, n * ks, 1),
        threadgroup=(32 * _DOWN_SG, 1, 1),
        output_shapes=[(n, ks, o)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------- the layer


_lhs: dict = {}


def _lhs_indices(n: int, k: int) -> tuple[mx.array, mx.array]:
    """``gather_qmm``'s lhs indices for N tokens of K experts (w13: the token, w2: its own row), kept: left out,
    MLX builds them with an ``arange`` kernel per call."""
    got = _lhs.get((n, k))
    if got is None:
        got = _lhs[(n, k)] = (mx.broadcast_to(mx.arange(n, dtype=mx.uint32)[:, None], (n, k)),
                              mx.arange(n * k, dtype=mx.uint32).reshape(n, k))
        mx.eval(got)
    return got


def _gather(proj, x, lhs, rhs):
    """``ExpertProj.__call__`` with explicit lhs indices."""
    return mx.gather_qmm(x, proj.weight, proj.scales, proj.get("biases"), lhs_indices=lhs, rhs_indices=rhs,
                         transpose=True, group_size=proj.group_size, bits=proj.bits, mode=proj.mode,
                         sorted_indices=False)


def layer_forward(layer, h, blocks, kv, states):
    """``Inkling2Layer.__call__`` for a short input (``supports``); returns (h, the four conv windows)."""
    from . import attn_decode, attn_sparse, kernels
    from .model import _softcapped_swiglu

    hidden = h.shape[-1]
    attn = layer.attn
    h_n, xs = pre(layer, h, blocks)
    q, k, v, rel, gate, sink, ks, vs = project(attn, xs, states[0], states[1], kv.offset if kv is not None else 0)
    left_padding = getattr(kv, "left_padding", None) if kv is not None else None
    sparse = kv is not None and attn.window == 0 and attn_sparse.enabled()
    if kv is not None:
        start = kv.offset
        k, v = kv.update_and_fetch(k, v)
        if sparse:
            attn_sparse.note_append(kv, start)
    o_in = None
    if sparse and attn_decode.supports_fused(q, left_padding, k, v):
        # 0ics OMLX_0ICS_INKLING_SPARSE: the selected blocks and the tail of the cache only (attn_sparse.py)
        o_in = attn_sparse.attention(kv, q, k, v, rel, sink, scale=attn.scale, gate=gate, min_chunk=MIN_CHUNK)
    if o_in is not None:
        a = attn.o_proj(o_in)
    elif attn_decode.supports_fused(q, left_padding, k, v):
        # the split attention, with the sink and the output gate in its combine: o_proj's input
        o_in = attn_decode.attention(q, k, v, rel, sink, scale=attn.scale, window=attn.window, gate=gate,
                                     min_chunk=MIN_CHUNK)
        a = attn.o_proj(o_in)
    else:
        out = kernels.banded_sink_attention(q, k, v, rel, sink, scale=attn.scale, window=attn.window,
                                            left_padding=left_padding)
        a = attn.output(out, gate)
    if layer.is_boundary:
        blocks.append(h_n)
    residual, x, s2 = post_attn(layer, a, states[2], h, blocks)
    mlp = layer.mlp
    flat = x.reshape(-1, hidden)
    inds, w = kernels.route_topk(router_logits(flat, mlp.router_weight), mlp.router_bias, mlp.top_k,
                                 mlp.route_scale)
    # Inkling2Experts before its weighted sum (affine carriers: no post-scale)
    experts = mlp.experts
    lhs13, lhs2 = _lhs_indices(*inds.shape)
    y = _gather(experts.w13, mx.expand_dims(flat, (-2, -3)), lhs13, inds)
    act = _softcapped_swiglu(y[..., 0::2], y[..., 1::2], experts._gcap, experts._ucap)
    if down_supported(experts.w2):
        z = expert_down(experts.w2, act.squeeze(-2), inds)
    else:
        z = _gather(experts.w2, act, lhs2, inds).squeeze(-2)
    h_out, s3 = post_moe(layer, z, w, states[3], residual)
    return h_out, (ks, vs, s2, s3)
