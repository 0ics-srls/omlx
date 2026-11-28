# SPDX-License-Identifier: Apache-2.0
"""0ics: the fused Inkling 2 layer for decode and verify (omlx/patches/inkling2/fused_decode.py)."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX or not mx.metal.is_available(), reason="MLX with Metal not available")

HIDDEN = 256
WINDOW = 1024
# 8 layers: 0-4 and 6-7 sliding, 5 global; attention-residual blocks start at 0 and 6
TEXT = {
    "hidden_size": HIDDEN, "num_hidden_layers": 8, "vocab_size": 512, "unpadded_vocab_size": 500,
    "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 128,
    "swa_num_attention_heads": 6, "swa_num_key_value_heads": 2, "swa_head_dim": 128,
    "sliding_window_size": WINDOW, "rel_extent": WINDOW, "d_rel": 16, "log_scaling_n_floor": 1000,
    "n_routed_experts": 32, "num_experts_per_tok": 4, "intermediate_size": 128,
}


@pytest.fixture(scope="module")
def model():
    """A small Inkling 2 with random weights, the experts in 2-bit affine groups of 64 as oQ2e stores them."""
    from mlx.utils import tree_flatten, tree_unflatten

    from omlx.patches.inkling2.model import Model, ModelArgs

    m = Model(ModelArgs(text_config=TEXT, quantization={"group_size": 64, "bits": 2, "mode": "affine"}))
    mx.random.seed(0)
    params = {}
    for name, p in tree_flatten(m.parameters()):
        if name.endswith(("w13.weight", "w2.weight", "w13.scales", "w2.scales", "w13.biases", "w2.biases")):
            continue
        # bf16 as the loader casts a checkpoint, the router and the attention-residual queries FP32
        dtype = mx.float32 if name.endswith(("router_weight", "router_bias", ".query")) else mx.bfloat16
        if name.endswith(("norm.weight", "q_scale", "r_scale")):
            params[name] = (1.0 + 0.1 * mx.random.normal(p.shape)).astype(dtype)
        else:
            params[name] = (0.3 * mx.random.normal(p.shape)).astype(dtype)
    for layer in m.model.layers:
        for proj in (layer.mlp.experts.w13, layer.mlp.experts.w2):
            e, o, packed = proj.weight.shape
            w = (0.05 * mx.random.normal((e, o, packed * 16))).astype(mx.bfloat16)
            proj.weight, proj.scales, proj.biases = mx.quantize(w, group_size=64, bits=2)
    m.update(tree_unflatten(list(params.items())))
    mx.eval(m.parameters())
    return m


@pytest.fixture()
def fused(monkeypatch):
    monkeypatch.setenv("OMLX_0ICS_INKLING_WINDOW", "1")
    monkeypatch.setenv("OMLX_0ICS_INKLING_ATTN", "1")
    return monkeypatch


def _logits(model, cache, tokens, on: bool, env):
    env.setenv("OMLX_0ICS_INKLING_FUSED", "1" if on else "0")
    out = model(tokens[None], cache=cache)
    mx.eval(out)
    return out[0].astype(mx.float32)


def test_switch_and_supported_layers(model, monkeypatch):
    from omlx.patches.inkling2 import fused_decode

    monkeypatch.delenv("OMLX_0ICS_INKLING_FUSED", raising=False)
    assert not fused_decode.enabled()
    monkeypatch.setenv("OMLX_0ICS_INKLING_FUSED", "1")
    assert fused_decode.enabled()
    layers = model.model.layers
    assert not fused_decode.supports(layers[0], 1)  # the first layer reads the embeddings directly
    assert fused_decode.supports(layers[1], 1) and fused_decode.supports(layers[5], fused_decode.MAX_TOKENS)
    assert not fused_decode.supports(layers[1], fused_decode.MAX_TOKENS + 1)  # prefill chunks keep the stock path
    assert fused_decode.down_supported(layers[1].mlp.experts.w2)


@pytest.mark.parametrize("index", [1, 5, 6])
def test_stages_match_the_reference_for_one_token(model, index):
    """Each kernel against the code it replaces (``_pre``, ``project``, ``_post``), for a sliding, a global and a
    block-boundary layer: bit for bit; the router GEMV within FP32 rounding."""
    from omlx.patches.inkling2 import fused_decode
    from omlx.patches.inkling2.model import _static_conv, _unit_rms

    layer = model.model.layers[index]
    attn = layer.attn
    mx.random.seed(index)
    h = (3 * mx.random.normal((1, 1, HIDDEN))).astype(mx.bfloat16)
    blocks = [_unit_rms(mx.random.normal((1, 1, HIDDEN))) for _ in range((index + 5) // 6)]
    st = [0.5 * mx.random.normal((1, 3, w)) for w in (attn.dkv, attn.dkv, HIDDEN, HIDDEN)]
    offset = 1500

    h_n = _unit_rms(h.astype(mx.float32))
    mixed = layer.attn_input(blocks, h_n)
    norms = [layer.attn_norm] if layer.sliding else [layer.q_input_norm, layer.k_input_norm, layer.v_input_norm]
    ref_xs = [n(mixed[:, :, j]).astype(mx.bfloat16) for j, n in enumerate(norms)]
    f_hn, f_xs = fused_decode.pre(layer, h, blocks)
    assert mx.array_equal(f_hn, h_n)
    for a, b in zip(f_xs, ref_xs, strict=True):
        assert mx.array_equal(a, b)

    tau = attn.log_tau(type("KV", (), {"offset": offset})(), 1)
    xk, xv = (None, None) if layer.sliding else (ref_xs[1], ref_xs[2])
    ref = attn.project(ref_xs[0], xk, xv, st[0], st[1], tau)
    got = fused_decode.project(attn, ref_xs, st[0], st[1], offset)
    for name, a, b in zip(("q", "k", "v", "rel", "gate", "sink", "conv k", "conv v"), got, ref, strict=True):
        assert mx.array_equal(a, b), name

    a = (0.5 * mx.random.normal((1, 1, HIDDEN))).astype(mx.bfloat16)
    mlp_blocks = [*blocks, h_n] if layer.is_boundary else blocks
    a2, ref_xp2 = _static_conv(a, layer.attn_conv, st[2], layer._conv_scale)
    residual = a2.astype(mx.float32) if layer.is_boundary else h.astype(mx.float32) + a2.astype(mx.float32)
    ref_x = layer.mlp_norm(layer.mlp_input(mlp_blocks, _unit_rms(residual))[:, :, 0]).astype(mx.bfloat16)
    f_res, f_x, f_xp2 = fused_decode.post_attn(layer, a, st[2], h, mlp_blocks)
    assert mx.array_equal(f_res, residual) and mx.array_equal(f_x, ref_x) and mx.array_equal(f_xp2, ref_xp2)

    flat = ref_x.reshape(-1, HIDDEN)
    logits = flat.astype(mx.float32) @ layer.mlp.router_weight.T
    assert mx.allclose(fused_decode.router_logits(flat, layer.mlp.router_weight), logits, rtol=1e-5, atol=1e-5)

    inds, w = layer.mlp.route(flat)
    z = mx.random.normal((1, inds.shape[1], HIDDEN)).astype(mx.bfloat16)
    zf = z.astype(mx.float32)
    m = (zf * w[..., None]).sum(axis=-2).astype(mx.bfloat16).reshape(1, 1, HIDDEN)
    m2, ref_xp3 = _static_conv(m, layer.mlp_conv, st[3], layer._conv_scale)
    f_h, f_xp3 = fused_decode.post_moe(layer, z, w, st[3], residual)
    assert mx.array_equal(f_h, (residual + m2.astype(mx.float32)).astype(mx.bfloat16))
    assert mx.array_equal(f_xp3, ref_xp3)


@pytest.mark.parametrize("tokens", [1, 5])
def test_expert_down_projection_matches_gather_qmm(model, tokens):
    from omlx.patches.inkling2 import fused_decode

    w2 = model.model.layers[2].mlp.experts.w2
    mx.random.seed(tokens)
    inds = mx.stack([mx.random.permutation(32)[:4] for _ in range(tokens)]).astype(mx.uint32)
    act = mx.random.normal((tokens, 4, 1, 128)).astype(mx.bfloat16)
    _, lhs2 = fused_decode._lhs_indices(tokens, 4)
    ref = fused_decode._gather(w2, act, lhs2, inds).squeeze(-2).astype(mx.float32)
    got = fused_decode.expert_down(w2, act.squeeze(-2), inds).astype(mx.float32)
    assert got.shape == ref.shape
    # one bf16 rounding apart at most, the sums run in another order
    assert float(mx.abs(got - ref).max()) <= float(mx.abs(ref).max()) * 2**-7


@pytest.mark.parametrize("window,keys", [(WINDOW, 700), (WINDOW, 1087), (WINDOW, 1343), (0, 3000)])
@pytest.mark.parametrize("length", [1, 5])
def test_split_attention_with_window_and_gate(window, keys, length):
    """The split kernel with a window and the output gate in its combine against the masked path."""
    from omlx.patches.inkling2 import attn_decode, kernels

    mx.random.seed(keys + length)
    q = mx.random.normal((1, 6, length, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, keys, 128)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, keys, 128)).astype(mx.bfloat16)
    rel = mx.random.normal((1, length, 6, WINDOW)).astype(mx.bfloat16)
    sink = mx.random.normal((1, length, 6)).astype(mx.bfloat16)
    gate = mx.random.normal((1, length, 6 * 128)).astype(mx.bfloat16)
    ref = kernels._masked_attention(q, k, v, rel, sink, 128**-0.5, window, None)
    out = attn_decode.attention(q, k, v, rel, sink, scale=128**-0.5, window=window, min_chunk=64)
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 1e-2
    ref_g = (ref.transpose(0, 2, 1, 3).reshape(1, length, -1).astype(mx.float32)
             * mx.sigmoid(gate.astype(mx.float32))).astype(mx.bfloat16)
    out_g = attn_decode.attention(q, k, v, rel, sink, scale=128**-0.5, window=window, gate=gate, min_chunk=64)
    assert out_g.shape == ref_g.shape
    assert float(mx.abs(out_g.astype(mx.float32) - ref_g.astype(mx.float32)).max()) < 1e-2


def test_decode_and_verify_track_the_reference(model, fused):
    """Two caches filled by the same prefill (past the sliding window), then decode, verifies of 5 with the
    rollback DSpark does, more decode: the fused layer, with the stock attention, keeps the stock layer's choices and
    logits (its glue is exact for one token; the router sums and the down projection round in another order).

    The split attention is checked on its own (``test_split_attention_with_window_and_gate``): its 1-ulp differences,
    through 8 layers of random weights, move the logits by whole units, as any other order of the sums would; on the
    real model (28/11, ``sperimenti/inkling/prova_nll.py``, 512 tokens of a source file one at a time) the fused path
    gives NLL 0.4494 against 0.4467, and changing only the router's order of sums moves it as much (0.4466)."""
    from omlx.patches.inkling2 import attn_decode

    fused.setattr(attn_decode, "supports_fused", lambda *a: False)
    serie, fusa = model.make_cache(), model.make_cache()
    prompt = mx.array([(i * 37) % 500 for i in range(1500)])
    for a in range(0, 1500, 512):
        ls = _logits(model, serie, prompt[a:a + 512], False, fused)
        _logits(model, fusa, prompt[a:a + 512], False, fused)
    tok = mx.argmax(ls[-1:], axis=-1)
    worst = 0.0

    def step(tokens):
        nonlocal worst
        ls, lf = _logits(model, serie, tokens, False, fused), _logits(model, fusa, tokens, True, fused)
        worst = max(worst, float(mx.abs(ls - lf).max() / mx.abs(ls).max()))
        assert mx.array_equal(mx.argmax(ls, axis=-1), mx.argmax(lf, axis=-1))
        return ls

    for _ in range(6):
        tok = mx.argmax(step(tok)[-1:], axis=-1)
    for r in range(2):
        block = mx.concatenate([tok, mx.array([(r * 31 + j * 7) % 500 for j in range(4)])])
        step(block)
        for c in (serie, fusa):
            assert model.mtp_partial_rollback(c, 1, 4)
        tok = block[1:2]
        for _ in range(3):
            tok = mx.argmax(step(tok)[-1:], axis=-1)
    assert worst < 0.05, worst
    assert serie[1][0].offset == fusa[1][0].offset
