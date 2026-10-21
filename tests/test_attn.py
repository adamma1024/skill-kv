"""Explicit-attention forward for one query token: must reproduce the model's logits."""

import mlx.core as mx
import pytest
from mlx_lm.models import llama, qwen3
from mlx_lm.models.cache import make_prompt_cache

from skillkv.attn import attention_profile, span_mass
from skillkv.pack import prefill

V = 97


def tiny(kind):
    mx.random.seed(1)
    if kind == "qwen3":
        m = qwen3.Model(qwen3.ModelArgs(
            model_type="qwen3", hidden_size=64, num_hidden_layers=3, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
            vocab_size=V, rope_theta=10000.0, max_position_embeddings=4096, tie_word_embeddings=True))
    else:
        m = llama.Model(llama.ModelArgs(
            model_type="llama", hidden_size=64, num_hidden_layers=3, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=V,
            rope_theta=500000.0, tie_word_embeddings=False))
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module", params=["qwen3", "llama"])
def model(request):
    return tiny(request.param)


CTX = [(3 + 7 * i) % V for i in range(30)]


def test_logits_match_model_forward(model):
    c1 = make_prompt_cache(model)
    prefill(model, c1, CTX[:-1])
    probs, logits = attention_profile(model, c1, CTX[-1])
    c2 = make_prompt_cache(model)
    prefill(model, c2, CTX[:-1])
    ref = prefill(model, c2, CTX[-1:])  # the model's own incremental step (one-shot prefill differs ~1e-3 in MLX)
    assert mx.allclose(logits, ref[0, -1], atol=1e-5)
    assert len(probs) == len(model.layers)
    for p in probs:
        assert p.shape == (len(CTX),)
        assert abs(mx.sum(p).item() - 1.0) < 1e-4
    assert c1[0].offset == len(CTX)


def test_span_mass_sums_named_spans(model):
    c = make_prompt_cache(model)
    prefill(model, c, CTX[:-1])
    probs, _ = attention_profile(model, c, CTX[-1])
    mass = span_mass(probs, {"a": (0, 10), "b": (10, 30)})
    for la, lb in zip(mass["a"], mass["b"]):
        assert abs(la + lb - 1.0) < 1e-4
