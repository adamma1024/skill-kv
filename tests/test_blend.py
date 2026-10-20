"""Selective recompute (CacheBlend-style) and the attention-mask control, on tiny Qwen3 / Llama."""

import mlx.core as mx
import pytest
from mlx_lm.models import llama, qwen3
from mlx_lm.models.cache import make_prompt_cache

from skillkv.blend import blend
from skillkv.pack import build_pack, prefill, splice

V = 97


def tiny(kind):
    mx.random.seed(0)
    if kind == "qwen3":
        m = qwen3.Model(qwen3.ModelArgs(
            model_type="qwen3", hidden_size=64, num_hidden_layers=3, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
            vocab_size=V, rope_theta=10000.0, max_position_embeddings=4096, tie_word_embeddings=True))
    else:
        m = llama.Model(llama.ModelArgs(
            model_type="llama", hidden_size=64, num_hidden_layers=3, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=V,
            rope_theta=500000.0, tie_word_embeddings=True,
            rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                          "high_freq_factor": 4.0, "original_max_position_embeddings": 64}))
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module", params=["qwen3", "llama"])
def model(request):
    return tiny(request.param)


def ids(n, start):
    return [(start + 11 * i) % V for i in range(n)]


SYS, TURN, WRAP, DOC, SUF = ids(12, 1), ids(9, 2), ids(3, 4), ids(20, 5), ids(4, 6)


def run(model, how, turn=TURN, **kw):
    """Prefix [SYS TURN WRAP], doc mounted via `how`, then SUF. Returns (last logits, cache)."""
    cache = make_prompt_cache(model)
    prefill(model, cache, SYS + turn + WRAP)
    pack = build_pack(model, SYS + WRAP, DOC, model_id="t")
    if how == "text":
        prefill(model, cache, DOC)
    elif how == "splice":
        splice(cache, pack, model, model_id="t", seam=kw.get("seam", 0))
        if kw.get("seam"):
            prefill(model, cache, DOC[-kw["seam"]:])
    else:
        stats = blend(model, cache, DOC, pack=pack, model_id="t", **kw)
        assert stats["n"] == len(DOC)
    return prefill(model, cache, SUF), cache


def test_full_recompute_equals_text(model):
    a, _ = run(model, "text")
    b, _ = run(model, "blend", select="all")
    assert mx.allclose(a, b, atol=1e-4)


def test_empty_selection_equals_plain_splice(model):
    a, _ = run(model, "splice")
    b, _ = run(model, "blend", select=[])
    assert mx.allclose(a, b, atol=1e-4)


def test_last_k_selection_equals_seam(model):
    a, _ = run(model, "splice", seam=5)
    b, _ = run(model, "blend", select=list(range(len(DOC) - 5, len(DOC))))
    assert mx.allclose(a, b, atol=1e-4)


def test_deviation_selection_size_and_full_ratio(model):
    cache = make_prompt_cache(model)
    prefill(model, cache, SYS + TURN + WRAP)
    pack = build_pack(model, SYS + WRAP, DOC, model_id="t")
    stats = blend(model, cache, DOC, pack=pack, model_id="t", select=("deviation", 0.15))
    assert stats["selected"] == 3  # ceil(0.15 * 20)
    assert stats["full_layers"] == 2
    a, _ = run(model, "text")
    b, _ = run(model, "blend", select=("deviation", 1.0))
    assert mx.allclose(a, b, atol=1e-4)


def test_random_selection_is_seeded(model):
    c1 = make_prompt_cache(model)
    prefill(model, c1, SYS + TURN + WRAP)
    pack = build_pack(model, SYS + WRAP, DOC, model_id="t")
    s1 = blend(model, c1, DOC, pack=pack, model_id="t", select=("random", 0.15, 3))
    c2 = make_prompt_cache(model)
    prefill(model, c2, SYS + TURN + WRAP)
    s2 = blend(model, c2, DOC, pack=pack, model_id="t", select=("random", 0.15, 3))
    assert s1["indices"] == s2["indices"] and len(s1["indices"]) == 3


def masked(model, turn):
    """T1' control: [SYS TURN] then [WRAP DOC] recomputed with no attention to TURN."""
    cache = make_prompt_cache(model)
    prefill(model, cache, SYS + turn)
    lo, hi = len(SYS), len(SYS) + len(turn)
    blend(model, cache, WRAP + DOC, select="all", exclude=(lo, hi))
    return cache


def test_masked_segment_is_independent_of_turn(model):
    c1, c2 = masked(model, TURN), masked(model, ids(9, 50))
    seg = slice(len(SYS) + len(TURN), len(SYS) + len(TURN) + len(WRAP) + len(DOC))
    for a, b in zip(c1, c2):
        assert mx.allclose(a.keys_and_values()[0][..., seg, :], b.keys_and_values()[0][..., seg, :], atol=1e-5)


def test_masked_equals_pack_without_shift_when_turn_empty(model):
    # with no turn there is nothing to mask: masked recompute == exact text
    cache = make_prompt_cache(model)
    prefill(model, cache, SYS)
    blend(model, cache, WRAP + DOC, select="all", exclude=(len(SYS), len(SYS)))
    a = prefill(model, cache, SUF)
    b = prefill(model, make_prompt_cache(model), SYS + WRAP + DOC + SUF)
    assert mx.allclose(a, b, atol=1e-4)


def test_pack_required_for_partial_selection(model):
    cache = make_prompt_cache(model)
    prefill(model, cache, SYS)
    with pytest.raises(ValueError, match="pack"):
        blend(model, cache, DOC, pack=None, select=[1, 2])
