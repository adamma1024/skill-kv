"""KV pack build / save / load / splice on a tiny random Qwen3 (no download)."""

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache

from skillkv.pack import (
    PackMismatchError,
    build_pack,
    load_pack,
    prefill,
    rope_shift,
    save_pack,
    splice,
)

MODEL_ID = "tiny-qwen3-test"


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        vocab_size=97, rope_theta=10000.0, max_position_embeddings=512, tie_word_embeddings=True,
    )
    m = qwen3.Model(args)
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


def ids(n, start=1):
    return [(start + 7 * i) % 97 for i in range(n)]


def full_logits(model, tokens):
    cache = make_prompt_cache(model)
    return prefill(model, cache, tokens, return_all=True)


def test_rope_shift_composes_with_rope(model):
    rope = model.layers[0].self_attn.rope
    k = mx.random.normal((1, 2, 6, 16))
    shifted = rope_shift(rope, rope(k, offset=5), 9)
    assert mx.allclose(shifted, rope(k, offset=14), atol=1e-5)


def test_splice_with_exact_prelude_reproduces_full_prefill(model):
    prefix, skill, suffix = ids(10), ids(12, start=3), ids(4, start=5)
    ref = full_logits(model, prefix + skill + suffix)

    pack = build_pack(model, prelude=prefix, segment=skill, model_id=MODEL_ID)
    cache = make_prompt_cache(model)
    prefill(model, cache, prefix)
    splice(cache, pack, model, model_id=MODEL_ID)
    out = prefill(model, cache, suffix, return_all=True)
    assert mx.allclose(out, ref[:, -len(suffix):], atol=1e-4)


def test_splice_at_new_position_shifts_rope(model):
    # Pack built with a 3-token prelude, mounted after a 10-token prefix that starts with the
    # same 3 tokens: only cross-attention to tokens 3..9 is missing, positions must still match.
    prefix, skill, suffix = ids(10), ids(12, start=3), ids(4, start=5)
    pack = build_pack(model, prelude=prefix[:3], segment=skill, model_id=MODEL_ID)
    assert pack.start_pos == 3
    cache = make_prompt_cache(model)
    prefill(model, cache, prefix)
    splice(cache, pack, model, model_id=MODEL_ID)
    assert cache[0].offset == len(prefix) + len(skill)
    out = prefill(model, cache, suffix, return_all=True)
    ref = full_logits(model, prefix + skill + suffix)[:, -len(suffix):]
    assert out.shape == ref.shape
    assert not mx.allclose(out, ref, atol=1e-4)  # approximation, not exact


def test_seam_recompute_keeps_positions(model):
    prefix, skill, suffix = ids(10), ids(12, start=3), ids(4, start=5)
    pack = build_pack(model, prelude=[], segment=skill, model_id=MODEL_ID)
    cache = make_prompt_cache(model)
    prefill(model, cache, prefix)
    splice(cache, pack, model, model_id=MODEL_ID, seam=4)
    assert cache[0].offset == len(prefix) + len(skill) - 4
    prefill(model, cache, skill[-4:])
    assert cache[0].offset == len(prefix) + len(skill)


def test_save_load_roundtrip(model, tmp_path):
    pack = build_pack(model, prelude=ids(3), segment=ids(8), model_id=MODEL_ID)
    path = tmp_path / "s.safetensors"
    save_pack(path, pack)
    back = load_pack(path)
    assert back.start_pos == pack.start_pos and back.tokens == pack.tokens
    assert back.model_id == MODEL_ID and back.prelude_hash == pack.prelude_hash
    for (k1, v1), (k2, v2) in zip(pack.layers, back.layers):
        assert mx.array_equal(k1, k2) and mx.array_equal(v1, v2)


def test_splice_rejects_pack_from_other_model(model):
    pack = build_pack(model, prelude=[], segment=ids(5), model_id="other-model")
    cache = make_prompt_cache(model)
    prefill(model, cache, ids(3))
    with pytest.raises(PackMismatchError, match="other-model"):
        splice(cache, pack, model, model_id=MODEL_ID)


def test_seam_larger_than_pack_is_rejected(model):
    pack = build_pack(model, prelude=[], segment=ids(5), model_id=MODEL_ID)
    cache = make_prompt_cache(model)
    with pytest.raises(ValueError, match="seam"):
        splice(cache, pack, model, model_id=MODEL_ID, seam=6)
