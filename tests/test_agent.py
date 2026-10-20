"""Prompt layouts for C0-C3 and the load_skill selection loop, on a tiny random Qwen3."""

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache

from skillkv.agent import Layout, parse_selection, post_selection
from skillkv.pack import build_pack, prefill

MODEL_ID = "tiny"


class CharTok:
    """Deterministic stand-in tokenizer: one id per character."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 97 for c in s]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        vocab_size=97, rope_theta=10000.0, max_position_embeddings=4096, tie_word_embeddings=True,
    )
    m = qwen3.Model(args)
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def layout():
    skills = [("alpha-audit", "Use for alpha audits.", "---\nname: alpha-audit\n---\nRule A1.\nRule A2."),
              ("beta-rollout", "Use for beta rollouts.", "---\nname: beta-rollout\n---\nRule B1.")]
    return Layout(CharTok().encode, skills)


@pytest.mark.parametrize("text,name", [
    ('<tool_call>\n{"name": "load_skill", "arguments": {"name": "alpha-audit"}}\n</tool_call>', "alpha-audit"),
    ('sure <tool_call>{"name":"load_skill","arguments":{"name":"beta-rollout"}}', "beta-rollout"),
    ("I will just answer directly.", None),
    ('<tool_call>{"name": "load_skill", "arguments": {"name": "nope"}}</tool_call>', "nope"),
])
def test_parse_selection(text, name):
    assert parse_selection(text) == name


def test_parse_selection_accepts_skill_called_as_tool():
    text = '{"name": "beta-rollout", "arguments": {"procedure": "1. do it"}}'
    assert parse_selection(text, known={"alpha-audit", "beta-rollout"}) == "beta-rollout"
    assert parse_selection(text) is None


def test_catalog_lists_every_skill_once(layout):
    for name, desc, _ in layout.skills:
        assert layout.sys_lazy_text.count(f"- {name}: {desc}") == 1
    assert "load_skill" in layout.sys_lazy_text


def test_eager_system_holds_all_skill_texts(layout):
    for _, _, text in layout.skills:
        assert text in layout.sys_eager_text


def test_c1_and_c3_see_identical_token_ids(layout):
    q = "please audit alpha"
    c1 = layout.c1_ids(q, "alpha-audit")
    assert c1 == (layout.sys_lazy + layout.user_turn(q) + layout.tool_call("alpha-audit")
                  + layout.wrap + layout.skill("alpha-audit") + layout.suffix)


def test_c2_prelude_is_c2_prefix(layout):
    q = "please audit alpha"
    c2 = layout.c2_ids(q, "alpha-audit")
    n = len(layout.c2_prelude)
    assert c2[:n] == layout.c2_prelude
    assert c2[n : n + len(layout.skill("alpha-audit"))] == layout.skill("alpha-audit")


def test_unknown_skill_fails_fast(layout):
    with pytest.raises(KeyError, match="nope"):
        layout.skill("nope")


def _first_logits_full(model, ids):
    return prefill(model, make_prompt_cache(model), ids)


@pytest.mark.parametrize("cond", ["C1", "C2"])
def test_exact_conditions_match_full_prefill(model, layout, cond):
    q, name = "please audit alpha", "alpha-audit"
    prelude = layout.pack_prelude if cond == "C1" else layout.c2_prelude
    pack = build_pack(model, prelude, layout.skill(name), model_id=MODEL_ID)
    cache = make_prompt_cache(model)
    prefill(model, cache, layout.sys_lazy if cond == "C1" else layout.c2_prelude)
    logits, ttft = post_selection(model, cache, layout, cond, q, name, pack=pack, model_id=MODEL_ID)
    ref = _first_logits_full(model, layout.c1_ids(q, name) if cond == "C1" else layout.c2_ids(q, name))
    assert ttft > 0
    assert mx.allclose(logits, ref, atol=1e-4)


def test_c3_approximates_c1_with_same_length(model, layout):
    q, name = "please audit alpha", "alpha-audit"
    pack = build_pack(model, layout.pack_prelude, layout.skill(name), model_id=MODEL_ID)
    cache = make_prompt_cache(model)
    prefill(model, cache, layout.sys_lazy)
    post_selection(model, cache, layout, "C3", q, name, pack=pack, model_id=MODEL_ID)
    assert cache[0].offset == len(layout.c1_ids(q, name))


def test_c0_uses_eager_cache(model, layout):
    q = "please audit alpha"
    cache = make_prompt_cache(model)
    prefill(model, cache, layout.sys_eager)
    logits, _ = post_selection(model, cache, layout, "C0", q, None)
    ref = _first_logits_full(model, layout.c0_ids(q))
    assert mx.allclose(logits, ref, atol=1e-4)


def test_generic_prelude_has_c1_structure(layout):
    pre = layout.generic_prelude("alpha-audit")
    assert pre == (layout.sys_lazy + layout.user_turn(layout.GENERIC_QUERY)
                   + layout.tool_call("alpha-audit") + layout.wrap)


def test_c3_with_query_prelude_pack_reproduces_c1(model, layout):
    # control: a pack whose prelude is the exact C1 prefix equals text prefill
    q, name = "please audit alpha", "alpha-audit"
    pre = layout.sys_lazy + layout.user_turn(q) + layout.tool_call(name) + layout.wrap
    pack = build_pack(model, pre, layout.skill(name), model_id=MODEL_ID)
    cache = make_prompt_cache(model)
    prefill(model, cache, layout.sys_lazy)
    logits, _ = post_selection(model, cache, layout, "C3", q, name, pack=pack, model_id=MODEL_ID)
    assert mx.allclose(logits, _first_logits_full(model, layout.c1_ids(q, name)), atol=1e-4)


def test_sampled_generation_is_reproducible_per_seed(model, layout):
    from mlx_lm.sample_utils import make_sampler

    from skillkv.agent import generate

    sampler = make_sampler(temp=1.0, top_p=0.95)
    outs = []
    for seed in (7, 7, 8):
        cache = make_prompt_cache(model)
        first = prefill(model, cache, layout.sys_lazy[:40])
        outs.append(generate(model, cache, first, 20, eos=set(), sampler=sampler, seed=seed))
    assert outs[0] == outs[1] and outs[0] != outs[2]
