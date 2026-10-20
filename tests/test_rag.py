"""W6 RAG layout: [sys][user: documents ... question][assistant], documents precomputed per chunk."""

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache

from skillkv.chat import LLAMA3, QWEN
from skillkv.inject import make_items, rag_docs, rag_question, rag_segments
from skillkv.pack import build_pack, prefill, splice


@pytest.fixture(scope="module")
def items():
    return make_items(n_docs=12, seed=0)


def ident(s):
    return [s]


def test_rag_question_drops_the_tool_instruction(items):
    q = rag_question(items[1])
    assert "read_doc" not in q and items[1].name in q and q.endswith("Answer in one sentence.")


def test_rag_docs_place_target_and_use_clean_distractors(items):
    for idx, it in enumerate(items):
        docs = rag_docs(items, idx, k=3)
        assert len(docs) == 3 and docs[idx % 3] == (it.name, it.doc)
        others = [d for j, d in enumerate(docs) if j != idx % 3]
        assert all(n != it.name for n, _ in others) and len({n for n, _ in others}) == 2
        assert all("AI assistant" not in t for _, t in others)
    assert rag_docs(items, 5, k=3) == rag_docs(items, 5, k=3)


@pytest.mark.parametrize("fmt", [QWEN, LLAMA3])
def test_rag_segments_order(fmt):
    seg = rag_segments(ident, fmt, [("a", "TEXT-A"), ("b", "TEXT-B")], "Q?")
    head, docs, tail = "".join(seg["head"]), ["".join(s) for s in seg["segs"]], "".join(seg["tail"])
    assert head.startswith(fmt.sys_open) and head.endswith(fmt.user_open + "Documents:\n")
    assert docs == ['\n<document name="a">\nTEXT-A\n</document>\n', '\n<document name="b">\nTEXT-B\n</document>\n']
    assert tail == "\nQuestion: Q?" + fmt.end + fmt.assistant_open


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    m = qwen3.Model(qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=97,
        rope_theta=10000.0, max_position_embeddings=4096, tie_word_embeddings=True))
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


def enc(s):
    return [ord(c) % 97 for c in s]


def mount_chunks(model, seg):
    """Each chunk precomputed after the shared head only (independent chunks, CacheBlend-style)."""
    cache = make_prompt_cache(model)
    prefill(model, cache, seg["head"])
    for s in seg["segs"]:
        splice(cache, build_pack(model, seg["head"], s, model_id="t"), model, model_id="t")
    return prefill(model, cache, seg["tail"]), cache


def test_single_chunk_is_exact_prefix(model):
    seg = rag_segments(enc, QWEN, [("a", "alpha text here")], "what?")
    a, _ = mount_chunks(model, seg)
    b = prefill(model, make_prompt_cache(model), seg["head"] + seg["segs"][0] + seg["tail"])
    assert mx.allclose(a, b, atol=1e-4)


def test_later_chunks_miss_cross_chunk_attention(model):
    seg = rag_segments(enc, QWEN, [("a", "alpha text here"), ("b", "bravo text there")], "what?")
    a, cache = mount_chunks(model, seg)
    full = seg["head"] + seg["segs"][0] + seg["segs"][1] + seg["tail"]
    b = prefill(model, make_prompt_cache(model), full)
    assert cache[0].offset == len(full)
    assert not mx.allclose(a, b, atol=1e-4)
