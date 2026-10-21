"""Explicit attention for a single query token, to see where the model looks when it starts
answering. mx.fast.scaled_dot_product_attention does not expose weights, so the last token is run
through the layers by hand (same ops as the mlx-lm Qwen3/Llama blocks), appending its K/V to the
cache. Returns per-layer attention distributions (mean over heads) and the final logits; tests
check the logits against the model's own forward.
"""

import mlx.core as mx

from skillkv.blend import _qkv


def attention_profile(model, cache, token: int):
    """Feed `token` at the cache's current position. Returns (per-layer [N] head-mean attention
    probabilities over all N key positions incl. itself, logits [V])."""
    P = cache[0].offset
    h = model.model.embed_tokens(mx.array([[token]]))
    probs_per_layer = []
    for layer, c in zip(model.model.layers, cache):
        attn = layer.self_attn
        q, k, v = _qkv(attn, layer.input_layernorm(h))
        q, k = attn.rope(q, offset=P), attn.rope(k, offset=P)
        K, Vv = c.update_and_fetch(k, v)
        rep = attn.n_heads // attn.n_kv_heads
        if rep > 1:
            K, Vv = mx.repeat(K, rep, axis=1), mx.repeat(Vv, rep, axis=1)
        scores = (q.astype(mx.float32) @ K.astype(mx.float32).transpose(0, 1, 3, 2)) * attn.scale
        p = mx.softmax(scores, axis=-1)  # [1, H, 1, N]
        out = (p @ Vv.astype(mx.float32)).astype(h.dtype)
        out = attn.o_proj(out.transpose(0, 2, 1, 3).reshape(1, 1, -1))
        hh = h + out
        h = hh + layer.mlp(layer.post_attention_layernorm(hh))
        probs_per_layer.append(mx.mean(p[0, :, 0, :], axis=0))
        mx.eval(h, probs_per_layer[-1])
    h = model.model.norm(h)
    tied = getattr(getattr(model, "args", None), "tie_word_embeddings", False) or not hasattr(model, "lm_head")
    logits = model.model.embed_tokens.as_linear(h) if tied else model.lm_head(h)
    return probs_per_layer, logits[0, -1]


def span_mass(probs_per_layer, spans: dict[str, tuple[int, int]]) -> dict[str, list[float]]:
    """Attention mass per named [lo, hi) span, per layer."""
    return {name: [mx.sum(p[lo:hi]).item() for p in probs_per_layer] for name, (lo, hi) in spans.items()}
