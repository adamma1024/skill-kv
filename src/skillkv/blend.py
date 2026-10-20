"""Selective recompute of a mounted segment (CacheBlend-style) and the attention-mask control.

`blend` appends a token segment to a live cache layer by layer. At every layer the tokens in
the active set S get fresh Q/K/V computed in the real context; all other segment tokens keep
the pack's K/V (RoPE-shifted to the mount position). Fresh tokens attend to the prefix, to
pack K/V of non-recomputed tokens and to fresh K/V of recomputed ones (causal by position).

select:
  "all"               recompute every token at every layer (== text prefill)
  [i, ...]            fixed set S at every layer (first-k / last-k / explicit)
  ("random", r, seed) fixed random ceil(r*n) tokens
  ("deviation", r)    CacheBlend: recompute all tokens on layers 0-1, then keep the ceil(r*n)
                      tokens whose layer-1 K/V deviate most from the pack (one-shot selection;
                      CacheBlend filters gradually, see Yao et al. 2405.16444 §4)
exclude=(lo, hi): segment tokens may not attend to prefix positions [lo, hi). With select="all"
this is the T1' control: a segment that never saw the conversation, at its true positions.
"""

import math
import random

import mlx.core as mx

from skillkv.pack import Pack, PackMismatchError, rope_shift


def _rope_at(rope, x: mx.array, idx: list[int], n: int, offset: int) -> mx.array:
    """RoPE for rows `idx` of an n-token segment starting at absolute position `offset`."""
    if idx == list(range(n)):
        return rope(x, offset=offset)
    B, H, _, D = x.shape
    full = mx.zeros((B, H, n, D), dtype=x.dtype)
    full[:, :, mx.array(idx), :] = x
    return rope(full, offset=offset)[:, :, mx.array(idx), :]


def _qkv(attn, x):
    B, L, _ = x.shape
    q = attn.q_proj(x).reshape(B, L, attn.n_heads, -1)
    k = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)
    v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1)
    if hasattr(attn, "q_norm"):
        q, k = attn.q_norm(q), attn.k_norm(k)
    return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)


def blend(model, cache, segment: list[int], pack: Pack | None = None, model_id: str | None = None,
          select="all", exclude: tuple[int, int] | None = None) -> dict:
    n, P = len(segment), cache[0].offset
    layers = model.model.layers
    L = len(layers)
    if select != "all" and pack is None:
        raise ValueError("partial recompute needs a pack for the tokens that are not recomputed")
    if pack is not None:
        if model_id is not None and pack.model_id != model_id:
            raise PackMismatchError(f"pack was built for {pack.model_id!r}, serving model is {model_id!r}")
        if pack.tokens != list(segment):
            raise PackMismatchError("pack tokens differ from the segment being mounted")

    full_layers, fixed = 0, None
    if select == "all":
        full_layers = L
    elif isinstance(select, list):
        fixed = sorted(select)
    elif select[0] == "random":
        fixed = sorted(random.Random(select[2]).sample(range(n), math.ceil(select[1] * n)))
    elif select[0] == "deviation":
        full_layers = min(2, L)
    else:
        raise ValueError(f"unknown select {select!r}")
    if fixed is not None and any(not 0 <= i < n for i in fixed):
        raise ValueError(f"selected indices must be in [0, {n})")

    act = list(range(n)) if full_layers else fixed
    h = model.model.embed_tokens(mx.array(segment)[None])
    h = h if act == list(range(n)) else (h[:, mx.array(act), :] if act else h[:, :0, :])
    kpos = mx.arange(P + n)
    keep_out = None if exclude is None else (kpos >= exclude[0]) & (kpos < exclude[1])

    for l, layer in enumerate(layers):
        attn = layer.self_attn
        if pack is not None:
            pk, pv = pack.layers[l]
            kd, vd = rope_shift(attn.rope, pk, P - pack.start_pos), pv
        if act:
            q, k, v = _qkv(attn, layer.input_layernorm(h))
            q, k = _rope_at(attn.rope, q, act, n, P), _rope_at(attn.rope, k, act, n, P)
            if len(act) == n:
                kseg, vseg = k, v
            else:
                kseg, vseg = mx.array(kd), mx.array(vd)
                kseg[:, :, mx.array(act), :] = k
                vseg[:, :, mx.array(act), :] = v
            if P:
                pk_, pv_ = cache[l].keys_and_values()
                K, Vv = mx.concatenate([pk_, kseg], axis=2), mx.concatenate([pv_, vseg], axis=2)
            else:
                K, Vv = kseg, vseg
            qpos = P + mx.array(act)
            allowed = kpos[None, :] <= qpos[:, None]
            if keep_out is not None:
                allowed = allowed & ~keep_out[None, :]
            mask = mx.where(allowed, 0.0, -float("inf")).astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, K, Vv, scale=attn.scale, mask=mask)
            out = attn.o_proj(out.transpose(0, 2, 1, 3).reshape(1, len(act), -1))
            hh = h + out
            h = hh + layer.mlp(layer.post_attention_layernorm(hh))
        else:
            kseg, vseg = kd, vd
        cache[l].update_and_fetch(kseg, vseg)
        if select != "all" and isinstance(select, tuple) and select[0] == "deviation" and l == full_layers - 1:
            dev = (mx.sum((kseg - kd).astype(mx.float32) ** 2, axis=(0, 1, 3))
                   + mx.sum((vseg - vd).astype(mx.float32) ** 2, axis=(0, 1, 3)))
            m = math.ceil(select[1] * n)
            act = sorted(mx.argsort(-dev)[:m].tolist())
            h = h[:, mx.array(act), :] if act else h[:, :0, :]
        mx.eval(h, kseg, vseg)

    sel = list(range(n)) if select == "all" else act
    recomputed = full_layers * n + (L - full_layers) * len(sel)
    return {"n": n, "selected": len(sel), "indices": sel, "full_layers": full_layers,
            "recompute_fraction": recomputed / (L * n)}
