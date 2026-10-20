"""KV packs: precomputed per-layer K/V for a token segment, mounted into a live mlx-lm cache.

A pack stores K/V for `tokens`, computed after an optional `prelude` (so the segment's first
token is not an attention sink and has seen a fixed context). Cached keys in mlx-lm are
post-RoPE, so mounting at a new position re-rotates keys by the position delta; standard RoPE
composes additively (R(p + d) = R(d) R(p)). Cross-attention to whatever precedes the mount
point but is not in the prelude is missing: that is the approximation under study.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

FORMAT = "skillkv/1"


class PackMismatchError(ValueError):
    pass


@dataclass
class Pack:
    layers: list[tuple[mx.array, mx.array]]  # per layer (keys, values), [1, kv_heads, n, head_dim]
    tokens: list[int]
    start_pos: int
    prelude_hash: str
    model_id: str

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in self.layers)


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def prefill(model, cache, tokens: list[int], chunk: int = 2048, return_all: bool = False) -> mx.array:
    """Run `tokens` through the model, extending `cache`. Returns last-token logits [1, 1, V]
    (or all fed tokens' logits with return_all, for small tests)."""
    if not tokens:
        raise ValueError("prefill needs at least one token")
    outs = []
    for i in range(0, len(tokens), chunk):
        logits = model(mx.array(tokens[i : i + chunk])[None], cache=cache)
        mx.eval(logits)
        outs.append(logits if return_all else logits[:, -1:, :])
    return mx.concatenate(outs, axis=1) if return_all else outs[-1]


def _rope(model, layer: int):
    try:
        return model.layers[layer].self_attn.rope
    except AttributeError as e:
        raise NotImplementedError(
            f"{type(model).__name__} has no layers[{layer}].self_attn.rope; "
            "KV packs support standard-RoPE decoder models (Qwen3, Llama) only"
        ) from e


def rope_shift(rope, keys: mx.array, delta: int) -> mx.array:
    """Rotate already-RoPE'd keys [B, H, L, D] by a constant position delta.

    Each token is fed to the model's own RoPE as a length-1 sequence at `offset=delta`, which
    reuses the exact frequency schedule (incl. Llama-3 style scaling). Computed in float32.
    """
    if delta == 0:
        return keys
    B, H, L, D = keys.shape
    x = keys.astype(mx.float32).reshape(B, H, L, 1, D)
    return rope(x, offset=delta).reshape(B, H, L, D).astype(keys.dtype)


def build_pack(model, prelude: list[int], segment: list[int], model_id: str) -> Pack:
    cache = make_prompt_cache(model)
    bad = {type(c).__name__ for c in cache if not isinstance(c, KVCache)}
    if bad:
        raise NotImplementedError(f"KV packs need plain KVCache layers, model uses {sorted(bad)}")
    if prelude:
        prefill(model, cache, prelude)
    prefill(model, cache, segment)
    n0 = len(prelude)
    layers = []
    for c in cache:
        k, v = c.keys_and_values()
        layers.append((k[..., n0:, :], v[..., n0:, :]))
    mx.eval(layers)
    return Pack(layers, list(segment), n0, token_hash(list(prelude)), model_id)


def splice(cache, pack: Pack, model, model_id: str, seam: int = 0) -> None:
    """Append `pack` to `cache` at the cache's current position.

    With seam=k the last k pack tokens are not mounted; the caller re-prefills them in the real
    context (cheap repair of the boundary, as in "Models Take Notes at Prefill").
    """
    if pack.model_id != model_id:
        raise PackMismatchError(f"pack was built for {pack.model_id!r}, serving model is {model_id!r}")
    if len(pack.layers) != len(cache):
        raise PackMismatchError(f"pack has {len(pack.layers)} layers, cache has {len(cache)}")
    if not 0 <= seam < len(pack.tokens):
        raise ValueError(f"seam={seam} must be in [0, {len(pack.tokens)}) for a {len(pack.tokens)}-token pack")
    n = len(pack.tokens) - seam
    delta = cache[0].offset - pack.start_pos
    for i, (c, (k, v)) in enumerate(zip(cache, pack.layers)):
        c.update_and_fetch(rope_shift(_rope(model, i), k[..., :n, :], delta), v[..., :n, :])
    mx.eval([c.keys_and_values() for c in cache])


def save_pack(path, pack: Pack) -> None:
    arrays = {}
    for i, (k, v) in enumerate(pack.layers):
        arrays[f"{i}.k"], arrays[f"{i}.v"] = k, v
    meta = {
        "format": FORMAT, "model_id": pack.model_id, "start_pos": str(pack.start_pos),
        "prelude_hash": pack.prelude_hash, "tokens": json.dumps(pack.tokens),
    }
    mx.save_safetensors(str(path), arrays, meta)


def load_pack(path) -> Pack:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no KV pack at {path}; build it with skillkv.pack.build_pack (scripts/run_w1.py does this)")
    arrays, meta = mx.load(str(path), return_metadata=True)
    if meta.get("format") != FORMAT:
        raise PackMismatchError(f"{path} has format {meta.get('format')!r}, expected {FORMAT!r}")
    n_layers = len(arrays) // 2
    layers = [(arrays[f"{i}.k"], arrays[f"{i}.v"]) for i in range(n_layers)]
    return Pack(layers, json.loads(meta["tokens"]), int(meta["start_pos"]), meta["prelude_hash"],
                meta["model_id"])
