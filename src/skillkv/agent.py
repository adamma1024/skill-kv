"""Agent prompt layouts (Qwen3 ChatML, thinking off) and the post-selection step of C0-C3.

Lazy layouts start from a shared, cacheable system prefix (instructions + skill catalog):
  C1 lazy-text   [sys][user q, assistant load_skill call][<tool_response>][SKILL.md][</tool_response>, assistant]
  C3 KV-splice   same token ids as C1; SKILL.md comes from a pack precomputed after [sys][<tool_response>]
  C2 sys-prefix  [sys + "# Loaded skill" + SKILL.md][user q, assistant load_skill call][ack][assistant]:
                 the loaded skill joins a dynamic system prompt, so its pack is an exact prefix
                 (zero KV error), the prompt-caching layout; the skill now precedes the question
  C0 eager       [sys with every SKILL.md in full][user q, assistant]
Segments are tokenized separately and concatenated, so all conditions share token ids exactly.
"""

import re
import time
from typing import Callable

import mlx.core as mx

from skillkv.chat import QWEN, ChatFormat
from skillkv.pack import Pack, prefill, splice


def lazy_instructions(fmt: ChatFormat) -> str:
    return (
        "You are an agent that completes tasks using skills.\n"
        "To use a skill, load it by replying with exactly:\n"
        f"{fmt.call('load_skill', '<skill-name>')}\n"
        f"load_skill is the only tool. Once a skill's text has been {fmt.tool_result_phrase}, "
        "do not call any tool again: write the deliverable directly as plain text, following every "
        "instruction in the skill exactly.\n\n# Available skills\n"
    )


EAGER_INSTRUCTIONS = (
    "You are an agent that completes tasks using skills. The full text of every skill is below. "
    "Pick the skill that matches the task and follow every instruction in it exactly.\n\n# Skills\n\n"
)
LOADED_HEADER = "\n\n# Loaded skill\n"
LOADED_ACK = 'Skill loaded: its full text is in the system prompt under "# Loaded skill".'
_SELECTION = re.compile(r'"arguments"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"')
_TOOL_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')


def parse_selection(text: str, known: set[str] | None = None) -> str | None:
    """Skill named in a load_skill call. With `known`, also accept the common small-model slip
    of calling the skill itself as the tool ({"name": "<skill>", ...})."""
    m = _SELECTION.search(text)
    if m and (known is None or m.group(1) in known):
        return m.group(1)
    if known:
        for name in _TOOL_NAME.findall(text):
            if name in known:
                return name
    return m.group(1) if m else None


class Layout:
    def __init__(self, enc: Callable[[str], list[int]], skills: list[tuple[str, str, str]],
                 fmt: ChatFormat = QWEN, instructions: str | None = None, tool: str = "load_skill",
                 loaded_header: str = LOADED_HEADER, loaded_ack: str = LOADED_ACK,
                 generic_query: str = "Please help me with a task using the right skill."):
        """skills: (name, description, full text). The keyword arguments let the same layouts
        serve other single-tool agents (e.g. read_doc for the RQ7 injection test); loaded_ack is
        the tool-result message of the C2 layout."""
        self.enc, self.skills, self.fmt = enc, skills, fmt
        self.tool, self.GENERIC_QUERY = tool, generic_query
        self.loaded_ack = fmt.tool_open + loaded_ack + fmt.tool_close + fmt.assistant_open
        instructions = lazy_instructions(fmt) if instructions is None else instructions
        self._text = {n: t for n, _, t in skills}
        self._ids: dict[str, list[int]] = {}
        catalog = "\n".join(f"- {n}: {d}" for n, d, _ in skills)
        sys_open = f"{fmt.sys_open}{instructions}{catalog}"
        self.sys_lazy_text = sys_open + fmt.end
        self.sys_eager_text = (fmt.sys_open + EAGER_INSTRUCTIONS
                               + "\n\n".join(t for _, _, t in skills) + fmt.end)
        self.sys_lazy = enc(self.sys_lazy_text)
        self.wrap = enc(fmt.tool_open)
        self.suffix = enc(fmt.tool_close + fmt.assistant_open)
        self.pack_prelude = self.sys_lazy + self.wrap  # C3 packs
        self.c2_prelude = enc(sys_open) + enc(loaded_header)  # C2 packs
        self._sys_eager: list[int] | None = None

    @property
    def sys_eager(self) -> list[int]:
        if self._sys_eager is None:
            self._sys_eager = self.enc(self.sys_eager_text)
        return self._sys_eager

    def set_text(self, name: str, text: str) -> None:
        """Serve a different text under an existing catalog entry (system prefix unchanged)."""
        if name not in self._text:
            raise KeyError(f"unknown entry {name!r}")
        self._text[name] = text
        self._ids.pop(name, None)

    def skill(self, name: str) -> list[int]:
        if name not in self._text:
            raise KeyError(f"unknown skill {name!r}; library has {len(self._text)} skills")
        if name not in self._ids:
            self._ids[name] = self.enc(self._text[name])
        return self._ids[name]

    def user_turn(self, q: str) -> list[int]:
        return self.enc(f"{self.fmt.user_open}{q}{self.fmt.end}{self.fmt.assistant_open}")

    def tool_call(self, name: str) -> list[int]:
        return self.enc(self.fmt.call(self.tool, name) + self.fmt.end)

    def c2_tail(self, q: str, name: str) -> list[int]:
        return self.enc(self.fmt.end) + self.user_turn(q) + self.tool_call(name) + self.enc(self.loaded_ack)

    def generic_prelude(self, name: str) -> list[int]:
        """C1-shaped prefix with a content-free query: packs built after it have seen *a*
        question and the load_skill call, but not the real question (W2 ablation C3G)."""
        return self.sys_lazy + self.user_turn(self.GENERIC_QUERY) + self.tool_call(name) + self.wrap

    def c0_ids(self, q: str) -> list[int]:
        return self.sys_eager + self.user_turn(q)

    def c1_ids(self, q: str, name: str) -> list[int]:
        return self.sys_lazy + self.user_turn(q) + self.tool_call(name) + self.wrap + self.skill(name) + self.suffix

    def c2_ids(self, q: str, name: str) -> list[int]:
        return self.c2_prelude + self.skill(name) + self.c2_tail(q, name)


def _settle(cache) -> None:
    """Finish pending lazy cache writes so the timed region only measures post-selection work."""
    mx.eval([c.keys_and_values() for c in cache])
    mx.synchronize()


def post_selection(model, cache, layout: Layout, cond: str, q: str, name: str | None,
                   pack: Pack | Callable[[], Pack] | None = None, model_id: str | None = None):
    """Extend `cache` from its shared prefix (sys_lazy for C1/C3, c2_prelude for C2, sys_eager
    for C0) to the full answer
    context. Returns (first-answer-token logits, seconds). The timed part is what happens after
    the skill is chosen: prefill / pack load + splice, until the first answer token's logits."""
    get_pack = (lambda: pack) if isinstance(pack, Pack) else pack
    if cond == "C0":
        _settle(cache)
        t0 = time.perf_counter()
        logits = prefill(model, cache, layout.user_turn(q))
    elif cond in ("C1", "C3"):
        prefill(model, cache, layout.user_turn(q) + layout.tool_call(name) + layout.wrap)
        _settle(cache)
        t0 = time.perf_counter()
        if cond == "C1":
            logits = prefill(model, cache, layout.skill(name) + layout.suffix)
        else:
            splice(cache, get_pack(), model, model_id=model_id)
            logits = prefill(model, cache, layout.suffix)
    elif cond == "C2":
        _settle(cache)
        t0 = time.perf_counter()
        splice(cache, get_pack(), model, model_id=model_id)
        logits = prefill(model, cache, layout.c2_tail(q, name))
    else:
        raise ValueError(f"unknown condition {cond!r}; expected C0, C1, C2 or C3")
    mx.eval(mx.argmax(logits[0, -1]))
    return logits, time.perf_counter() - t0


def generate(model, cache, first_logits, max_tokens: int, eos: set[int],
             decode: Callable[[list[int]], str] | None = None, stop: str | None = None,
             sampler: Callable[[mx.array], mx.array] | None = None, seed: int | None = None) -> list[int]:
    """Decode from `first_logits`, extending `cache`. Greedy unless `sampler` (an mlx-lm sampler
    over log-probs) is given; `seed` fixes the random stream so conditions can share it."""
    if seed is not None:
        mx.random.seed(seed)

    def pick(logits):
        if sampler is None:
            return mx.argmax(logits[0, -1]).item()
        lp = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], axis=-1, keepdims=True)
        return sampler(lp).item()

    out, y = [], pick(first_logits)
    while len(out) < max_tokens and y not in eos:
        out.append(y)
        if stop and decode and stop in decode(out[-12:]):
            break
        y = pick(model(mx.array([[y]]), cache=cache))
    return out
