"""Chat-template pieces for the prompt layouts. Segments are tokenized separately and
concatenated, so a format is just the strings around each turn.

QWEN reproduces the exact W1-W3 prompts (Qwen3 ChatML, thinking off).
LLAMA3 follows the Llama 3.x instruct template; tool results use the `ipython` role and
tool calls are bare JSON with a "parameters" key, as in Meta's Llama 3.2 tool-calling format.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatFormat:
    name: str
    sys_open: str
    end: str  # closes a system / user / assistant / tool turn
    user_open: str
    assistant_open: str  # opens the assistant turn (incl. Qwen's empty think block)
    tool_open: str
    tool_close: str
    call_key: str
    call_pre: str
    call_post: str
    tool_result_phrase: str  # how the instructions refer to a returned tool result

    def call(self, tool: str, arg: str) -> str:
        return self.call_pre + json.dumps({"name": tool, self.call_key: {"name": arg}}) + self.call_post


QWEN = ChatFormat(
    name="qwen", sys_open="<|im_start|>system\n", end="<|im_end|>\n",
    user_open="<|im_start|>user\n", assistant_open="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    tool_open="<|im_start|>user\n<tool_response>\n", tool_close="\n</tool_response><|im_end|>\n",
    call_key="arguments", call_pre="<tool_call>\n", call_post="\n</tool_call>",
    tool_result_phrase="returned in a <tool_response>",
)

LLAMA3 = ChatFormat(
    name="llama3", sys_open="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    end="<|eot_id|>", user_open="<|start_header_id|>user<|end_header_id|>\n\n",
    assistant_open="<|start_header_id|>assistant<|end_header_id|>\n\n",
    tool_open="<|start_header_id|>ipython<|end_header_id|>\n\n", tool_close="<|eot_id|>",
    call_key="parameters", call_pre="", call_post="",
    tool_result_phrase="returned by the tool",
)


def detect_format(tokenizer) -> ChatFormat:
    template = getattr(tokenizer, "chat_template", None) or ""
    if "<|start_header_id|>" in template:
        return LLAMA3
    if "<|im_start|>" in template:
        return QWEN
    raise ValueError("unsupported chat template: expected Qwen ChatML (<|im_start|>) or "
                     "Llama 3 (<|start_header_id|>); add a ChatFormat in skillkv/chat.py")
