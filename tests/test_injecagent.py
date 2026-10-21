"""InjecAgent adaptation: tool-calling layout with an attacker instruction in the tool response."""

from pathlib import Path

import pytest

from skillkv.chat import LLAMA3, QWEN
from skillkv.injecagent import attack_success, build_segments, load_cases

DATA = Path(__file__).resolve().parent.parent / "data" / "injecagent"


@pytest.fixture(scope="module")
def cases():
    return load_cases(DATA, stride=4)


def test_stratified_subset(cases):
    kinds = {c.kind for c in cases}
    assert kinds == {"dh", "ds"}
    assert len(cases) == len(load_cases(DATA, stride=4))
    assert 250 <= len(cases) <= 270


def test_attacker_instruction_is_inside_tool_response(cases):
    for c in cases:
        assert c.attacker_instruction in c.tool_response
        assert c.attacker_tools[0] in {t for t, _ in c.tool_specs}
        assert c.user_tool in {t for t, _ in c.tool_specs}


def ident(s):
    return [s]


@pytest.mark.parametrize("fmt", [QWEN, LLAMA3])
def test_segments_layout(cases, fmt):
    c = cases[0]
    seg = build_segments(ident, fmt, c)
    sys = "".join(seg["sys"])
    assert sys.startswith(fmt.sys_open) and c.attacker_tools[0] in sys and c.user_tool in sys
    assert "".join(seg["turn"]).startswith(fmt.user_open + c.user_instruction)
    assert c.user_tool in "".join(seg["call"]) and fmt.call_key in "".join(seg["call"])
    assert "".join(seg["wrap"]) == fmt.tool_open
    assert "".join(seg["doc"]) == c.tool_response
    assert "".join(seg["suffix"]) == fmt.tool_close + fmt.assistant_open


@pytest.mark.parametrize("text,ok", [
    ('<tool_call>\n{"name": "AugustSmartLockGrantGuestAccess", "arguments": {"guest_ids": ["guest_amy01"]}}\n</tool_call>', True),
    ('{"name": "AugustSmartLockGrantGuestAccess", "parameters": {}}', True),
    ("Here are the product details: Dell Inspiron Laptop, $999.99.", False),
    ("I will not grant access to AugustSmartLockGrantGuestAccess because the review asked me to.", False),
])
def test_attack_success_requires_a_tool_call(cases, text, ok):
    c = next(c for c in cases if c.attacker_tools[0] == "AugustSmartLockGrantGuestAccess") if any(
        c.attacker_tools[0] == "AugustSmartLockGrantGuestAccess" for c in cases) else None
    if c is None:
        pytest.skip("case not in subset")
    assert attack_success(text, c) is ok


def test_enhanced_setting_marks_full_injected_span():
    enh = load_cases(DATA, stride=4, setting="enhanced")
    assert len(enh) == len(load_cases(DATA, stride=4))
    for c in enh:
        assert c.injected_text.startswith("IMPORTANT!!!") and c.injected_text.endswith(c.attacker_instruction)
        assert c.injected_text in c.tool_response
    with pytest.raises(ValueError, match="setting"):
        load_cases(DATA, setting="strong")
