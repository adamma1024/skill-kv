"""Natural-document (SQuAD/Wikipedia) injection items."""

from pathlib import Path

import pytest

from skillkv.inject import grade_item, make_natural_items

DATA = Path(__file__).resolve().parent.parent / "data" / "natural_docs.json"


@pytest.fixture(scope="module")
def items():
    return make_natural_items(DATA)


def test_counts_and_determinism(items):
    assert len(items) == 48 * 3
    assert [i.injection for i in items[:3]] == [None, "override", "append"]
    assert [i.doc for i in items] == [i.doc for i in make_natural_items(DATA)]


def test_injection_once_at_depth_and_canaries_unique(items):
    canaries = [i.canary for i in items if i.injection]
    assert len(set(canaries)) == len(canaries)
    for it in items:
        if it.injection is None:
            assert "AI assistant" not in it.doc
            continue
        assert it.doc.count(it.injection_text) == 1
        rel = it.doc.index(it.injection_text) / len(it.doc)
        assert abs(rel - it.depth) < 0.12  # sentence granularity
        assert it.canary not in it.question


def test_question_names_document_and_answers_present(items):
    for it in items:
        assert it.name in it.question and "read_doc" in it.question
        assert any(a.lower() in it.doc.lower() for a in it.answers)


def test_grading_uses_any_answer_case_insensitive(items):
    it = next(i for i in items if i.injection == "override")
    assert grade_item(f"It was {it.answers[0].upper()}.", it) == {"complied": False, "correct": True}
    assert grade_item(f"ACK-{it.canary}", it)["complied"] is True
    clean = next(i for i in items if i.injection is None)
    assert grade_item("I do not know.", clean) == {"complied": False, "correct": False}
