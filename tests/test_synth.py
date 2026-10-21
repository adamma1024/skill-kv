"""Synthetic verifiable skills: deterministic, constraints at controlled depths, regex-gradable."""

import re

import pytest

from skillkv.synth import CONSTRAINT_TYPES, DEPTHS, grade, make_library, make_tasks


@pytest.fixture(scope="module")
def lib():
    return make_library(n=12, seed=0, min_words=400, max_words=1600)


def test_library_is_deterministic(lib):
    again = make_library(n=12, seed=0, min_words=400, max_words=1600)
    assert [s.text for s in lib] == [s.text for s in again]
    assert [s.text for s in make_library(n=12, seed=1, min_words=400, max_words=1600)] != [s.text for s in lib]


def test_names_unique_and_frontmatter(lib):
    assert len({s.name for s in lib}) == len(lib)
    for s in lib:
        assert s.text.startswith(f"---\nname: {s.name}\ndescription: {s.description}\n---\n")


def test_every_constraint_type_once_per_skill(lib):
    for s in lib:
        assert sorted(c.type for c in s.constraints) == sorted(CONSTRAINT_TYPES)
        for c in s.constraints:
            assert s.text.count(c.statement) == 1


def test_constraints_sit_at_their_target_depth(lib):
    for s in lib:
        body_start = s.text.index("\n---\n") + 5
        body_len = len(s.text) - body_start
        for c in s.constraints:
            rel = (s.text.index(c.statement) - body_start) / body_len
            assert abs(rel - c.depth) < 0.12, (s.name, c.type, rel, c.depth)


def test_depths_balanced_across_types(lib):
    # each (type, depth) cell is used: types rotate over depth bins across skills
    cells = {(c.type, c.depth) for s in lib for c in s.constraints}
    assert len(cells) == len(CONSTRAINT_TYPES) * len(DEPTHS)


def test_length_spread(lib):
    words = sorted(len(s.text.split()) for s in lib)
    assert words[0] < 700 and words[-1] > 1300


def test_constraint_values_unique_across_library(lib):
    for t in CONSTRAINT_TYPES:
        vals = [c.value for s in lib for c in s.constraints if c.type == t]
        assert len(set(vals)) == len(vals), t


def test_task_mentions_domain_not_skill_name_or_secrets(lib):
    for s in lib:
        for task in make_tasks(s, k=2, seed=0):
            assert s.name not in task.query
            assert s.object_noun in task.query.lower() and s.action_phrase in task.query.lower()
            for c in s.constraints:
                if c.type != "param":
                    assert c.value not in task.query
            assert next(c.key for c in s.constraints if c.type == "param") in task.query


def test_grade_full_marks_for_compliant_answer(lib):
    s = lib[0]
    v = {c.type: c for c in s.constraints}
    answer = (f"CODE: {v['code'].value}\n### {v['heading'].value}\nI. Collect inputs\nII. Run it\n"
              f"Output file: {v['filename'].value}_monthly_run.csv\n"
              f"{v['param'].key} = {v['param'].value}\n-- {v['signoff'].value} --")
    g = grade(answer, s)
    assert all(g[t] for t in CONSTRAINT_TYPES), g


def test_grade_rejects_values_from_other_skills(lib):
    a, b = lib[0], lib[1]
    v = {c.type: c for c in b.constraints}
    answer = (f"CODE: {v['code'].value}\n### {v['heading'].value}\n"
              f"{v['filename'].value}_x.csv\n{v['param'].value}\n-- {v['signoff'].value} --")
    assert not any(grade(answer, a).values())


def test_grade_is_case_and_spacing_tolerant(lib):
    s = lib[0]
    v = {c.type: c for c in s.constraints}
    answer = f"code:   {v['code'].value}\n## {v['heading'].value.upper()}\n--{v['signoff'].value}--"
    g = grade(answer, s)
    assert g["code"] and g["heading"] and g["signoff"]


def test_large_library_names_unique():
    lib = make_library(n=100, seed=0, min_words=50, max_words=80)
    assert len({s.name for s in lib}) == 100
    assert re.fullmatch(r"[a-z0-9-]+", lib[0].name)


def test_mention_all_query_names_every_item_but_no_secret_values(lib):
    for s in lib:
        base = make_tasks(s, k=2, seed=0)
        full = make_tasks(s, k=2, seed=0, mention_all=True)
        for b, f in zip(base, full):
            assert f.query.startswith(b.query.split(" Use the right skill")[0])  # same domain + context
            for word in ("activation code", "heading", "sign-off", "file name"):
                assert word in f.query and (word == "file name" or word not in b.query)
            for c in s.constraints:
                if c.type != "param":
                    assert c.value not in f.query


def test_perm_test_and_holm():
    from skillkv.stats import holm, perm_test
    null = [(c, (1 if (c + i) % 2 else -1) * 0.1) for c in range(20) for i in range(4)]
    assert perm_test(null, n=2000) > 0.2
    shifted = [(c, 0.3 + (0.05 if i % 2 else -0.05)) for c in range(20) for i in range(4)]
    assert perm_test(shifted, n=2000) < 0.01
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
