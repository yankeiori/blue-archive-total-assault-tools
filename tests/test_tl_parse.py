# -*- coding: utf-8 -*-
"""TLテキストパーサ (app.backend.tl_parse) のテスト。"""

from app.backend import skill_order
from app.backend.tl_parse import (
    KIND_COPY,
    aliases,
    display_name,
    KIND_RETREAT,
    KIND_USE,
    parse_timeline,
)

NAMES = ["リオ", "マリー", "ホシノ", "ドアル", "ハルカ", "キサキ"]
RIO = 0


def kinds(steps):
    return [(s.kind, NAMES[s.skill]) for s in steps]


def test_strips_timing_annotations():
    text = """
    即 マリー
    11 ドアル
    10.5 ホシノ
    3:03.000 キサキ
    オート ハルカ
    コストカット後 マリー
    """
    steps, warns = parse_timeline(text, NAMES)
    assert [n for _, n in kinds(steps)] == [
        "マリー", "ドアル", "ホシノ", "キサキ", "ハルカ", "マリー"]
    assert not warns


def test_comments_are_ignored():
    text = ("2:22.500 マリー イロハ 【254】イロハで255台。目標は254\n"
            "即 ホシノ //0:54.000くらい\n"
            "11 ドアル /キサキが落ちてたら飛ばす\n"
            "10 キサキ ※お祈り\n"
            "目標 2.4億付近\n")
    steps, _ = parse_timeline(text, NAMES)
    # 【 / // / / / ※ 以降は注釈。目標行も無視される
    assert [n for _, n in kinds(steps)] == [
        "マリー", "ホシノ", "ドアル", "キサキ"]


def test_ns_annotation_is_not_a_card_use():
    steps, _ = parse_timeline("マリーNS後 ホシノ\nns後（10c) キサキ", NAMES)
    assert [n for _, n in kinds(steps)] == ["ホシノ", "キサキ"]


def test_copy_notation():
    steps, _ = parse_timeline("cマリー Cホシノ ハルカ(コピー)", NAMES)
    assert kinds(steps) == [(KIND_COPY, "マリー"), (KIND_COPY, "ホシノ"),
                            (KIND_COPY, "ハルカ")]


def test_copier_target():
    steps, warns = parse_timeline("即 リオ（マリー） cマリー", NAMES, {RIO})
    assert steps[0].kind == KIND_USE and steps[0].skill == RIO
    assert steps[0].target == NAMES.index("マリー")
    assert steps[1].kind == KIND_COPY
    assert not warns


def test_copier_target_inferred_from_later_copy():
    steps, warns = parse_timeline("即 リオ\n11 ドアル\n即 cハルカ", NAMES, {RIO})
    assert steps[0].target == NAMES.index("ハルカ")
    assert any("複製対象が書かれていない" in w for w in warns)


def test_copier_target_unresolvable_warns():
    steps, warns = parse_timeline("即 リオ\n11 ドアル", NAMES, {RIO})
    assert steps[0].target is None
    assert any("判別できません" in w for w in warns)


def test_non_student_target_becomes_memo():
    steps, _ = parse_timeline("2:33.500 マリー(緑壺)", NAMES)
    assert steps[0].memo == "緑壺"
    assert steps[0].target is None


def test_student_target_of_non_copier_becomes_memo():
    # マリーは複製キャラではないので、対象は探索に影響しない注記として残す
    steps, _ = parse_timeline("即 マリー（ホシノ）", NAMES)
    assert steps[0].memo == "ホシノ"
    assert steps[0].target is None


def test_retreat():
    steps, _ = parse_timeline("11 ドアル\n即 ハルカ撤退\n10 キサキ", NAMES)
    assert kinds(steps) == [(KIND_USE, "ドアル"), (KIND_RETREAT, "ハルカ"),
                            (KIND_USE, "キサキ")]


def test_concatenated_names_are_split():
    steps, warns = parse_timeline("2:40.700 マリーホシノドアル", NAMES)
    assert [n for _, n in kinds(steps)] == ["マリー", "ホシノ", "ドアル"]
    assert not warns


def test_unknown_word_warns():
    steps, warns = parse_timeline("11 cハルカ アル キサキ", NAMES)
    assert [n for _, n in kinds(steps)] == ["ハルカ", "キサキ"]
    assert any("アル" in w for w in warns)


def test_draw_is_inferred_for_repeated_card():
    steps, warns = parse_timeline("即 ハルカ\nハルカNS後 ハルカ", NAMES)
    assert steps[0].draw and not steps[1].draw
    assert any("ドロー" in w for w in warns)
    # 実際にドロー無しでは成立しないことを確認
    plan = [skill_order.Step(steps[0].skill), skill_order.Step(steps[1].skill)]
    assert not skill_order.solve(6, set(), plan)[0]


def test_draw_after_copy_use():
    steps, _ = parse_timeline("即 cハルカ ハルカ", NAMES, {RIO})
    assert steps[0].kind == KIND_COPY and steps[0].draw


def test_multiple_timelines_warn():
    text = "即 マリー\n目標 2.4億付近\n10.5 ホシノ\n"
    _, warns = parse_timeline(text, NAMES)
    assert any("1回の戦闘ぶんずつ" in w for w in warns)


def test_aliases():
    names = ["リオ", "マリー", "ホシノ", "ドアル/アル", "ハルカ", "キサキ"]
    steps, warns = parse_timeline("11 cハルカ アル キサキ\n10 ドアル", names)
    assert [s.skill for s in steps] == [4, 3, 5, 3]
    assert not warns
    assert display_name("ドアル/アル") == "ドアル"
    assert aliases("ドアル/アル") == ["ドアル", "アル"]
    assert display_name("") == ""


def test_no_names_warns():
    steps, warns = parse_timeline("即 マリー", ["", "", ""])
    assert not steps and warns


def test_parsed_plan_is_solvable():
    """読み取った手順がそのまま探索に通ること。"""
    text = """
    即 リオ（マリー） cマリー ホシノ
    11 ドアル
    10.5 マリー
    即 リオ（ハルカ）
    10 ハルカ キサキ cハルカ
    """
    steps, warns = parse_timeline(text, NAMES, {RIO})
    assert not warns
    plan = []
    for st in steps:
        if st.kind == KIND_COPY:
            plan.append(skill_order.Step(st.skill, use_copy=True, draw=st.draw))
        elif st.kind == KIND_RETREAT:
            plan.append(skill_order.Step(st.skill, retreat=True))
        else:
            plan.append(skill_order.Step(st.skill, copy_target=st.target,
                                         draw=st.draw))
    res, _ = skill_order.solve(6, {RIO}, plan)
    assert res
