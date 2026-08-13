# -*- coding: utf-8 -*-
"""スキル順探索 (app.backend.skill_order) のテスト。"""

import random
from collections import Counter
from itertools import permutations

import pytest

from app.backend.skill_order import (
    HAND_SIZE_DECISIVE,
    default_node_budget,
    SearchBudgetExceeded,
    Step,
    different_slots,
    distinct_layouts,
    solve,
    total_layouts,
    trace_entry_label,
)

NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


# ---------------------------------------------------------------------------
# 参照実装: 初期配置を全列挙して素直にシミュレートする(遅いが単純)
# ---------------------------------------------------------------------------
def _ref_owner(card):
    return card[1] if card is not None else None


def _ref_solve(n, copiers, plan, hand_size=3):
    """(初期配置, 使用スロット列) の一覧を全順列から求める。"""
    out = []

    def dfs(hand, deck, i, layout, slots):
        if i == len(plan):
            out.append((layout, tuple(slots)))
            return
        step = plan[i]

        if step.retreat:
            for j, c in enumerate(hand):
                if _ref_owner(c) == step.skill:
                    nh, nd = list(hand), list(deck)
                    nh[j] = nd.pop(0) if nd else None
                    slots.append(j + 1)
                    dfs(nh, nd, i + 1, layout, slots)
                    slots.pop()
                    return
            for j, c in enumerate(deck):
                if _ref_owner(c) == step.skill:
                    nd = list(deck)
                    nd.pop(j)
                    slots.append(0)
                    dfs(list(hand), nd, i + 1, layout, slots)
                    slots.pop()
                    return
            return                                   # 場に無い → 不成立

        cand = ([step.slot - 1] if step.slot is not None
                else range(hand_size))
        for idx in cand:
            card = hand[idx]
            if card is None:
                continue
            if step.skill is None:
                if card[0] == "N" and card[1] in copiers:
                    continue
            elif step.use_copy:
                if not (card[0] == "C" and card[2] == step.skill):
                    continue
            else:
                if not (card[0] == "N" and card[1] == step.skill):
                    continue

            nh, nd = list(hand), list(deck)
            if card[0] == "N" and card[1] in copiers:
                nh[idx] = ("C", card[1], step.copy_target)
            else:
                if card[0] == "C":
                    nd.append(("N", card[1]))
                    origin = ("N", card[2])
                else:
                    nd.append(card)
                    origin = card
                nh[idx] = None
                if step.draw and origin in nd:
                    nd.remove(origin)
                    nh[idx] = origin
                else:
                    nh[idx] = nd.pop(0) if nd else None
            slots.append(idx + 1)
            dfs(nh, nd, i + 1, layout, slots)
            slots.pop()

    for layout in permutations(range(n)):
        hand = [("N", i) for i in layout[:hand_size]]
        hand += [None] * max(0, hand_size - n)
        deck = [("N", i) for i in layout[hand_size:]]
        dfs(hand, deck, 0, layout, [])
    return out


def _actual(n, copiers, plan, hand_size=3):
    res, _ = solve(n, copiers, plan, hand_size=hand_size)
    got = Counter()
    for sol in res:
        seq = tuple(e[0] for e in sol.trace)
        expanded = sol.expand()
        assert len(expanded) == sol.count      # 通り数と展開数が一致すること
        for lay in expanded:
            got[(lay, seq)] += 1
    return got


def _expected(n, copiers, plan, hand_size=3):
    return Counter(_ref_solve(n, copiers, plan, hand_size))


# ---------------------------------------------------------------------------
# 基本動作
# ---------------------------------------------------------------------------
def test_basic_plan_has_solutions():
    plan = [Step(0), Step(1), Step(2), Step(3), Step(0, slot=1)]
    res, truncated = solve(6, set(), plan)
    assert res and not truncated


def test_copy_transform_and_use():
    # A(=0, 複製) が B(=1) を複製 → B(コピー) を使用
    plan = [Step(0, copy_target=1), Step(1, use_copy=True)]
    res, _ = solve(6, {0}, plan)
    assert res
    layout, trace = res[0]
    # 変化はその場で起きるので、コピー使用は同じスロット
    assert trace[0][0] == trace[1][0]
    assert trace[0][2] == "transform"
    assert trace[1][1] == ("C", 0, 1)
    labels = [trace_entry_label(e, NAMES) for e in trace]
    assert labels[0].endswith("A→B(コピー)")
    assert labels[1].endswith("B(コピー)")


def test_draw_flag_returns_card_to_hand():
    # ドローフラグ付きで使うと自分のカードが手札に戻るため連打できる
    res, _ = solve(6, set(), [Step(1, draw=True), Step(1)])
    assert res
    for _, trace in res:
        assert trace[0][0] == trace[1][0]  # 同じスロットで連打

    # フラグ無しでは同じカードは2連打できない
    res_nodraw, _ = solve(6, set(), [Step(1), Step(1)])
    assert not res_nodraw


def test_draw_flag_on_copy_pulls_original_from_deck():
    # B(=1, 複製) が A(=0) を複製 → A(コピー) をドロー付きで使用 →
    # B に戻って山札の底へ行き、山札に A があれば同じスロットへ引き抜かれる
    plan = [
        Step(1, copy_target=0),
        Step(0, use_copy=True, draw=True),
        Step(0),
    ]
    res, _ = solve(6, {1}, plan)
    assert res
    checked = False
    for sol in res:
        for layout in sol.expand():
            if 0 in layout[3:]:  # 元の A が初期山札にいる配置
                checked = True
                assert sol.trace[1][0] == sol.trace[2][0]
    assert checked


def test_constraints():
    res, _ = solve(6, set(), [Step(None), Step(None)],
                   [different_slots(0, 1)])
    assert res
    assert all(t[0][0] != t[1][0] for _, t in res)


def test_wildcard_skips_copier_original():
    res, _ = solve(6, {0}, [Step(None)])
    assert res
    for sol in res:
        used_pos = sol.trace[0][1]
        for layout in sol.expand():
            # ワイルドカードで使われた位置に複製キャラが来てはいけない
            assert layout[used_pos[1]] != 0


def test_max_results_truncation():
    res, truncated = solve(6, set(), [Step(None)], max_results=2)
    assert truncated and len(res) == 2


def test_stats_reports_reached_depth():
    # 3手目で成立しなくなる手順 (B は山札を1周しないと戻らない)
    plan = [Step(1), Step(2), Step(1)]
    stats = {}
    res, _ = solve(6, set(), plan, stats=stats)
    assert not res
    assert stats["max_depth"] == 2          # 2手目までは到達できた

    stats = {}
    res, _ = solve(6, set(), [Step(1), Step(2)], stats=stats)
    assert res and stats["max_depth"] == 2  # 最後まで到達


def test_default_node_budget_scales_with_plan_length():
    assert default_node_budget([Step(0)] * 3) == 2_000_000
    assert default_node_budget([Step(0)] * 98) == 98 * 200_000


def test_long_plan_is_not_cut_off_by_the_budget():
    # 手順が長いだけで打ち切られないこと (既定予算は手順長に比例)
    plan = [Step(i % 6) for i in range(60)]
    with pytest.raises(SearchBudgetExceeded):
        solve(6, set(), plan, node_budget=1000)   # 明示指定はそのまま効く
    res, _ = solve(6, set(), plan)
    assert res


def test_node_budget():
    plan = [Step(None)] * 12
    with pytest.raises(SearchBudgetExceeded):
        solve(6, set(), plan, node_budget=1000)


# ---------------------------------------------------------------------------
# 撤退
# ---------------------------------------------------------------------------
def test_retreat_removes_card_permanently():
    # C(=2) が撤退した後に C は使えない
    res, _ = solve(6, set(), [Step(2, retreat=True), Step(2)])
    assert not res


def test_retreat_from_hand_draws_replacement():
    # 手札の A(=0) が撤退 → そのスロットに山札トップが来るので直後に使える
    plan = [Step(0, retreat=True, slot=None), Step(1)]
    res, _ = solve(6, set(), plan)
    assert res
    for sol in res:
        r_slot, _, action, _ = sol.trace[0]
        assert action == "retreat"
        if r_slot:                       # 手札から撤退した場合
            # 補充されたカードをそのスロットで使えている解が存在する
            assert sol.trace[1][0] >= 1


def test_retreat_label():
    res, _ = solve(6, set(), [Step(3, retreat=True)])
    assert res
    labels = {trace_entry_label(sol.trace[0], NAMES) for sol in res}
    assert "[山札]D撤退" in labels
    assert "[左]D撤退" in labels


def test_retreat_of_copier_removes_copy_card():
    # A(=0, 複製) が B(=1) を複製 → A が撤退 → B(コピー) も消えるので使えない
    plan = [Step(0, copy_target=1), Step(0, retreat=True),
            Step(1, use_copy=True)]
    res, _ = solve(6, {0}, plan)
    assert not res


def test_retreat_shrinks_deck_to_empty_slot():
    # 2枚しかない状態で1枚撤退 → 山札が空なのでスロットは空欄のまま
    plan = [Step(0, retreat=True), Step(1), Step(1)]
    res, _ = solve(2, set(), plan, hand_size=3)
    assert res
    # 残った B は山札を経由して戻ってくるので2回使える
    for sol in res:
        assert sol.trace[1][0] == sol.trace[2][0]


# ---------------------------------------------------------------------------
# 枚数バリエーション(1〜5枚 / 決戦モード)
# ---------------------------------------------------------------------------
def test_single_card_cycles():
    res, _ = solve(1, set(), [Step(0), Step(0), Step(0)], hand_size=3)
    assert res
    for sol in res:
        assert {e[0] for e in sol.trace} == {1}     # 常に左スロット
        assert sol.layout == (0,)


def test_fewer_cards_than_hand_leaves_empty_slots():
    # 2枚なら手札は左・中のみ。右スロットは常に空欄で使えない
    res, _ = solve(2, set(), [Step(0), Step(1)], hand_size=3)
    assert res
    assert all(e[0] in (1, 2) for sol in res for e in sol.trace)


def test_decisive_mode_uses_five_slots():
    plan = [Step(i) for i in range(5)]
    res, _ = solve(10, set(), plan, hand_size=HAND_SIZE_DECISIVE)
    assert res
    assert {e[0] for sol in res for e in sol.trace} == {1, 2, 3, 4, 5}
    # 手順に出てこない5枚は任意 → 1解あたり 5! = 120 通り
    assert all(sol.count == 120 for sol in res)


def test_decisive_mode_label_uses_five_slot_names():
    res, _ = solve(10, set(), [Step(0, slot=4)], hand_size=HAND_SIZE_DECISIVE)
    assert res
    assert trace_entry_label(res[0].trace[0], NAMES,
                             hand_size=HAND_SIZE_DECISIVE) == "[4番目]A"


def test_total_layouts():
    res, _ = solve(6, set(), [Step(0, slot=1)])
    assert len(res) == 1
    assert total_layouts(res) == 120        # 残り5枚は任意
    assert distinct_layouts(res) == (120, True)


def test_distinct_layouts_dedupes_across_traces():
    # ワイルドカードは同じ初期配置から複数の使用順が生まれるため、
    # 延べ数(total_layouts)と初期配置の種類数は一致しない
    res, _ = solve(6, set(), [Step(None), Step(None)])
    assert total_layouts(res) == 6480
    assert distinct_layouts(res) == (720, True)     # = 6! (全配置が条件を満たす)


def test_distinct_layouts_cap():
    res, _ = solve(6, set(), [Step(None)])
    assert distinct_layouts(res, cap=10) == (10, False)


# ---------------------------------------------------------------------------
# 参照実装との突き合わせ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,hand_size,copiers,plan", [
    (6, 3, set(), [Step(0), Step(1), Step(2), Step(0)]),
    (6, 3, set(), [Step(1, draw=True), Step(1), Step(2, draw=True), Step(0)]),
    (6, 3, set(), [Step(0, slot=1), Step(3), Step(0, slot=1), Step(2)]),
    (6, 3, {0}, [Step(0, copy_target=1), Step(1, use_copy=True), Step(2)]),
    (6, 3, {0}, [Step(0, copy_target=2), Step(2, use_copy=True, draw=True),
                 Step(2), Step(1)]),
    (6, 3, set(), [Step(None), Step(None), Step(3)]),
    (6, 3, {4}, [Step(None), Step(4, copy_target=0), Step(0, use_copy=True)]),
    (6, 3, set(), [Step(0, retreat=True), Step(1), Step(2)]),
    (6, 3, set(), [Step(1), Step(1, retreat=True), Step(None), Step(2)]),
    (6, 3, {0}, [Step(0, copy_target=1), Step(2, retreat=True),
                 Step(1, use_copy=True)]),
    (5, 3, set(), [Step(0), Step(1, retreat=True), Step(2), Step(0)]),
    (3, 3, set(), [Step(0), Step(1, retreat=True), Step(2), Step(0)]),
    (2, 3, set(), [Step(0), Step(1), Step(0)]),
    (7, 5, set(), [Step(0), Step(1), Step(6), Step(0, draw=True)]),
    (7, 5, {2}, [Step(2, copy_target=0), Step(0, use_copy=True, draw=True),
                 Step(3, retreat=True), Step(1)]),
])
def test_matches_reference(n, hand_size, copiers, plan):
    assert _actual(n, copiers, plan, hand_size) == \
        _expected(n, copiers, plan, hand_size)


def test_matches_reference_random():
    """ランダム生成した手順で参照実装と一致することを確認する。"""
    rng = random.Random(20260813)
    for _ in range(120):
        n = rng.choice([4, 5, 6, 6, 6])
        copiers = set(rng.sample(range(n), rng.choice([0, 0, 1])))
        plan = []
        for _ in range(rng.randint(2, 6)):
            r = rng.random()
            slot = rng.choice([None, None, None, 1, 2, 3])
            draw = rng.random() < 0.3
            if r < 0.15:
                plan.append(Step(None, slot=slot, draw=draw))
            elif r < 0.3:
                plan.append(Step(rng.randrange(n), retreat=True))
            elif r < 0.45 and copiers:
                tgt = rng.choice([x for x in range(n) if x not in copiers])
                plan.append(Step(tgt, use_copy=True, slot=slot, draw=draw))
            else:
                sk = rng.randrange(n)
                tgt = None
                if sk in copiers:
                    cand = [x for x in range(n) if x not in copiers]
                    tgt = rng.choice(cand)
                plan.append(Step(sk, copy_target=tgt, slot=slot, draw=draw))
        assert _actual(n, copiers, plan) == _expected(n, copiers, plan), plan
