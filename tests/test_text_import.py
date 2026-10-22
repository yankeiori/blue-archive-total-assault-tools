"""テキスト貼り付け → カード生成の UI 配線 (app.frontend.callbacks.text_add_cards)。

解釈そのものは tests/test_ocr.py が見る。ここでは入力欄の値がカードに
届いているか — とくに備考の prefix が欠けていないか — だけを確認する。
"""
from app.frontend import callbacks

_PASTED = """\
ヒット1-2 (165.33%)
18,164 - 25,247
会心
35,239 - 48,979
"""


def _memos(children):
    """生成されたカード群から備考欄の値を取り出す。"""
    out = []
    for card in children:
        out.append(next(
            c.value for c in _walk(card)
            if isinstance(getattr(c, "id", None), dict)
            and c.id.get("type") == "memo"))
    return out


def _walk(node):
    if isinstance(node, (list, tuple)):
        for c in node:
            yield from _walk(c)
        return
    if not hasattr(node, "_prop_names"):
        return
    yield node
    yield from _walk(getattr(node, "children", None))


def _import(prefix):
    children, indices, next_idx, _order, status, _hp, _groups = \
        callbacks.text_add_cards(1, _PASTED, prefix, None, [], [], 0, None)
    assert indices == [0] and next_idx == 1, status
    return _memos(children)


def test_prefix_reaches_the_card_memo():
    assert _import("ミカ1射目") == ["ミカ1射目 ヒット1-2 攻撃力165.33%"]


def test_empty_prefix_leaves_the_memo_as_before():
    assert _import("") == ["ヒット1-2 攻撃力165.33%"]
    assert _import(None) == ["ヒット1-2 攻撃力165.33%"]


# ---------------------------------------------------------------------------
# 取り込んだまとまりの再追加 (text-groups)
# ---------------------------------------------------------------------------
_TWO_CARDS = _PASTED + """\
ヒット3 (1322.61%)
145,315 - 201,977
"""


def _params(children):
    out = []
    for card in children:
        out.append({c.id["param"]: c.value for c in _walk(card)
                    if isinstance(getattr(c, "id", None), dict)
                    and c.id.get("type") == "param"})
    return out


def _import_into(text, prefix, groups, children=None, crit=None):
    children = children or []
    n = len(children)
    out = callbacks.text_add_cards(1, text, prefix, crit, children,
                                   list(range(n)), n, groups)
    children, _indices, _next, _order, status, _hp, groups = out
    return children, groups, status


def test_import_registers_the_cards_under_the_prefix():
    _children, groups, _ = _import_into(_TWO_CARDS, "ミカEX", None)
    g, = groups["groups"]
    assert (g["name"], g["count"], len(g["cards"])) == ("ミカEX", 1, 2)
    # 名前抜きの本体で持つ (再追加のたびに #n を付け直すため)
    assert g["cards"][0]["memo"] == "ヒット1-2 攻撃力165.33%"
    assert groups["selected"] == "ミカEX"


def test_readding_numbers_the_memo_and_keeps_the_params():
    children, groups, _ = _import_into(_TWO_CARDS, "ミカEX", None)
    out = callbacks.text_group_add(1, "ミカEX", "", groups, children, [0, 1], 2)
    children, indices, next_idx, order, status, groups = out
    assert indices == [0, 1, 2, 3] and next_idx == 4 and order == indices
    assert _memos(children) == [
        "ミカEX ヒット1-2 攻撃力165.33%",
        "ミカEX ヒット3 攻撃力1322.61% 確定会心",
        "ミカEX #2 ヒット1-2 攻撃力165.33%",
        "ミカEX #2 ヒット3 攻撃力1322.61% 確定会心",
    ]
    p = _params(children)
    assert p[2] == p[0] and p[3] == p[1]
    # パーサが返さない項目はカードの既定値のまま (None で潰さない)
    assert p[3]["enemies"] == 1
    assert groups["groups"][0]["count"] == 2
    assert "#2" in status


def test_pasting_the_same_text_again_counts_as_another_shot():
    children, groups, _ = _import_into(_PASTED, "ミカEX", None)
    children, groups, _ = _import_into(_PASTED, "ミカEX", groups, children)
    assert _memos(children)[1] == "ミカEX #2 ヒット1-2 攻撃力165.33%"
    assert len(groups["groups"]) == 1


def test_different_text_under_the_same_name_replaces_it_but_keeps_counting():
    _children, groups, _ = _import_into(_PASTED, "ミカEX", None)
    children, groups, _ = _import_into(_TWO_CARDS, "ミカEX", groups)
    g, = groups["groups"]
    assert (len(g["cards"]), g["count"]) == (2, 2)
    assert _memos(children)[0] == "ミカEX #2 ヒット1-2 攻撃力165.33%"


def test_unnamed_imports_get_their_own_names():
    children, groups, _ = _import_into(_PASTED, "", None)
    assert _memos(children) == ["ヒット1-2 攻撃力165.33%"]
    _children, groups, _ = _import_into(_TWO_CARDS, "", groups)
    assert [g["name"] for g in groups["groups"]] == ["取り込み1", "取り込み2"]
    # 同じテキストをもう一度 → 既存の無名まとまりの 2 回目
    children, groups, _ = _import_into(_PASTED, None, groups)
    assert _memos(children) == ["取り込み1 #2 ヒット1-2 攻撃力165.33%"]
    assert len(groups["groups"]) == 2


def test_removing_a_group_leaves_the_cards():
    _children, groups, _ = _import_into(_PASTED, "ミカEX", None)
    _children, groups, _ = _import_into(_TWO_CARDS, "ホシノ", groups)
    groups, _status = callbacks.text_group_remove(1, "ミカEX", groups)
    assert [g["name"] for g in groups["groups"]] == ["ホシノ"]
    assert groups["selected"] == "ホシノ"
    options, value = callbacks.text_group_choices(groups)
    assert value == "ホシノ"
    assert options[0]["label"] == "ホシノ (2枚・3Hit・1回追加済み)"


def test_common_crit_rate_skips_guaranteed_crits_and_is_kept_for_readding():
    children, groups, _ = _import_into(_TWO_CARDS, "ミカEX", None, crit=75)
    assert [p["crit_rate"] for p in _params(children)] == [75, 100]
    out = callbacks.text_group_add(1, "ミカEX", "", groups, children, [0, 1], 2)
    assert [p["crit_rate"] for p in _params(out[0])] == [75, 100, 75, 100]


def test_empty_crit_rate_leaves_the_global_setting_in_charge():
    children, _groups, _ = _import_into(_PASTED, "ミカEX", None, crit=None)
    assert _params(children)[0]["crit_rate"] is None


def test_readding_with_a_different_prefix():
    children, groups, _ = _import_into(_PASTED, "ミカEX", None)
    out = callbacks.text_group_add(1, "ミカEX", " ミカ2射目 ", groups,
                                   children, [0], 1)
    children, groups = out[0], out[5]
    assert _memos(children)[1] == "ミカ2射目 ヒット1-2 攻撃力165.33%"
    # 回数は進むので、次に空欄で足すと #3
    out = callbacks.text_group_add(1, "ミカEX", "", groups, children, [0, 1], 2)
    assert _memos(out[0])[2] == "ミカEX #3 ヒット1-2 攻撃力165.33%"
