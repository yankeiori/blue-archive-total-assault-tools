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
    children, indices, next_idx, _order, status, _hp = callbacks.text_add_cards(
        1, _PASTED, prefix, [], [], 0)
    assert indices == [0] and next_idx == 1, status
    return _memos(children)


def test_prefix_reaches_the_card_memo():
    assert _import("ミカ1射目") == ["ミカ1射目 ヒット1-2 攻撃力165.33%"]


def test_empty_prefix_leaves_the_memo_as_before():
    assert _import("") == ["ヒット1-2 攻撃力165.33%"]
    assert _import(None) == ["ヒット1-2 攻撃力165.33%"]
