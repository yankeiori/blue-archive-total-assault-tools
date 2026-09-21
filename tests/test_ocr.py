"""ocr.parse_cards のレイアウト解釈テスト (Vision API 非依存)。

実画像の代わりに、Vision の textAnnotations を模した
{text,x,y,h} トークン列を合成して検証する。
"""

from app.backend import ocr


def _tokens(rows: list[str], *, row_h: int = 20, gap: int = 30) -> list[dict]:
    """行テキストのリストをトークン列に変換する。

    各行を語ごとに分割し、同一 y・x 昇順で配置する。
    """
    tokens: list[dict] = []
    for r, line in enumerate(rows):
        y = 50 + r * gap
        for w, word in enumerate(line.split()):
            tokens.append({"text": word, "x": 50 + w * 80, "y": y, "h": row_h})
    return tokens


def test_no_random_single_values():
    """乱数なし: 範囲記号が無く下限=上限。通常+会心。"""
    tokens = _tokens(
        [
            "攻撃力 % 1293.11% (7ヒット)",
            "ヒット1-7 (184.73%) 6,199",
            "会心 10,290",
            "平均ダメージ 60,046",
        ]
    )
    result = ocr.parse_cards(tokens)
    cards = result["cards"]
    assert len(cards) == 1
    p = cards[0]["params"]
    assert p["hits"] == 7
    assert p["normal_min"] == p["normal_max"] == 6199
    assert p["crit_min"] == p["crit_max"] == 10290
    assert "crit_rate" not in p  # 確定会心ではない
    assert result["hp_dependent"] is False


def test_general_two_groups():
    """一般: 通常レンジ+会心レンジ。ヒット1 と ヒット2-6 の2カード。"""
    tokens = _tokens(
        [
            "攻撃力 % 642.04% (6ヒット)",
            "ヒット1 (107.22%) 1,362 - 1,740",
            "会心 3,172 - 4,053",
            "ヒット2-6 (106.96%) 1,359 - 1,736",
            "会心 3,164 - 4,043",
            "平均ダメージ 16,057",
        ]
    )
    cards = ocr.parse_cards(tokens)["cards"]
    assert len(cards) == 2

    a = cards[0]["params"]
    assert a["hits"] == 1
    assert (a["normal_min"], a["normal_max"]) == (1362, 1740)
    assert (a["crit_min"], a["crit_max"]) == (3172, 4053)

    b = cards[1]["params"]
    assert b["hits"] == 5
    assert (b["normal_min"], b["normal_max"]) == (1359, 1736)
    assert (b["crit_min"], b["crit_max"]) == (3164, 4043)


def test_guaranteed_crit_hp_dependent():
    """確定会心+HP依存: 会心行が無くヒット行のみ。同値の連続ヒットを統合。"""
    rows = ["攻撃力 % 1540.36% (11ヒット)"]
    for i in range(1, 11):
        rows.append(f"ヒット{i} (77.02%) 7,235 - 9,286")
    rows.append("ヒット11 (770.18%) 72,350 - 92,860")
    rows.append("平均ダメージ 165,215")
    rows.append("現在HP 0%")

    result = ocr.parse_cards(_tokens(rows))
    cards = result["cards"]
    assert result["hp_dependent"] is True
    # ヒット1-10 (同値) が1枚に統合され、ヒット11 が別カード
    assert len(cards) == 2

    a = cards[0]["params"]
    assert a["hits"] == 10
    assert (a["crit_min"], a["crit_max"]) == (7235, 9286)
    assert a["crit_rate"] == 100
    assert "normal_min" not in a

    b = cards[1]["params"]
    assert b["hits"] == 1
    assert (b["crit_min"], b["crit_max"]) == (72350, 92860)
    assert b["crit_rate"] == 100


def test_row_grouping_tolerates_y_jitter():
    """同一行内のトークンに多少の y ばらつきがあっても1行に収まる。"""
    tokens = [
        {"text": "ヒット1", "x": 50, "y": 100, "h": 20},
        {"text": "(50.00%)", "x": 130, "y": 103, "h": 20},
        {"text": "1,000", "x": 300, "y": 98, "h": 20},
        {"text": "-", "x": 360, "y": 100, "h": 20},
        {"text": "2,000", "x": 380, "y": 101, "h": 20},
    ]
    cards = ocr.parse_cards(tokens)["cards"]
    assert len(cards) == 1
    p = cards[0]["params"]
    assert (p["crit_min"], p["crit_max"]) == (1000, 2000)


# ---------------------------------------------------------------------------
# テキスト貼り付け (ocr.parse_text)
# ---------------------------------------------------------------------------
_PASTED = """\
ヒット1-2 (165.33%)
18,164 - 25,247
会心
35,239 - 48,979
ヒット3-10 (82.66%)
9,082 - 12,623
会心
17,619 - 24,489
"""


def test_text_memo_keeps_the_hit_label():
    """prefix なしでは従来どおりヒットラベル(+攻撃力%)だけ。"""
    cards = ocr.parse_text(_PASTED)["cards"]
    assert [c["memo"] for c in cards] == [
        "ヒット1-2 攻撃力165.33%",
        "ヒット3-10 攻撃力82.66%",
    ]


def test_text_prefix_is_prepended_to_every_memo():
    """prefix は取り込んだ全カードの備考の頭に付く。"""
    cards = ocr.parse_text(_PASTED, "ミカ1射目")["cards"]
    assert [c["memo"] for c in cards] == [
        "ミカ1射目 ヒット1-2 攻撃力165.33%",
        "ミカ1射目 ヒット3-10 攻撃力82.66%",
    ]
    # ダメージ値は prefix の有無で変わらない
    assert cards[0]["params"]["crit_min"] == 35239


def test_text_prefix_is_trimmed_and_optional():
    """空白だけの prefix は付けない。前後の空白は落とす。"""
    assert ocr.parse_text(_PASTED, "   ")["cards"][0]["memo"] == "ヒット1-2 攻撃力165.33%"
    assert ocr.parse_text(_PASTED, None)["cards"][0]["memo"] == "ヒット1-2 攻撃力165.33%"
    assert ocr.parse_text(_PASTED, "  ミカ1射目 ")["cards"][0]["memo"].startswith(
        "ミカ1射目 ヒット1-2")


def test_text_prefix_survives_merging_consecutive_hits():
    """同値の連続ヒットを1枚に統合した後も prefix は1回だけ付く。"""
    text = "\n".join(f"ヒット{i} (10.00%)\n1,000 - 2,000" for i in range(1, 4))
    cards = ocr.cards_from_text(text, "ミカ2射目")["cards"]
    assert len(cards) == 1
    assert cards[0]["params"]["hits"] == 3
    assert cards[0]["memo"] == "ミカ2射目 ヒット1 攻撃力10.00% 確定会心"
