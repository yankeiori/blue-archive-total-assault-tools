"""スクリーンショット OCR → ダメージカード自動生成。

Google Cloud Vision API (DOCUMENT_TEXT_DETECTION, REST + API キー) で
画像からテキスト+座標を取得し、座標ベースで行を組み立てて
「ヒット / 通常 / 会心 / 範囲」を解釈し、make_damage_card 用の
パラメータ辞書リストへ変換する。

元ツールが複数/不定なのでレイアウト座標に依存せず、テキストパターンと
行の y 座標近接だけでグルーピングする方針。

対応する表示パターン:
  - 乱数なし   : 数値に範囲記号(-, ~ 等)が無い → 下限=上限
  - 一般       : ヒット行(通常レンジ) + 直下の「会心」行(会心レンジ)
  - 確定会心   : ヒット行のみ(会心行が続かない) → 会心側に格納し会心率=100
  - HP依存     : 「現在HP」等を検出 → メタ情報として返す(全体設定で扱う)
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import requests

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
_API_KEY_ENV = "GOOGLE_VISION_API_KEY"

# プロジェクトルート (.env 読み込み用)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OcrError(RuntimeError):
    """OCR 処理中のユーザー向けエラー (API キー未設定・API 失敗など)。"""


# ---------------------------------------------------------------------------
# API キー
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    """環境変数、無ければプロジェクトルートの .env から API キーを取得する。"""
    key = os.environ.get(_API_KEY_ENV)
    if key:
        return key.strip()

    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == _API_KEY_ENV:
                return value.strip().strip('"').strip("'")

    raise OcrError(
        f"環境変数 {_API_KEY_ENV} が未設定です。"
        "Google Vision API キーを .env もしくは環境変数に設定してください。"
    )


# ---------------------------------------------------------------------------
# Vision API 呼び出し
# ---------------------------------------------------------------------------
def _strip_data_url(content: str) -> str:
    """data URL (data:image/png;base64,xxxx) なら base64 本体だけ返す。"""
    if content.startswith("data:"):
        _, _, b64 = content.partition(",")
        return b64
    return content


def vision_annotate(image: str | bytes, *, api_key: str | None = None) -> list[dict]:
    """画像を Vision API に投げ、textAnnotations[1:] (単語+座標) を返す。

    image: data URL 文字列 / base64 文字列 / 生バイト列のいずれか。
    戻り値: [{"text": str, "x": int, "y": int, "h": int}, ...] (単語単位)。
    """
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode("ascii")
    else:
        b64 = _strip_data_url(image)

    key = api_key or _load_api_key()
    body = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                # 日本語+英数字を優先
                "imageContext": {"languageHints": ["ja", "en"]},
            }
        ]
    }
    try:
        resp = requests.post(
            VISION_ENDPOINT, params={"key": key}, json=body, timeout=30
        )
    except requests.RequestException as exc:  # ネットワーク障害等
        raise OcrError(f"Vision API への接続に失敗しました: {exc}") from exc

    if resp.status_code != 200:
        raise OcrError(
            f"Vision API がエラーを返しました (HTTP {resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json()
    responses = data.get("responses", [{}])
    first = responses[0] if responses else {}
    if "error" in first:
        raise OcrError(f"Vision API エラー: {first['error'].get('message', first['error'])}")

    annotations = first.get("textAnnotations", [])
    return _annotations_to_tokens(annotations)


def _annotations_to_tokens(annotations: list[dict]) -> list[dict]:
    """Vision の textAnnotations[1:] を {text,x,y,h} のトークン列に変換。"""
    tokens: list[dict] = []
    # [0] は画像全体テキストなのでスキップ
    for ann in annotations[1:]:
        text = ann.get("description", "")
        verts = ann.get("boundingPoly", {}).get("vertices", [])
        if not text or not verts:
            continue
        xs = [v.get("x", 0) for v in verts]
        ys = [v.get("y", 0) for v in verts]
        tokens.append(
            {
                "text": text,
                "x": min(xs),
                "y": (min(ys) + max(ys)) / 2,  # 中心 y
                "h": max(ys) - min(ys),
            }
        )
    return tokens


# ---------------------------------------------------------------------------
# 行の組み立て
# ---------------------------------------------------------------------------
def group_rows(tokens: list[dict]) -> list[dict]:
    """トークンを y 座標近接でグルーピングし、行ごとのテキストを組み立てる。

    戻り値: [{"text": "ヒット1 (107.22%) 1,362 - 1,740", "y": ..., "h": ...}, ...]
    (y 昇順)。
    """
    if not tokens:
        return []

    heights = sorted(t["h"] for t in tokens if t["h"] > 0)
    median_h = heights[len(heights) // 2] if heights else 10
    threshold = max(median_h * 0.6, 6)  # 同一行とみなす y 距離

    rows: list[dict] = []
    for tok in sorted(tokens, key=lambda t: t["y"]):
        placed = False
        for row in rows:
            if abs(row["y"] - tok["y"]) <= threshold:
                row["tokens"].append(tok)
                # 行 y を加重平均でなだらかに更新
                row["y"] = (row["y"] * row["n"] + tok["y"]) / (row["n"] + 1)
                row["n"] += 1
                placed = True
                break
        if not placed:
            rows.append({"y": tok["y"], "h": tok["h"], "n": 1, "tokens": [tok]})

    result = []
    for row in rows:
        ordered = sorted(row["tokens"], key=lambda t: t["x"])
        result.append(
            {
                "text": " ".join(t["text"] for t in ordered),
                "y": row["y"],
                "h": row["h"],
            }
        )
    result.sort(key=lambda r: r["y"])
    return result


# ---------------------------------------------------------------------------
# 数値・ヒット解析
# ---------------------------------------------------------------------------
_HIT_RE = re.compile(r"ヒット\s*(\d+)\s*(?:[-–~〜]\s*(\d+))?")
_RANGE_SEP = r"[-–—~〜]"
# 7,235 - 9,286 / 6,199 / 72,350 ~ 92,860 などにマッチ
_RANGE_RE = re.compile(
    rf"(\d[\d,]*)\s*(?:{_RANGE_SEP}\s*(\d[\d,]*))?"
)


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def _extract_damage(text: str) -> tuple[int, int] | None:
    """行末側の数値(範囲 or 単一)を (min, max) で返す。見つからなければ None。

    パーセント表記 (xx.xx%) は除外する。
    """
    # パーセント値を除去 (例: (107.22%))
    cleaned = re.sub(r"\(?\d[\d,]*\.\d+\s*%\)?", " ", text)
    cleaned = re.sub(r"\d+\s*ヒット", " ", cleaned)  # "7ヒット" 等の総数表記を除外
    matches = list(_RANGE_RE.finditer(cleaned))
    # 末尾(右側)のダメージ数値を採用。小数を含むものは弾く。
    best = None
    for m in matches:
        lo = m.group(1)
        if "." in m.group(0):
            continue
        hi = m.group(2) if m.group(2) else lo
        best = (_to_int(lo), _to_int(hi))
    return best


def _hit_count(text: str) -> tuple[int, str] | None:
    """ヒット行ならヒット数とラベルを返す。例 'ヒット2-6' → (5, 'ヒット2-6')。"""
    m = _HIT_RE.search(text)
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    label = m.group(0).replace(" ", "")
    return (max(1, end - start + 1), label)


def _is_crit_row(text: str) -> bool:
    """『会心』ラベル行か(ヒット表記を含まず会心を含む)。"""
    return "会心" in text and "ヒット" not in text


def _has_hp_dependency(rows: list[dict]) -> bool:
    joined = " ".join(r["text"] for r in rows)
    return "現在HP" in joined or "現在 HP" in joined


# ---------------------------------------------------------------------------
# カード組み立て
# ---------------------------------------------------------------------------
def parse_cards(tokens: list[dict]) -> dict:
    """トークン列からカードパラメータと検出メタ情報を構築する。

    戻り値: {"cards": [{"params": {...}, "memo": str}, ...],
             "hp_dependent": bool}
    """
    rows = group_rows(tokens)

    # 各ヒットを「エントリ」に分解する
    #   entry = {hits, normal:(min,max)|None, crit:(min,max)|None,
    #            guaranteed_crit:bool, label}
    entries: list[dict] = []
    for row in rows:
        text = row["text"]
        hit = _hit_count(text)
        if hit is not None:
            count, label = hit
            dmg = _extract_damage(text)
            entries.append(
                {
                    "hits": count,
                    "label": label,
                    "first": dmg,        # ヒット行に並ぶ数値 (通常 or 確定会心)
                    "crit": None,        # 後続の会心行で埋まる
                    "pct": _extract_pct(text),
                }
            )
            continue
        if _is_crit_row(text) and entries:
            entries[-1]["crit"] = _extract_damage(text)

    cards = [_entry_to_card(e) for e in entries if e["first"] is not None]
    cards = _merge_consecutive(cards)

    return {"cards": cards, "hp_dependent": _has_hp_dependency(rows)}


_PCT_RE = re.compile(r"(\d[\d,]*\.\d+)\s*%")


def _extract_pct(text: str) -> str | None:
    m = _PCT_RE.search(text)
    return m.group(1) if m else None


def _entry_to_card(entry: dict) -> dict:
    """1 エントリ → カードパラメータ辞書。"""
    params: dict = {"hits": entry["hits"]}
    memo_parts = [entry["label"]]
    if entry["pct"]:
        memo_parts.append(f"攻撃力{entry['pct']}%")

    if entry["crit"] is not None:
        # 一般: first=通常, crit=会心
        params["normal_min"], params["normal_max"] = entry["first"]
        params["crit_min"], params["crit_max"] = entry["crit"]
    else:
        # 確定会心: 会心行が続かない → first を会心として扱い会心率100
        params["crit_min"], params["crit_max"] = entry["first"]
        params["crit_rate"] = 100
        memo_parts.append("確定会心")

    return {"params": params, "memo": " ".join(memo_parts)}


def _merge_consecutive(cards: list[dict]) -> list[dict]:
    """ダメージ値が同一の連続カードを1枚に統合し hits を合算する。

    確定会心モードで各ヒットが個別行になっているケース
    (ヒット1..ヒット10 が同値) を1枚にまとめる。
    """
    merged: list[dict] = []
    for card in cards:
        if merged and _same_damage(merged[-1]["params"], card["params"]):
            merged[-1]["params"]["hits"] += card["params"]["hits"]
        else:
            merged.append(card)
    return merged


_DAMAGE_KEYS = ("normal_min", "normal_max", "crit_min", "crit_max", "crit_rate")


def _same_damage(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in _DAMAGE_KEYS)


def _apply_memo_prefix(cards: list[dict], prefix: str) -> list[dict]:
    """全カードの備考の頭に共通の prefix を付ける (空なら何もしない)。"""
    prefix = (prefix or "").strip()
    if not prefix:
        return cards
    for card in cards:
        card["memo"] = f"{prefix} {card['memo']}".strip()
    return cards


# ---------------------------------------------------------------------------
# テキスト貼り付けからのカード生成
# ---------------------------------------------------------------------------
# OCR (parse_cards) と異なり、貼り付けテキストはヒット行・ダメージ行・「会心」行
# がそれぞれ別の行に分かれている。
#   一般       : ヒット行 → 通常ダメージ行 → 「会心」行 → 会心ダメージ行
#   確定会心/確定非会心 : ヒット行 → ダメージ行 (会心行が続かない)
# そのため OCR の「ヒット行に数値が並ぶ」前提は使えず、行ごとに状態を持って
# entry を組み立てる。entry の形は parse_cards と同じなので _entry_to_card /
# _merge_consecutive を共用する。


def parse_text(text: str, prefix: str = "") -> dict:
    """貼り付けテキストからカードパラメータと検出メタ情報を構築する。

    prefix: 全カードの備考の頭に付ける共通の文字列 (例:「ミカ1射目」)。
            1 回の取り込みがどの攻撃のものか、後から見て分かるようにする。

    戻り値: {"cards": [{"params": {...}, "memo": str}, ...],
             "hp_dependent": bool}
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    entries: list[dict] = []
    expect_crit = False  # 直前に「会心」行が来た → 次のダメージ行は会心
    for line in lines:
        hit = _hit_count(line)
        if hit is not None:
            count, label = hit
            # 同一行に数値が並ぶ表記にも備え、ヒットラベルを除いてから抽出する
            # ("ヒット1-2" の "1-2" を誤ってダメージとして拾わないため)。
            remainder = _HIT_RE.sub(" ", line, count=1)
            entries.append(
                {
                    "hits": count,
                    "label": label,
                    "first": _extract_damage(remainder),
                    "crit": None,
                    "pct": _extract_pct(line),
                }
            )
            expect_crit = False
            continue

        if _is_crit_row(line):
            expect_crit = True
            # 「会心 35,239 - 48,979」のように同一行に数値があれば即取り込む
            dmg = _extract_damage(line)
            if dmg is not None and entries:
                entries[-1]["crit"] = dmg
                expect_crit = False
            continue

        # 数値のみの行 (通常 or 会心ダメージ)
        dmg = _extract_damage(line)
        if dmg is None or not entries:
            continue
        entry = entries[-1]
        if expect_crit:
            entry["crit"] = dmg
            expect_crit = False
        elif entry["first"] is None:
            entry["first"] = dmg
        elif entry["crit"] is None:
            # 「会心」ラベルが省略され通常・会心の2行が続くケースの保険
            entry["crit"] = dmg

    cards = [_entry_to_card(e) for e in entries if e["first"] is not None]
    cards = _merge_consecutive(cards)
    # prefix は統合が済んでから付ける (統合後に残る備考にだけ付けば良い)
    _apply_memo_prefix(cards, prefix)

    rows = [{"text": ln} for ln in lines]
    return {"cards": cards, "hp_dependent": _has_hp_dependency(rows)}


# ---------------------------------------------------------------------------
# 公開エントリポイント
# ---------------------------------------------------------------------------
def cards_from_image(image: str | bytes, *, api_key: str | None = None) -> dict:
    """画像 → {"cards": [...], "hp_dependent": bool}。"""
    tokens = vision_annotate(image, api_key=api_key)
    return parse_cards(tokens)


def cards_from_text(text: str, prefix: str = "") -> dict:
    """貼り付けテキスト → {"cards": [...], "hp_dependent": bool}。

    prefix を渡すと、生成する全カードの備考の頭に付く。
    """
    return parse_text(text, prefix)
