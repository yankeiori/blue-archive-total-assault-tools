"""蓄積 (チャージ) 型スキルの分布を手で動かして確かめる 独立した Dash ミニアプリ。

理論は docs/accumulate.md、計算は app/backend/accumulate.py。
ダメージカード (edgeworth_app と同じ UI) と蓄積プールをブラウザで設定し、
「計算実行」で次を並べて表示する:

    - グリッド合成 (build_accum_dist) と MC (mc_accum) の裾確率 P(T >= x)・密度
    - 蓄積なし (素の和モデル) との比較 (通過確率の差・分位点ごとの上乗せダメージ)
    - 目標ダメージ D での通過確率
    - プールごとの診断量 (飽和確率・プール平均・期待溢れ・爆発平均)
    - 分布の形状の変化 (標準化密度・飽和/非飽和の内訳・上限スイープでのモーメント)

カードはゲームのダメージ表示のテキストを貼り付けて取り込むこともできる
(本番アプリと同じ app/backend/ocr.py の cards_from_text を使う)。

プールの「対象カード」「上限カード」はカード見出しの番号 (「カード N」の N) を
"1-3,5" のように書く。削除や取り込みで番号が飛んでも見出しの番号で指す。カードの Hit はすべてそのプールに寄与する。

実行例:
    uv run python -m experiments.accum_app
    # → http://127.0.0.1:8062/
"""
from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.accumulate import (  # noqa: E402
    AccumWindow,
    CapSpec,
    _sample_mixture,
    build_accum_dist,
    burst_amount,
    mc_accum,
)
from app.backend import ocr  # noqa: E402
from app.backend.cos import Uniform  # noqa: E402
from experiments.edgeworth_animation import build_all_hits  # noqa: E402
from experiments.edgeworth_app import (  # noqa: E402
    LABEL_STYLE,
    PRIMARY,
    SECTION_STYLE,
    _assemble_cards,
    _labelled,
    make_card,
    manage_cards,
)


# ---------------------------------------------------------------------------
# 既定値
# ---------------------------------------------------------------------------
DEFAULT_N_MC = 400_000
DEFAULT_SEED = 0
MC_CHUNK = 100_000          # MC を分割して回す単位 (Hit 数 × この本数の配列を持つ)
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

DEFAULT_POOL: dict = {
    "name": "", "cards": "1-2", "rate": 100, "cap_kind": "fixed",
    "cap_value": 6_000_000, "cap_lo": 5_000_000, "cap_hi": 6_000_000,
    "cap_cards": "1", "cap_pct": 400, "burst_mult": 100, "burst_decay": [],
    "emit": ["on"],
}

# experiments/accum_compare.py と同じ攻撃列 (EX 4Hit + NS 3Hit + サブ 6Hit)
_COMPARE_CARDS = [
    {"crit_min": 900_000, "crit_max": 1_100_000, "normal_min": 450_000,
     "normal_max": 550_000, "hits": 4, "crit_rate": 60, "evade_rate": 0, "memo": "EX"},
    {"crit_min": 1_400_000, "crit_max": 1_800_000, "normal_min": 700_000,
     "normal_max": 900_000, "hits": 3, "crit_rate": 50, "evade_rate": 5, "memo": "NS"},
    {"crit_min": 300_000, "crit_max": 380_000, "normal_min": 150_000,
     "normal_max": 190_000, "hits": 6, "crit_rate": 40, "evade_rate": 0, "memo": "サブ"},
]

# イロハ(水着) のスーツケース: イロハ1 の攻撃 (3Hit) × 120% が上限、ハルカの 3Hit が 150% で蓄積。
# イロハは確定会心なので非会心欄は計算に使われない (入力画面の値をそのまま残している)。
_IROHA_HARUKA_CARDS = [
    {"crit_min": 1_677_970, "crit_max": 2_414_520, "normal_min": 1_000_000,
     "normal_max": 1_500_000, "hits": 2, "crit_rate": 100, "evade_rate": None,
     "memo": "イロハ1 ヒット1-2 攻撃力502.47% 確定会心"},
    {"crit_min": 2_237_293, "crit_max": 3_219_360, "normal_min": 1_000_000,
     "normal_max": 1_500_000, "hits": 1, "crit_rate": 100, "evade_rate": None,
     "memo": "イロハ1 ヒット3 攻撃力669.96% 確定会心"},
    {"crit_min": 2_516_955, "crit_max": 3_621_781, "normal_min": 1_000_000,
     "normal_max": 1_500_000, "hits": 2, "crit_rate": 100, "evade_rate": None,
     "memo": "イロハ2 ヒット1-2 攻撃力502.47% 確定会心"},
    {"crit_min": 3_355_939, "crit_max": 4_663_233, "normal_min": 1_000_000,
     "normal_max": 1_500_000, "hits": 1, "crit_rate": 100, "evade_rate": None,
     "memo": "イロハ2 ヒット3 攻撃力669.96% 確定会心"},
    {"crit_min": 2_267_624, "crit_max": 2_634_362, "normal_min": 434_427,
     "normal_max": 504_686, "hits": 2, "crit_rate": 64.94, "evade_rate": None,
     "memo": "ハルカ ヒット1-2 攻撃力415.11%"},
    {"crit_min": 2_268_304, "crit_max": 2_635_152, "normal_min": 434_558,
     "normal_max": 504_838, "hits": 1, "crit_rate": 64.94, "evade_rate": None,
     "memo": "ハルカ ヒット3 攻撃力415.23%"},
]

PRESETS: dict = {
    "単発・固定上限 (accum_compare 1)": {
        "cards": _COMPARE_CARDS,
        "pools": [{**DEFAULT_POOL, "name": "蓄積", "cards": "1-2", "rate": 100,
                   "cap_kind": "fixed", "cap_value": 6_000_000}],
    },
    "単発・乱数上限 120% (accum_compare 2 相当)": {
        "cards": _COMPARE_CARDS,
        "pools": [{**DEFAULT_POOL, "name": "蓄積", "cards": "1-2", "rate": 120,
                   "cap_kind": "range", "cap_lo": 5_000_000, "cap_hi": 9_500_000}],
    },
    "上限 = カード1 × 200% (相関あり)": {
        "cards": _COMPARE_CARDS,
        "pools": [{**DEFAULT_POOL, "name": "蓄積", "cards": "1-2", "rate": 100,
                   "cap_kind": "cards", "cap_cards": "1", "cap_pct": 200}],
    },
    "3 回撃ち (カードごとに窓)": {
        "cards": _COMPARE_CARDS,
        "pools": [{**DEFAULT_POOL, "name": f"蓄積{k + 1}", "cards": str(k + 1),
                   "cap_kind": "range", "cap_lo": 2_000_000, "cap_hi": 2_600_000}
                  for k in range(3)],
    },
    "全体 120%・上限 1000 万・爆発に減衰": {
        "cards": _COMPARE_CARDS,
        "pools": [{**DEFAULT_POOL, "name": "蓄積", "cards": "1-3", "rate": 120,
                   "cap_kind": "fixed", "cap_value": 10_000_000, "burst_decay": ["on"]}],
    },
    "イロハ(水着) × ハルカ (上限 = イロハ1 の 3Hit × 120%)": {
        "cards": _IROHA_HARUKA_CARDS,
        "pools": [{**DEFAULT_POOL, "name": "スーツケース", "cards": "5-6", "rate": 150,
                   "cap_kind": "cards", "cap_cards": "1-2", "cap_pct": 120,
                   "burst_mult": 100, "burst_decay": []}],
    },
}
DEFAULT_PRESET = next(iter(PRESETS))


# ---------------------------------------------------------------------------
# プール入力カード
# ---------------------------------------------------------------------------

def make_pool(idx: int, params: dict | None = None) -> html.Div:
    """蓄積プール 1 本分の入力カード。id は {"type": "acc-param", "param", "index"}。"""
    p = dict(DEFAULT_POOL)
    if params:
        p.update(params)

    def pid(param: str) -> dict:
        return {"type": "acc-param", "param": param, "index": idx}

    def field(label: str, param: str, kind: str = "number", **kw) -> html.Div:
        return html.Div(
            [html.Label(label, style=LABEL_STYLE),
             dcc.Input(id=pid(param), type=kind, value=p[param],
                       style={"width": "100%"}, **kw)],
            style={"flex": "1", "minWidth": "110px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Strong("⚡ 蓄積プール"),
                    dcc.Input(id=pid("name"), type="text", value=p["name"],
                              placeholder="名前",
                              style={"marginLeft": "8px", "flex": "1", "fontSize": "0.85rem"}),
                    html.Button("✕", id={"type": "acc-remove", "index": idx}, n_clicks=0,
                                title="プールを削除",
                                style={"marginLeft": "6px", "background": "none",
                                       "border": "none", "cursor": "pointer",
                                       "fontSize": "1.1rem"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    field("対象カード (例 1-3,5)", "cards", "text"),
                    field("蓄積率 (%)", "rate"),
                    field("爆発倍率 (%)", "burst_mult"),
                    html.Div(
                        [dcc.Checklist(id=pid("burst_decay"),
                                       options=[{"label": " 爆発に減衰を通す", "value": "on"}],
                                       value=p["burst_decay"]),
                         dcc.Checklist(id=pid("emit"),
                                       options=[{"label": " 爆発する", "value": "on"}],
                                       value=p["emit"])],
                        style={"flex": "1", "minWidth": "140px", "fontSize": "0.85rem"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("上限の種類", style=LABEL_STYLE),
                         dcc.Dropdown(
                             id=pid("cap_kind"),
                             options=[
                                 {"label": "固定値", "value": "fixed"},
                                 {"label": "一様乱数 (独立)", "value": "range"},
                                 {"label": "カードのダメージ × %", "value": "cards"},
                             ],
                             value=p["cap_kind"], clearable=False)],
                        style={"flex": "1.4", "minWidth": "170px"},
                    ),
                    field("固定上限", "cap_value"),
                    field("乱数上限 下限", "cap_lo"),
                    field("乱数上限 上限", "cap_hi"),
                    field("上限カード", "cap_cards", "text"),
                    field("上限 %", "cap_pct"),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "6px"},
            ),
        ],
        id={"type": "acc-pool", "index": idx},
        style={"border": "1px solid #e0b000", "borderRadius": "8px", "padding": "12px",
               "marginBottom": "10px", "background": "#fffdf2"},
    )


def _parse_card_list(text) -> list[int]:
    """"1-3,5" → [1, 2, 3, 5] (カード見出しの番号。重複は除く)。"""
    out: list[int] = []
    for part in str(text or "").replace("、", ",").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            rng = range(int(a), int(b) + 1)
        else:
            rng = range(int(part), int(part) + 1)
        out.extend(k for k in rng if k not in out)
    return out


def _build_windows(pools: list[dict], boundaries: list[int],
                   labels: list[int] | None = None) -> list[AccumWindow]:
    """プール入力をカード → Hit 番号に展開して AccumWindow にする。

    labels[j] は上から j 枚目のカードの見出し番号 (「カード N」の N)。
    省略時は 1, 2, ... (見出し番号 = 位置)。存在しない番号はエラーにする
    (黙って落とすと「蓄積なし」と同じ結果が出て気づけないため)。
    """
    starts = [0] + boundaries[:-1]
    pos = {n: j for j, n in enumerate(labels or range(1, len(boundaries) + 1))}

    def hits_of(text, name: str, what: str) -> list[int]:
        nums = _parse_card_list(text)
        missing = [n for n in nums if n not in pos]
        if missing:
            raise ValueError(f"{name}: {what}に存在しないカード番号があります: {missing}"
                             f" (今あるカード: {sorted(pos)})")
        return [i for n in nums for i in range(starts[pos[n]], boundaries[pos[n]])]

    windows = []
    for k, p in enumerate(pools):
        name = p.get("name") or f"蓄積{k + 1}"
        hits = hits_of(p.get("cards"), name, "対象カード")
        if not hits:
            raise ValueError(f"{name}: 対象カードが空です")
        kind = p.get("cap_kind") or "fixed"
        if kind == "fixed":
            cap = CapSpec(kind="fixed", value=float(p.get("cap_value") or 0))
        elif kind == "range":
            lo, hi = float(p.get("cap_lo") or 0), float(p.get("cap_hi") or 0)
            cap = CapSpec(kind="mixture", mixture=[Uniform(1.0, min(lo, hi), max(lo, hi))])
        else:
            src = hits_of(p.get("cap_cards"), name, "上限カード")
            if not src:
                raise ValueError(f"{name}: 上限カードが空です")
            cap = CapSpec(kind="hits", hits=src, coef=float(p.get("cap_pct") or 0) / 100)
        windows.append(AccumWindow(
            hits=hits, rate=float(p.get("rate") or 0) / 100, cap=cap, name=name,
            burst_mult=float(p.get("burst_mult") or 0) / 100,
            burst_decay=bool(p.get("burst_decay")), emit=bool(p.get("emit")),
        ))
    return windows


def _assemble_pools(values, ids, order) -> list[dict]:
    per: dict[int, dict] = {}
    for v, sid in zip(values, ids):
        per.setdefault(sid["index"], {})[sid["param"]] = v
    return [per[i] for i in order if i in per]


# ---------------------------------------------------------------------------
# レイアウト
# ---------------------------------------------------------------------------

def _text_panel() -> html.Details:
    """テキスト貼り付け → カード取り込みパネル (本番アプリの _text_panel 相当)。"""
    placeholder = ("ダメージ表示のテキストを貼り付け\n例:\nヒット1-2 (165.33%)\n"
                   "18,164 - 25,247\n会心\n35,239 - 48,979")
    return html.Details(
        [
            html.Summary("📝 テキストからカード取り込み",
                         style={"cursor": "pointer", "fontWeight": "bold"}),
            html.Div(
                [
                    dcc.Input(id="acc-text-prefix", type="text", value="",
                              placeholder="備考の先頭に付ける文字列 (任意, 例: ミカ1射目)",
                              style={"flex": "1"}),
                    dcc.Checklist(id="acc-text-replace",
                                  options=[{"label": " 既存カードを置き換える", "value": "on"}],
                                  value=[], style={"fontSize": "0.85rem"}),
                ],
                style={"display": "flex", "gap": "12px", "alignItems": "center",
                       "marginTop": "8px"},
            ),
            dcc.Textarea(id="acc-text-input", placeholder=placeholder,
                         style={"width": "100%", "boxSizing": "border-box", "height": "120px",
                                "marginTop": "8px", "fontFamily": "monospace",
                                "fontSize": "0.82rem", "resize": "vertical"}),
            html.Button("テキストから取り込み", id="acc-text-import-btn", n_clicks=0,
                        style={"background": "#7c5cd9", "color": "white", "border": "none",
                               "borderRadius": "6px", "padding": "6px 16px",
                               "cursor": "pointer", "marginTop": "6px"}),
            html.Div(id="acc-text-status",
                     style={"fontSize": "0.82rem", "color": "#666", "marginTop": "6px",
                            "minHeight": "1.2em"}),
        ],
        style={**SECTION_STYLE, "background": "#f7f3ff"},
    )


def _sidebar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("プリセット"),
                    _labelled("シナリオ", dcc.Dropdown(
                        id="acc-preset", options=[{"label": k, "value": k} for k in PRESETS],
                        value=DEFAULT_PRESET, clearable=False)),
                    html.Button("プリセット読込", id="acc-preset-load-btn", n_clicks=0,
                                style={"marginTop": "8px", "width": "100%", "cursor": "pointer"}),
                ],
                style={**SECTION_STYLE, "background": "#fffbe6"},
            ),
            html.Div(
                [
                    html.Strong("ダメージ生成モード"),
                    dcc.RadioItems(
                        id="acc-damage-mode",
                        options=[{"label": "減衰考慮済み（推奨）", "value": "post_decay"},
                                 {"label": "減衰考慮前", "value": "pre_decay"}],
                        value="post_decay",
                        style={"display": "flex", "flexDirection": "column",
                               "gap": "4px", "marginTop": "6px"}),
                ],
                style={**SECTION_STYLE, "background": "#f0f8f0"},
            ),
            html.Div(
                [
                    html.Strong("一括設定"),
                    _labelled("会心率(%)", dcc.Input(id="acc-global-crit", type="number",
                                                    value=50, style={"width": "100%"})),
                    _labelled("回避率(%)", dcc.Input(id="acc-global-evade", type="number",
                                                    value=0, style={"width": "100%"})),
                ],
                style={**SECTION_STYLE, "background": "#f5f5ff"},
            ),
            html.Div(
                [
                    html.Strong("比較設定"),
                    _labelled("目標ダメージ D (空欄なら中央値)",
                              dcc.Input(id="acc-target", type="number",
                                        style={"width": "100%"})),
                    _labelled("MC サンプル数 (0 で省略)",
                              dcc.Input(id="acc-n-mc", type="number", value=DEFAULT_N_MC,
                                        min=0, step=100_000, style={"width": "100%"})),
                    _labelled("乱数シード", dcc.Input(id="acc-seed", type="number",
                                                   value=DEFAULT_SEED, step=1,
                                                   style={"width": "100%"})),
                    dcc.Checklist(id="acc-show-base",
                                  options=[{"label": " 蓄積なしと比較", "value": "show"}],
                                  value=["show"], style={"marginTop": "8px"}),
                    dcc.Checklist(id="acc-show-shape",
                                  options=[{"label": " 形状の変化を解析", "value": "show"}],
                                  value=["show"], style={"marginTop": "4px"}),
                ],
                style={**SECTION_STYLE, "background": "#fff5f5"},
            ),
        ],
        style={"width": "220px", "flexShrink": "0", "position": "sticky", "top": "20px",
               "alignSelf": "flex-start"},
    )


def create_layout() -> html.Div:
    preset = PRESETS[DEFAULT_PRESET]
    btn = {"marginLeft": "12px", "background": PRIMARY, "color": "white", "border": "none",
           "borderRadius": "4px", "padding": "6px 16px", "cursor": "pointer",
           "fontWeight": "bold"}
    return html.Div(
        [
            html.H2("蓄積スキル 分布実験UI"),
            html.Div(
                [
                    _sidebar(),
                    html.Div(
                        [
                            html.H4("ダメージカード", style={"margin": "0 0 8px"}),
                            _text_panel(),
                            html.Div(id="exp-cards-container",
                                     children=[make_card(i, params=c, memo=c.get("memo", ""))
                                               for i, c in enumerate(preset["cards"])]),
                            html.Button("+ カード追加", id="exp-add-btn", n_clicks=0,
                                        style={"marginBottom": "16px"}),
                            html.H4("蓄積プール", style={"margin": "0 0 8px"}),
                            html.Div(
                                "対象カード・上限カードは見出し「カード N」の番号で指定。窓は互いに素である必要があります"
                                " (同じカードを 2 つのプールに入れると後ろは無視されます)。",
                                style={"fontSize": "0.8rem", "color": "#666", "marginBottom": "6px"}),
                            html.Div(id="acc-pools-container",
                                     children=[make_pool(i, p)
                                               for i, p in enumerate(preset["pools"])]),
                            html.Div(
                                [
                                    html.Button("+ プール追加", id="acc-add-btn", n_clicks=0),
                                    html.Button("計算実行", id="acc-run-btn", n_clicks=0,
                                                style=btn),
                                ],
                                style={"marginBottom": "12px"},
                            ),
                            dcc.Loading(html.Div(id="acc-output", style={"marginTop": "12px"}),
                                        type="circle", color=PRIMARY),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
            ),
            dcc.Store(id="exp-card-indices", data=list(range(len(preset["cards"])))),
            dcc.Store(id="exp-next-index", data=len(preset["cards"])),
            dcc.Store(id="acc-pool-indices", data=list(range(len(preset["pools"])))),
            dcc.Store(id="acc-pool-next", data=len(preset["pools"])),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px",
               "fontFamily": "sans-serif"},
    )


app = Dash(__name__)
app.title = "蓄積スキル 分布実験"
app.layout = create_layout


# ---------------------------------------------------------------------------
# カード追加・削除 (edgeworth_app の manage_cards をそのまま再利用)
# ---------------------------------------------------------------------------
app.callback(
    Output("exp-cards-container", "children"),
    Output("exp-card-indices", "data"),
    Output("exp-next-index", "data"),
    Input("exp-add-btn", "n_clicks"),
    Input({"type": "exp-remove", "index": ALL}, "n_clicks"),
    Input({"type": "exp-duplicate", "index": ALL}, "n_clicks"),
    Input({"type": "exp-move-up", "index": ALL}, "n_clicks"),
    Input({"type": "exp-move-down", "index": ALL}, "n_clicks"),
    State("exp-cards-container", "children"),
    State("exp-card-indices", "data"),
    State("exp-next-index", "data"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)(manage_cards)


@app.callback(
    Output("acc-pools-container", "children"),
    Output("acc-pool-indices", "data"),
    Output("acc-pool-next", "data"),
    Input("acc-add-btn", "n_clicks"),
    Input({"type": "acc-remove", "index": ALL}, "n_clicks"),
    State({"type": "acc-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "acc-param", "param": ALL, "index": ALL}, "id"),
    State("acc-pool-indices", "data"),
    State("acc-pool-next", "data"),
    prevent_initial_call=True,
)
def manage_pools(_add, _removes, values, ids, order, next_idx):
    """プールの追加・削除。入力中の値は State から組み直して保つ。"""
    trig = ctx.triggered_id
    pools = dict(zip(order, _assemble_pools(values, ids, order)))
    order = list(order)
    if trig == "acc-add-btn":
        pools[next_idx] = dict(DEFAULT_POOL)
        order.append(next_idx)
        next_idx += 1
    elif isinstance(trig, dict) and trig.get("type") == "acc-remove":
        if not ctx.triggered[0]["value"]:
            return no_update, no_update, no_update
        order = [i for i in order if i != trig["index"]]
    else:
        return no_update, no_update, no_update
    return [make_pool(i, pools[i]) for i in order], order, next_idx


@app.callback(
    Output("exp-cards-container", "children", allow_duplicate=True),
    Output("exp-card-indices", "data", allow_duplicate=True),
    Output("exp-next-index", "data", allow_duplicate=True),
    Output("acc-pools-container", "children", allow_duplicate=True),
    Output("acc-pool-indices", "data", allow_duplicate=True),
    Output("acc-pool-next", "data", allow_duplicate=True),
    Input("acc-preset-load-btn", "n_clicks"),
    State("acc-preset", "value"),
    prevent_initial_call=True,
)
def load_preset(n_clicks, name):
    if not n_clicks or name not in PRESETS:
        return (no_update,) * 6
    cards, pools = PRESETS[name]["cards"], PRESETS[name]["pools"]
    return ([make_card(i, params=c, memo=c.get("memo", "")) for i, c in enumerate(cards)],
            list(range(len(cards))), len(cards),
            [make_pool(i, p) for i, p in enumerate(pools)],
            list(range(len(pools))), len(pools))


@app.callback(
    Output("exp-cards-container", "children", allow_duplicate=True),
    Output("exp-card-indices", "data", allow_duplicate=True),
    Output("exp-next-index", "data", allow_duplicate=True),
    Output("acc-text-status", "children"),
    Input("acc-text-import-btn", "n_clicks"),
    State("acc-text-input", "value"),
    State("acc-text-prefix", "value"),
    State("acc-text-replace", "value"),
    State("exp-cards-container", "children"),
    State("exp-card-indices", "data"),
    State("exp-next-index", "data"),
    prevent_initial_call=True,
)
def text_import(n_clicks, text, prefix, replace, children, indices, next_idx):
    """貼り付けテキストを解析してカードを追加 (または置き換え) する。

    children は State なので入力中の値ごと届く。追加時はそのまま後ろに足す。
    """
    no_change = (no_update, no_update, no_update)
    if not n_clicks or not (text or "").strip():
        return (*no_change, "⚠ テキストが空です。")
    try:
        result = ocr.cards_from_text(text, prefix or "")
    except Exception as e:  # noqa: BLE001 — 解析失敗はそのまま画面に出す
        return (*no_change, f"⚠ 解析に失敗しました: {e}")
    parsed = result["cards"]
    if not parsed:
        return (*no_change, "⚠ カードを検出できませんでした。テキストを確認してください。")

    if replace:
        children, indices, next_idx = [], [], 0
    children, indices = list(children or []), list(indices or [])
    for card in parsed:
        children.append(make_card(next_idx, params=card["params"], memo=card["memo"]))
        indices.append(next_idx)
        next_idx += 1

    verb = "置き換えました" if replace else "追加しました"
    msg = f"✅ {len(parsed)} 枚のカードを{verb} (計 {len(indices)} 枚)。"
    if result.get("hp_dependent"):
        msg += " ⚠ HP依存の表記を検出しましたが、この実験UIは和モデルのみ対応です。"
    return children, indices, next_idx, msg


# ---------------------------------------------------------------------------
# 計算
# ---------------------------------------------------------------------------

def _mc(hm, windows, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = [mc_accum(hm, windows, min(MC_CHUNK, n - s), rng) for s in range(0, n, MC_CHUNK)]
    return np.sort(np.concatenate(parts))


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def _table(header: list[str], rows: list[list]) -> html.Table:
    cell = {"border": "1px solid #ddd", "padding": "3px 8px", "textAlign": "right",
            "fontSize": "0.85rem"}
    return html.Table(
        [html.Tr([html.Th(h, style={**cell, "background": "#f3f3f3"}) for h in header])]
        + [html.Tr([html.Td(v, style=cell) for v in r]) for r in rows],
        style={"borderCollapse": "collapse", "marginBottom": "12px"},
    )


def _quantile(dist, ps) -> np.ndarray:
    """セル質量から分位点を引く (累積が p に達する最初のセルの中心)。"""
    cum = np.cumsum(dist.mass)
    idx = np.searchsorted(cum, np.asarray(ps, dtype=float), side="left")
    return dist.centers[np.clip(idx, 0, dist.mass.size - 1)]


def _compare_figures(dist, base, D: float) -> tuple[go.Figure, go.Figure]:
    """蓄積あり / なしの比較。

    1 枚目: 通過確率 P(T ≥ D) を D の関数として重ね、下段に差 (あり − なし)。
    2 枚目: 分位点ごとの上乗せダメージ Q_あり(p) − Q_なし(p)。
            蓄積が「どの運の良さのときに」どれだけ効くかが分かる
            (上限で頭打ちになると上位の分位点ほど上乗せが縮む)。
    """
    lo = float(_quantile(base, [1e-4])[0])
    hi = float(_quantile(dist, [1 - 1e-4])[0])
    xs = np.linspace(lo, hi, 800)
    p_acc, p_base = dist.sf(xs), base.sf(xs)

    fp = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                       vertical_spacing=0.06)
    fp.add_scatter(x=xs, y=p_acc, name="蓄積あり", line={"color": PRIMARY, "width": 2},
                   row=1, col=1)
    fp.add_scatter(x=xs, y=p_base, name="蓄積なし", line={"color": "#888", "dash": "dash"},
                   row=1, col=1)
    fp.add_scatter(x=xs, y=p_acc - p_base, name="差 (あり − なし)", fill="tozeroy",
                   line={"color": "#00b894"}, row=2, col=1)
    fp.add_vline(x=D, line={"color": "#2d3436", "width": 1})
    fp.update_yaxes(title="P(T ≥ D)", range=[0, 1.02], row=1, col=1)
    fp.update_yaxes(title="確率の差", row=2, col=1)
    fp.update_xaxes(title="目標ダメージ D", tickformat=",", row=2, col=1)
    fp.update_layout(height=560, margin={"l": 60, "r": 20, "t": 40, "b": 40},
                     legend={"orientation": "h", "y": 1.06}, hovermode="x unified")

    ps = np.linspace(0.001, 0.999, 999)
    gain = _quantile(dist, ps) - _quantile(base, ps)
    burst_mean = dist.mean - base.mean
    fq = go.Figure()
    fq.add_scatter(x=ps * 100, y=gain, name="上乗せ Q_あり(p) − Q_なし(p)",
                   line={"color": "#6c5ce7", "width": 2})
    fq.add_hline(y=burst_mean, line={"color": "#888", "dash": "dash"},
                 annotation_text=f"平均の上乗せ {_fmt(burst_mean)}")
    fq.update_xaxes(title="分位点 p (%)  (大きいほど運が良い側)")
    fq.update_yaxes(title="上乗せダメージ", tickformat=",", rangemode="tozero")
    fq.update_layout(height=380, margin={"l": 60, "r": 20, "t": 40, "b": 40},
                     legend={"orientation": "h", "y": 1.1}, hovermode="x unified")
    return fp, fq


# ---------------------------------------------------------------------------
# 形状の変化 (docs/accumulate.md §2 の ψ が分布をどう曲げるか)
#
# 窓の寄与は Z = S + g(mult·min(C, αS))。非飽和側では傾き 1 + α·mult で S を
# 引き伸ばし、飽和側では S の傾きが 1 に戻って代わりに上限 C のばらつきが乗る。
# つまり「運の悪い側は拡大、運の良い側は上限で押し潰す」写像なので、平均と
# 分散が増えるだけでなく、上側の裾が詰まって歪度が負に寄る。その強さは
# 飽和確率が中間 (一部だけ上限に当たる) のときに最大になる。
# ---------------------------------------------------------------------------
SHAPE_MC = 200_000          # 内訳・散布図用の MC 本数
SHAPE_SCATTER = 4_000       # 散布図に描く点の数
SWEEP_FACTORS = np.round(np.concatenate([np.linspace(0.0, 3.0, 31), [1.0]]), 4)


def _moments(dist) -> tuple[float, float, float, float]:
    """(平均, 標準偏差, 歪度, 超過尖度)。セル質量から直接。"""
    c, m = dist.centers, dist.mass
    mu = float((m * c).sum())
    sd = float(np.sqrt((m * (c - mu) ** 2).sum()))
    z = (c - mu) / sd if sd > 0 else np.zeros_like(c)
    return mu, sd, float((m * z ** 3).sum()), float((m * z ** 4).sum() - 3.0)


def _scale_caps(windows: list[AccumWindow], f: float) -> list[AccumWindow]:
    """全プールの上限を f 倍した窓のコピー (種類ごとに値・範囲・係数を f 倍)。"""
    out = copy.deepcopy(windows)
    for w in out:
        if w.cap.kind == "fixed":
            w.cap.value *= f
        elif w.cap.kind == "mixture":
            w.cap.mixture = [Uniform(u.weight, u.lo * f, u.hi * f) for u in w.cap.mixture or []]
        else:
            w.cap.coef *= f
    return out


def _mc_detail(hm, windows: list[AccumWindow], n: int, seed: int):
    """合計 T と、プールごとの (窓内ダメージ S, 上限 C, 爆発) の MC サンプル。

    mc_accum と同じ抽選だが、形状の内訳を見るために途中の量も返す。
    """
    rng = np.random.default_rng(seed)
    xs = [_sample_mixture(mix, n, rng) for mix in hm]
    total = np.sum(xs, axis=0)
    parts = []
    for w in windows:
        if not w.emit or w.rate <= 0 or not w.hits or w.burst_mult <= 0:
            continue
        s = np.sum([xs[i] for i in w.hits], axis=0)
        if w.cap.kind == "fixed":
            cap = np.full(n, float(w.cap.value))
        elif w.cap.kind == "mixture":
            cap = _sample_mixture(w.cap.mixture or [], n, rng)
        else:
            cap = w.cap.coef * np.sum([xs[i] for i in w.cap.source_hits()], axis=0)
        burst = w.burst_mult * np.minimum(cap, w.rate * s)
        if w.burst_decay:
            burst = burst_amount(burst, np.inf, 1.0, True)
        total = total + burst
        parts.append((w, s, cap, burst))
    return total, parts


def _shape_section(dist, base, hm, windows: list[AccumWindow], seed: int) -> list:
    mo_a, mo_b = _moments(dist), _moments(base)
    qa = _quantile(dist, [0.01, 0.5, 0.99])
    qb = _quantile(base, [0.01, 0.5, 0.99])

    def spread(q):  # 上側の広がり / 下側の広がり。1 より小さいほど上が詰まっている
        return (q[2] - q[1]) / (q[1] - q[0]) if q[1] > q[0] else float("nan")

    rows = [["平均", _fmt(mo_b[0]), _fmt(mo_a[0]), f"× {mo_a[0] / mo_b[0]:.3f}"],
            ["標準偏差", _fmt(mo_b[1]), _fmt(mo_a[1]), f"× {mo_a[1] / mo_b[1]:.3f}"],
            ["変動係数 σ/μ", f"{mo_b[1] / mo_b[0]:.4f}", f"{mo_a[1] / mo_a[0]:.4f}", ""],
            ["歪度", f"{mo_b[2]:+.3f}", f"{mo_a[2]:+.3f}", f"{mo_a[2] - mo_b[2]:+.3f}"],
            ["超過尖度", f"{mo_b[3]:+.3f}", f"{mo_a[3]:+.3f}", f"{mo_a[3] - mo_b[3]:+.3f}"],
            ["上下の広がり比 (Q99−Q50)/(Q50−Q1)", f"{spread(qb):.3f}", f"{spread(qa):.3f}", ""]]
    table = _table(["", "蓄積なし", "蓄積あり", "変化"], rows)

    lay = {"margin": {"l": 60, "r": 20, "t": 40, "b": 40},
           "legend": {"orientation": "h", "y": 1.1}}

    # --- ① 標準化密度: 位置と尺度を揃えて「形」だけを比べる ---------------
    fz = go.Figure()
    zs = np.linspace(-4.5, 4.5, 700)
    for d, mo, name, line in ((base, mo_b, "蓄積なし", {"color": "#888", "dash": "dash"}),
                              (dist, mo_a, "蓄積あり", {"color": PRIMARY, "width": 2})):
        fz.add_scatter(x=zs, y=d.pdf(mo[0] + zs * mo[1]) * mo[1], name=name, line=line)
    fz.add_scatter(x=zs, y=np.exp(-zs ** 2 / 2) / np.sqrt(2 * np.pi), name="正規分布",
                   line={"color": "#b2bec3", "width": 1})
    fz.update_layout(height=380, **lay)
    fz.update_xaxes(title="標準化した合計ダメージ (T − μ) / σ")
    fz.update_yaxes(title="密度")

    # --- ② / ③ MC による内訳 --------------------------------------------
    total, parts = _mc_detail(hm, windows, SHAPE_MC, seed)
    n_sat = np.zeros(total.size, dtype=int)
    for w, s, cap, _b in parts:
        n_sat += (w.rate * s > cap)
    fs = go.Figure()
    colors = ["#0984e3", "#e17055", "#6c5ce7", "#00b894", "#fdcb6e"]
    lo_h = min(float(total.min()), base.support_lo)
    width = (float(total.max()) - lo_h) / 160
    bins = {"start": lo_h, "end": float(total.max()) + width, "size": width}  # 積み上げ用に共通
    for k in range(int(n_sat.max()) + 1):
        sel = total[n_sat == k]
        if sel.size:
            fs.add_histogram(x=sel, xbins=bins,
                             name=f"飽和したプール {k} 本 ({sel.size / total.size:.1%})",
                             marker={"color": colors[k % len(colors)]}, opacity=0.75)
    bx = np.linspace(base.support_lo, base.support_hi, 400)
    fs.add_scatter(x=bx, y=base.pdf(bx) * total.size * width, name="蓄積なし (同じ本数換算)",
                   line={"color": "#888", "dash": "dash"})
    fs.update_layout(barmode="stack", bargap=0, height=400, **lay)
    fs.update_xaxes(title="合計ダメージ", tickformat=",")
    fs.update_yaxes(title="本数")

    rng = np.random.default_rng(seed + 1)
    pick = rng.choice(total.size, size=min(SHAPE_SCATTER, total.size), replace=False)
    fb = make_subplots(rows=1, cols=len(parts), subplot_titles=[w.name for w, *_ in parts])
    for j, (w, s, cap, burst) in enumerate(parts, start=1):
        sat = (w.rate * s > cap)[pick]
        for mask, label, col in ((~sat, "非飽和", "#0984e3"), (sat, "飽和 (上限で頭打ち)", "#e17055")):
            fb.add_scattergl(x=s[pick][mask], y=burst[pick][mask], mode="markers", name=label,
                             marker={"size": 3, "color": col, "opacity": 0.5},
                             showlegend=(j == 1), row=1, col=j)
        sx = np.linspace(0, float(s.max()), 50)
        fb.add_scatter(x=sx, y=burst_amount(sx, np.inf, w.rate, bool(w.burst_decay), w.burst_mult),
                       name="上限なしなら", line={"color": "#2d3436", "dash": "dot"},
                       showlegend=(j == 1), row=1, col=j)
        fb.update_xaxes(title="窓内ダメージ S", tickformat=",", row=1, col=j)
        fb.update_yaxes(title="爆発ダメージ" if j == 1 else None, tickformat=",", row=1, col=j)
    fb.update_layout(height=400, **lay)

    # --- ④ 上限スイープ: 飽和の度合いと形の関係 ---------------------------
    sweep = []
    for f in np.unique(SWEEP_FACTORS):
        d = build_accum_dist(hm, _scale_caps(windows, float(f)))
        sat = float(np.mean([st.sat_prob for st in d.window_stats])) if d.window_stats else 0.0
        sweep.append((float(f), sat, *_moments(d)))
    sw = np.array(sweep)
    fw = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       subplot_titles=["飽和確率 (プール平均)", "標準偏差", "歪度・超過尖度"])
    fw.add_scatter(x=sw[:, 0], y=sw[:, 1], name="飽和確率", line={"color": "#e17055"},
                   row=1, col=1)
    fw.add_scatter(x=sw[:, 0], y=sw[:, 3], name="標準偏差", line={"color": PRIMARY}, row=2, col=1)
    fw.add_hline(y=mo_b[1], line={"color": "#888", "dash": "dash"}, row=2, col=1,
                 annotation_text="蓄積なし")
    fw.add_scatter(x=sw[:, 0], y=sw[:, 4], name="歪度", line={"color": "#6c5ce7"}, row=3, col=1)
    fw.add_scatter(x=sw[:, 0], y=sw[:, 5], name="超過尖度", line={"color": "#00b894"}, row=3, col=1)
    fw.add_vline(x=1.0, line={"color": "#2d3436", "width": 1})
    fw.update_xaxes(title="上限の倍率 (1 = 現在の設定)", row=3, col=1)
    fw.update_yaxes(tickformat=",", row=2, col=1)
    fw.update_layout(height=640, **lay)

    note = {"fontSize": "0.82rem", "color": "#555", "margin": "0 0 4px"}
    return [
        html.H4("形状の変化"), table,
        html.P("① 平均と標準偏差で標準化した密度。位置と広がりの違いを消して、形の違いだけを比べる。",
               style=note), dcc.Graph(figure=fz),
        html.P("② 合計ダメージを「上限に当たったプールの本数」で色分けした内訳 (MC)。",
               style=note), dcc.Graph(figure=fs),
        html.P("③ 窓内ダメージ S と爆発の関係 (MC)。点線より下に外れた点が上限で削られた分。",
               style=note), dcc.Graph(figure=fb),
        html.P("④ 全プールの上限を一律に倍率 f で動かしたときの形の変化 (グリッド計算)。",
               style=note), dcc.Graph(figure=fw),
    ]


def _figures(dist, base, mc, D: float) -> tuple[go.Figure, go.Figure]:
    lo = float(_quantile(dist, [1e-4])[0])
    xs = np.linspace(lo, dist.support_hi, 600)
    tail, dens = go.Figure(), go.Figure()
    tail.add_scatter(x=xs, y=dist.sf(xs), name="グリッド (蓄積あり)",
                     line={"color": PRIMARY, "width": 2})
    dens.add_scatter(x=xs, y=dist.pdf(xs), name="グリッド (蓄積あり)",
                     line={"color": PRIMARY, "width": 2})
    if mc is not None:
        sf_mc = 1.0 - np.searchsorted(mc, xs, side="left") / mc.size
        tail.add_scatter(x=xs, y=np.where(sf_mc > 0, sf_mc, np.nan), name="MC",
                         line={"color": "#0984e3", "dash": "dot"})
        dens.add_histogram(x=mc, histnorm="probability density", nbinsx=150,
                           name="MC", marker={"color": "rgba(9,132,227,0.35)"})
    if base is not None:
        bx = np.linspace(base.support_lo, base.support_hi, 400)
        tail.add_scatter(x=bx, y=base.sf(bx), name="蓄積なし",
                         line={"color": "#888", "dash": "dash"})
        dens.add_scatter(x=bx, y=base.pdf(bx), name="蓄積なし",
                         line={"color": "#888", "dash": "dash"})
    for f in (tail, dens):
        f.add_vline(x=D, line={"color": "#2d3436", "width": 1},
                    annotation_text=f"D = {_fmt(D)}")
        f.update_layout(height=420, margin={"l": 60, "r": 20, "t": 40, "b": 40},
                        legend={"orientation": "h", "y": 1.08}, bargap=0)
        f.update_xaxes(title="合計ダメージ", tickformat=",")
    tail.update_yaxes(type="log", title="P(T ≥ x)", exponentformat="e")
    dens.update_yaxes(title="密度")
    return tail, dens


@app.callback(
    Output("acc-output", "children"),
    Input("acc-run-btn", "n_clicks"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "exp-param", "param": ALL, "index": ALL}, "id"),
    State({"type": "exp-memo", "index": ALL}, "value"),
    State({"type": "exp-memo", "index": ALL}, "id"),
    State("exp-card-indices", "data"),
    State({"type": "acc-param", "param": ALL, "index": ALL}, "value"),
    State({"type": "acc-param", "param": ALL, "index": ALL}, "id"),
    State("acc-pool-indices", "data"),
    State("acc-global-crit", "value"),
    State("acc-global-evade", "value"),
    State("acc-damage-mode", "value"),
    State("acc-target", "value"),
    State("acc-n-mc", "value"),
    State("acc-seed", "value"),
    State("acc-show-base", "value"),
    State("acc-show-shape", "value"),
    prevent_initial_call=True,
)
def run(n_clicks, values, ids, memo_values, memo_ids, card_order,
        pool_values, pool_ids, pool_order, global_crit, global_evade, damage_mode,
        target, n_mc, seed, show_base, show_shape):
    if not n_clicks:
        return no_update
    try:
        cards = _assemble_cards(values, ids, memo_values, memo_ids, card_order or [])
        if not cards:
            return html.Div("カードが0枚です。", style={"color": "red"})
        hm, boundaries = build_all_hits(cards, float(global_crit or 0),
                                        float(global_evade or 0), damage_mode)
        pools = _assemble_pools(pool_values, pool_ids, pool_order or [])
        # 見出し番号 = 内部 index + 1 (make_card の「カード {idx + 1}」)
        present = {sid["index"] for sid in ids}
        labels = [i + 1 for i in (card_order or []) if i in present]
        windows = _build_windows(pools, boundaries, labels)

        t0 = time.perf_counter()
        dist = build_accum_dist(hm, windows)
        t_grid = time.perf_counter() - t0
        base = build_accum_dist(hm, []) if (show_base or show_shape) else None
        n_mc = int(n_mc or 0)
        t0 = time.perf_counter()
        mc = _mc(hm, windows, n_mc, int(seed or 0)) if n_mc > 0 else None
        t_mc = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 — 入力ミスはそのまま画面に出す
        return html.Div(f"エラー: {e!s}", style={"color": "red", "whiteSpace": "pre-wrap"})

    D = float(target) if target else float(_quantile(dist, [0.5])[0])

    # --- 要約 --------------------------------------------------------------
    def diff(a: float, b: float, prob: bool = False) -> str:
        return f"{a - b:+.6f}" if prob else f"{a - b:+,.0f}"

    rows = [["平均", _fmt(dist.mean), _fmt(mc.mean()) if mc is not None else "-",
             _fmt(base.mean) if base else "-", diff(dist.mean, base.mean) if base else "-"],
            ["標準偏差", _fmt(np.sqrt(dist.var)), _fmt(mc.std()) if mc is not None else "-",
             _fmt(np.sqrt(base.var)) if base else "-",
             diff(np.sqrt(dist.var), np.sqrt(base.var)) if base else "-"],
            [f"P(T ≥ {_fmt(D)})", f"{dist.pass_prob(D):.6f}",
             f"{(mc >= D).mean():.6f}" if mc is not None else "-",
             f"{base.pass_prob(D):.6f}" if base else "-",
             diff(dist.pass_prob(D), base.pass_prob(D), prob=True) if base else "-"]]
    q_acc = _quantile(dist, QUANTILES)
    q_base = _quantile(base, QUANTILES) if base else None
    for j, q in enumerate(QUANTILES):
        rows.append([f"{q:.0%} 点", _fmt(q_acc[j]),
                     _fmt(np.quantile(mc, q)) if mc is not None else "-",
                     _fmt(q_base[j]) if base else "-",
                     diff(q_acc[j], q_base[j]) if base else "-"])
    summary = _table(["", "グリッド", "MC", "蓄積なし", "差 (あり − なし)"], rows)

    note = f"Hit数 {len(hm)} / グリッド {t_grid:.2f}s"
    if mc is not None:
        err = float(np.abs(dist.cdf(np.quantile(mc, QUANTILES)) - np.array(QUANTILES)).max())
        note += (f" / MC {n_mc:,} 本 {t_mc:.2f}s / 分位点での |CDF 差| 最大 {err:.2e}"
                 f" (MC のばらつき ~{1 / np.sqrt(n_mc):.1e})")

    # --- プール診断 ---------------------------------------------------------
    win_rows = []
    for st in dist.window_stats:
        win_rows.append([st.name, f"{len(st.hits)}", f"{st.rate:.0%}", _fmt(st.damage_mean),
                         _fmt(st.cap_mean), f"{st.sat_prob:.3f}", _fmt(st.pool_mean),
                         _fmt(st.overflow_mean), _fmt(st.burst_mean),
                         f"{_fmt(st.burst_lo)} ~ {_fmt(st.burst_hi)}"])
    win_table = (_table(["プール", "Hit数", "率", "窓内ダメージ平均", "上限平均", "飽和確率",
                         "プール平均", "期待溢れ", "爆発平均", "爆発 10–90%"], win_rows)
                 if win_rows else html.Div("有効なプールがありません (蓄積なしと同じ)。",
                                           style={"color": "#888"}))

    tail, dens = _figures(dist, base, mc, D)
    compare = []
    if base is not None:
        fp, fq = _compare_figures(dist, base, D)
        compare = [html.H4("蓄積なしとの比較: 通過確率"), dcc.Graph(figure=fp),
                   html.H4("蓄積なしとの比較: 分位点ごとの上乗せダメージ"), dcc.Graph(figure=fq)]
    return html.Div([
        html.Div(note, style={"fontSize": "0.85rem", "color": "#555", "marginBottom": "6px"}),
        html.H4("要約"), summary,
        html.H4("プールごとの診断"), win_table,
        *compare,
        *(_shape_section(dist, base, hm, windows, int(seed or 0))
          if show_shape and dist.window_stats else []),
        html.H4("裾確率 P(T ≥ x)"), dcc.Graph(figure=tail),
        html.H4("密度"), dcc.Graph(figure=dens),
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8062))
    debug = "PORT" not in os.environ
    app.run(host="127.0.0.1", port=port, debug=debug)
