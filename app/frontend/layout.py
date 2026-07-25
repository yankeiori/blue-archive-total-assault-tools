import sys
from pathlib import Path

from dash import html, dcc

from app import OCR_ENABLED

# PyInstaller バンドル時は _MEIPASS、通常時はプロジェクトルート
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys._MEIPASS)
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent

_MANUAL_MD = (_BASE_DIR / "docs" / "manual.md").read_text(encoding="utf-8")

DEFAULT_CRIT_RATE = 60
DEFAULT_EVADE_RATE = 0

LABEL_STYLE = {"fontSize": "0.85rem", "whiteSpace": "nowrap"}


_DEFAULT_PARAMS = {
    "crit_min": 100000,
    "crit_max": 120000,
    "normal_min": 50000,
    "normal_max": 60000,
    "hits": 10,
    "crit_rate": None,
    "evade_rate": None,
    "enemies": 1,
    "hp_dep": [1],
}


def _hp_dep_checklist_value(v) -> list:
    """hp_dep パラメータ (Checklist の list / bool / int / None) を Checklist の
    value へ正規化する。未指定 (None) は従来互換で「HP依存」扱い。"""
    if v is None:
        return [1]
    if isinstance(v, (list, tuple)):
        return [1] if len(v) > 0 else []
    return [1] if v else []


def make_damage_card(
    index: int,
    crit_rate=None,
    evade_rate=None,
    *,
    params: dict | None = None,
    memo: str = "",
) -> html.Div:
    """ダメージデータ入力カードを1つ生成する。

    params が渡された場合はその値を使い、なければデフォルト値を使う。
    crit_rate / evade_rate は後方互換のために残す（params 未指定時のみ有効）。
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)
    else:
        if crit_rate is not None:
            p["crit_rate"] = crit_rate
        if evade_rate is not None:
            p["evade_rate"] = evade_rate

    def field(label, param, value):
        return html.Div(
            [
                html.Label(label, style=LABEL_STYLE),
                dcc.Input(
                    id={"type": "param", "param": param, "index": index},
                    type="number",
                    value=value,
                    style={"width": "100%"},
                ),
            ],
            style={"flex": "1", "minWidth": "100px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("⠿", className="drag-handle"),
                    html.Strong(f"ダメージ {index + 1}"),
                    dcc.Input(
                        id={"type": "memo", "index": index},
                        type="text",
                        placeholder="備考",
                        value=memo or "",
                        style={"marginLeft": "8px", "flex": "1", "fontSize": "0.85rem"},
                    ),
                    html.Button(
                        "📋",
                        id={"type": "duplicate-btn", "index": index},
                        n_clicks=0,
                        title="カードを複製",
                        style={
                            "marginLeft": "auto",
                            "background": "none",
                            "border": "none",
                            "cursor": "pointer",
                            "fontSize": "1.1rem",
                        },
                    ),
                    html.Button(
                        "✕",
                        id={"type": "remove-btn", "index": index},
                        n_clicks=0,
                        style={
                            "background": "none",
                            "border": "none",
                            "cursor": "pointer",
                            "fontSize": "1.1rem",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    field("会心ダメージ下限", "crit_min", p["crit_min"]),
                    html.Span("~", style={"alignSelf": "end", "paddingBottom": "4px"}),
                    field("会心ダメージ上限", "crit_max", p["crit_max"]),
                    field("非会心ダメージ下限", "normal_min", p["normal_min"]),
                    html.Span("~", style={"alignSelf": "end", "paddingBottom": "4px"}),
                    field("非会心ダメージ上限", "normal_max", p["normal_max"]),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    field("Hit数", "hits", p["hits"]),
                    field("会心率 (%)", "crit_rate", p["crit_rate"]),
                    field("回避率 (%)", "evade_rate", p["evade_rate"]),
                    field("敵の数", "enemies", p["enemies"]),
                    html.Div(
                        dcc.Checklist(
                            id={"type": "param", "param": "hp_dep", "index": index},
                            options=[{"label": " HP依存", "value": 1}],
                            value=_hp_dep_checklist_value(p.get("hp_dep")),
                            style={"fontSize": "0.85rem", "whiteSpace": "nowrap"},
                        ),
                        className="hp-dep-box",
                        title="このカードの与ダメージが敵の現在HPに比例するならチェック。"
                              "チェックしたカードのダメージ欄には「HP0時のダメージ」を入力して"
                              "ください(現在HPに応じた倍率が掛かります)。外すと通常ダメージとして扱われ、"
                              "混在時は混在モデルで計算します。",
                        style={"alignSelf": "end", "paddingBottom": "2px"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "6px"},
            ),
        ],
        id={"type": "card", "index": index},
        style={
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "padding": "12px",
            "marginBottom": "10px",
            "background": "#fafafa",
        },
    )


def _top_settings_panel() -> html.Div:
    """カード生成の上に配置する「目標ダメージ」「一括設定」パネル。"""
    box_style = {
        "flex": "1",
        "minWidth": "240px",
        "padding": "10px",
        "border": "1px solid #ddd",
        "borderRadius": "8px",
    }
    return html.Div(
        [
            # 目標ダメージ
            html.Div(
                [
                    html.Strong("目標ダメージ"),
                    dcc.Input(id="target-damage", type="number", value=1000000, style={"width": "100%", "marginTop": "6px"}),
                ],
                style={**box_style, "background": "#fff5f5"},
            ),
            # 一括設定
            html.Div(
                [
                    html.Strong("一括設定"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("会心率(%)", style=LABEL_STYLE),
                                    dcc.Input(id="global-crit-rate", type="number", value=DEFAULT_CRIT_RATE, style={"width": "100%"}),
                                ],
                                style={"flex": "1", "minWidth": "100px"},
                            ),
                            html.Div(
                                [
                                    html.Label("回避率(%)", style=LABEL_STYLE),
                                    dcc.Input(id="global-evade-rate", type="number", value=DEFAULT_EVADE_RATE, style={"width": "100%"}),
                                ],
                                style={"flex": "1", "minWidth": "100px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "marginTop": "6px"},
                    ),
                    html.Button("一括適用", id="apply-global-btn", n_clicks=0, style={"marginTop": "8px", "width": "100%"}),
                ],
                style={**box_style, "background": "#f5f5ff"},
            ),
        ],
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"},
    )


def _ocr_panel() -> html.Div:
    """スクリーンショット → カード自動生成パネル。"""
    return html.Div(
        [
            html.Strong("📷 スクショからカード生成", style={"fontSize": "0.95rem"}),
            html.Div(
                [
                    dcc.Upload(
                        id="ocr-upload",
                        children=html.Div(
                            "ここに画像をドロップ / クリックして選択",
                            style={"fontSize": "0.85rem", "color": "#555"},
                        ),
                        accept="image/*",
                        multiple=False,
                        style={
                            "width": "100%",
                            "boxSizing": "border-box",
                            "border": "2px dashed #4a90d9",
                            "borderRadius": "6px",
                            "padding": "10px",
                            "textAlign": "center",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "🖥 画面スニップ",
                        id="ocr-snip-btn",
                        n_clicks=0,
                        title="画面の一部をドラッグで範囲選択して取り込む",
                        style={
                            "background": "#4a90d9",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "8px 16px",
                            "cursor": "pointer",
                            "whiteSpace": "nowrap",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "10px",
                       "alignItems": "stretch", "marginTop": "8px"},
            ),
            dcc.Loading(
                html.Div(
                    id="ocr-status",
                    style={"fontSize": "0.82rem", "color": "#666", "marginTop": "6px", "minHeight": "1.2em"},
                ),
                type="dot",
                color="#4a90d9",
            ),
        ],
        style={
            "border": "1px solid #4a90d9",
            "borderRadius": "8px",
            "padding": "12px",
            "marginBottom": "16px",
            "background": "#f3f8ff",
        },
    )


def _text_panel() -> html.Div:
    """テキスト貼り付け → カード自動生成パネル。"""
    placeholder = (
        "ダメージ表示のテキストを貼り付け\n"
        "例:\n"
        "ヒット1-2 (165.33%)\n"
        "18,164 - 25,247\n"
        "会心\n"
        "35,239 - 48,979"
    )
    return html.Div(
        [
            html.Strong("📝 テキストからカード生成", style={"fontSize": "0.95rem"}),
            dcc.Textarea(
                id="text-input",
                placeholder=placeholder,
                style={
                    "width": "100%",
                    "boxSizing": "border-box",
                    "height": "120px",
                    "marginTop": "8px",
                    "fontFamily": "monospace",
                    "fontSize": "0.82rem",
                    "resize": "vertical",
                },
            ),
            html.Button(
                "テキストから取り込み",
                id="text-import-btn",
                n_clicks=0,
                style={
                    "background": "#7c5cd9",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "6px",
                    "padding": "8px 16px",
                    "cursor": "pointer",
                    "width": "100%",
                    "marginTop": "8px",
                },
            ),
            html.Div(
                id="text-status",
                style={"fontSize": "0.82rem", "color": "#666", "marginTop": "6px", "minHeight": "1.2em"},
            ),
        ],
        style={
            "border": "1px solid #7c5cd9",
            "borderRadius": "8px",
            "padding": "12px",
            "marginBottom": "16px",
            "background": "#f7f3ff",
        },
    )


def _io_panel() -> html.Div:
    """入力情報のエクスポート / インポートパネル(カード + 全体設定 + 多段リスタ設定)。"""
    return html.Div(
        [
            html.Strong("💾 入力の保存 / 読込", style={"fontSize": "0.95rem"}),
            html.Div(
                [
                    html.Button(
                        "⬇ エクスポート (JSON)",
                        id="export-btn",
                        n_clicks=0,
                        title="現在の全カード・全体設定・足切りライン最適化の設定を JSON で保存",
                        style={
                            "background": "#2d8659", "color": "white", "border": "none",
                            "borderRadius": "6px", "padding": "8px 16px", "cursor": "pointer",
                            "whiteSpace": "nowrap",
                        },
                    ),
                    dcc.Upload(
                        id="import-upload",
                        children=html.Div(
                            "⬆ インポート: JSON をドロップ / クリックして選択",
                            style={"fontSize": "0.85rem", "color": "#555"},
                        ),
                        accept=".json,application/json",
                        multiple=False,
                        style={
                            "width": "100%", "boxSizing": "border-box", "border": "2px dashed #2d8659",
                            "borderRadius": "6px", "padding": "10px", "textAlign": "center",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "10px",
                       "alignItems": "stretch", "marginTop": "8px"},
            ),
            html.Div(
                id="io-status",
                style={"fontSize": "0.82rem", "color": "#666", "marginTop": "6px", "minHeight": "1.2em"},
            ),
            dcc.Download(id="export-download"),
        ],
        style={
            "border": "1px solid #2d8659", "borderRadius": "8px", "padding": "12px",
            "marginBottom": "16px", "background": "#f1faf4",
        },
    )


def _sidebar() -> html.Div:
    """左サイドバー: 設定パネル。"""
    section_style = {
        "marginBottom": "16px",
        "padding": "10px",
        "border": "1px solid #ddd",
        "borderRadius": "8px",
    }
    def hp_field(label, hid, value):
        return html.Div(
            [
                html.Label(label, style=LABEL_STYLE),
                dcc.Input(id=hid, type="number", value=value, style={"width": "100%"}),
            ],
            style={"marginTop": "6px"},
        )

    # --- 積or和モデル (HP依存ダメージ) ---
    hp_section = html.Div(
        [
            html.Strong("HP依存ダメージ"),
            dcc.RadioItems(
                id="hp-mode",
                options=[
                    {"label": "なし（合計＝和モデル）", "value": "off"},
                    {"label": "あり（ミカ型。カード別に混在可）", "value": "on"},
                ],
                value="off",
                style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginTop": "6px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            "⚠ HP依存カードのダメージ欄には",
                            html.Strong("「HP0時のダメージ」"),
                            "を入力してください。入力値に倍率(現在HPに応じて R0〜R1)が掛かります。",
                        ],
                        style={"fontSize": "0.8rem", "color": "#b35900",
                               "fontWeight": "bold", "background": "#fff3e0",
                               "border": "1px solid #f0b060", "borderRadius": "6px",
                               "padding": "6px 8px", "marginTop": "6px"},
                    ),
                    hp_field("敵の最大HP", "hp-H", 1000000),
                    hp_field("開始時HP", "hp-H1", 1000000),
                    hp_field("HP満タン時の倍率 (R1)", "hp-R1", 2),
                    hp_field("HP0時の倍率 (R0)", "hp-R0", 1),
                ],
                id="hp-params",
                style={"display": "none", "marginTop": "4px"},
            ),
        ],
        style={**section_style, "background": "#fff7ec"},
    )

    # --- その他: 計算方式 ---
    calc_section = html.Div(
        [
            html.Strong("計算方式"),
            dcc.RadioItems(
                id="calc-method",
                options=[
                    {"label": "COS法（準厳密・推奨）", "value": "cos"},
                    {"label": "モンテカルロ", "value": "mc"},
                ],
                value="cos",
                style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginTop": "6px"},
            ),
        ],
        style={**section_style, "background": "#eef6ff"},
    )

    # --- その他: ダメージ生成モード ---
    damage_section = html.Div(
        [
            html.Strong("ダメージ生成モード"),
            dcc.RadioItems(
                id="damage-mode",
                options=[
                    {"label": "減衰考慮済み（推奨）", "value": "post_decay"},
                    {"label": "減衰考慮前", "value": "pre_decay"},
                ],
                value="post_decay",
                style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginTop": "6px"},
            ),
        ],
        style={**section_style, "background": "#f0f8f0"},
    )

    # --- その他: 安定値 (全体設定) ---
    stability_section = html.Div(
        [
            html.Strong("安定値"),
            dcc.Input(
                id="global-stability",
                type="number",
                value=None,
                placeholder="未入力で無効",
                style={"width": "100%", "marginTop": "6px"},
            ),
            html.Div(
                "最大ダメージが上限(10,966,999)に張り付く場合のみ使用。最小ダメージと"
                "安定値から最大ダメージを逆算します(未入力なら通常計算)。",
                style={"fontSize": "0.72rem", "color": "#888", "marginTop": "6px"},
            ),
        ],
        style={**section_style, "background": "#f3f0ff"},
    )

    return html.Div(
        [
            # 積or和モデル → スクショ → テキスト → 保存/読込 → 安定値 → 計算方式 → 生成モード
            hp_section,
            # OCR (スクショ→カード) はローカル専用。外部公開時は非表示。
            *([_ocr_panel()] if OCR_ENABLED else []),
            _text_panel(),
            _io_panel(),
            stability_section,
            calc_section,
            damage_section,
        ],
        id="sim-sidebar",
        className="sim-sidebar",
        style={
            "width": "260px",
            "flexShrink": "0",
            "position": "sticky",
            "top": "20px",
            "alignSelf": "flex-start",
        },
    )


def _restart_page() -> html.Div:
    """多段リスタ(複数足切り関門)スループット最適化ページ。スライダーは使わず、
    カードごとにチェックポイント指定・時間割合を設定して最適足切りをサーバ計算する。"""
    return html.Div(
        [
            html.H3("足切りライン最適化", style={"marginTop": "0"}),
            html.P(
                "「シミュレータ」で設定した攻撃列を使い、複数チェックポイントで"
                "リセットする運用の最適足切りラインを計算します。各関門は "
                "「コスト/成功 = 期待時間/成功確率」を最小化する Bermudan 後ろ向き帰納で"
                "決定(HP依存=積モデルにも対応)。",
                style={"fontSize": "0.88rem", "color": "#555"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button("カード読込 / 更新", id="restart-reload-btn",
                                        n_clicks=0,
                                        style={"cursor": "pointer", "padding": "6px 12px"}),
                            html.Span(" カードを変更したら押してください",
                                      style={"fontSize": "0.8rem", "color": "#888",
                                             "marginLeft": "8px"}),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        "攻撃列(参考)です。足切り(チェックポイント)で区切られた"
                        "区間ごとに所要時間の割合を入力します。",
                        style={"fontSize": "0.82rem", "color": "#666", "marginBottom": "8px"},
                    ),
                    html.Div(id="restart-cards-table"),
                    html.Div(
                        [
                            html.Label("足切り(チェックポイント)を追加", style=LABEL_STYLE),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="restart-cp-dropdown",
                                        options=[],
                                        placeholder="チェックポイントにするカードを選択",
                                        style={"flex": "1", "minWidth": "260px"},
                                    ),
                                    html.Button("+ 追加", id="restart-cp-add-btn",
                                                n_clicks=0,
                                                style={"cursor": "pointer",
                                                       "padding": "6px 14px",
                                                       "whiteSpace": "nowrap"}),
                                ],
                                style={"display": "flex", "gap": "8px",
                                       "alignItems": "center", "marginTop": "4px"},
                            ),
                            html.Div(
                                [
                                    "足切りを追加すると区間カードが分割されます。各区間カードに"
                                    "「時間割合(相対)」を入力してください(コスト = 期待時間 / 成功確率)。"
                                    "区間カードの「✕」でその足切りを解除できます。",
                                    html.Br(),
                                    html.Span(
                                        "🎲 ダメージ外 成功率% ",
                                        style={"color": "#0984e3", "fontWeight": "bold"}),
                                    html.Span(
                                        "= ダメージ(足切り)とは無関係な成功要因の確率。"
                                        "区間を回しきって次へ進める割合で、失敗するとリスタート"
                                        "(その区間の所要時間は消費)。100%なら従来どおりダメージ"
                                        "足切りのみで判定します。",
                                        style={"color": "#0984e3"}),
                                ],
                                style={"fontSize": "0.8rem", "color": "#888",
                                       "margin": "8px 0 6px"},
                            ),
                            html.Div(id="restart-cp-cards", style={"marginTop": "8px"}),
                        ],
                        style={"marginTop": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label("目標ダメージ D", style=LABEL_STYLE),
                            dcc.Input(id="restart-D", type="number", value=1_000_000,
                                      step=100_000,
                                      style={"width": "200px", "marginLeft": "8px"}),
                        ],
                        style={"marginTop": "12px"},
                    ),
                    html.Button("解析実行", id="restart-run-btn", n_clicks=0,
                                style={"background": "#d63031", "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "padding": "8px 18px", "cursor": "pointer",
                                       "fontWeight": "bold", "marginTop": "12px"}),
                ],
                style={"background": "#fff0f0", "border": "2px solid #d63031",
                       "borderRadius": "8px", "padding": "14px", "marginBottom": "16px"},
            ),
            dcc.Loading(
                [
                    html.Div(id="restart-summary",
                             style={"fontSize": "0.95rem", "marginBottom": "10px"}),
                    dcc.Graph(id="restart-graph"),
                ],
                type="circle", color="#d63031",
            ),
            # --- リスタライン手動調整 (スライダーで確率の変化を確認) ---
            html.Div(
                [
                    html.H4("リスタラインを手で調整して確率の変化を見る",
                            style={"marginBottom": "4px"}),
                    html.Div(
                        "「解析実行」後に表示されます。各足切りの残りダメージを"
                        "スライダーで動かすと、成功率・スループットが再計算され、"
                        "最適ライン(灰点線)と重ねて表示されます。",
                        style={"fontSize": "0.82rem", "color": "#666",
                               "marginBottom": "10px"},
                    ),
                    html.Div(id="restart-gate-sliders"),
                    dcc.Loading(
                        [
                            html.Div(id="restart-interactive-summary",
                                     style={"fontSize": "0.92rem", "margin": "6px 0"}),
                            dcc.Graph(id="restart-interactive-graph"),
                        ],
                        type="circle", color="#0984e3",
                    ),
                ],
                style={"background": "#f0f7ff", "border": "2px solid #0984e3",
                       "borderRadius": "8px", "padding": "14px", "marginTop": "16px"},
            ),
            dcc.Store(id="restart-config", data=None),
        ],
        style={"maxWidth": "900px"},
    )


# ---------------------------------------------------------------------------
# スキル順探索ページ
# ---------------------------------------------------------------------------
SO_N_SKILLS = 6
_SO_DEFAULT_NAMES = ["", "", "", "", "", ""]


def so_skill_options(names: list, copiers: set) -> list:
    """手順ステップの「使うカード」ドロップダウン選択肢。

    value 形式: "any" / "n{i}" (スキルiの元カード) / "c{i}" (スキルiのコピー)。
    名前を後から変えても添字参照なので選択は維持される。
    """
    def nm(i):
        return (names[i] or "").strip() or f"カード{i + 1}"

    opts = [{"label": "指定なし(何でも)", "value": "any"}]
    for i in range(len(names)):
        suffix = " ※複製スキル" if i in copiers else ""
        opts.append({"label": nm(i) + suffix, "value": f"n{i}"})
    if copiers:
        for i in range(len(names)):
            if i not in copiers:
                opts.append({"label": f"{nm(i)}(コピー)", "value": f"c{i}"})
    return opts


def so_target_options(names: list, copiers: set) -> list:
    """複製対象ドロップダウンの選択肢(複製キャラ自身は対象外)。"""
    return [
        {"label": (names[i] or "").strip() or f"カード{i + 1}", "value": str(i)}
        for i in range(len(names)) if i not in copiers
    ]


def make_so_step(index: int, skill_options: list, target_options: list, *,
                 skill=None, target=None, slot: str = "any",
                 draw: bool = False, memo: str = "") -> html.Div:
    """手順(PLAN)の1ステップ行を生成する。行の並び順 = 手順の順番。

    skill が None(未選択)の行は実行時に無視される。
    """
    return html.Div(
        [
            dcc.Dropdown(
                id={"type": "so-step-skill", "index": index},
                options=skill_options,
                value=skill,
                placeholder="生徒を選択",
                clearable=False,
                className="so-dd-skill",
                style={"width": "160px", "flexShrink": "0"},
            ),
            # 複製対象: 複製スキルを選択した行でのみコールバックが表示する
            dcc.Dropdown(
                id={"type": "so-step-target", "index": index},
                options=target_options,
                value=target,
                placeholder="複製対象",
                clearable=True,
                style={"width": "130px", "flexShrink": "0", "display": "none"},
            ),
            html.Span("枠", style={"fontSize": "0.85rem", "flexShrink": "0",
                                   "marginLeft": "2px"}),
            dcc.Dropdown(
                id={"type": "so-step-slot", "index": index},
                options=[
                    {"label": "任意", "value": "any"},
                    {"label": "左", "value": "1"},
                    {"label": "中", "value": "2"},
                    {"label": "右", "value": "3"},
                ],
                value=slot,
                clearable=False,
                searchable=False,
                style={"width": "80px", "flexShrink": "0"},
            ),
            dcc.Checklist(
                id={"type": "so-step-draw", "index": index},
                options=[{"label": "ドロー", "value": "draw"}],
                value=["draw"] if draw else [],
                style={"whiteSpace": "nowrap", "fontSize": "0.85rem"},
            ),
            dcc.Input(
                id={"type": "so-step-memo", "index": index},
                type="text",
                value=memo,
                placeholder="メモ",
                style={"flex": "1", "minWidth": "80px",
                       "fontSize": "0.85rem"},
            ),
            html.Button("✕", id={"type": "so-step-remove", "index": index},
                        n_clicks=0, title="このステップを削除",
                        className="so-mini-btn"),
        ],
        id={"type": "so-step", "index": index},
        className="so-step-row",
    )


def make_so_constraint(index: int, *, ctype: str = "diff", steps: str = "") -> html.Div:
    """手順間制約の1行を生成する。"""
    return html.Div(
        [
            dcc.Dropdown(
                id={"type": "so-con-type", "index": index},
                options=[
                    {"label": "別スロットにする", "value": "diff"},
                    {"label": "同じスロットにする", "value": "same"},
                ],
                value=ctype,
                clearable=False,
                style={"width": "190px"},
            ),
            html.Span("対象手順:", style={"fontSize": "0.85rem", "whiteSpace": "nowrap"}),
            dcc.Input(
                id={"type": "so-con-steps", "index": index},
                type="text",
                value=steps,
                placeholder="手順番号をカンマ区切り 例: 1,3",
                style={"flex": "1", "minWidth": "140px"},
            ),
            html.Button("✕", id={"type": "so-con-remove", "index": index},
                        n_clicks=0, title="この制約を削除", className="so-mini-btn"),
        ],
        id={"type": "so-con", "index": index},
        style={"display": "flex", "gap": "8px", "alignItems": "center",
               "marginBottom": "6px"},
    )


def _skill_order_page() -> html.Div:
    """スキル順(開始スキル設定)探索ページ。"""
    initial_skill_opts = so_skill_options(_SO_DEFAULT_NAMES, set())
    initial_target_opts = so_target_options(_SO_DEFAULT_NAMES, set())

    skill_rows = []
    for i in range(SO_N_SKILLS):
        skill_rows.append(
            html.Div(
                [
                    html.Span(f"カード{i + 1}", style={"width": "62px", "flexShrink": "0",
                                                      "fontSize": "0.85rem"}),
                    dcc.Input(
                        id={"type": "so-name", "index": i},
                        type="text",
                        value=_SO_DEFAULT_NAMES[i],
                        placeholder=f"生徒名{i + 1}",
                        maxLength=12,
                        style={"width": "110px", "flexShrink": "0"},
                    ),
                    dcc.Checklist(
                        id={"type": "so-copier", "index": i},
                        options=[{"label": "複製スキル", "value": "copier"}],
                        value=[],
                        style={"whiteSpace": "nowrap", "fontSize": "0.82rem"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center",
                       "width": "calc(33.3% - 9px)", "minWidth": "240px"},
            )
        )

    return html.Div(
        [
            html.H3("スキル順探索(β版)", style={"marginTop": "0"}),
            html.P(
                "使いたいスキル順(手順)を満たす「開始スキル設定(手札3枚+山札3枚の初期配置)」"
                "を全探索します。カードは6枚固定で、カードを使うと山札の一番下へ行き、"
                "山札の一番上がドローされます。",
                style={"fontSize": "0.88rem", "color": "#555"},
            ),
            # --- カード(スキル)設定 ---
            html.Div(
                [
                    html.Strong("カード設定(6枚)"),
                    html.Div(
                        "キャラ名を入力してください。複製スキル持ち(対象を指定して撃つと"
                        "自分のカードが対象のコピーに変化するキャラ)は「複製スキル」に"
                        "チェックを入れます。",
                        style={"fontSize": "0.8rem", "color": "#888", "margin": "4px 0 8px"},
                    ),
                    html.Div(skill_rows,
                             style={"display": "flex", "flexWrap": "wrap", "gap": "6px 12px"}),
                ],
                style={"border": "1px solid #ddd", "borderRadius": "8px",
                       "padding": "12px", "marginBottom": "14px", "background": "#f7fbff"},
            ),
            # --- 手順 (PLAN) ---
            html.Div(
                [
                    html.Strong("手順(使いたいスキル順)"),
                    html.Div(
                        [
                            "上から順にスキルを使います。生徒を選ぶと自動で次の行が"
                            "追加されます。未選択(空)の行は無視されます。"
                            "「指定なし」は繋ぎの1枚"
                            "(何を使ってもよい。ただし複製スキルの元カードは複製対象を"
                            "決められないため使いません)。",
                            html.Br(),
                            "複製スキルのカードを選んだ場合は「複製対象」を指定してください。"
                            "撃つとその場でカードが「対象(コピー)」に変化します"
                            "(山札への移動・ドローなし)。コピーを使う手順は"
                            "「◯◯(コピー)」を選びます。使用後はカードが複製キャラに戻って"
                            "山札の一番下へ行きます。",
                            html.Br(),
                            html.Span("ドロー", style={"fontWeight": "bold"}),
                            " = そのステップでスキルカードのドローが発生する場合にチェック。"
                            "使用カードが山札へ行った後、そのスキルの元カードが山札にあれば"
                            "同じスロットに引き抜かれます(無ければ通常ドロー)。",
                        ],
                        style={"fontSize": "0.8rem", "color": "#888", "margin": "4px 0 8px"},
                    ),
                    html.Div(id="so-steps-container",
                             children=[make_so_step(0, initial_skill_opts, initial_target_opts)]),
                    html.Button("+ ステップ追加", id="so-add-step-btn", n_clicks=0,
                                style={"marginTop": "6px"}),
                ],
                style={"border": "1px solid #ddd", "borderRadius": "8px",
                       "padding": "12px", "marginBottom": "14px", "background": "#fffdf5"},
            ),
            # --- 制約 ---
            html.Div(
                [
                    html.Strong("手順間の制約(任意)"),
                    html.Div(
                        "「1,3」のように手順番号(1始まり)をカンマ区切りで指定すると、"
                        "その手順どうしを別スロット / 同じスロットに限定できます。"
                        "裏にあるスキルの後掛けを回避したい場合などに設定してください。",
                        style={"fontSize": "0.8rem", "color": "#888", "margin": "4px 0 8px"},
                    ),
                    html.Div(id="so-cons-container", children=[]),
                    html.Button("+ 制約追加", id="so-add-con-btn", n_clicks=0,
                                style={"marginTop": "6px"}),
                ],
                style={"border": "1px solid #ddd", "borderRadius": "8px",
                       "padding": "12px", "marginBottom": "14px", "background": "#f5fff7"},
            ),
            # --- 実行 ---
            html.Div(
                [
                    html.Label("表示件数上限", style=LABEL_STYLE),
                    dcc.Input(id="so-limit", type="number", value=60, min=1, max=1000,
                              style={"width": "90px", "margin": "0 16px 0 8px"}),
                    html.Button("探索実行", id="so-run-btn", n_clicks=0,
                                style={"background": "#d63031", "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "padding": "8px 18px", "cursor": "pointer",
                                       "fontWeight": "bold"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "12px"},
            ),
            dcc.Loading(
                html.Div(id="so-results"),
                type="circle", color="#d63031",
            ),
            # 手順行の表示順 (step index のリスト)
            dcc.Store(id="so-step-order", data=[0]),
            dcc.Store(id="so-next-step", data=1),
            dcc.Store(id="so-next-con", data=0),
        ],
        style={"maxWidth": "900px"},
    )


def _nav_bar() -> html.Div:
    # 初期表示は「シミュレータ」ページ (= active)
    return html.Div(
        [
            html.Button("📊 シミュレータ", id="nav-sim", n_clicks=0,
                        className="nav-btn active"),
            html.Button("🎯 足切りライン最適化", id="nav-restart", n_clicks=0,
                        className="nav-btn", style={"marginLeft": "6px"}),
            html.Button("🃏 スキル順探索(β版)", id="nav-skill", n_clicks=0,
                        className="nav-btn", style={"marginLeft": "6px"}),
        ],
        className="nav-bar",
        style={"display": "flex", "marginBottom": "14px", "gap": "0",
               "borderBottom": "2px solid #d63031", "paddingBottom": "0"},
    )


def create_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("ブルアカダメージ足切り最適化", style={"marginBottom": "0"}),
                    html.Div(
                        [
                            html.Button(
                                "📖 マニュアル",
                                id="open-manual-btn",
                                n_clicks=0,
                                style={
                                    "background": "#4a90d9",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "padding": "6px 16px",
                                    "cursor": "pointer",
                                    "fontSize": "0.9rem",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                            
                            html.Span(
                                [
                                    "不具合報告、要望などは",
                                    html.A(
                                        "こちら",
                                        href="https://x.com/yankeiori",
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        style={"color": "#4a90d9"},
                                    ),
                                    "まで",
                                ],
                                style={
                                    "fontSize": "0.85rem",
                                    "color": "#555",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                        style={"marginLeft": "auto", "display": "flex", "alignItems": "center", "gap": "12px"},
                    ),
                ],
                className="app-header",
                style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                       "gap": "8px", "marginBottom": "16px"},
            ),
            # マニュアルモーダル
            html.Div(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong("マニュアル", style={"fontSize": "1.2rem"}),
                                html.Button(
                                    "✕",
                                    id="close-manual-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "auto",
                                        "background": "none",
                                        "border": "none",
                                        "cursor": "pointer",
                                        "fontSize": "1.3rem",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "borderBottom": "1px solid #ddd",
                                "paddingBottom": "8px",
                                "marginBottom": "12px",
                            },
                        ),
                        dcc.Markdown(_MANUAL_MD, style={"overflowY": "auto", "flex": "1"}),
                    ],
                    className="manual-modal-content",
                ),
                id="manual-modal",
                className="manual-modal-overlay",
                style={"display": "none"},
            ),
            _nav_bar(),
            html.Div(
                html.Div(
                [
                    html.Button("≡", id="sidebar-toggle", n_clicks=0,
                                title="設定パネルの表示/非表示",
                                style={"alignSelf": "flex-start", "flexShrink": "0",
                                       "border": "1px solid #ccc", "background": "#f4f4f4",
                                       "borderRadius": "4px", "padding": "6px 10px",
                                       "cursor": "pointer", "fontSize": "1.1rem"}),
                    # サイドバー
                    _sidebar(),
                    # メインコンテンツ
                    html.Div(
                        [
                            _top_settings_panel(),
                            html.Div(id="cards-container", children=[make_damage_card(0, DEFAULT_CRIT_RATE, DEFAULT_EVADE_RATE)]),
                            html.Div(
                                [
                                    html.Button("+ ダメージ追加", id="add-btn", n_clicks=0),
                                    html.Button(
                                        "シミュレーション実行",
                                        id="run-btn",
                                        n_clicks=0,
                                        style={"marginLeft": "12px"},
                                    ),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                            dcc.Loading(
                                [
                                    html.Div(id="pass-rate-text", style={"fontSize": "1.2rem", "fontWeight": "bold", "marginBottom": "8px"}),
                                    dcc.Store(id="cdf-table-store"),
                                    html.Div(
                                        [
                                            html.Div(
                                                "通過確率 ⇄ ダメージ 変換",
                                                style={"fontWeight": "bold", "marginBottom": "6px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("ダメージ", style={"marginRight": "6px"}),
                                                    dcc.Input(
                                                        id="conv-damage-input",
                                                        type="number",
                                                        placeholder="ダメージ",
                                                        style={"width": "260px"},
                                                    ),
                                                    html.Span(" → 通過確率: ", style={"margin": "0 6px"}),
                                                    html.Span(id="conv-prob-output", children="—", style={"fontWeight": "bold"}),
                                                ],
                                                style={"marginBottom": "4px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("通過確率 (%)", style={"marginRight": "6px"}),
                                                    dcc.Input(
                                                        id="conv-prob-input",
                                                        type="number",
                                                        placeholder="%",
                                                        min=0,
                                                        max=100,
                                                        style={"width": "140px"},
                                                    ),
                                                    html.Span(" → ダメージ: ", style={"margin": "0 6px"}),
                                                    html.Span(id="conv-damage-output", children="—", style={"fontWeight": "bold"}),
                                                ],
                                            ),
                                        ],
                                        style={
                                            "border": "1px solid #ccc",
                                            "borderRadius": "6px",
                                            "padding": "8px 12px",
                                            "marginBottom": "12px",
                                            "background": "#f7f7f7",
                                        },
                                    ),
                                    dcc.Graph(id="result-graph"),
                                    dcc.Graph(
                                        id="result-cdf-graph",
                                        figure={
                                            "data": [],
                                            "layout": {
                                                "title": "それ以上になる確率 P(D≥x)",
                                                "xaxis": {"title": {"text": "合計ダメージ"}},
                                                "yaxis": {"title": {"text": "それ以上になる確率 (%)"}, "range": [0, 100]},
                                            },
                                        },
                                    ),
                                ],
                                type="circle",
                                color="#4a90d9",
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                className="main-flex",
                style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
                ),
                id="page-sim",
            ),
            html.Div(_restart_page(), id="page-restart", style={"display": "none"}),
            html.Div(_skill_order_page(), id="page-skill", style={"display": "none"}),
            # 非表示 Store 群
            dcc.Store(id="drag-order", data=""),
            dcc.Store(id="card-indices", data=[0]),
            dcc.Store(id="sorted-indices", data=[]),
            dcc.Store(id="next-index", data=1),
            # スニップした画像 (data URL) を JS から受け取る
            dcc.Store(id="ocr-image-store", data=None),
            # 多段リスタ: 選択済みチェックポイント (累積ヒット数のリスト)
            dcc.Store(id="restart-cp-store", data=[]),
            # 多段リスタ: 区間ごとの時間割合 (区間開始境界の累積ヒット数 → 相対重み)
            dcc.Store(id="restart-seg-time-store", data={"0": 1.0}),
            # 多段リスタ: 区間ごとのダメージ独立成功確率 % (区間開始境界 → 0..100)
            dcc.Store(id="restart-seg-success-store", data={"0": 100.0}),
            # 多段リスタ: 総ヒット数 (区間描画用)
            dcc.Store(id="restart-nhits", data=0),
        ],
        className="app-root",
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px", "fontFamily": "sans-serif"},
    )
