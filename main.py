import os

from dash import ALL, Input, Output, State
from app import app as application, OCR_ENABLED
from app.frontend.layout import create_layout
import app.frontend.callbacks  # noqa: F401 - コールバック登録
# 入力の自動保存 / 全クリア。callbacks.py と出力が重なるので後に登録する。
import app.frontend.persist  # noqa: F401,E402 - コールバック登録

application.layout = create_layout()

# ===========================================================================
# クライアントサイドコールバック (assets/simulation.js の関数を参照)
# ===========================================================================

# --- ドラッグ順同期 ---
application.clientside_callback(
    "dash_clientside.sim.syncDragOrder",
    Output("sorted-indices", "data"),
    Input("drag-order", "data"),
    State("card-indices", "data"),
    prevent_initial_call=True,
)

# --- 一括適用 ---
application.clientside_callback(
    "dash_clientside.sim.applyGlobal",
    Output({"type": "param", "param": "crit_rate", "index": ALL}, "value"),
    Output({"type": "param", "param": "evade_rate", "index": ALL}, "value"),
    Input("apply-global-btn", "n_clicks"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State({"type": "param", "param": "crit_rate", "index": ALL}, "value"),
    prevent_initial_call=True,
)

# --- シミュレーション実行 ---
application.clientside_callback(
    "dash_clientside.sim.runSimulation",
    Output("result-graph", "figure"),
    Output("pass-rate-text", "children"),
    Output("result-cdf-graph", "figure"),
    Output("cdf-table-store", "data"),
    Output("conv-damage-input", "value"),
    Output("accum-summary", "children"),
    Input("run-btn", "n_clicks"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("global-stability", "value"),
    State("target-damage", "value"),
    State("damage-mode", "value"),
    State("calc-method", "value"),
    State("hp-mode", "value"),
    State("hp-H", "value"),
    State("hp-H1", "value"),
    State("hp-R0", "value"),
    State("hp-R1", "value"),
    # 蓄積 (チャージ) 型スキルの設定一式 (assets/cos_accumulate.js が使う)
    State({"type": "accum", "field": ALL, "index": ALL}, "value"),
    State({"type": "accum", "field": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)

# --- 通過確率 ⇄ ダメージ 変換 ---
application.clientside_callback(
    "dash_clientside.sim.damageToProb",
    Output("conv-prob-output", "children"),
    Input("conv-damage-input", "value"),
    Input("cdf-table-store", "data"),
    prevent_initial_call=True,
)

application.clientside_callback(
    "dash_clientside.sim.probToDamage",
    Output("conv-damage-output", "children"),
    Input("conv-prob-input", "value"),
    Input("cdf-table-store", "data"),
    prevent_initial_call=True,
)

# --- マニュアルモーダル開閉 ---
application.clientside_callback(
    "dash_clientside.sim.toggleManualModal",
    Output("manual-modal", "style"),
    Input("open-manual-btn", "n_clicks"),
    Input("close-manual-btn", "n_clicks"),
    prevent_initial_call=True,
)

# --- HP依存パラメータ入力欄の表示切替 ---
application.clientside_callback(
    "dash_clientside.sim.toggleHpParams",
    Output("hp-params", "style"),
    Input("hp-mode", "value"),
)

# --- HP依存モード「なし」のときカードの HP依存チェックを非表示 (CSS クラス切替) ---
application.clientside_callback(
    "function(m) { return m === 'on' ? '' : 'hp-mode-off'; }",
    Output("cards-container", "className"),
    Input("hp-mode", "value"),
)

# --- 画面スニップ (getDisplayMedia → 範囲選択 → ocr-image-store) ---
# OCR はローカル専用。外部公開時 (ENABLE_OCR=false) は登録しない。
if OCR_ENABLED:
    application.clientside_callback(
        "dash_clientside.sim.snipScreen",
        Output("ocr-image-store", "data"),
        Input("ocr-snip-btn", "n_clicks"),
        prevent_initial_call=True,
    )

# --- ページ切替 (シミュレータ / 多段リスタ解析 / スキル順探索) ---
application.clientside_callback(
    """
    function(nsim, nrest, nskill) {
        var ctx = window.dash_clientside.callback_context;
        var trig = (ctx.triggered && ctx.triggered.length) ? ctx.triggered[0].prop_id : '';
        var page = 'sim';
        if (trig.indexOf('nav-restart') === 0) page = 'restart';
        else if (trig.indexOf('nav-skill') === 0) page = 'skill';
        return [
            {display: page === 'sim' ? 'block' : 'none'},
            {display: page === 'restart' ? 'block' : 'none'},
            {display: page === 'skill' ? 'block' : 'none'},
            page === 'sim' ? 'nav-btn active' : 'nav-btn',
            page === 'restart' ? 'nav-btn active' : 'nav-btn',
            page === 'skill' ? 'nav-btn active' : 'nav-btn'
        ];
    }
    """,
    Output("page-sim", "style"),
    Output("page-restart", "style"),
    Output("page-skill", "style"),
    Output("nav-sim", "className"),
    Output("nav-restart", "className"),
    Output("nav-skill", "className"),
    Input("nav-sim", "n_clicks"),
    Input("nav-restart", "n_clicks"),
    Input("nav-skill", "n_clicks"),
    prevent_initial_call=True,
)

# --- 目標ダメージの双方向同期 (シミュレータ <-> 多段リスタ) ---
application.clientside_callback(
    """
    function(simVal, restartVal) {
        var ctx = window.dash_clientside.callback_context;
        var nu = window.dash_clientside.no_update;
        // 双方向同期は Dash のコールバックグラフ上ただ 1 つの循環なので、
        // 「同値なら何も返さない」で確実に止める必要がある。素の値を === で
        // 比べるだけだと、片方が '' もう片方が null (数値入力の空欄) のように
        // 表現が食い違ったまま互いを書き換え続け、React の更新上限
        // (Maximum update depth exceeded) に達する。値を正規化して比較し、
        // 書き込む値も正規化後のものにして、1 往復で必ず一致させる。
        var norm = function (v) {
            if (v === null || v === undefined || v === '') return null;
            var n = Number(v);
            return isNaN(n) ? null : n;
        };
        var a = norm(simVal), b = norm(restartVal);
        if (a === b) return [nu, nu];
        var trig = (ctx.triggered && ctx.triggered.length) ? ctx.triggered[0].prop_id : '';
        if (trig.indexOf('target-damage') === 0) return [nu, a];
        if (trig.indexOf('restart-D') === 0) return [b, nu];
        return [nu, nu];
    }
    """,
    Output("target-damage", "value", allow_duplicate=True),
    Output("restart-D", "value", allow_duplicate=True),
    Input("target-damage", "value"),
    Input("restart-D", "value"),
    prevent_initial_call=True,
)

# --- サイドバー開閉 (スライド) ---
application.clientside_callback(
    """
    function(n) {
        return ((n || 0) % 2 === 1) ? 'sim-sidebar collapsed' : 'sim-sidebar';
    }
    """,
    Output("sim-sidebar", "className"),
    Input("sidebar-toggle", "n_clicks"),
    prevent_initial_call=True,
)

# --- 入力の自動保存 (ブラウザの localStorage) ---
# 引数の並びは assets/autosave.js の FIELDS と 1:1 で対応する。
# 片方だけ並べ替えるとスナップショットの中身がずれるので必ず両方を直すこと。
# 復元は app/frontend/persist.py が担当する。
application.clientside_callback(
    "dash_clientside.persist.collect",
    Output("autosave-store", "data", allow_duplicate=True),
    # --- ダメージシミュレータ ---
    Input({"type": "param", "param": ALL, "index": ALL}, "value"),
    Input({"type": "memo", "index": ALL}, "value"),
    Input("sorted-indices", "data"),
    Input("card-indices", "data"),
    Input("next-index", "data"),
    Input("target-damage", "value"),
    Input("global-crit-rate", "value"),
    Input("global-evade-rate", "value"),
    Input("global-stability", "value"),
    Input("calc-method", "value"),
    Input("damage-mode", "value"),
    Input("hp-mode", "value"),
    Input("hp-H", "value"),
    Input("hp-H1", "value"),
    Input("hp-R0", "value"),
    Input("hp-R1", "value"),
    Input("text-input", "value"),
    Input("text-prefix", "value"),
    Input({"type": "accum", "field": ALL, "index": ALL}, "value"),
    Input("accum-next-index", "data"),
    # --- 足切りライン最適化 ---
    Input("restart-D", "value"),
    Input("restart-cp-store", "data"),
    Input("restart-seg-time-store", "data"),
    Input("restart-seg-success-store", "data"),
    Input("restart-save-store", "data"),
    # --- スキル順探索 ---
    Input("so-hand-size", "value"),
    Input("so-card-count", "value"),
    Input("so-limit", "value"),
    Input("so-tl-text", "value"),
    Input({"type": "so-name", "index": ALL}, "value"),
    Input({"type": "so-copier", "index": ALL}, "value"),
    Input({"type": "so-step-skill", "index": ALL}, "value"),
    Input({"type": "so-step-target", "index": ALL}, "value"),
    Input({"type": "so-step-slot", "index": ALL}, "value"),
    Input({"type": "so-step-draw", "index": ALL}, "value"),
    Input({"type": "so-step-memo", "index": ALL}, "value"),
    Input("so-step-order", "data"),
    Input("so-next-step", "data"),
    Input({"type": "so-con-type", "index": ALL}, "value"),
    Input({"type": "so-con-steps", "index": ALL}, "value"),
    Input("so-next-con", "data"),
    # --- 値と添字を対応づけるための id 群 ---
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "id"),
    State({"type": "so-step-skill", "index": ALL}, "id"),
    State({"type": "so-con-type", "index": ALL}, "id"),
    State({"type": "accum", "field": ALL, "index": ALL}, "id"),
    # --- 復元が済むまで保存しないためのフラグ ---
    State("persist-armed", "data"),
    prevent_initial_call=True,
)

# gunicorn から参照される WSGI サーバー (gunicorn main:server)
server = application.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = "PORT" not in os.environ
    application.run(host="0.0.0.0", port=port, debug=debug)
