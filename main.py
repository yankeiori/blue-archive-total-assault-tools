import os

from dash import ALL, Input, Output, State
from app import app as application, OCR_ENABLED
from app.frontend.layout import create_layout
import app.frontend.callbacks  # noqa: F401 - コールバック登録

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
        var trig = (ctx.triggered && ctx.triggered.length) ? ctx.triggered[0].prop_id : '';
        if (trig.indexOf('target-damage') === 0) {
            return (restartVal === simVal) ? [nu, nu] : [nu, simVal];
        }
        if (trig.indexOf('restart-D') === 0) {
            return (simVal === restartVal) ? [nu, nu] : [restartVal, nu];
        }
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

# gunicorn から参照される WSGI サーバー (gunicorn main:server)
server = application.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = "PORT" not in os.environ
    application.run(host="0.0.0.0", port=port, debug=debug)
