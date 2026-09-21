import base64
import hashlib
import json

import dash
import plotly.graph_objects as go
from dash import callback, Input, Output, State, ALL, MATCH, ctx, dcc, html
from dash.exceptions import PreventUpdate

from app import OCR_ENABLED
from app.backend import (
    ocr, restart_accum, restart_cos, restart_mixed, restart_save, skill_order,
    tl_parse,
)
from app.backend.accumulate import AccumWindow, CapSpec
from app.backend.cos import HPParams, build_hit_mixtures, y_mixture
from app.backend.mixed import card_is_hp_dep, hit_specs_from_cards, mixed_support
from app.frontend.layout import (
    ACCUM_PRESETS,
    SO_DEFAULT_CARDS,
    SO_MAX_CARDS,
    accum_options,
    make_accum_card,
    make_damage_card,
    make_so_constraint,
    make_so_step,
    so_card_count_options,
    so_skill_options,
    so_slot_options,
    so_target_options,
)

# スキル順探索: 解を集める下限件数(表示件数がこれより少なくてもここまでは数える)
SO_SEARCH_CAP_MIN = 500

# エクスポート/インポートで扱うカードパラメータ項目とフォーマット版。
_CARD_PARAMS = ["crit_min", "crit_max", "normal_min", "normal_max",
                "hits", "crit_rate", "evade_rate", "enemies", "hp_dep"]
_IO_VERSION = 3

# 蓄積 (チャージ) 型スキルのエクスポート項目。カード参照 (cards / cap_cards) は
# 表示順の位置で書き出す — インポートはカードを 0..n-1 で作り直すため。
_ACCUM_FIELDS = ["preset", "name", "cards", "rate", "cap_mode", "cap_value",
                 "atk", "atk_pct", "cap_cards", "cap_pct", "burst_mult",
                 "burst_decay", "burst_after"]


def _accum_by_index(values, ids) -> dict:
    """蓄積スキルの (値, id) 列を {index: {field: 値}} にする。"""
    by: dict = {}
    for v, aid in zip(values or [], ids or []):
        if isinstance(aid, dict) and "index" in aid and "field" in aid:
            by.setdefault(aid["index"], {})[aid["field"]] = v
    return by


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _triggered_clicked() -> bool:
    """今回のトリガー当人の値が真か (= 実際にクリックされたか) を返す。

    動的にカードが追加されると n_clicks=0 の新ボタンが ALL 入力に加わり
    本コールバックが再発火する。n_clicks は累積保持されるため集計では
    判定できず、トリガーされた当人の値 (ctx.triggered[0]) を見る。
    """
    trig = ctx.triggered
    return bool(trig and trig[0].get("value"))


def _order_from_children(children: list) -> list:
    """children の並び順から sorted-indices 用の順序リストを構築する。"""
    order = []
    for c in children:
        if isinstance(c, dict):
            cid = c.get("props", {}).get("id", {})
        else:
            cid = getattr(c, "id", {})
        if not isinstance(cid, dict):
            continue
        if cid.get("type") == "card":
            order.append(cid["index"])
    return order


# ---------------------------------------------------------------------------
# カード追加・削除 (コンポーネント生成が必要なためサーバーサイド)
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children"),
    Output("card-indices", "data"),
    Output("next-index", "data"),
    Output("sorted-indices", "data", allow_duplicate=True),
    Input("add-btn", "n_clicks"),
    Input({"type": "remove-btn", "index": ALL}, "n_clicks"),
    Input({"type": "duplicate-btn", "index": ALL}, "n_clicks"),
    State("card-indices", "data"),
    State("next-index", "data"),
    State("cards-container", "children"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_cards(
    add_clicks, remove_clicks, duplicate_clicks,
    indices, next_idx, children, global_crit, global_evade,
    param_values, param_ids, memo_values, memo_ids,
):
    trigger = ctx.triggered_id
    # 初期状態はカード0枚のため、State が None になり得る
    indices = indices or []
    children = children or []

    if trigger == "add-btn":
        indices.append(next_idx)
        children.append(make_damage_card(next_idx, global_crit, global_evade))
        return children, indices, next_idx + 1, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "remove-btn":
        # カード動的追加でボタンが増えると本コールバックが再発火するため、
        # 実際にクリックされた (トリガー当人の n_clicks が真) 時のみ処理する。
        # n_clicks は累積保持されるので集計ではなくトリガー値を見る。
        if not _triggered_clicked():
            raise PreventUpdate
        remove_idx = trigger["index"]
        indices = [i for i in indices if i != remove_idx]
        children = [
            c for c in children
            if not (c["props"]["id"].get("type") == "card" and c["props"]["id"].get("index") == remove_idx)
        ]
        return children, indices, next_idx, _order_from_children(children)

    if isinstance(trigger, dict) and trigger.get("type") == "duplicate-btn":
        # 同上: 実際に複製ボタンが押された時のみ処理する。
        if not _triggered_clicked():
            raise PreventUpdate
        src_idx = trigger["index"]
        src_params = {}
        for val, pid in zip(param_values, param_ids):
            if pid["index"] == src_idx:
                src_params[pid["param"]] = val
        src_memo = ""
        for val, mid in zip(memo_values, memo_ids):
            if mid["index"] == src_idx:
                src_memo = val or ""
                break
        new_card = make_damage_card(next_idx, params=src_params, memo=src_memo)
        children.append(new_card)
        indices.append(next_idx)
        return children, indices, next_idx + 1, _order_from_children(children)

    raise PreventUpdate


# ---------------------------------------------------------------------------
# スクリーンショット OCR → カード自動生成 (サーバーサイド)
# OCR はローカル専用機能。外部公開時 (ENABLE_OCR=false) はコールバック自体を
# 登録しないため、Vision API へのアクセス経路が存在しなくなる。
# ---------------------------------------------------------------------------
if OCR_ENABLED:
    @callback(
        Output("cards-container", "children", allow_duplicate=True),
        Output("card-indices", "data", allow_duplicate=True),
        Output("next-index", "data", allow_duplicate=True),
        Output("sorted-indices", "data", allow_duplicate=True),
        Output("ocr-status", "children"),
        Output("hp-mode", "value", allow_duplicate=True),
        Input("ocr-upload", "contents"),
        Input("ocr-image-store", "data"),
        State("cards-container", "children"),
        State("card-indices", "data"),
        State("next-index", "data"),
        prevent_initial_call=True,
    )
    def ocr_add_cards(upload_contents, snip_data, children, indices, next_idx):
        """アップロード / スニップ画像を OCR し、抽出カードを追加する。"""
        trigger = ctx.triggered_id
        image = upload_contents if trigger == "ocr-upload" else snip_data
        if not image:
            raise PreventUpdate

        no_change = (dash.no_update, dash.no_update, dash.no_update, dash.no_update)

        try:
            result = ocr.cards_from_image(image)
        except ocr.OcrError as exc:
            return (*no_change, f"⚠ {exc}", dash.no_update)
        except Exception as exc:  # noqa: BLE001 - 予期せぬ失敗もユーザーに表示
            return (*no_change, f"⚠ 解析に失敗しました: {exc}", dash.no_update)

        parsed = result["cards"]
        if not parsed:
            return (*no_change, "⚠ カードを検出できませんでした。画像を確認してください。", dash.no_update)

        children = children or []
        indices = indices or []
        for card in parsed:
            children.append(make_damage_card(next_idx, params=card["params"], memo=card["memo"]))
            indices.append(next_idx)
            next_idx += 1

        msgs = [f"✅ {len(parsed)} 枚のカードを追加しました。"]
        hp_mode = dash.no_update
        if result.get("hp_dependent"):
            hp_mode = "on"
            msgs.append("HP依存を検出 → サイドバーのHP依存モードをONにしました。")

        return (
            children,
            indices,
            next_idx,
            _order_from_children(children),
            " ".join(msgs),
            hp_mode,
        )


# ---------------------------------------------------------------------------
# テキスト貼り付け → カード自動生成 (サーバーサイド)
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children", allow_duplicate=True),
    Output("card-indices", "data", allow_duplicate=True),
    Output("next-index", "data", allow_duplicate=True),
    Output("sorted-indices", "data", allow_duplicate=True),
    Output("text-status", "children"),
    Output("hp-mode", "value", allow_duplicate=True),
    Input("text-import-btn", "n_clicks"),
    State("text-input", "value"),
    State("text-prefix", "value"),
    State("cards-container", "children"),
    State("card-indices", "data"),
    State("next-index", "data"),
    prevent_initial_call=True,
)
def text_add_cards(n_clicks, text, prefix, children, indices, next_idx):
    """貼り付けテキストを解析し、抽出カードを追加する。

    prefix は取り込む全カードの備考の頭に付く (例:「ミカ1射目 ヒット1-10」)。
    """
    if not n_clicks or not (text or "").strip():
        raise PreventUpdate

    no_change = (dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    try:
        result = ocr.cards_from_text(text, prefix)
    except Exception as exc:  # noqa: BLE001 - 予期せぬ失敗もユーザーに表示
        return (*no_change, f"⚠ 解析に失敗しました: {exc}", dash.no_update)

    parsed = result["cards"]
    if not parsed:
        return (*no_change, "⚠ カードを検出できませんでした。テキストを確認してください。", dash.no_update)

    children = children or []
    indices = indices or []
    for card in parsed:
        children.append(make_damage_card(next_idx, params=card["params"], memo=card["memo"]))
        indices.append(next_idx)
        next_idx += 1

    msgs = [f"✅ {len(parsed)} 枚のカードを追加しました。"]
    hp_mode = dash.no_update
    if result.get("hp_dependent"):
        hp_mode = "on"
        msgs.append("HP依存を検出 → サイドバーのHP依存モードをONにしました。")

    return (
        children,
        indices,
        next_idx,
        _order_from_children(children),
        " ".join(msgs),
        hp_mode,
    )


# ---------------------------------------------------------------------------
# 多段リスタ最適化 (サーバーサイド: COS + Bermudan 後ろ向き帰納)
# ---------------------------------------------------------------------------
def _assemble_cards_ordered(order, card_indices, param_values, param_ids):
    """param の State から、表示順の (カード index, カード dict) 列を組み立てる。"""
    by_index: dict = {}
    for val, pid in zip(param_values, param_ids):
        by_index.setdefault(pid["index"], {})[pid["param"]] = val
    seq = [i for i in (order or []) if isinstance(i, int)] or list(card_indices or [])
    return [(i, by_index[i]) for i in seq if i in by_index]


def _card_hit_spans(ordered) -> dict:
    """カード index → そのカードが占める Hit 番号のリスト。

    Hit の数え方は build_hit_mixtures と同じ (カードの「Hit数」。足切りページは
    従来から「敵の数」を掛けない)。
    """
    spans, pos = {}, 0
    for idx, params in ordered:
        h = max(int(params.get("hits") or 1), 0)
        spans[idx] = list(range(pos, pos + h))
        pos += h
    return spans


def _accum_windows(accum_values, accum_ids, ordered) -> list:
    """蓄積スキルの入力欄から AccumWindow のリストを作る (Hit 番号は通し番号)。

    クライアント側 assets/simulation.js の buildPools と同じ変換。
    """
    spans = _card_hit_spans(ordered)
    wins = []
    for k, a in sorted(_accum_by_index(accum_values, accum_ids).items()):
        cards = [c for c in (a.get("cards") or []) if c in spans]
        try:
            rate = float(a.get("rate") or 0) / 100.0
            mult = float(a.get("burst_mult") or 0) / 100.0
        except (TypeError, ValueError):
            continue
        hits = [h for c in cards for h in spans[c]]
        if not hits or rate <= 0 or mult <= 0:
            continue
        name = (a.get("name") or "").strip() or f"蓄積スキル{k + 1}"
        mode = a.get("cap_mode") or "atk"
        if mode == "cards":
            src = [h for c in (a.get("cap_cards") or []) if c in spans
                   for h in spans[c]]
            if not src:
                raise ValueError(f"「{name}」の上限カードが指定されていません。")
            cap = CapSpec(kind="hits", hits=src,
                          coef=float(a.get("cap_pct") or 0) / 100.0)
        elif mode == "atk":
            cap = CapSpec(kind="fixed",
                          value=float(a.get("atk") or 0)
                          * float(a.get("atk_pct") or 0) / 100.0)
        else:
            cap = CapSpec(kind="fixed", value=float(a.get("cap_value") or 0))
        hits = sorted(hits)
        after = a.get("burst_after")
        burst_hit = spans[after][-1] if (after in spans and spans[after]) else hits[-1]
        if burst_hit < max([*hits, *cap.source_hits()]):
            raise ValueError(
                f"「{name}」の爆発カードが蓄積の終わりより前です。"
                "爆発は蓄積が止まったあとに起きるので、蓄積対象の最後のカード以降を"
                "指定してください。")
        wins.append(AccumWindow(hits=hits, rate=rate, cap=cap,
                                burst_mult=mult,
                                burst_decay=bool(a.get("burst_decay")),
                                name=name, burst_hit=burst_hit))
    return wins


def _restart_fingerprint(order, card_indices, param_values, param_ids, cp_store,
                         seg_times, seg_success, save_store, D,
                         global_crit, global_evade,
                         damage_mode, hp_mode, hp_H, hp_H1, hp_R0, hp_R1,
                         accum_values=None, accum_ids=None) -> str:
    """足切り最適化の結果を左右する入力だけを 1 本の指紋にまとめる。

    run_restart は実行ボタンでしか発火しないため、実行後にカードを並べ替えたり
    パラメータを変えたりしても図・表は古いまま残る。実行時の指紋を restart-config
    に載せておき、現在の指紋と突き合わせて「再実行してください」を出すのに使う。

    カード名 (メモ) は数値結果に影響しないので含めない (入力中の点滅を避ける)。
    HP 依存パラメータは hp_mode が on のときだけ効くので、off なら無視する。
    """
    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    payload = {
        "cards": [{k: params.get(k) for k in _CARD_PARAMS} for _idx, params in ordered],
        "cps": sorted({int(x) for x in (cp_store or [])}),
        "seg_times": seg_times or {},
        "seg_success": seg_success or {},
        "saves": sorted({int(x) for x in (save_store or [])}),
        "D": D,
        "crit": global_crit,
        "evade": global_evade,
        "damage_mode": damage_mode,
        "hp_mode": hp_mode,
        "hp": [hp_H, hp_H1, hp_R0, hp_R1] if hp_mode == "on" else None,
        # 蓄積スキルも結果を変えるので指紋に含める
        "accum": sorted(_accum_by_index(accum_values, accum_ids).items()),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _export_accum(values, ids, pos_of: dict) -> list:
    """蓄積スキルの設定を JSON 用に書き出す (カード参照を表示順の位置へ)。"""
    out = []
    for index in sorted(_accum_by_index(values, ids)):
        a = _accum_by_index(values, ids)[index]
        row = {k: a.get(k) for k in _ACCUM_FIELDS}
        for key in ("cards", "cap_cards"):
            row[key] = [pos_of[i] for i in (a.get(key) or []) if i in pos_of]
        b = a.get("burst_after")
        row["burst_after"] = pos_of.get(b) if b is not None else None
        row["burst_decay"] = bool(a.get("burst_decay"))
        out.append(row)
    return out


def _import_accum(rows, options) -> tuple[list, int]:
    """JSON の蓄積スキル設定から入力カードを組み直す (位置 = 新しいカード index)。"""
    children = []
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        params = {k: row.get(k) for k in _ACCUM_FIELDS}
        params["burst_decay"] = [1] if row.get("burst_decay") else []
        children.append(make_accum_card(len(children), params=params,
                                        options=options))
    return children, len(children)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 入力情報のエクスポート (カード + 全体設定 + 多段リスタ設定 → JSON ダウンロード)
# ---------------------------------------------------------------------------
@callback(
    Output("export-download", "data"),
    Input("export-btn", "n_clicks"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    State("restart-cp-store", "data"),
    State("restart-seg-time-store", "data"),
    State("restart-seg-success-store", "data"),
    State("restart-save-store", "data"),
    State("target-damage", "value"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("global-stability", "value"),
    State("calc-method", "value"),
    State("damage-mode", "value"),
    State("hp-mode", "value"),
    State("hp-H", "value"), State("hp-H1", "value"),
    State("hp-R0", "value"), State("hp-R1", "value"),
    State("restart-D", "value"),
    State({"type": "accum", "field": ALL, "index": ALL}, "value"),
    State({"type": "accum", "field": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def export_input(n_clicks, order, card_indices, param_values, param_ids,
                 memo_values, memo_ids, cp_store, seg_times, seg_success,
                 save_store, target_damage, gcrit, gevade, gstab, calc_method,
                 damage_mode,
                 hp_mode, hp_H, hp_H1, hp_R0, hp_R1, restart_D,
                 accum_values, accum_ids):
    if not n_clicks:
        raise PreventUpdate

    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    memo_by = {mid["index"]: (v or "") for v, mid in zip(memo_values, memo_ids)}
    n = sum(int(params.get("hits") or 1) for _idx, params in ordered)
    cps = sorted({int(c) for c in (cp_store or []) if 0 < int(c) < n})
    # 区間開始境界 (0, cps...) の時間割合だけを書き出す
    seg_times = seg_times or {}
    seg_success = seg_success or {}
    boundaries = [0, *cps]
    segment_times = {str(b): float(seg_times.get(str(b), 1.0)) for b in boundaries}
    segment_success = {str(b): float(seg_success.get(str(b), 100.0)) for b in boundaries}

    cards = []
    for idx, params in ordered:
        cards.append({
            "params": {k: params.get(k) for k in _CARD_PARAMS},
            "memo": memo_by.get(idx, ""),
        })
    data = {
        "version": _IO_VERSION,
        "globals": {
            "target_damage": target_damage,
            "global_crit": gcrit, "global_evade": gevade, "global_stability": gstab,
            "calc_method": calc_method, "damage_mode": damage_mode, "hp_mode": hp_mode,
            "hp_H": hp_H, "hp_H1": hp_H1, "hp_R0": hp_R0, "hp_R1": hp_R1,
            "restart_D": restart_D,
        },
        "cards": cards,
        "accum": _export_accum(accum_values, accum_ids,
                               {idx: pos for pos, (idx, _p) in enumerate(ordered)}),
        "restart": {"checkpoints": cps, "segment_times": segment_times,
                    "segment_success": segment_success,
                    "save_points": sorted({int(x) for x in (save_store or [])
                                           if int(x) in cps})},
    }
    return dict(content=json.dumps(data, ensure_ascii=False, indent=2),
                filename="damage_cutoff_input.json")


# ---------------------------------------------------------------------------
# 入力情報のインポート (JSON → カード再構築 + 全体設定 + 多段リスタ設定復元)
# ---------------------------------------------------------------------------
@callback(
    Output("cards-container", "children", allow_duplicate=True),
    Output("card-indices", "data", allow_duplicate=True),
    Output("next-index", "data", allow_duplicate=True),
    Output("sorted-indices", "data", allow_duplicate=True),
    Output("restart-cp-store", "data", allow_duplicate=True),
    Output("restart-seg-time-store", "data", allow_duplicate=True),
    Output("restart-seg-success-store", "data", allow_duplicate=True),
    Output("restart-save-store", "data", allow_duplicate=True),
    Output("io-status", "children"),
    Output("target-damage", "value"),
    Output("global-crit-rate", "value"),
    Output("global-evade-rate", "value"),
    Output("global-stability", "value"),
    Output("calc-method", "value"),
    Output("damage-mode", "value"),
    Output("hp-mode", "value", allow_duplicate=True),
    Output("hp-H", "value"), Output("hp-H1", "value"),
    Output("hp-R0", "value"), Output("hp-R1", "value"),
    Output("restart-D", "value"),
    Output("accum-container", "children", allow_duplicate=True),
    Output("accum-next-index", "data", allow_duplicate=True),
    Input("import-upload", "contents"),
    prevent_initial_call=True,
)
def import_input(contents):
    if not contents:
        raise PreventUpdate
    nu = dash.no_update
    # globals 出力 13 個 (target..restart_D) の「変更なし」ベクトル
    globals_nu = (nu,) * 12

    try:
        _meta, b64 = contents.split(",", 1)
        data = json.loads(base64.b64decode(b64).decode("utf-8"))
        cards = data.get("cards", [])
        if not isinstance(cards, list) or not cards:
            raise ValueError("カードが空です。")
    except Exception as exc:  # noqa: BLE001 - 不正ファイルはユーザーに表示
        return (nu, nu, nu, nu, nu, nu, nu, nu, f"⚠ インポート失敗: {exc}",
                *globals_nu, nu, nu)

    children = []
    total_hits = 0
    for i, c in enumerate(cards):
        params = {k: (c.get("params", {}) or {}).get(k) for k in _CARD_PARAMS}
        children.append(make_damage_card(i, params=params, memo=c.get("memo", "")))
        total_hits += int(params.get("hits") or 1)
    n = len(cards)
    indices = list(range(n))

    # 多段リスタ設定 (新フォーマット: restart.checkpoints / restart.segment_times)
    restart = data.get("restart", {}) or {}
    cps = sorted({int(c) for c in restart.get("checkpoints", [])
                  if 0 < int(c) < total_hits})
    seg_times = {str(k): float(v)
                 for k, v in (restart.get("segment_times", {}) or {}).items()}
    seg_times.setdefault("0", 1.0)
    seg_success = {str(k): float(v)
                   for k, v in (restart.get("segment_success", {}) or {}).items()}
    seg_success.setdefault("0", 100.0)
    saves = sorted({int(x) for x in (restart.get("save_points", []) or [])
                    if int(x) in cps})

    g = data.get("globals", {}) or {}
    def gv(key):
        return g[key] if key in g else nu
    accum_children, accum_next = _import_accum(
        data.get("accum"),
        accum_options(indices, {i: (c.get("memo") or "")
                                for i, c in enumerate(cards)}))
    msg = f"✅ {n} 枚のカードと設定を読み込みました。足切りライン最適化の設定も復元済みです。"
    if accum_children:
        msg += f" 蓄積スキル {len(accum_children)} 件も復元しました。"
    return (
        children, indices, n, indices, cps, seg_times, seg_success, saves, msg,
        gv("target_damage"), gv("global_crit"), gv("global_evade"), gv("global_stability"),
        gv("calc_method"), gv("damage_mode"), gv("hp_mode"),
        gv("hp_H"), gv("hp_H1"), gv("hp_R0"), gv("hp_R1"), gv("restart_D"),
        accum_children, accum_next,
    )


# ---------------------------------------------------------------------------
# 多段リスタ: カード別 チェックポイント / 時間割合 テーブル
# ---------------------------------------------------------------------------
@callback(
    Output("restart-cards-table", "children"),
    Output("restart-cp-dropdown", "options"),
    Output("restart-nhits", "data"),
    Input("nav-restart", "n_clicks"),
    Input("restart-reload-btn", "n_clicks"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    State("hp-mode", "value"),
    prevent_initial_call=True,
)
def populate_restart_table(_n1, _n2, order, card_indices, param_values, param_ids,
                           memo_values, memo_ids, hp_mode):
    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    if not ordered:
        return html.Div("カードがありません。", style={"color": "#d63031"}), [], 0

    memo_by = {mid["index"]: (v or "") for v, mid in zip(memo_values, memo_ids)}

    show_type = hp_mode == "on"
    header_cells = [html.Th("カード"), html.Th("ヒット"), html.Th("累積")]
    if show_type:
        header_cells.append(html.Th("型"))
    rows = [html.Tr(header_cells)]
    options = []
    cum = 0
    last = len(ordered) - 1
    for pos, (idx, card) in enumerate(ordered):
        h = int(card.get("hits") or 1)
        cum += h
        memo = memo_by.get(idx, "")
        label = f"{pos + 1}: {memo}" if memo else f"カード{pos + 1}"
        # 最終セグメント以外を足切り候補としてプルダウンに出す
        if pos != last:
            options.append({"label": f"{label}(累積 {cum} ヒット目で足切り)",
                            "value": cum})
        cells = [html.Td(label), html.Td(str(h)), html.Td(str(cum))]
        if show_type:
            cells.append(html.Td("HP依存" if card_is_hp_dep(card) else "通常"))
        rows.append(html.Tr(cells))
    table = html.Table(rows, className="restart-cards")
    return table, options, cum


# ---------------------------------------------------------------------------
# 多段リスタ: チェックポイントの追加 / 削除 (プルダウン + カード方式)
# ---------------------------------------------------------------------------
@callback(
    Output("restart-cp-store", "data", allow_duplicate=True),
    Output("restart-cp-dropdown", "value"),
    Input("restart-cp-add-btn", "n_clicks"),
    Input({"type": "restart-cp-remove", "index": ALL}, "n_clicks"),
    State("restart-cp-dropdown", "value"),
    State("restart-cp-store", "data"),
    prevent_initial_call=True,
)
def manage_restart_cp(_add, _removes, dropdown_value, store):
    if not _triggered_clicked():
        raise PreventUpdate
    store = list(store or [])
    trig = ctx.triggered_id
    if trig == "restart-cp-add-btn":
        if dropdown_value is None:
            raise PreventUpdate
        cum = int(dropdown_value)
        if cum not in store:
            store.append(cum)
            store.sort()
        return store, None
    if isinstance(trig, dict) and trig.get("type") == "restart-cp-remove":
        cum = int(trig["index"])
        return [c for c in store if c != cum], dash.no_update
    raise PreventUpdate


# ---------------------------------------------------------------------------
# 多段リスタ: 区間カード (足切りで区切られた各区間 = 1 カード)
#   各カードに「時間割合」入力を内蔵し、末尾以外は「✕」でその足切りを解除できる。
# ---------------------------------------------------------------------------
def _segments(cps, n):
    """足切り cps と総ヒット n から区間 [(start, end), ...] を作る。"""
    cps = sorted({int(c) for c in (cps or []) if 0 < int(c) < int(n or 0)})
    bounds = [0, *cps, int(n or 0)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _seg_card(idx, total, s, e, weight, end_label, success=100.0, is_save=False):
    """区間カード 1 枚 (横長) を生成する。末尾以外は ✕ で足切り (境界 e) を解除。

    is_save=True の境界は凸区切り (セーブポイント) で、そこまでのダメージが確定し、
    以後のリセットでもその境界より前には戻らない。
    """
    is_last = idx == total - 1
    head = "完走(最終区間)" if is_last else f"足切り{idx + 1}"
    if is_save and not is_last:
        head += "・🚩凸区切り"
    title = f"区間{idx + 1}・{head}"
    sub = f"ヒット {s + 1}–{e}"
    if not is_last and end_label:
        sub += f"  /  {end_label}"
    cells = [
        # 左: 区間名 + ヒット範囲
        html.Div(
            [
                html.Div(title, style={"fontWeight": "bold", "fontSize": "0.85rem"}),
                html.Div(sub, style={"fontSize": "0.78rem", "color": "#666"}),
            ],
            style={"flex": "1", "minWidth": "0"},
        ),
        # 中: 所要時間入力 (相対値でよい — 比だけが結果に影響する)
        html.Div(
            [
                html.Label("所要時間 ",
                           title="この区間を回すのにかかる時間。おおよその秒数でOK"
                                 "(比だけが結果に影響します)。",
                           style={"fontSize": "0.8rem"}),
                dcc.Input(id={"type": "restart-seg-time", "index": s}, type="number",
                          value=weight, min=0, step=0.1,
                          style={"width": "90px", "marginLeft": "4px"}),
            ],
            style={"whiteSpace": "nowrap"},
        ),
        # 中2: ダメージと独立な成功率 (%) 入力。この区間を回しきって次へ進める確率。
        # 足切り(ダメージ)とは別要因であることが分かるよう、ラベル・色・注記で区別する。
        html.Div(
            [
                html.Label(
                    "🎲 ダメージ外 成功率% ",
                    title="ダメージ(足切り)とは無関係な成功要因。この区間を回しきって"
                          "次へ進める確率です。失敗するとリスタート(その区間の時間は消費)。"
                          "100%=この要因では失敗しない(=ダメージ足切りのみ)。",
                    style={"fontSize": "0.8rem", "color": "#0984e3",
                           "fontWeight": "bold"}),
                dcc.Input(id={"type": "restart-seg-success", "index": s}, type="number",
                          value=success, min=0, max=100, step=0.01,
                          style={"width": "64px", "marginLeft": "4px",
                                 "border": "1px solid #0984e3", "color": "#0984e3"}),
                html.Span("%", style={"fontSize": "0.8rem", "color": "#0984e3",
                                      "marginLeft": "2px"}),
            ],
            style={"whiteSpace": "nowrap",
                   "borderLeft": "1px solid #dfe6e9", "paddingLeft": "12px"},
        ),
    ]
    # 中3: 凸区切り (セーブポイント) のトグル。末尾区間の後ろには境界が無い。
    if not is_last:
        cells.append(html.Div(
            [
                dcc.Checklist(
                    id={"type": "restart-seg-save", "index": e},
                    options=[{"label": " 🚩 ここで凸を区切る", "value": "on"}],
                    value=(["on"] if is_save else []),
                    inputStyle={"marginRight": "3px"},
                    style={"fontSize": "0.8rem",
                           "color": "#b35900" if is_save else "#666",
                           "fontWeight": "bold" if is_save else "normal"},
                ),
            ],
            title="この境界で凸が終わります。ここまでのダメージは確定 (セーブ) され、"
                  "以後リセットしてもこの境界より前には戻りません。"
                  "凸区切りの足切りは「この凸を確定させてよいか」の判断になります。",
            style={"whiteSpace": "nowrap",
                   "borderLeft": "1px solid #dfe6e9", "paddingLeft": "12px"},
        ))
    # 右: 足切り解除 (末尾区間以外)
    cells.append(html.Button(
        "✕", id={"type": "restart-cp-remove", "index": e if not is_last else -1},
        n_clicks=0, title="この足切りを解除",
        style={"border": "none", "background": "transparent",
               "cursor": "pointer" if not is_last else "default",
               "color": "#d63031" if not is_last else "transparent",
               "fontWeight": "bold", "marginLeft": "12px",
               "visibility": "visible" if not is_last else "hidden"}))
    return html.Div(
        cells,
        style={"display": "flex", "alignItems": "center", "gap": "12px",
               "border": ("2px solid #b35900" if (is_save and not is_last)
                          else "1px solid #d63031"),
               "borderRadius": "8px",
               "padding": "8px 14px", "marginBottom": "8px",
               "background": "#fffaf3" if (is_save and not is_last) else "#fff",
               "width": "100%", "boxSizing": "border-box"},
    )


@callback(
    Output("restart-cp-cards", "children"),
    Input("restart-cp-store", "data"),
    Input("restart-nhits", "data"),
    Input("restart-cp-dropdown", "options"),
    State("restart-seg-time-store", "data"),
    State("restart-seg-success-store", "data"),
    State("restart-save-store", "data"),
    prevent_initial_call=True,
)
def render_restart_cards(cp_store, n, options, seg_times, seg_success, save_store):
    segs = _segments(cp_store, n)
    if not segs:
        return html.Div("攻撃列が未読込です。「カード読込 / 更新」を押してください。",
                        style={"fontSize": "0.82rem", "color": "#d63031"})
    label_by = {opt["value"]: opt["label"] for opt in (options or [])}
    seg_times = seg_times or {}
    seg_success = seg_success or {}
    saves = {int(x) for x in (save_store or [])}
    cards = []
    for i, (s, e) in enumerate(segs):
        cards.append(_seg_card(i, len(segs), s, e,
                               seg_times.get(str(s), 1.0), label_by.get(e, ""),
                               seg_success.get(str(s), 100.0), e in saves))
    return html.Div(cards, style={"display": "flex", "flexDirection": "column"})


@callback(
    Output("restart-save-store", "data", allow_duplicate=True),
    Input({"type": "restart-seg-save", "index": ALL}, "value"),
    State({"type": "restart-seg-save", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_restart_save(values, ids):
    """凸区切り (セーブポイント) にした境界の累積ヒット数を Store へ。

    表示中のトグルから毎回作り直すので、足切りを解除するとその凸区切りも自動的に
    消える (境界そのものが無くなるため)。
    """
    return sorted({int(sid["index"]) for v, sid in zip(values, ids) if v})


@callback(
    Output("restart-seg-time-store", "data", allow_duplicate=True),
    Input({"type": "restart-seg-time", "index": ALL}, "value"),
    State({"type": "restart-seg-time", "index": ALL}, "id"),
    State("restart-seg-time-store", "data"),
    prevent_initial_call=True,
)
def update_restart_seg_time(values, ids, store):
    store = dict(store or {})
    for v, sid in zip(values, ids):
        try:
            store[str(sid["index"])] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            store[str(sid["index"])] = 0.0
    return store


@callback(
    Output("restart-seg-success-store", "data", allow_duplicate=True),
    Input({"type": "restart-seg-success", "index": ALL}, "value"),
    State({"type": "restart-seg-success", "index": ALL}, "id"),
    State("restart-seg-success-store", "data"),
    prevent_initial_call=True,
)
def update_restart_seg_success(values, ids, store):
    """区間ごとのダメージ独立成功率 % (0..100) を Store へ。空欄/不正は 100% 扱い。"""
    store = dict(store or {})
    for v, sid in zip(values, ids):
        try:
            store[str(sid["index"])] = min(100.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            store[str(sid["index"])] = 100.0
    return store


# ---------------------------------------------------------------------------
# 多段リスタ: 図・表の共通ヘルパー (最適表示 / インタラクティブ表示で共用)
# ---------------------------------------------------------------------------
def _restart_disp(res, cum_to_label, last_label, D):
    """解析結果 res から表示用の行を作る。

    行は [ラベル, 残りダメージ, 区間通過率, 累積通過率, 完走?, 凸区切り?,
    残り下端, 残り上端]。累積通過率は凸区切りでリセットされる (凸区切り付きでは
    各行の通過率は「その凸の 1 試行のうち、ここまで到達して通過した割合」で、
    凸をまたいで掛け合わせるものではない)。残りの下端/上端は、足切りが直前の
    セーブ地点の累積ダメージに依存して動く幅 (1凸目の中は幅ゼロ)。
    """
    disp = []
    prev_cum = 1.0
    for r in res["rows"]:
        cum_pass = r["pass_rate"]
        sect = (cum_pass / prev_cum) if prev_cum > 0 else 0.0
        label = cum_to_label.get(str(r["checkpoint"]), f"{r['checkpoint']}ヒット目")
        is_save = bool(r.get("save"))
        disp.append([label, D - r["gate"], sect, cum_pass, False, is_save,
                     D - r.get("gate_hi", r["gate"]),
                     D - r.get("gate_lo", r["gate"])])
        prev_cum = 1.0 if is_save else cum_pass
    final_cum = res["success"]
    final_sect = (final_cum / prev_cum) if prev_cum > 0 else 0.0
    disp.append([f"{last_label}(完走/目標達成)", 0.0, final_sect, final_cum,
                 True, False, 0.0, 0.0])
    return disp


def _cutoff_figure(disp, title, *, color="#d63031", ref_disp=None):
    """足切りライン(残りダメージ)の折れ線図。ref_disp があれば最適ラインを点線で重ねる。

    x軸は「足切り1, 2, …, 完走」の短い表記。カード名・区間/累積通過率は
    ホバーで表示する。点上の常時テキストは「残りダメージ」のみ(完走点は
    残り0固定なので省略し、y=0 の注記と重ねない)。"""
    xs = ["完走" if row[4] else (f"足切り{i + 1}🚩" if row[5] else f"足切り{i + 1}")
          for i, row in enumerate(disp)]

    def hover(d):
        out = []
        for x, row in zip(xs, d):
            lbl, rem, sect, cumr, _fin, is_save, r_lo, r_hi = row
            band = ("" if abs(r_hi - r_lo) < 1
                    else f"<br>幅 {r_lo:,.0f}〜{r_hi:,.0f}(セーブ地点の累積次第)")
            mark = "<br>🚩 ここで凸が確定 (セーブ)" if is_save else ""
            out.append(f"{x}({lbl})<br>残りダメージ {rem:,.0f}{band}<br>"
                       f"区間通過率 {sect:.1%} / 累積通過率 {cumr:.1%}{mark}")
        return out

    fig = go.Figure()
    if ref_disp is not None:
        fig.add_trace(go.Scatter(
            x=xs, y=[d[1] for d in ref_disp],
            mode="lines+markers", line=dict(color="#999", dash="dot"),
            marker=dict(size=8, color="#999"), name="最適ライン",
            hovertext=hover(ref_disp), hoverinfo="text",
        ))
    band = [(d[7] - d[1], d[1] - d[6]) for d in disp]
    err = dict(type="data", symmetric=False,
               array=[max(a, 0.0) for a, _b in band],
               arrayminus=[max(b, 0.0) for _a, b in band],
               color="rgba(150,150,150,0.55)", thickness=1.4, width=6)
    fig.add_trace(go.Scatter(
        x=xs, y=[d[1] for d in disp],
        mode="lines+markers+text",
        error_y=err if any(a > 1 or b > 1 for a, b in band) else None,
        text=["" if d[4] else f"残り{d[1]:,.0f}" for d in disp],
        textposition="top center", cliponaxis=False,
        marker=dict(size=11, color=color,
                    symbol=["diamond" if d[5] else "circle" for d in disp],
                    line=dict(width=[2 if d[5] else 0 for d in disp],
                              color="#b35900")),
        line=dict(color=color),
        name="設定ライン" if ref_disp is not None else "最適足切り(残りダメージ)",
        hovertext=hover(disp), hoverinfo="text",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="black",
                  annotation_text="目標達成(残り0)",
                  annotation_position="bottom right")
    fig.update_layout(
        title=title, xaxis_title="チェックポイント",
        yaxis_title="足切りライン(目標までの残りダメージ)",
        height=460, margin=dict(t=60),
    )
    return fig


@callback(
    Output("restart-graph", "figure"),
    Output("restart-summary", "children"),
    Output("restart-config", "data"),
    Output("restart-gate-sliders", "children"),
    Input("restart-run-btn", "n_clicks"),
    State("restart-D", "value"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "param", "param": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    State("restart-cp-store", "data"),
    State("restart-seg-time-store", "data"),
    State("restart-seg-success-store", "data"),
    State("restart-save-store", "data"),
    State("global-crit-rate", "value"),
    State("global-evade-rate", "value"),
    State("damage-mode", "value"),
    State("hp-mode", "value"),
    State("hp-H", "value"),
    State("hp-H1", "value"),
    State("hp-R0", "value"),
    State("hp-R1", "value"),
    State({"type": "accum", "field": ALL, "index": ALL}, "value"),
    State({"type": "accum", "field": ALL, "index": ALL}, "id"),
    prevent_initial_call=True,
)
def run_restart(n_clicks, D, order, card_indices, param_values, param_ids,
                memo_values, memo_ids, cp_store, seg_times, seg_success_store,
                save_store, global_crit, global_evade,
                damage_mode, hp_mode, hp_H, hp_H1, hp_R0, hp_R1,
                accum_values, accum_ids):
    if not n_clicks:
        raise PreventUpdate

    # 指紋用に、正規化前 (float 化・or {} 前) の生値を控えておく。flag_restart_stale
    # 側は生値しか持たないので、ここで正規化後の値を混ぜると常に不一致になる。
    D_raw, seg_times_raw, seg_success_raw = D, seg_times, seg_success_store
    save_raw = save_store

    empty = go.Figure()

    def err(msg):
        return (empty, html.Div(f"⚠ {msg}", style={"color": "#d63031"}),
                None, [])

    ordered = _assemble_cards_ordered(order, card_indices, param_values, param_ids)
    if not ordered:
        return err("攻撃カードがありません。「カード読込」を押してください。")

    cards = [c for _i, c in ordered]
    hits = build_hit_mixtures(cards, float(global_crit or 0),
                              float(global_evade or 0), damage_mode or "post_decay")
    n = len(hits)
    if n < 2:
        return err("総ヒット数が2以上必要です。")

    # チェックポイント = プルダウンで追加した累積ヒット数
    cps = sorted({int(x) for x in (cp_store or [])})
    cps = [c for c in cps if 0 < c < n]
    if not cps:
        return err("足切り(チェックポイント)を1つ以上追加してください。")

    # 凸区切り (セーブポイント) = 足切り境界のうち「ここで凸が終わる」もの
    saves = sorted({int(x) for x in (save_store or []) if int(x) in cps})

    # 時間割合: 区間(足切り間)ごとの相対重みを各区間内のヒットへ等分。
    # 区間は開始境界の累積ヒット数 (0, cps...) でキー付けされる。
    seg_times = seg_times or {}
    bounds = [0, *cps, n]
    hit_times = [0.0] * n
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        try:
            w = float(seg_times.get(str(s), 1.0))
        except (TypeError, ValueError):
            w = 1.0
        length = e - s
        per = (w / length) if length > 0 else 0.0
        for j in range(s, e):
            hit_times[j] = per
    if sum(hit_times) <= 0:
        hit_times = [1.0] * n          # 全部0なら一様にフォールバック

    # 区間ごとのダメージ独立成功確率: ストアは区間開始境界 → % (0..100)。
    # 区間順 (0, cps...) に並べ、フラクション [0,1] へ変換 (未設定は 1.0)。
    seg_success_store = seg_success_store or {}
    seg_success = []
    for s in bounds[:-1]:
        try:
            pct = float(seg_success_store.get(str(s), 100.0))
        except (TypeError, ValueError):
            pct = 100.0
        seg_success.append(min(1.0, max(0.0, pct / 100.0)))

    D = float(D or 0)
    if D <= 0:
        return err("目標ダメージ D を正の値で入力してください。")

    try:
        accum_wins = _accum_windows(accum_values, accum_ids, ordered)
    except ValueError as exc:
        return err(str(exc))

    dep_flags = [card_is_hp_dep(c) for c in cards]
    if accum_wins:
        # 蓄積スキルあり (docs/accumulate.md §4)。チェックポイントが蓄積窓の境界に
        # あれば増分は状態非依存のままなので、区間の増分分布を差し替えるだけで
        # 既存の Dinkelbach + 後ろ向き帰納がそのまま使える。
        if hp_mode == "on" and any(dep_flags):
            return err("蓄積スキルと HP依存ダメージ(ミカ型)の併用は未対応です。"
                       "サイドバーの「HP依存ダメージ」を「なし」にしてください。")
        if saves:
            return err("蓄積スキルと凸区切り(セーブポイント)の併用は未対応です。"
                       "凸の区切りを外してから実行してください。")
        try:
            res = restart_accum.analyze_accum(hits, accum_wins, cps, hit_times, D,
                                              seg_success=seg_success)
        except ValueError as exc:
            return err(str(exc))
        model_note = "和モデル + 蓄積スキル"
        model_key = "accum"
    elif hp_mode == "on" and any(dep_flags):
        try:
            hp = HPParams(H=float(hp_H), H1=float(hp_H1),
                          R0=float(hp_R0), R1=float(hp_R1))
            if hp.H == 0 or hp.beta == 0:
                raise ValueError
        except (TypeError, ValueError):
            return err("HP依存パラメータ (H, H1, R0, R1) を正しく入力してください。")
        if all(dep_flags):
            if hp.Htil > 0 and D >= hp.Htil:
                return err(f"目標 D は H̃₁={hp.Htil:,.0f} 未満にしてください(到達不能)。")
            ymix = [y_mixture(m, hp.beta) for m in hits]
            if saves:
                try:
                    res = restart_save.analyze_product(
                        ymix, hp, cps, saves, hit_times, D,
                        seg_success=seg_success)
                except ValueError as exc:
                    return err(str(exc))
            else:
                res = restart_cos.analyze_product(ymix, hp, cps, hit_times, D,
                                                  seg_success=seg_success)
            model_note = "積モデル(HP依存)"
            model_key = "product"
        else:
            specs = hit_specs_from_cards(cards, float(global_crit or 0),
                                         float(global_evade or 0),
                                         damage_mode or "post_decay")
            d_max = mixed_support(specs, hp)[1]
            if D >= d_max:
                return err(f"目標 D は最大可能ダメージ {d_max:,.0f} 未満にしてください(到達不能)。")
            if saves:
                try:
                    res = restart_save.analyze_mixed(
                        specs, hp, cps, saves, hit_times, D,
                        seg_success=seg_success)
                except ValueError as exc:
                    return err(str(exc))
            else:
                res = restart_mixed.analyze_mixed(specs, hp, cps, hit_times, D,
                                                  seg_success=seg_success)
            model_note = "混在モデル(HP依存+通常、グリッドDP)"
            model_key = "mixed"
    else:
        if saves:
            try:
                res = restart_save.analyze(hits, cps, saves, hit_times, D,
                                           seg_success=seg_success)
            except ValueError as exc:
                return err(str(exc))
        else:
            res = restart_cos.analyze(hits, cps, hit_times, D,
                                      seg_success=seg_success)
        model_note = ("和モデル(HP依存カードなし)" if hp_mode == "on"
                      else "和モデル(HP非依存)")
        model_key = "sum"

    # 最適足切りラインの単調化(表示用)。残りダメージが途中で増加する関門は、累積
    # ダメージが単調増加する以上「手前のより高い関門を通過した時点で自動的に満たされ
    # る冗長な関門」である(cp_i 通過 ⇒ cum>=g_i>g_j なので g_j は誰も足切りしない)。
    # 各関門を running max に引き上げても足切り判定・通過率・スループットは不変なので、
    # 実効的で単調な足切りラインを表示する。手動調整パス(update_restart_gates)は
    # 利用者入力をそのまま尊重するため触らない。
    # 凸区切りありでは足切りがセーブ地点の累積ダメージに依存し、表に出しているのは
    # その代表値 (重み付き平均) なので、running max で持ち上げると実際の方策と
    # ずれる。単調化は凸区切りなし (関門がスカラー) のときだけ行う。
    if not saves:
        run_max = 0.0
        for r in res["rows"]:
            run_max = max(run_max, r["gate"])
            r["gate"] = run_max
            r["gate_lo"] = r["gate_hi"] = run_max

    # チェックポイントのヒット数 → カード名 の対応と、最終(完走)カード名を作る。
    memo_by = {mid["index"]: (v or "") for v, mid in zip(memo_values, memo_ids)}
    cum = 0
    cum_to_label = {}        # {str(累積ヒット): カード名}
    last_label = ""
    for pos, (idx, card) in enumerate(ordered):
        cum += int(card.get("hits") or 1)
        # 位置番号を前置してカード名を一意化 (同名カードでも図/表で衝突しない)
        memo = memo_by.get(idx)
        label = f"{pos + 1}: {memo}" if memo else f"カード{pos + 1}"
        cum_to_label[str(cum)] = label
        last_label = label

    # 表示用の行 (label, remaining, section_rate, cumulative_rate, is_final)。
    #   区間通過率 = P(関門通過 | 到達) = 累積_j / 累積_{j-1}
    #   累積通過率 = P(関門1..j を全通過) = forward の pass_rate(joint)
    # 「最終(完走=目標達成)」行を足切り0(残り0)で自動追加する。
    disp = _restart_disp(res, cum_to_label, last_label, D)

    # --- 図: カード名別の最適足切りライン (残りダメージ + 区間/累積通過率) ---
    fig = _cutoff_figure(
        disp, f"最適足切りライン(時短率 {res['speedup']:.2f}x)")

    # --- リスタライン手動調整用の設定 (Store) と スライダー ---
    config = {
        "model": model_key,
        # 実行時点の入力の指紋。以後の変更検知 (flag_restart_stale) に使う。
        "fingerprint": _restart_fingerprint(
            order, card_indices, param_values, param_ids, cp_store,
            seg_times_raw, seg_success_raw, save_raw, D_raw,
            global_crit, global_evade,
            damage_mode, hp_mode, hp_H, hp_H1, hp_R0, hp_R1,
            accum_values, accum_ids),
        "accum": restart_accum.windows_to_json(accum_wins),
        "cards": cards, "crit": float(global_crit or 0),
        "evade": float(global_evade or 0), "damage_mode": damage_mode or "post_decay",
        "cps": cps, "saves": saves, "hit_times": hit_times,
        "seg_success": seg_success, "D": D,
        "hp": ({"H": float(hp_H), "H1": float(hp_H1),
                "R0": float(hp_R0), "R1": float(hp_R1)}
               if model_key in ("product", "mixed") else None),
        "cum_to_label": cum_to_label, "last_label": last_label,
        # 最適ライン (重ね描き用) と基準
        "opt_disp": disp,
        "opt_gates": [r["gate"] for r in res["rows"]],
        "opt_success": res["success"], "opt_throughput": res["throughput"],
        "opt_exp_time": res["exp_time"], "opt_speedup": res["speedup"],
        "base_success": res["baseline"]["success"],
        "base_throughput": res["baseline"]["g"],
        "base_exp_time": res["baseline"]["exp_time"],
    }
    sliders = _gate_sliders(cps, res["rows"], cum_to_label, D)

    # 冗長な関門の注記。区間通過率がほぼ100%の関門は、そのチェックポイントに到達した
    # 試行をほとんど足切りしていない(手前のより厳しい関門で既に絞られている、または
    # まだ見切る段階でない)= 設定から外しても結果は変わらない。最終(完走)行は除外。
    redundant = any(row[2] >= 0.995 for row in disp if not row[4])

    # --- サマリ ---
    base = res["baseline"]
    head = [html.Th("チェックポイント(カード)"),
            html.Th("最適足切り(残りダメージ)"),
            html.Th("区間通過率"), html.Th("累積通過率")]
    if saves:
        head.insert(1, html.Th("凸"))
    rows = [html.Tr(head)]
    for i, row in enumerate(disp):
        label, rem, sect, cumr, is_final, is_save, r_lo, r_hi = row
        style = {"background": "#fff3e0"} if is_final else (
            {"background": "#fff3e0"} if is_save else {})
        rem_txt = f"{rem:,.0f}"
        if abs(r_hi - r_lo) >= 1:
            rem_txt += f"  ({r_lo:,.0f}〜{r_hi:,.0f})"
        cells = [
            html.Td(label if is_final else f"足切り{i + 1}: {label}"),
            html.Td(rem_txt),
            html.Td(f"{sect:.1%}"),
            html.Td(f"{cumr:.1%}"),
        ]
        if saves:
            cells.insert(1, html.Td("🚩 区切り" if is_save else ""))
        rows.append(html.Tr(cells, style=style))
    table = html.Table(rows, style={"borderCollapse": "collapse", "marginTop": "6px"},
                       className="restart-table")
    children = [html.Div(model_note, style={"fontSize": "0.85rem", "color": "#888"})]
    if saves:
        children += [
            html.Div([
                html.Strong("結果: "),
                f"目標 {D:,.0f} 到達までの期待総時間 {res['exp_time']:.2f}",
            ]),
            html.Div(
                f"凸の中で足切りしない場合: {base['exp_time']:.2f}"
                f"  →  時短率 {res['speedup']:.2f}x"
                "(凸区切りの関門は基準側でも最適のまま。"
                "そこまで外すと到達不能な確定が起きて期待時間が発散するため)",
                style={"color": "#555", "fontSize": "0.9rem"},
            ),
            _blocks_table(res, cps, cum_to_label),
        ]
    else:
        children += [
            html.Div([
                html.Strong("結果: "),
                f"成功率 {res['success']:.3%} / 平均所要時間 {res['exp_time']:.2f} / "
                f"スループット {res['throughput']:.3e}(成功/時間)",
            ]),
            html.Div(
                f"足切り無し: 成功率 {base['success']:.3%} / 時間 {base['exp_time']:.2f} / "
                f"スループット {base['g']:.3e}  →  時短率 {res['speedup']:.2f}x",
                style={"color": "#555", "fontSize": "0.9rem"},
            ),
            # 凸区切りありの表示 (期待総時間) と読み比べられるよう併記する。
            # 「平均所要時間」は 1 試行の時間、こちらは成功までの延べ時間。
            html.Div(
                "目標到達までの期待総時間(成功するまで繰り返した場合): "
                + (f"{1.0 / res['throughput']:.2f}"
                   if res["throughput"] > 0 else "—"),
                style={"color": "#555", "fontSize": "0.85rem"},
            ),
        ]
    children.append(table)
    if saves:
        children.append(html.Div(
            "※ 凸区切りありでは、足切りラインは直前の凸区切り時点の累積ダメージに"
            "依存します。表・図の値は入口分布での代表値で、括弧内が動く幅です"
            "(前の凸で稼げているほど足切りは上がります)。"
            "通過率は「その凸の1試行のうち、ここまで通過した割合」で、凸をまたいで"
            "掛け合わせるものではありません。",
            style={"color": "#b35900", "fontSize": "0.85rem", "marginTop": "8px"},
        ))
        if not res.get("feasible", True):
            children.append(html.Div(
                "⚠ 一部の状態から目標に到達できない経路が残っています。"
                "凸区切りの位置か目標ダメージを見直してください。",
                style={"color": "#d63031", "fontSize": "0.85rem", "marginTop": "6px"},
            ))
    if redundant:
        children.append(html.Div(
            "※ 区間通過率がほぼ100%の関門は実質的に足切りしておらず、"
            "設定から外しても結果は変わりません。",
            style={"color": "#b35900", "fontSize": "0.85rem", "marginTop": "8px"},
        ))
    children += _accum_notes(res)
    children += _accum_pending_notes(res, cum_to_label, D)
    summary = html.Div(children)
    return fig, summary, config, sliders


def _accum_notes(res, detail: bool = True) -> list:
    """蓄積スキルごとの診断行 (飽和確率・爆発平均・溢れ)。"""
    stats = res.get("accum_stats") or []
    if not stats:
        return []
    out = [html.Div(
        "⚡ 蓄積スキルを含めて計算しています。足切りラインは「実ダメージ + 確定した"
        "蓄積分」で判定してください(画面に出る値と同じです)。",
        style={"color": "#8a6d00", "fontSize": "0.85rem", "marginTop": "8px"})]
    if not detail:
        return out
    for st in stats:
        out.append(html.Div(
            f"⚡ {st['name']}: 蓄積上限 {st['cap_mean']:,.0f} / "
            f"窓内ダメージ平均 {st['damage_mean']:,.0f} / "
            f"飽和確率 {st['sat_prob'] * 100:.1f}% / "
            f"爆発ダメージ平均 {st['burst_mean']:,.0f} / "
            f"期待溢れ {st['overflow_mean']:,.0f}",
            style={"color": "#555", "fontSize": "0.85rem"}))
    return out


def _accum_pending_notes(res, cum_to_label, D) -> list:
    """爆発がまだ着弾していない関門について、画面基準の読み替えを出す。

    足切りラインは「実ダメージ + 確定した蓄積分」(実効ダメージ) で出ているため、
    爆発より手前の関門では画面の数字にその分が乗っていない。最適方策は変わらない
    (確定プールは確保済みの情報として使える) が、読むときに足し戻す必要がある。
    """
    stats = res.get("accum_stats") or []
    if not stats or not cum_to_label:
        return []
    lines = []
    for row in res["rows"]:
        m = int(row["checkpoint"])
        # 爆発は burst_hit の直後 → 先頭 m Hit に含まれるのは burst_hit < m のとき
        late = [st for st in stats if int(st.get("burst_hit", -1)) >= m]
        if not late:
            continue
        mean = sum(st["burst_mean"] for st in late)
        lo = sum(st.get("burst_lo") or 0.0 for st in late)
        hi = sum(st.get("burst_hi") or 0.0 for st in late)
        label = cum_to_label.get(str(m), f"{m} Hit")
        rem = float(D) - float(row["gate"])
        rng = f" ({lo:,.0f}〜{hi:,.0f})" if hi - lo >= 1 else ""
        lines.append(html.Div(
            f"・{label}: {'・'.join(st['name'] for st in late)} の爆発 "
            f"{mean:,.0f}{rng} が未着弾 → 画面の「残りダメージ」では "
            f"{rem + mean:,.0f} 前後以下なら続行",
            style={"color": "#555", "fontSize": "0.85rem"}))
    if not lines:
        return []
    return [html.Div(
        "⚡ 次の関門は爆発より手前なので、画面にはまだ蓄積分が乗っていません。"
        "表の「残りダメージ」に爆発ぶんを足して読んでください"
        "(チェックポイントを爆発の後に置けば、そのまま読めます)。",
        style={"color": "#8a6d00", "fontSize": "0.85rem", "marginTop": "8px"}),
        *lines]


def _blocks_table(res, cps, cum_to_label):
    """凸 (セーブポイントで区切られたブロック) ごとの期待時間・平均試行回数の表。"""
    ends = [*res.get("save_points", []), None]
    rows = [html.Tr([html.Th("凸"), html.Th("終わり(カード)"),
                     html.Th("期待時間"), html.Th("1試行の完走率"),
                     html.Th("平均試行回数")])]
    for i, b in enumerate(res.get("blocks", [])):
        end = ends[i] if i < len(ends) else None
        label = ("最後まで" if end is None
                 else cum_to_label.get(str(end), f"{end}ヒット目"))
        rows.append(html.Tr([
            html.Td(f"{b['block']}凸目"),
            html.Td(label),
            html.Td(f"{b['exp_time']:.2f}"),
            html.Td(f"{b['completion']:.1%}"),
            html.Td(f"{b['attempts']:.2f}" if b["attempts"] < 1e6 else "—"),
        ]))
    return html.Table(rows,
                      style={"borderCollapse": "collapse", "margin": "8px 0"},
                      className="restart-table")


# ---------------------------------------------------------------------------
# 多段リスタ: リスタライン手動調整 (スライダー → 成功率/スループット 再計算)
# ---------------------------------------------------------------------------
def _gate_sliders(cps, rows, cum_to_label, D):
    """各足切りの「残りダメージ」スライダーを生成。初期値 = 最適ライン。"""
    sliders = []
    Dmax = int(round(D))
    step = 1
    for k, m in enumerate(cps):
        label = cum_to_label.get(str(m), f"{m}ヒット目")
        if rows[k].get("save"):
            label = f"🚩{label}(凸区切り)"
        opt_remain = min(Dmax, max(0, int(round(D - rows[k]["gate"]))))
        sliders.append(html.Div(
            [
                html.Div(f"足切り{k + 1}:{label}(残りダメージ)",
                         style={"fontSize": "0.83rem", "fontWeight": "bold"}),
                dcc.Slider(
                    id={"type": "restart-gate-slider", "index": m},
                    min=0, max=Dmax, step=step, value=opt_remain,
                    marks={0: "0", Dmax: f"{Dmax:,}"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            style={"marginBottom": "10px"},
        ))
    if not sliders:
        return []
    sliders.append(html.Button(
        "最適ラインに戻す", id="restart-gate-reset-btn", n_clicks=0,
        style={"cursor": "pointer", "padding": "5px 12px", "marginTop": "2px"}))
    return sliders


# ---------------------------------------------------------------------------
# 多段リスタ: 解析結果が古くなったことの通知
#   run_restart は「解析実行」ボタンでしか発火しないため、実行後に並べ替え・
#   パラメータ変更をしても図/表/スライダーは古い前提のまま残る。特に手動調整
#   グラフは restart-config に焼き付いたカード配列で再計算するので、スライダー
#   を動かせば「更新されているように見える」のが厄介。指紋の不一致で警告する。
# ---------------------------------------------------------------------------
_STALE_STYLE = {
    "marginTop": "10px", "padding": "8px 10px", "borderRadius": "4px",
    "background": "#fff3cd", "border": "1px solid #e0a800",
    "color": "#7a5c00", "fontSize": "0.85rem",
}


@callback(
    Output("restart-stale-note", "children"),
    Output("restart-stale-note-interactive", "children"),
    Input("restart-config", "data"),
    Input("restart-D", "value"),
    Input("sorted-indices", "data"),
    Input("card-indices", "data"),
    Input({"type": "param", "param": ALL, "index": ALL}, "value"),
    Input("restart-cp-store", "data"),
    Input("restart-seg-time-store", "data"),
    Input("restart-seg-success-store", "data"),
    Input("restart-save-store", "data"),
    Input("global-crit-rate", "value"),
    Input("global-evade-rate", "value"),
    Input("damage-mode", "value"),
    Input("hp-mode", "value"),
    Input("hp-H", "value"),
    Input("hp-H1", "value"),
    Input("hp-R0", "value"),
    Input("hp-R1", "value"),
    Input({"type": "accum", "field": ALL, "index": ALL}, "value"),
    State({"type": "param", "param": ALL, "index": ALL}, "id"),
    State({"type": "accum", "field": ALL, "index": ALL}, "id"),
)
def flag_restart_stale(cfg, D, order, card_indices, param_values, cp_store,
                       seg_times, seg_success, save_store,
                       global_crit, global_evade,
                       damage_mode, hp_mode, hp_H, hp_H1, hp_R0, hp_R1,
                       accum_values, param_ids, accum_ids):
    """実行後に入力が変わっていたら「再実行してください」を出す。"""
    if not cfg or not cfg.get("fingerprint"):
        return "", ""          # まだ一度も実行していない
    now = _restart_fingerprint(order, card_indices, param_values, param_ids,
                               cp_store, seg_times, seg_success, save_store, D,
                               global_crit, global_evade, damage_mode,
                               hp_mode, hp_H, hp_H1, hp_R0, hp_R1,
                               accum_values, accum_ids)
    if now == cfg["fingerprint"]:
        return "", ""
    return (
        html.Div("⚠ 解析実行のあとにカード(並び順・パラメータ)または設定が"
                 "変更されています。下の結果は変更前のものです。"
                 "「解析実行」を押し直してください。", style=_STALE_STYLE),
        html.Div("⚠ 変更前のカード構成で計算しています。"
                 "スライダーを動かしても最新の並び順・パラメータは反映されません。"
                 "先に「解析実行」を押し直してください。", style=_STALE_STYLE),
    )


def _rebuild_for_config(cfg):
    """config からヒット混合・hit_times・(積モデルなら)ymix/hp を再構築する。"""
    hits = build_hit_mixtures(cfg["cards"], cfg["crit"], cfg["evade"],
                              cfg["damage_mode"])
    return hits


@callback(
    Output({"type": "restart-gate-slider", "index": ALL}, "value"),
    Input("restart-gate-reset-btn", "n_clicks"),
    State({"type": "restart-gate-slider", "index": ALL}, "id"),
    State("restart-config", "data"),
    prevent_initial_call=True,
)
def reset_restart_gates(n_clicks, slider_ids, cfg):
    """「最適ラインに戻す」: 各スライダーを最適ラインの残りダメージへ。"""
    if not n_clicks or not cfg or not slider_ids:
        raise PreventUpdate
    D = float(cfg["D"])
    Dmax = int(round(D))
    remains_by = {m: min(Dmax, max(0, int(round(D - g))))
                  for m, g in zip(cfg["cps"], cfg["opt_gates"])}
    return [remains_by.get(sid["index"], 0) for sid in slider_ids]


@callback(
    Output("restart-interactive-graph", "figure"),
    Output("restart-interactive-summary", "children"),
    Input({"type": "restart-gate-slider", "index": ALL}, "value"),
    State({"type": "restart-gate-slider", "index": ALL}, "id"),
    State("restart-config", "data"),
    prevent_initial_call=True,
)
def update_restart_interactive(slider_values, slider_ids, cfg):
    if not cfg or not slider_ids:
        raise PreventUpdate

    D = float(cfg["D"])
    cps = cfg["cps"]
    remains_by = {sid["index"]: (v if v is not None else 0.0)
                  for sid, v in zip(slider_ids, slider_values)}
    # cps の順に残りダメージ → 累積ダメージしきい値 (gate) へ変換
    manual_gates = [D - float(remains_by.get(m, D)) for m in cps]

    hits = _rebuild_for_config(cfg)
    seg_success = cfg.get("seg_success")
    saves = cfg.get("saves") or []
    if saves:
        try:
            if cfg["model"] == "product":
                hp = HPParams(**cfg["hp"])
                ymix = [y_mixture(mm, hp.beta) for mm in hits]
                res = restart_save.analyze_product(
                    ymix, hp, cps, saves, cfg["hit_times"], D,
                    manual_gates=manual_gates, seg_success=seg_success)
            elif cfg["model"] == "mixed":
                hp = HPParams(**cfg["hp"])
                specs = hit_specs_from_cards(cfg["cards"], cfg["crit"],
                                             cfg["evade"], cfg["damage_mode"])
                res = restart_save.analyze_mixed(
                    specs, hp, cps, saves, cfg["hit_times"], D,
                    manual_gates=manual_gates, seg_success=seg_success)
            else:
                res = restart_save.analyze(
                    hits, cps, saves, cfg["hit_times"], D,
                    manual_gates=manual_gates, seg_success=seg_success)
        except ValueError as exc:
            return go.Figure(), html.Div(f"⚠ {exc}", style={"color": "#d63031"})
    elif cfg["model"] == "product":
        hp = HPParams(**cfg["hp"])
        ymix = [y_mixture(mm, hp.beta) for mm in hits]
        res = restart_cos.analyze_product(ymix, hp, cps, cfg["hit_times"], D,
                                          manual_gates=manual_gates,
                                          seg_success=seg_success)
    elif cfg["model"] == "mixed":
        hp = HPParams(**cfg["hp"])
        specs = hit_specs_from_cards(cfg["cards"], cfg["crit"], cfg["evade"],
                                     cfg["damage_mode"])
        res = restart_mixed.analyze_mixed(specs, hp, cps, cfg["hit_times"], D,
                                          manual_gates=manual_gates,
                                          seg_success=seg_success)
    elif cfg["model"] == "accum":
        try:
            res = restart_accum.analyze_accum(
                hits, restart_accum.windows_from_json(cfg.get("accum")),
                cps, cfg["hit_times"], D, manual_gates=manual_gates,
                seg_success=seg_success)
        except ValueError as exc:
            return go.Figure(), html.Div(f"⚠ {exc}", style={"color": "#d63031"})
    else:
        res = restart_cos.analyze(hits, cps, cfg["hit_times"], D,
                                  manual_gates=manual_gates,
                                  seg_success=seg_success)

    disp = _restart_disp(res, cfg["cum_to_label"], cfg["last_label"], D)
    fig = _cutoff_figure(disp, "あなたの設定したリスタライン vs 最適",
                         color="#0984e3", ref_disp=cfg["opt_disp"])

    # 最適・基準との比較サマリ
    def pct(x):
        return f"{x:.3%}"
    opt_s, opt_g, opt_t = cfg["opt_success"], cfg["opt_throughput"], cfg["opt_exp_time"]
    base_t = cfg.get("base_exp_time") or 0.0
    if saves:
        ratio = (res["exp_time"] / opt_t) if opt_t > 0 else float("nan")
        speedup = (base_t / res["exp_time"]) if res["exp_time"] > 0 else float("nan")
        summary = html.Div([
            html.Div([
                html.Strong("あなたの設定: "),
                f"期待総時間 {res['exp_time']:.2f}(時短率 {speedup:.2f}x)",
            ]),
            html.Div(
                f"最適比: 期待総時間 {ratio:.1%}"
                f"(最適 = {opt_t:.2f}・時短率 {cfg['opt_speedup']:.2f}x)",
                style={"color": "#555", "fontSize": "0.88rem"},
            ),
        ])
        return fig, summary
    base_g = cfg["base_throughput"]
    speedup = (res["throughput"] / base_g) if base_g > 0 else float("nan")
    d_succ = res["success"] - opt_s
    g_ratio = (res["throughput"] / opt_g) if opt_g > 0 else float("nan")
    summary = html.Div([
        html.Div([
            html.Strong("あなたの設定: "),
            f"成功率 {pct(res['success'])} / 平均時間 {res['exp_time']:.2f} / "
            f"スループット {res['throughput']:.3e}(時短率 {speedup:.2f}x)",
        ]),
        html.Div(
            f"最適比: 成功率 {d_succ:+.3%}pt / スループット {g_ratio:.1%}"
            f"(最適 = 成功率 {pct(opt_s)}・スループット {opt_g:.3e}・時短率 "
            f"{cfg['opt_speedup']:.2f}x)",
            style={"color": "#555", "fontSize": "0.88rem"},
        ),
        *_accum_notes(res, detail=False),
    ])
    return fig, summary


# ---------------------------------------------------------------------------
# スキル順探索: ヘルパー
# ---------------------------------------------------------------------------
def _so_names_copiers(name_values, name_ids, copier_values, copier_ids,
                      n_cards=SO_MAX_CARDS):
    """カード名リスト(空欄はデフォルト名で補完)と複製キャラ添字集合を返す。

    使用枚数 n_cards より後ろのカード設定行は無視する。
    """
    names = [""] * SO_MAX_CARDS
    for v, nid in zip(name_values, name_ids):
        names[nid["index"]] = (v or "").strip()
    copiers = {cid["index"] for v, cid in zip(copier_values, copier_ids)
               if v and cid["index"] < n_cards}
    # 表示名は「ドアル/アル」のうち先頭だけ (残りは TL 照合用の別名)
    disp_names = [tl_parse.display_name(nm) or f"カード{i + 1}"
                  for i, nm in enumerate(names)]
    return names, disp_names, copiers


def _so_int(value, default):
    """ドロップダウンの文字列値を int に。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _so_step_order_from(children):
    """手順コンテナ children の並び順から step index のリストを返す。"""
    order = []
    for c in children:
        if isinstance(c, dict):
            cid = c.get("props", {}).get("id")
        else:
            cid = getattr(c, "id", None)
        if isinstance(cid, dict) and cid.get("type") == "so-step":
            order.append(cid["index"])
    return order


def _so_step_desc(step, names, hand_size=skill_order.HAND_SIZE_NORMAL):
    """手順1ステップの表示文字列。"""
    if step.retreat:
        return f"↩{names[step.skill]}撤退"
    if step.skill is None:
        s = "＊"
    elif step.use_copy:
        s = f"{names[step.skill]}(コピー)"
    else:
        s = names[step.skill]
        if step.copy_target is not None:
            s += f"→{names[step.copy_target]}(コピー)"
    if step.slot is not None:
        s += f"@{skill_order.slot_labels(hand_size)[step.slot - 1]}"
    if step.draw:
        s += "+ドロー"
    return s


def _so_error(msg):
    return html.Div(f"⚠ {msg}", style={"color": "#d63031", "fontWeight": "bold"})


# ---------------------------------------------------------------------------
# スキル順探索: モード(手札枚数) / カード枚数
# ---------------------------------------------------------------------------
@callback(
    Output("so-card-count", "options"),
    Output("so-card-count", "value"),
    Input("so-hand-size", "value"),
    Input("so-restore-count", "data"),
    State("so-card-count", "value"),
)
def so_sync_card_count(hand_size_raw, restore_count, count_raw):
    """モードに応じてカード枚数の選択肢を切り替える(決戦は10枚まで)。

    自動保存からの復元 / 全クリア (app/frontend/persist.py) はモードと枚数を
    同時に戻す。so-hand-size だけを見ると「モード切替 → 既定枚数」の分岐に
    吸い込まれて指定した枚数が消えるので、復元側は so-restore-count 経由で
    枚数を渡す ({"count": "6", "nonce": ...})。
    """
    hand_size = _so_int(hand_size_raw, 3)
    options = so_card_count_options(hand_size)
    hi = SO_MAX_CARDS if hand_size >= 5 else SO_DEFAULT_CARDS
    count = _so_int(count_raw, SO_DEFAULT_CARDS)
    triggered = {t["prop_id"] for t in (ctx.triggered or [])}
    if "so-restore-count.data" in triggered and isinstance(restore_count, dict):
        count = _so_int(restore_count.get("count"), SO_DEFAULT_CARDS)
    # モード切替時は既定枚数(通常6 / 決戦10)へ、範囲外なら丸める
    elif "so-hand-size.value" in triggered:
        count = hi
    return options, str(min(max(count, 1), hi))


@callback(
    Output({"type": "so-name-row", "index": ALL}, "style"),
    Output("so-cards-title", "children"),
    Input("so-card-count", "value"),
    State({"type": "so-name-row", "index": ALL}, "style"),
    State({"type": "so-name-row", "index": ALL}, "id"),
)
def so_toggle_name_rows(count_raw, styles, row_ids):
    """使用枚数を超えるカード設定行を隠す。"""
    n_cards = _so_int(count_raw, SO_DEFAULT_CARDS)
    out = []
    for style, rid in zip(styles, row_ids):
        style = dict(style or {})
        if rid["index"] < n_cards:
            style["display"] = "flex"
        else:
            style["display"] = "none"
        out.append(style)
    return out, f"カード設定({n_cards}枚)"


@callback(
    Output({"type": "so-step-slot", "index": ALL}, "options"),
    Output({"type": "so-step-slot", "index": ALL}, "value"),
    Input("so-hand-size", "value"),
    State({"type": "so-step-slot", "index": ALL}, "value"),
)
def so_sync_slot_options(hand_size_raw, slot_values):
    """手札枚数に合わせてスロット指定の選択肢を更新する。"""
    hand_size = _so_int(hand_size_raw, 3)
    options = so_slot_options(hand_size)
    values = [v if (v == "any" or _so_int(v, 1) <= hand_size) else "any"
              for v in slot_values]
    return [options] * len(slot_values), values


# ---------------------------------------------------------------------------
# スキル順探索: 手順ステップの追加 / 削除 / 生徒選択時の自動行追加
# ---------------------------------------------------------------------------
@callback(
    Output("so-steps-container", "children"),
    Output("so-step-order", "data"),
    Output("so-next-step", "data"),
    Output("so-tl-msg", "children"),
    Input("so-add-step-btn", "n_clicks"),
    Input("so-tl-import-btn", "n_clicks"),
    Input({"type": "so-step-remove", "index": ALL}, "n_clicks"),
    Input({"type": "so-step-up", "index": ALL}, "n_clicks"),
    Input({"type": "so-step-down", "index": ALL}, "n_clicks"),
    Input({"type": "so-step-insert", "index": ALL}, "n_clicks"),
    Input({"type": "so-step-skill", "index": ALL}, "value"),
    State("so-tl-text", "value"),
    State("so-steps-container", "children"),
    State("so-next-step", "data"),
    State({"type": "so-step-skill", "index": ALL}, "id"),
    State({"type": "so-name", "index": ALL}, "value"),
    State({"type": "so-name", "index": ALL}, "id"),
    State({"type": "so-copier", "index": ALL}, "value"),
    State({"type": "so-copier", "index": ALL}, "id"),
    State("so-card-count", "value"),
    State("so-hand-size", "value"),
    prevent_initial_call=True,
)
def so_update_steps(_add, _import, _rm, _up, _down, _ins, skill_values,
                    tl_text, children, next_idx, skill_ids,
                    name_values, name_ids, copier_values, copier_ids,
                    count_raw, hand_size_raw):
    trigger = ctx.triggered_id
    children = children or []
    n_cards = _so_int(count_raw, SO_DEFAULT_CARDS)
    hand_size = _so_int(hand_size_raw, 3)
    names, disp_names, copiers = _so_names_copiers(
        name_values, name_ids, copier_values, copier_ids, n_cards)
    skill_opts = so_skill_options(disp_names, copiers, n_cards)
    target_opts = so_target_options(disp_names, copiers, n_cards)

    def new_step(index, **kw):
        return make_so_step(index, skill_opts, target_opts,
                            hand_size=hand_size, **kw)

    if trigger == "so-add-step-btn":
        children.append(new_step(next_idx))
        return (children, _so_step_order_from(children), next_idx + 1,
                dash.no_update)

    if trigger == "so-tl-import-btn":
        if not _triggered_clicked():
            raise PreventUpdate
        return _so_import_tl(tl_text, names, copiers, n_cards, next_idx,
                             new_step, children)

    if not (isinstance(trigger, dict) and _triggered_clicked()):
        raise PreventUpdate

    ttype = trigger.get("type")
    tidx = trigger.get("index")
    order = _so_step_order_from(children)
    if tidx not in order:
        raise PreventUpdate

    if ttype == "so-step-remove":
        if len(order) <= 1:
            raise PreventUpdate
        children = [c for c in children
                    if _so_step_order_from([c]) != [tidx]]
        return (children, _so_step_order_from(children), next_idx,
                dash.no_update)

    # 行の並べ替え / 挿入 (手順番号は CSS カウンタなので自動で振り直される)
    if ttype in ("so-step-up", "so-step-down"):
        pos = order.index(tidx)
        dst = pos - 1 if ttype == "so-step-up" else pos + 1
        if not 0 <= dst < len(order):
            raise PreventUpdate
        children[pos], children[dst] = children[dst], children[pos]
        return (children, _so_step_order_from(children), next_idx,
                dash.no_update)

    if ttype == "so-step-insert":
        children.insert(order.index(tidx) + 1, new_step(next_idx))
        return (children, _so_step_order_from(children), next_idx + 1,
                dash.no_update)

    if ttype == "so-step-skill":
        # 生徒を選択したら、最後の行が埋まっている場合に空の行を自動追加する
        skill_by = {i["index"]: v for v, i in zip(skill_values, skill_ids)}
        if skill_by.get(order[-1]):
            children.append(new_step(next_idx))
            return (children, _so_step_order_from(children), next_idx + 1,
                    dash.no_update)
        raise PreventUpdate

    raise PreventUpdate


def _so_import_tl(text, names, copiers, n_cards, next_idx, new_step, children):
    """TLテキストを手順行に変換する。"""
    if not (text or "").strip():
        return (dash.no_update, dash.no_update, dash.no_update,
                _so_error("TLテキストが空です。"))
    if not any((names[i] or "").strip() for i in range(n_cards)):
        return (dash.no_update, dash.no_update, dash.no_update,
                _so_error("先にカード設定へキャラ名を入力してください。"))

    steps, warns = tl_parse.parse_timeline(text, names[:n_cards], copiers)
    if not steps:
        return (dash.no_update, dash.no_update, dash.no_update,
                _so_error("カード使用を1つも読み取れませんでした。"
                          "カード設定のキャラ名とTLの表記が一致しているか"
                          "確認してください。"))

    kind_value = {tl_parse.KIND_COPY: "c", tl_parse.KIND_RETREAT: "r"}
    rows, idx = [], next_idx
    for st in steps:
        rows.append(new_step(
            idx,
            skill=f"{kind_value.get(st.kind, 'n')}{st.skill}",
            target=None if st.target is None else str(st.target),
            draw=st.draw,
            memo=st.memo,
        ))
        idx += 1
    rows.append(new_step(idx))          # 末尾の空行
    idx += 1

    msg = [html.Div(f"✅ {len(steps)}手を読み込みました。",
                    style={"color": "#0a7c2f", "fontWeight": "bold"})]
    msg += [html.Div(f"⚠ {w}", style={"color": "#b8860b"}) for w in warns]
    return rows, _so_step_order_from(rows), idx, html.Div(msg)


# ---------------------------------------------------------------------------
# スキル順探索: 複製スキルを選択した行だけ「複製対象」を表示する
# ---------------------------------------------------------------------------
_SO_TARGET_STYLE = {"width": "130px", "flexShrink": "0"}


@callback(
    Output({"type": "so-step-target", "index": MATCH}, "style"),
    Input({"type": "so-step-skill", "index": MATCH}, "value"),
    Input({"type": "so-copier", "index": ALL}, "value"),
    State({"type": "so-copier", "index": ALL}, "id"),
    State("so-card-count", "value"),
)
def so_toggle_target(skill_value, copier_values, copier_ids, count_raw):
    n_cards = _so_int(count_raw, SO_DEFAULT_CARDS)
    copiers = {cid["index"] for v, cid in zip(copier_values, copier_ids)
               if v and cid["index"] < n_cards}
    if (skill_value and skill_value.startswith("n")
            and int(skill_value[1:]) in copiers):
        return _SO_TARGET_STYLE
    return {**_SO_TARGET_STYLE, "display": "none"}


# ---------------------------------------------------------------------------
# スキル順探索: カード名 / 複製フラグ変更 → ドロップダウン選択肢を更新
# ---------------------------------------------------------------------------
@callback(
    Output({"type": "so-step-skill", "index": ALL}, "options"),
    Output({"type": "so-step-target", "index": ALL}, "options"),
    Input({"type": "so-name", "index": ALL}, "value"),
    Input({"type": "so-copier", "index": ALL}, "value"),
    Input("so-card-count", "value"),
    State({"type": "so-name", "index": ALL}, "id"),
    State({"type": "so-copier", "index": ALL}, "id"),
    State({"type": "so-step-skill", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def so_refresh_options(name_values, copier_values, count_raw,
                       name_ids, copier_ids, skill_dd_ids):
    n_cards = _so_int(count_raw, SO_DEFAULT_CARDS)
    _, disp_names, copiers = _so_names_copiers(
        name_values, name_ids, copier_values, copier_ids, n_cards)
    n = len(skill_dd_ids)
    return ([so_skill_options(disp_names, copiers, n_cards)] * n,
            [so_target_options(disp_names, copiers, n_cards)] * n)


# ---------------------------------------------------------------------------
# スキル順探索: 制約行の追加 / 削除
# ---------------------------------------------------------------------------
@callback(
    Output("so-cons-container", "children"),
    Output("so-next-con", "data"),
    Input("so-add-con-btn", "n_clicks"),
    Input({"type": "so-con-remove", "index": ALL}, "n_clicks"),
    State("so-cons-container", "children"),
    State("so-next-con", "data"),
    prevent_initial_call=True,
)
def so_update_constraints(_add, _rm, children, next_idx):
    trigger = ctx.triggered_id
    children = children or []

    if trigger == "so-add-con-btn":
        children.append(make_so_constraint(next_idx))
        return children, next_idx + 1

    if isinstance(trigger, dict) and trigger.get("type") == "so-con-remove":
        if not _triggered_clicked():
            raise PreventUpdate
        rm = trigger["index"]
        children = [
            c for c in children
            if not (isinstance(c, dict)
                    and c.get("props", {}).get("id", {}).get("type") == "so-con"
                    and c["props"]["id"].get("index") == rm)
        ]
        return children, next_idx

    raise PreventUpdate


# ---------------------------------------------------------------------------
# スキル順探索: 実行
# ---------------------------------------------------------------------------
@callback(
    Output("so-results", "children"),
    Input("so-run-btn", "n_clicks"),
    State("so-step-order", "data"),
    State({"type": "so-name", "index": ALL}, "value"),
    State({"type": "so-name", "index": ALL}, "id"),
    State({"type": "so-copier", "index": ALL}, "value"),
    State({"type": "so-copier", "index": ALL}, "id"),
    State({"type": "so-step-skill", "index": ALL}, "value"),
    State({"type": "so-step-skill", "index": ALL}, "id"),
    State({"type": "so-step-target", "index": ALL}, "value"),
    State({"type": "so-step-target", "index": ALL}, "id"),
    State({"type": "so-step-slot", "index": ALL}, "value"),
    State({"type": "so-step-slot", "index": ALL}, "id"),
    State({"type": "so-step-draw", "index": ALL}, "value"),
    State({"type": "so-step-draw", "index": ALL}, "id"),
    State({"type": "so-step-memo", "index": ALL}, "value"),
    State({"type": "so-step-memo", "index": ALL}, "id"),
    State({"type": "so-con-type", "index": ALL}, "value"),
    State({"type": "so-con-type", "index": ALL}, "id"),
    State({"type": "so-con-steps", "index": ALL}, "value"),
    State({"type": "so-con-steps", "index": ALL}, "id"),
    State("so-limit", "value"),
    State("so-card-count", "value"),
    State("so-hand-size", "value"),
    prevent_initial_call=True,
)
def so_run(_n, step_order,
           name_values, name_ids, copier_values, copier_ids,
           skill_values, skill_ids, target_values, target_ids,
           slot_values, slot_ids, draw_values, draw_ids,
           memo_values, memo_ids,
           con_types, con_type_ids, con_steps, con_step_ids,
           limit, count_raw, hand_size_raw):
    n_cards = _so_int(count_raw, SO_DEFAULT_CARDS)
    hand_size = _so_int(hand_size_raw, skill_order.HAND_SIZE_NORMAL)
    _, disp_names, copiers = _so_names_copiers(
        name_values, name_ids, copier_values, copier_ids, n_cards)

    # ステップ属性を index → 値 の辞書に集約し、表示順 (step_order) で組み立てる
    skill_by = {i["index"]: v for v, i in zip(skill_values, skill_ids)}
    target_by = {i["index"]: v for v, i in zip(target_values, target_ids)}
    slot_by = {i["index"]: v for v, i in zip(slot_values, slot_ids)}
    draw_by = {i["index"]: bool(v) for v, i in zip(draw_values, draw_ids)}
    memo_by = {i["index"]: (v or "").strip()
               for v, i in zip(memo_values, memo_ids)}

    # 未選択(空)の手順は無視する
    step_order = [i for i in (step_order or [])
                  if i in skill_by and skill_by.get(i)]
    if not step_order:
        return _so_error("手順がありません。手順で生徒を選択してください。")

    plan = []
    memos = []
    retreat_steps = set()      # 撤退ステップの手順番号(1始まり)
    retreated = set()
    for pos, sidx in enumerate(step_order, start=1):
        raw = skill_by[sidx]
        memos.append(memo_by.get(sidx, ""))
        slot_raw = slot_by.get(sidx) or "any"
        slot = int(slot_raw) if slot_raw != "any" else None
        if slot is not None and slot > hand_size:
            slot = None
        draw = draw_by.get(sidx, False)

        if raw == "any":
            plan.append(skill_order.Step(None, slot=slot, draw=draw))
            continue

        if raw.startswith("r"):
            skill = int(raw[1:])
            if skill >= n_cards:
                return _so_error(f"手順{pos}: 使用しないカードが指定されています。")
            if skill in retreated:
                return _so_error(
                    f"手順{pos}: {disp_names[skill]} は既に撤退しています。")
            retreated.add(skill)
            retreat_steps.add(pos)
            plan.append(skill_order.Step(skill, retreat=True))
            continue

        use_copy = raw.startswith("c")
        skill = int(raw[1:])
        if skill >= n_cards:
            return _so_error(f"手順{pos}: 使用しないカードが指定されています。")
        if use_copy:
            # コピー札は複製元が撤退しない限り残るので、複製対象の撤退は問わない
            if skill in copiers:
                return _so_error(
                    f"手順{pos}: 複製スキル自身のコピーは指定できません。")
            plan.append(skill_order.Step(
                skill, use_copy=True, slot=slot, draw=draw))
            continue

        if skill in retreated:
            return _so_error(
                f"手順{pos}: {disp_names[skill]} は撤退済みなので使用できません。")

        target = None
        if skill in copiers:
            traw = target_by.get(sidx)
            if traw is None or traw == "":
                return _so_error(
                    f"手順{pos}: {disp_names[skill]} は複製スキルです。"
                    "複製対象を選択してください。")
            target = int(traw)
            if target >= n_cards or target == skill or target in copiers:
                return _so_error(f"手順{pos}: 複製対象が不正です。")
            if target in retreated:
                return _so_error(
                    f"手順{pos}: {disp_names[target]} は撤退済みなので"
                    "複製できません。")
        plan.append(skill_order.Step(
            skill, copy_target=target, slot=slot, draw=draw))

    # 制約の組み立て
    con_type_by = {i["index"]: v for v, i in zip(con_types, con_type_ids)}
    con_steps_by = {i["index"]: v for v, i in zip(con_steps, con_step_ids)}
    constraints = []
    for cidx, ctype in con_type_by.items():
        text = (con_steps_by.get(cidx) or "").strip()
        if not text:
            continue
        try:
            nums = [int(x) for x in text.replace("、", ",").split(",") if x.strip()]
        except ValueError:
            return _so_error(f"制約「{text}」: 手順番号はカンマ区切りの数値で"
                             "指定してください(例: 1,3)。")
        if len(set(nums)) < 2:
            return _so_error(f"制約「{text}」: 手順番号を2つ以上指定してください。")
        bad = [x for x in nums if not (1 <= x <= len(plan))]
        if bad:
            return _so_error(f"制約「{text}」: 手順番号 {bad} が範囲外です"
                             f"(1〜{len(plan)})。")
        bad = [x for x in nums if x in retreat_steps]
        if bad:
            return _so_error(f"制約「{text}」: 手順番号 {bad} は撤退ステップなので"
                             "スロット制約の対象にできません。")
        idx0 = [x - 1 for x in nums]
        if ctype == "same":
            constraints.append(skill_order.same_slot(*idx0))
        else:
            constraints.append(skill_order.different_slots(*idx0))

    limit = max(1, min(int(limit or 60), 1000))
    # 探索は表示に必要な件数だけ集めて打ち切る。解が多い手順では
    # 全件そろえるのに時間がかかる一方、表示に使われるのは先頭 limit 件だけ。
    # (下限 500 は「実際に絞り込めている手順なら正確な件数が出る」ようにするため)
    search_cap = max(limit, SO_SEARCH_CAP_MIN)
    stats = {}
    try:
        results, truncated = skill_order.solve(
            n_cards, copiers, plan, constraints,
            hand_size=hand_size, max_results=search_cap, stats=stats)
    except skill_order.SearchBudgetExceeded:
        return _so_error(
            "探索の組合せが多すぎて打ち切りました。「指定なし」ステップを減らすか、"
            "スロット指定を追加して絞り込んでください。")

    n_layouts, layouts_exact = skill_order.distinct_layouts(results)
    plan_str = " → ".join(
        _so_step_desc(s, disp_names, hand_size) + (f"「{m}」" if m else "")
        for s, m in zip(plan, memos))
    header = [
        html.Div([html.Strong("手順: "), plan_str],
                 style={"fontSize": "0.88rem", "marginBottom": "4px"}),
        html.Div(
            [
                html.Strong(f"解の数: {len(results)}{'+' if truncated else ''}"),
                html.Span(
                    f"(初期配置 {n_layouts} 通り"
                    f"{'' if layouts_exact and not truncated else '以上'})",
                    style={"color": "#666", "marginLeft": "6px",
                           "fontSize": "0.85rem"}),
            ]
            + ([html.Span(
                f"※ {search_cap}件見つかった時点で打ち切りました。"
                "正確な件数が要る場合は枠指定や制約で絞ってください。",
                style={"color": "#888", "marginLeft": "6px",
                       "fontSize": "0.8rem"})] if truncated else []),
            style={"marginBottom": "10px"},
        ),
    ]
    if not results:
        header.append(html.Div(
            "条件を満たす初期配置は見つかりませんでした。",
            style={"color": "#d63031", "fontWeight": "bold"}))
        # 探索が到達できた最深の手順を示すと原因を絞りやすい
        reached = stats.get("max_depth", 0)
        bad = None if reached >= len(plan) else reached + 1
        if bad is None:
            header.append(html.Div(
                "手順そのものは最後まで成立します。手順間の制約が"
                "厳しすぎる可能性があります。",
                style={"fontSize": "0.85rem", "color": "#555"}))
        else:
            desc = _so_step_desc(plan[bad - 1], disp_names, hand_size)
            memo = memos[bad - 1]
            header.append(html.Div([
                html.Div([
                    f"{bad - 1}手目までは成立し、",
                    html.Strong(f"{bad}手目「{desc}」"
                                + (f"「{memo}」" if memo else "")),
                    "で成立しなくなります。",
                ]),
                html.Div(
                    "この手順の前で同じカードを使っている場合は、その手順に"
                    "「ドロー」が要るか、どちらかが「◯◯(コピー)」の使用では"
                    "ないかを確認してください。枠指定を付けている場合は"
                    "その指定も疑ってください。",
                    style={"color": "#888", "marginTop": "2px"}),
            ], style={"fontSize": "0.85rem", "color": "#555"}))
        return html.Div(header)

    # 開始スキル設定画面のタップ順: 1..hand_size=手札(左から) / 以降=山札(上から)
    # (最後の1枚は残りで自動的に決まるため表示しない)
    n_shown = max(1, n_cards - 1)
    _legend = {"borderRadius": "4px", "padding": "1px 6px", "fontWeight": "bold",
               "marginLeft": "6px", "whiteSpace": "nowrap"}
    def _range_label(lo, hi):
        return f"{lo} = " if lo == hi else f"{lo}〜{hi} = "

    legend = ["数字 = 開始スキル画面でカードをタップする順番。"]
    legend.append(html.Span(
        _range_label(1, min(n_shown, hand_size)) + "手札(左から)",
        style={**_legend, "background": "#f1c40f", "color": "#333"}))
    if n_shown > hand_size:
        legend.append(html.Span(
            _range_label(hand_size + 1, n_shown) + "山札(上から)",
            style={**_legend, "background": "#35a2ff", "color": "#fff"}))
    legend.append(html.Span("「任意」= 残りのどのカードでもよい",
                            style={**_legend, "background": "#ddd",
                                   "color": "#555"}))
    header.append(html.Div(
        legend,
        style={"fontSize": "0.8rem", "color": "#666", "marginBottom": "8px"},
    ))

    rows = []
    for sol in results[:limit]:
        # 番号はゲームの開始スキル画面と同じ色分け(黄=手札 / 青=山札)。
        order_parts = []
        for pos, i in enumerate(sol.layout[:n_shown], start=1):
            badge_bg, badge_fg = (("#f1c40f", "#333") if pos <= hand_size
                                  else ("#35a2ff", "#fff"))
            order_parts.append(html.Span([
                html.Span(str(pos), style={
                    "display": "inline-block", "minWidth": "18px",
                    "textAlign": "center", "borderRadius": "4px",
                    "background": badge_bg, "color": badge_fg,
                    "fontWeight": "bold", "fontSize": "0.8rem",
                    "marginRight": "4px", "padding": "0 3px"}),
                (html.Strong(disp_names[i]) if i is not None
                 else html.Span("任意", style={"color": "#999"})),
            ], style={"marginRight": "12px", "whiteSpace": "nowrap"}))
        if sol.count > 1:
            order_parts.append(html.Span(
                f"({sol.count}通り)",
                style={"color": "#999", "fontSize": "0.8rem"}))
        seq = " ".join(
            skill_order.trace_entry_label(e, disp_names, hand_size)
            for e in sol.trace)
        rows.append(html.Div(
            [
                html.Div(order_parts),
                # 手順が長いと一覧が読みにくいので、使用順は畳んでおく
                html.Details([
                    html.Summary("使用順", style={"cursor": "pointer",
                                                 "fontSize": "0.8rem",
                                                 "color": "#888"}),
                    html.Div(seq, style={"fontSize": "0.85rem",
                                         "color": "#555"}),
                ], style={"marginTop": "2px"}),
            ],
            style={"border": "1px solid #e0e0e0", "borderRadius": "6px",
                   "padding": "8px 10px", "marginBottom": "6px",
                   "background": "#fafafa"},
        ))
    if len(results) > limit:
        rows.append(html.Div(
            f"... 他 {len(results) - limit} 件(表示件数上限)",
            style={"color": "#888", "fontSize": "0.85rem"}))
    return html.Div(header + rows)


# ===========================================================================
# 蓄積 (チャージ) 型スキル
#   モデルと計算は docs/accumulate.md / app/backend/accumulate.py、
#   実際の分布計算はクライアント側 assets/cos_accumulate.js が行う。
#   ここは入力欄の増減・選択肢の更新・プリセット適用だけを担当する。
# ===========================================================================
def _accum_options_now(order, indices, memo_values, memo_ids) -> list:
    """現在のカード一覧 (表示順) から蓄積スキルの選択肢を作る。"""
    order = [i for i in (order or []) if isinstance(i, int)]
    for i in (indices or []):
        if i not in order:
            order.append(i)
    memo_by = {m["index"]: v for v, m in zip(memo_values or [], memo_ids or [])
               if isinstance(m, dict)}
    return accum_options(order, memo_by)


@callback(
    Output("accum-container", "children"),
    Output("accum-next-index", "data"),
    Input("accum-add-btn", "n_clicks"),
    Input({"type": "accum-remove", "index": ALL}, "n_clicks"),
    State("accum-container", "children"),
    State("accum-next-index", "data"),
    State("sorted-indices", "data"),
    State("card-indices", "data"),
    State({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_accum_cards(_add, _remove, children, next_idx, order, card_indices,
                       memo_values, memo_ids):
    trigger = ctx.triggered_id
    children = children or []
    next_idx = next_idx or 0

    if trigger == "accum-add-btn":
        options = _accum_options_now(order, card_indices, memo_values, memo_ids)
        children.append(make_accum_card(next_idx, options=options))
        return children, next_idx + 1

    if isinstance(trigger, dict) and trigger.get("type") == "accum-remove":
        # カード追加でボタンが増えると再発火するので、押された当人のときだけ処理する。
        if not _triggered_clicked():
            raise PreventUpdate
        rm = trigger["index"]
        children = [
            c for c in children
            if not (c["props"]["id"].get("type") == "accum-card"
                    and c["props"]["id"].get("index") == rm)
        ]
        return children, next_idx

    raise PreventUpdate


@callback(
    Output({"type": "accum", "field": "cards", "index": ALL}, "options"),
    Output({"type": "accum", "field": "cap_cards", "index": ALL}, "options"),
    Output({"type": "accum", "field": "burst_after", "index": ALL}, "options"),
    Input("sorted-indices", "data"),
    Input("card-indices", "data"),
    Input({"type": "memo", "index": ALL}, "value"),
    State({"type": "memo", "index": ALL}, "id"),
    State({"type": "accum", "field": "cards", "index": ALL}, "id"),
)
def accum_card_options(order, indices, memo_values, memo_ids, accum_ids):
    """カード一覧が変わったとき、蓄積スキルの選択肢を作り直す。

    accum-container.children は **Input にしない**。中の options を書き換えると
    Dash が children の変化とみなして再発火し、無限ループになる。
    蓄積スキルを新しく作るときの選択肢は make_accum_card(options=...) で渡す。
    """
    options = _accum_options_now(order, indices, memo_values, memo_ids)
    n = len(accum_ids or [])
    return [options] * n, [options] * n, [options] * n


@callback(
    Output({"type": "accum-ui", "field": "hint", "index": MATCH}, "children"),
    Input({"type": "accum", "field": "preset", "index": MATCH}, "value"),
)
def accum_hint(preset):
    return ACCUM_PRESETS.get(preset or "custom", {}).get("hint", "")


@callback(
    Output({"type": "accum-ui", "field": "cap_atk_box", "index": MATCH}, "style"),
    Output({"type": "accum-ui", "field": "cap_fixed_box", "index": MATCH}, "style"),
    Output({"type": "accum-ui", "field": "cap_cards_box", "index": MATCH}, "style"),
    Input({"type": "accum", "field": "cap_mode", "index": MATCH}, "value"),
)
def accum_cap_mode_visibility(mode):
    show = {"display": "flex", "gap": "8px", "flexWrap": "wrap", "flex": "1"}
    hide = {"display": "none"}
    return (show if mode == "atk" else hide,
            show if mode == "fixed" else hide,
            show if mode == "cards" else hide)


@callback(
    Output({"type": "accum", "field": "name", "index": MATCH}, "value"),
    Output({"type": "accum", "field": "rate", "index": MATCH}, "value"),
    Output({"type": "accum", "field": "cap_mode", "index": MATCH}, "value"),
    Output({"type": "accum", "field": "atk_pct", "index": MATCH}, "value"),
    Output({"type": "accum", "field": "cap_pct", "index": MATCH}, "value"),
    Output({"type": "accum", "field": "burst_mult", "index": MATCH}, "value"),
    Output({"type": "accum", "field": "burst_decay", "index": MATCH}, "value"),
    Input({"type": "accum-preset-btn", "index": MATCH}, "n_clicks"),
    State({"type": "accum", "field": "preset", "index": MATCH}, "value"),
    State({"type": "accum", "field": "name", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def accum_apply_preset(n_clicks, preset, name):
    """プリセットの既定値を入力欄へ流し込む。

    自動保存からの復元でカードを作り直したときに上書きされないよう、
    ドロップダウンの変更ではなく明示的な「適用」ボタンでのみ発火させる。
    """
    if not n_clicks:
        raise PreventUpdate
    pr = ACCUM_PRESETS.get(preset or "custom")
    if not pr:
        raise PreventUpdate
    return ((name or pr["name"]), pr["rate"], pr["cap_mode"], pr["atk_pct"],
            pr["cap_pct"], pr["burst_mult"], [1] if pr["burst_decay"] else [])
