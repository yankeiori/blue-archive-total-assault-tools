"""入力の自動保存 (ブラウザの localStorage) と「入力を全クリア」。

ブラウザを閉じても入力が残るように、全入力のスナップショットを
dcc.Store(id="autosave-store", storage_type="local") に置く。

  収集: assets/autosave.js の dash_clientside.persist.collect
        (main.py でクライアントサイドコールバックとして登録)。サーバーを
        叩かないので、入力のたびに保存しても通信は発生しない。
  復元: 本モジュールの restore_or_clear。読込直後に persist-restore-tick が
        1 回だけ発火して呼ばれる。

スナップショットは Dash の生の値をそのまま詰めた内部形式で、ユーザーに見せる
JSON (エクスポート / インポート) とは別物。JS 側は値を素通しするだけで、
カードの組み立てなどの意味付けは全てここで行う。形式を変えたら
SNAPSHOT_VERSION を上げること (版が違う保存は読み捨てる)。

復元より先に自動保存が走ると、まだ空の初期状態で前回の入力を上書きして
しまう。そのため復元が済むまでは persist-armed が False で、JS 側が保存を
拒むようにしている。復元するものが無い場合でも armed だけは必ず True を
返すこと (さもないと以後一切保存されない)。
"""
import uuid

import dash
from dash import ALL, Input, Output, State, callback, ctx

from app.backend import tl_parse
from app.frontend.layout import (
    DEFAULT_CALC_METHOD,
    DEFAULT_CRIT_RATE,
    DEFAULT_DAMAGE_MODE,
    DEFAULT_EVADE_RATE,
    DEFAULT_HP_H,
    DEFAULT_HP_H1,
    DEFAULT_HP_MODE,
    DEFAULT_HP_R0,
    DEFAULT_HP_R1,
    DEFAULT_SO_HAND_SIZE,
    DEFAULT_SO_LIMIT,
    DEFAULT_STABILITY,
    DEFAULT_TARGET_DAMAGE,
    SO_DEFAULT_CARDS,
    SO_MAX_CARDS,
    accum_options,
    make_accum_card,
    make_damage_card,
    make_so_constraint,
    make_so_step,
    so_skill_options,
    so_target_options,
)

# スナップショットの形式版。assets/autosave.js の VERSION と揃える。
SNAPSHOT_VERSION = 1


def _int(value, default):
    """ドロップダウン等の文字列値を int に (壊れていれば既定値)。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _restore_count(count):
    """カード枚数を so_sync_card_count (so-card-count の持ち主) へ渡す形。

    同じ枚数を続けて渡しても確実に発火するよう、毎回変わる nonce を添える。
    値が変わらないと Dash はコールバックを起こさず、例えば「6 枚で復元 →
    手で 4 枚に変更 → 全クリア」で枚数だけ 4 のまま取り残される。
    """
    return {"count": str(count), "nonce": uuid.uuid4().hex}


def _list(snap, key):
    """スナップショットのリスト項目 (欠けていれば空リスト)。"""
    v = snap.get(key)
    return v if isinstance(v, list) else []


def _indexed(values, ids):
    """ALL パターンの (値, id) 列を {index: 値} にする。"""
    return {i["index"]: v for v, i in zip(values, ids)
            if isinstance(i, dict) and "index" in i}


# ---------------------------------------------------------------------------
# スナップショット → 各ページの状態
# ---------------------------------------------------------------------------
def _restore_cards(snap):
    """ダメージカードを組み直す。戻り値は (children, 表示順の index リスト)。"""
    by_index: dict = {}
    for v, pid in zip(_list(snap, "param_values"), _list(snap, "param_ids")):
        if isinstance(pid, dict) and "index" in pid and "param" in pid:
            by_index.setdefault(pid["index"], {})[pid["param"]] = v
    memo_by = _indexed(_list(snap, "memo_values"), _list(snap, "memo_ids"))

    # 表示順 (sorted-indices) を正とし、漏れた分は card-indices の順で末尾に。
    order = []
    for i in [*_list(snap, "order"), *_list(snap, "card_indices")]:
        if i in by_index and i not in order:
            order.append(i)

    children = [
        make_damage_card(i, params=by_index[i], memo=memo_by.get(i) or "")
        for i in order
    ]
    return children, order


def _restore_accum(snap, options):
    """蓄積 (チャージ) 型スキルの入力カードを組み直す。

    戻り値は (children, 次の index)。古い保存 (蓄積スキル導入前) には
    accum_values / accum_ids が無く、空リストとして正しく復元される。
    カード選択肢は生成時に渡す (コールバックで後入れすると Dash が無限ループする)。
    """
    by_index: dict = {}
    for v, aid in zip(_list(snap, "accum_values"), _list(snap, "accum_ids")):
        if isinstance(aid, dict) and "index" in aid and "field" in aid:
            by_index.setdefault(aid["index"], {})[aid["field"]] = v
    order = sorted(by_index)
    children = [make_accum_card(i, params=by_index[i], options=options)
                for i in order]
    next_idx = max(_int(snap.get("accum_next_index"), 0),
                   (max(order) + 1) if order else 0)
    return children, next_idx


def _so_names(snap):
    """カード名・複製フラグを (Input 用の値, Checklist 用の値, 表示名, 複製集合) で返す。"""
    names = [""] * SO_MAX_CARDS
    for i, v in enumerate(_list(snap, "so_names")[:SO_MAX_CARDS]):
        names[i] = v or ""
    copier_values = [[] for _ in range(SO_MAX_CARDS)]
    for i, v in enumerate(_list(snap, "so_copiers")[:SO_MAX_CARDS]):
        copier_values[i] = ["copier"] if v else []

    n_cards = _int(snap.get("so_card_count"), SO_DEFAULT_CARDS)
    copiers = {i for i, v in enumerate(copier_values)
               if v and i < n_cards}
    disp = [tl_parse.display_name(nm) or f"カード{i + 1}"
            for i, nm in enumerate(names)]
    return names, copier_values, disp, copiers


def _restore_so_steps(snap, disp_names, copiers):
    """手順(PLAN)の行を組み直す。戻り値は (children, 表示順, 次の index)。"""
    hand_size = _int(snap.get("so_hand_size"), 3)
    n_cards = _int(snap.get("so_card_count"), SO_DEFAULT_CARDS)
    skill_opts = so_skill_options(disp_names, copiers, n_cards)
    target_opts = so_target_options(disp_names, copiers, n_cards)

    step_ids = _list(snap, "so_step_ids")
    skill = _indexed(_list(snap, "so_step_skill"), step_ids)
    target = _indexed(_list(snap, "so_step_target"), step_ids)
    slot = _indexed(_list(snap, "so_step_slot"), step_ids)
    draw = _indexed(_list(snap, "so_step_draw"), step_ids)
    memo = _indexed(_list(snap, "so_step_memo"), step_ids)

    known = [i["index"] for i in step_ids if isinstance(i, dict)]
    order = [i for i in _list(snap, "so_step_order") if i in known]
    order += [i for i in known if i not in order]
    if not order:
        return _default_so_steps(disp_names, copiers, n_cards, hand_size)

    children = [
        make_so_step(
            i, skill_opts, target_opts,
            skill=skill.get(i), target=target.get(i),
            slot=slot.get(i) or "any", draw=bool(draw.get(i)),
            memo=memo.get(i) or "", hand_size=hand_size,
        )
        for i in order
    ]
    next_step = max(_int(snap.get("so_next_step"), 0), max(order) + 1)
    return children, order, next_step


def _default_so_steps(disp_names=None, copiers=None,
                      n_cards=SO_DEFAULT_CARDS, hand_size=3):
    """空の手順1行だけの初期状態。"""
    disp_names = disp_names if disp_names is not None else [""] * SO_MAX_CARDS
    copiers = copiers or set()
    step = make_so_step(0,
                        so_skill_options(disp_names, copiers, n_cards),
                        so_target_options(disp_names, copiers, n_cards),
                        hand_size=hand_size)
    return [step], [0], 1


def _restore_so_constraints(snap):
    """手順間制約の行を組み直す。戻り値は (children, 次の index)。"""
    con_ids = _list(snap, "so_con_ids")
    ctypes = _indexed(_list(snap, "so_con_type"), con_ids)
    csteps = _indexed(_list(snap, "so_con_steps"), con_ids)
    order = sorted(ctypes)
    children = [
        make_so_constraint(i, ctype=ctypes.get(i) or "diff",
                           steps=csteps.get(i) or "")
        for i in order
    ]
    next_con = max(_int(snap.get("so_next_con"), 0),
                   (max(order) + 1) if order else 0)
    return children, next_con


# ---------------------------------------------------------------------------
# 復元 / 全クリア
# ---------------------------------------------------------------------------
def _cleared_state():
    """「入力を全クリア」後の状態 (= 初期表示と同じ)。"""
    steps, step_order, next_step = _default_so_steps()
    return {
        "cards": [], "card_indices": [], "next_index": 0, "sorted_indices": [],
        "target_damage": DEFAULT_TARGET_DAMAGE,
        "global_crit": DEFAULT_CRIT_RATE, "global_evade": DEFAULT_EVADE_RATE,
        "global_stability": DEFAULT_STABILITY,
        "calc_method": DEFAULT_CALC_METHOD, "damage_mode": DEFAULT_DAMAGE_MODE,
        "hp_mode": DEFAULT_HP_MODE,
        "hp_H": DEFAULT_HP_H, "hp_H1": DEFAULT_HP_H1,
        "hp_R0": DEFAULT_HP_R0, "hp_R1": DEFAULT_HP_R1,
        "text_input": "", "text_prefix": "",
        "accum": [], "accum_next_index": 0,
        "restart_D": DEFAULT_TARGET_DAMAGE,
        "restart_cp": [], "restart_seg_time": {"0": 1.0},
        "restart_seg_success": {"0": 100.0}, "restart_save": [],
        "so_hand_size": DEFAULT_SO_HAND_SIZE,
        "so_restore_count": _restore_count(SO_DEFAULT_CARDS),
        "so_limit": DEFAULT_SO_LIMIT, "so_tl_text": "",
        "so_names": [""] * SO_MAX_CARDS,
        "so_copiers": [[] for _ in range(SO_MAX_CARDS)],
        "so_steps": steps, "so_step_order": step_order,
        "so_next_step": next_step,
        "so_cons": [], "so_next_con": 0,
        "autosave": None,
        "status": "🗑 入力を全クリアしました。",
        "armed": True,
    }


def _restored_state(snap):
    """スナップショットから復元した状態。"""
    cards, order = _restore_cards(snap)
    names, copier_values, disp, copiers = _so_names(snap)
    steps, step_order, next_step = _restore_so_steps(snap, disp, copiers)
    cons, next_con = _restore_so_constraints(snap)
    accum, accum_next = _restore_accum(
        snap, accum_options(order, _indexed(_list(snap, "memo_values"),
                                            _list(snap, "memo_ids"))))
    return {
        "cards": cards, "card_indices": order, "sorted_indices": order,
        # 新しいカードの index は、復元した中の最大値より必ず後ろに取る
        "next_index": max(max([*order, -1]) + 1,
                          _int(snap.get("next_index"), 0)),
        "target_damage": snap.get("target_damage"),
        "global_crit": snap.get("global_crit"),
        "global_evade": snap.get("global_evade"),
        "global_stability": snap.get("global_stability"),
        "calc_method": snap.get("calc_method") or DEFAULT_CALC_METHOD,
        "damage_mode": snap.get("damage_mode") or DEFAULT_DAMAGE_MODE,
        "hp_mode": snap.get("hp_mode") or DEFAULT_HP_MODE,
        "hp_H": snap.get("hp_H"), "hp_H1": snap.get("hp_H1"),
        "hp_R0": snap.get("hp_R0"), "hp_R1": snap.get("hp_R1"),
        "text_input": snap.get("text_input") or "",
        "text_prefix": snap.get("text_prefix") or "",
        "accum": accum, "accum_next_index": accum_next,
        "restart_D": snap.get("restart_D"),
        "restart_cp": _list(snap, "restart_cp"),
        "restart_seg_time": snap.get("restart_seg_time") or {"0": 1.0},
        "restart_seg_success": snap.get("restart_seg_success") or {"0": 100.0},
        "restart_save": _list(snap, "restart_save"),
        "so_hand_size": snap.get("so_hand_size") or DEFAULT_SO_HAND_SIZE,
        "so_restore_count": _restore_count(
            _int(snap.get("so_card_count"), SO_DEFAULT_CARDS)),
        "so_limit": snap.get("so_limit") or DEFAULT_SO_LIMIT,
        "so_tl_text": snap.get("so_tl_text") or "",
        "so_names": names, "so_copiers": copier_values,
        "so_steps": steps, "so_step_order": step_order,
        "so_next_step": next_step,
        "so_cons": cons, "so_next_con": next_con,
        "autosave": dash.no_update,
        "status": f"🔄 前回の入力を復元しました（カード {len(cards)} 枚）。",
        "armed": True,
    }


def _no_restore():
    """復元するものが無いとき。自動保存だけは有効にする。"""
    state = {k: dash.no_update for k in _cleared_state()}
    # ワイルドカード出力は「触らない」でも要素数分のリストで返す必要がある
    state["so_names"] = [dash.no_update] * SO_MAX_CARDS
    state["so_copiers"] = [dash.no_update] * SO_MAX_CARDS
    state["armed"] = True
    return state


# 出力一覧。返り値の dict とキーが 1:1 で対応する (テストで突き合わせる)。
_OUTPUTS = dict(
    cards=Output("cards-container", "children", allow_duplicate=True),
    card_indices=Output("card-indices", "data", allow_duplicate=True),
    next_index=Output("next-index", "data", allow_duplicate=True),
    sorted_indices=Output("sorted-indices", "data", allow_duplicate=True),
    target_damage=Output("target-damage", "value", allow_duplicate=True),
    global_crit=Output("global-crit-rate", "value", allow_duplicate=True),
    global_evade=Output("global-evade-rate", "value", allow_duplicate=True),
    global_stability=Output("global-stability", "value", allow_duplicate=True),
    calc_method=Output("calc-method", "value", allow_duplicate=True),
    damage_mode=Output("damage-mode", "value", allow_duplicate=True),
    hp_mode=Output("hp-mode", "value", allow_duplicate=True),
    hp_H=Output("hp-H", "value", allow_duplicate=True),
    hp_H1=Output("hp-H1", "value", allow_duplicate=True),
    hp_R0=Output("hp-R0", "value", allow_duplicate=True),
    hp_R1=Output("hp-R1", "value", allow_duplicate=True),
    text_input=Output("text-input", "value", allow_duplicate=True),
    text_prefix=Output("text-prefix", "value", allow_duplicate=True),
    accum=Output("accum-container", "children", allow_duplicate=True),
    accum_next_index=Output("accum-next-index", "data", allow_duplicate=True),
    restart_D=Output("restart-D", "value", allow_duplicate=True),
    restart_cp=Output("restart-cp-store", "data", allow_duplicate=True),
    restart_seg_time=Output("restart-seg-time-store", "data",
                            allow_duplicate=True),
    restart_seg_success=Output("restart-seg-success-store", "data",
                               allow_duplicate=True),
    restart_save=Output("restart-save-store", "data", allow_duplicate=True),
    so_hand_size=Output("so-hand-size", "value", allow_duplicate=True),
    so_restore_count=Output("so-restore-count", "data", allow_duplicate=True),
    so_limit=Output("so-limit", "value", allow_duplicate=True),
    so_tl_text=Output("so-tl-text", "value", allow_duplicate=True),
    so_names=Output({"type": "so-name", "index": ALL}, "value",
                    allow_duplicate=True),
    so_copiers=Output({"type": "so-copier", "index": ALL}, "value",
                      allow_duplicate=True),
    so_steps=Output("so-steps-container", "children", allow_duplicate=True),
    so_step_order=Output("so-step-order", "data", allow_duplicate=True),
    so_next_step=Output("so-next-step", "data", allow_duplicate=True),
    so_cons=Output("so-cons-container", "children", allow_duplicate=True),
    so_next_con=Output("so-next-con", "data", allow_duplicate=True),
    autosave=Output("autosave-store", "data", allow_duplicate=True),
    status=Output("io-status", "children", allow_duplicate=True),
    armed=Output("persist-armed", "data", allow_duplicate=True),
)


@callback(
    output=_OUTPUTS,
    inputs=dict(
        _tick=Input("persist-restore-tick", "n_intervals"),
        clear_clicks=Input("clear-all-confirm", "submit_n_clicks"),
    ),
    state=dict(snap=State("autosave-store", "data")),
    prevent_initial_call=True,
)
def restore_or_clear(_tick, clear_clicks, snap):
    """読込時の復元と「入力を全クリア」。出力が重なるので 1 つにまとめてある。"""
    if ctx.triggered_id == "clear-all-confirm":
        if not clear_clicks:
            return _no_restore()
        return _cleared_state()

    if not isinstance(snap, dict) or snap.get("v") != SNAPSHOT_VERSION:
        # 保存が無い / 形式が変わった → 初期表示のまま使う
        return _no_restore()
    return _restored_state(snap)
