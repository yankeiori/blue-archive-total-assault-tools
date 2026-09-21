"""入力の自動保存 (localStorage) と「入力を全クリア」の検証。

スナップショットの収集は assets/autosave.js が値を素通しするだけで、意味付けは
すべて app/frontend/persist.py が持つ。よってここでは

  - スナップショット → 画面状態 の復元が正しいか
  - 「全クリア」が初期表示とぴったり同じ状態に戻るか
  - JS 側の引数の並び (FIELDS) と Python 側が読むキーがずれていないか

を見る。最後のものが要で、main.py の Input を 1 つ足して JS を直し忘れると
スナップショットの中身が丸ごとずれるが、画面上はただ「復元されない」だけで
気づきにくい。
"""
import re
from pathlib import Path

import dash
import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict

import main  # noqa: F401 - コールバック登録
from app.frontend import callbacks, persist
from app.frontend.layout import (
    SO_DEFAULT_CARDS, SO_MAX_CARDS, create_layout,
)

_ROOT = Path(__file__).resolve().parent.parent
_PARAMS = ["crit_min", "crit_max", "normal_min", "normal_max",
           "hits", "crit_rate", "evade_rate", "enemies", "hp_dep"]


def _card(index, values):
    ids = [{"type": "param", "param": p, "index": index} for p in _PARAMS]
    return list(values), ids


def _snapshot(**over):
    """2 枚のカードと 3 手順を持つスナップショット。"""
    v0, i0 = _card(7, [111, 222, 33, 44, 5, 60, 0, 1, []])
    v1, i1 = _card(2, [900, 1000, 100, 200, 3, 50, 10, 2, [1]])
    snap = {
        "v": persist.SNAPSHOT_VERSION,
        "param_values": v0 + v1, "param_ids": i0 + i1,
        "memo_values": ["ホシノ", "ミカ"],
        "memo_ids": [{"type": "memo", "index": 7}, {"type": "memo", "index": 2}],
        "order": [2, 7], "card_indices": [7, 2], "next_index": 8,
        "target_damage": 4321000,
        "global_crit": 75, "global_evade": 5, "global_stability": 1234,
        "calc_method": "mc", "damage_mode": "pre_decay", "hp_mode": "on",
        "hp_H": 30000000, "hp_H1": 29000000, "hp_R0": 1.5, "hp_R1": 2.5,
        "text_input": "貼り付け途中のテキスト",
        "restart_D": 4321000, "restart_cp": [5],
        "restart_seg_time": {"0": 1.0, "5": 2.0},
        "restart_seg_success": {"0": 100.0, "5": 80.0},
        "restart_save": [5],
        "so_hand_size": "5", "so_card_count": "7", "so_limit": 123,
        "so_tl_text": "即 リオ",
        "so_names": ["リオ", "マリー", "ホシノ", "", "", "", "", "", "", ""],
        "so_copiers": [["copier"]] + [[]] * 9,
        "so_step_skill": ["n1", "n0", None],
        "so_step_target": [None, "1", None],
        "so_step_slot": ["2", "any", "any"],
        "so_step_draw": [[], ["draw"], []],
        "so_step_memo": ["a", "b", ""],
        "so_step_ids": [{"type": "so-step-skill", "index": i} for i in (0, 1, 2)],
        "so_step_order": [1, 0, 2], "so_next_step": 3,
        "so_con_type": ["same"], "so_con_steps": ["1,2"],
        "so_con_ids": [{"type": "so-con-type", "index": 0}],
        "so_next_con": 1,
    }
    snap.update(over)
    return snap


# ---------------------------------------------------------------------------
# レイアウトから「id → 初期値」を拾うヘルパー
# ---------------------------------------------------------------------------
def _walk(node):
    if isinstance(node, (list, tuple)):
        for c in node:
            yield from _walk(c)
        return
    if not hasattr(node, "_prop_names"):
        return
    yield node
    yield from _walk(getattr(node, "children", None))


def _by_id(layout):
    out = {}
    for c in _walk(layout):
        cid = getattr(c, "id", None)
        if isinstance(cid, str):
            out[cid] = c
        elif isinstance(cid, dict):
            out[(cid.get("type"), cid.get("index"))] = c
    return out


def _value(comp):
    return getattr(comp, "value", getattr(comp, "data", None))


# ---------------------------------------------------------------------------
# 復元
# ---------------------------------------------------------------------------
def _card_values(child):
    """復元したカード 1 枚から {param: 値} と備考を取り出す。"""
    params, memo = {}, ""
    for c in _walk(child):
        cid = getattr(c, "id", None)
        if not isinstance(cid, dict):
            continue
        if cid.get("type") == "param":
            params[cid["param"]] = c.value
        elif cid.get("type") == "memo":
            memo = c.value
    return params, memo


def test_restore_rebuilds_cards_in_display_order():
    state = persist._restored_state(_snapshot())
    # sorted-indices の並び (2, 7) が表示順。card-indices の順 (7, 2) ではない。
    assert state["card_indices"] == [2, 7]
    assert state["sorted_indices"] == [2, 7]
    assert state["next_index"] == 8
    first, second = state["cards"]
    p, memo = _card_values(first)
    assert (p["crit_min"], p["hits"], p["enemies"], p["hp_dep"]) == (900, 3, 2, [1])
    assert memo == "ミカ"
    p, memo = _card_values(second)
    assert (p["crit_min"], p["hits"], p["hp_dep"]) == (111, 5, [])
    assert memo == "ホシノ"


def test_restore_keeps_globals_and_cutoff_settings():
    state = persist._restored_state(_snapshot())
    assert state["target_damage"] == 4321000
    assert (state["global_crit"], state["global_evade"]) == (75, 5)
    assert state["global_stability"] == 1234
    assert (state["calc_method"], state["damage_mode"]) == ("mc", "pre_decay")
    assert state["hp_mode"] == "on"
    assert (state["hp_H"], state["hp_H1"], state["hp_R0"], state["hp_R1"]) \
        == (30000000, 29000000, 1.5, 2.5)
    assert state["text_input"] == "貼り付け途中のテキスト"
    assert state["restart_cp"] == [5]
    assert state["restart_seg_time"] == {"0": 1.0, "5": 2.0}
    assert state["restart_seg_success"] == {"0": 100.0, "5": 80.0}
    assert state["restart_save"] == [5]


def test_restore_rebuilds_skill_order_rows():
    state = persist._restored_state(_snapshot())
    assert state["so_hand_size"] == "5"
    # 枚数は so-card-count を所有する so_sync_card_count 経由で戻す
    assert state["so_restore_count"]["count"] == "7"
    assert state["so_limit"] == 123
    assert state["so_names"][:3] == ["リオ", "マリー", "ホシノ"]
    assert state["so_copiers"][0] == ["copier"]
    assert len(state["so_names"]) == SO_MAX_CARDS

    assert state["so_step_order"] == [1, 0, 2]
    assert state["so_next_step"] == 3
    rows = [dict(_step_values(s)) for s in state["so_steps"]]
    assert [r["so-step-skill"] for r in rows] == ["n0", "n1", None]
    assert [r["so-step-slot"] for r in rows] == ["any", "2", "any"]
    assert [r["so-step-draw"] for r in rows] == [["draw"], [], []]
    assert [r["so-step-memo"] for r in rows] == ["b", "a", ""]
    assert rows[0]["so-step-target"] == "1"

    con, = state["so_cons"]
    vals = dict(_step_values(con))
    assert vals["so-con-type"] == "same"
    assert vals["so-con-steps"] == "1,2"
    assert state["so_next_con"] == 1


def _step_values(row):
    for c in _walk(row):
        cid = getattr(c, "id", None)
        if isinstance(cid, dict) and str(cid.get("type", "")).startswith(
                ("so-step-", "so-con-")):
            yield cid["type"], getattr(c, "value", None)


def test_restore_survives_a_snapshot_with_holes():
    """壊れた / 途中までのスナップショットでも例外にしない。"""
    state = persist._restored_state({"v": persist.SNAPSHOT_VERSION})
    assert state["cards"] == []
    assert state["card_indices"] == []
    assert state["next_index"] == 0
    assert len(state["so_steps"]) == 1      # 空の手順が 1 行だけ
    assert state["so_cons"] == []
    assert state["so_names"] == [""] * SO_MAX_CARDS


def test_restore_falls_back_to_index_order_without_sorted_indices():
    """並べ替え情報が無くても card-indices の順で並べる。"""
    state = persist._restored_state(_snapshot(order=[]))
    assert state["card_indices"] == [7, 2]


# ---------------------------------------------------------------------------
# 全クリア
# ---------------------------------------------------------------------------
def test_clear_matches_the_initial_layout():
    """「入力を全クリア」後の値が初期表示と一致すること。

    レイアウト側の既定値だけを変えるとクリア後の状態がずれるので、
    実際に組み立てたレイアウトから初期値を拾って突き合わせる。
    """
    comps = _by_id(create_layout())
    cleared = persist._cleared_state()
    same = {
        "target-damage": "target_damage",
        "global-crit-rate": "global_crit",
        "global-evade-rate": "global_evade",
        "global-stability": "global_stability",
        "calc-method": "calc_method",
        "damage-mode": "damage_mode",
        "hp-mode": "hp_mode",
        "hp-H": "hp_H", "hp-H1": "hp_H1", "hp-R0": "hp_R0", "hp-R1": "hp_R1",
        "restart-D": "restart_D",
        "restart-cp-store": "restart_cp",
        "restart-seg-time-store": "restart_seg_time",
        "restart-seg-success-store": "restart_seg_success",
        "restart-save-store": "restart_save",
        "so-hand-size": "so_hand_size",
        "so-limit": "so_limit",
        "so-step-order": "so_step_order",
        "so-next-step": "so_next_step",
        "so-next-con": "so_next_con",
        "card-indices": "card_indices",
        "next-index": "next_index",
        "sorted-indices": "sorted_indices",
    }
    for cid, key in same.items():
        assert _value(comps[cid]) == cleared[key], cid
    assert cleared["so_restore_count"]["count"] == comps["so-card-count"].value
    assert cleared["cards"] == []
    assert cleared["so_names"] == [""] * SO_MAX_CARDS
    assert cleared["so_copiers"] == [[]] * SO_MAX_CARDS
    assert cleared["text_input"] == (getattr(comps["text-input"], "value", "") or "")
    # 自動保存そのものも消す (次に開いたときに戻ってこない)
    assert cleared["autosave"] is None


def test_clear_rebuilds_one_empty_step_row():
    steps = persist._cleared_state()["so_steps"]
    assert len(steps) == 1
    assert dict(_step_values(steps[0]))["so-step-skill"] is None
    # 既定枚数ぶんの選択肢が入っている (「指定なし」+ 元カード + 撤退)
    dd = next(c for c in _walk(steps[0])
              if getattr(c, "id", {}).get("type") == "so-step-skill")
    assert len(dd.options) == 1 + 2 * SO_DEFAULT_CARDS


# ---------------------------------------------------------------------------
# 復元するものが無いとき
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("snap", [None, {}, {"v": persist.SNAPSHOT_VERSION + 1}])
def test_no_snapshot_only_enables_autosaving(snap):
    """保存が無い / 形式が変わったときは画面に触らない。

    ただし armed だけは必ず立てること。立てないと以後一切保存されなくなる。
    """
    state = persist._no_restore()
    assert state["armed"] is True
    touched = {k for k, v in state.items()
               if k != "armed" and v is not dash.no_update
               and v != [dash.no_update] * SO_MAX_CARDS}
    assert touched == set()


# ---------------------------------------------------------------------------
# 出力と返り値、JS と Python のキーの対応
# ---------------------------------------------------------------------------
def test_every_output_is_filled_on_every_path():
    keys = set(persist._OUTPUTS)
    assert set(persist._cleared_state()) == keys
    assert set(persist._restored_state(_snapshot())) == keys
    assert set(persist._no_restore()) == keys


def _js_fields():
    js = (_ROOT / "assets" / "autosave.js").read_text(encoding="utf-8")
    body = re.search(r"var FIELDS = \[(.*?)\];", js, re.S).group(1)
    return re.findall(r'"([^"]+)"', body)


def _js_version():
    js = (_ROOT / "assets" / "autosave.js").read_text(encoding="utf-8")
    return int(re.search(r"var VERSION = (\d+);", js).group(1))


def test_js_snapshot_version_matches_python():
    assert _js_version() == persist.SNAPSHOT_VERSION


def _autosave_callback():
    """自動保存 (クライアントサイド) の登録内容を取り出す。"""
    cbs = [c for c in main.application._callback_list      # noqa: SLF001
           if c["output"].startswith("autosave-store.data")]
    assert len(cbs) == 1
    return cbs[0]


def test_js_field_list_matches_the_registered_callback():
    """JS の引数の本数と main.py の Input/State の本数が合っていること。"""
    cb = _autosave_callback()
    fields = _js_fields()
    assert len(fields) == len(cb["inputs"]) + len(cb["state"])
    # 最後の 1 つは「復元が済んだか」のフラグ
    assert fields[-1] == "armed"
    last = cb["state"][-1]
    assert (last["id"], last["property"]) == ("persist-armed", "data")


def test_python_only_reads_fields_that_js_writes():
    """persist.py が読むキーが JS の FIELDS に必ずあること (綴り間違い検出)。"""
    src = (_ROOT / "app" / "frontend" / "persist.py").read_text(encoding="utf-8")
    read = set(re.findall(r'snap\.get\("([^"]+)"\)', src))
    read |= set(re.findall(r'_list\(snap, "([^"]+)"\)', src))
    assert read - {"v"} <= set(_js_fields())
    # 逆に、保存しているのに一度も読んでいない項目が無いこと
    unread = set(_js_fields()) - read - {"armed"}
    assert unread == set(), f"保存しているのに復元で使っていない: {unread}"


# ---------------------------------------------------------------------------
# 復元 / 全クリアが so-card-count (so_sync_card_count の持ち物) を戻せること
# ---------------------------------------------------------------------------
def _sync_card_count(triggered, hand_size, restore_count, current):
    context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": p} for p in triggered]))
    return callbacks.so_sync_card_count(hand_size, restore_count, current)


def test_restored_card_count_survives_the_mode_switch():
    """復元はモードと枚数を同時に戻す。枚数が既定値に潰されないこと。"""
    _, value = _sync_card_count(["so-hand-size.value", "so-restore-count.data"],
                                "5", persist._restore_count(7), "6")
    assert value == "7"
    # モードに対して多すぎる枚数は丸める
    _, value = _sync_card_count(["so-hand-size.value", "so-restore-count.data"],
                                "3", persist._restore_count(7), "6")
    assert value == "6"


def test_mode_switch_alone_still_resets_to_the_default_count():
    """従来どおり、モードを切り替えたら既定枚数 (通常6 / 決戦10) へ。"""
    assert _sync_card_count(["so-hand-size.value"], "5", None, "4")[1] == "10"
    assert _sync_card_count(["so-hand-size.value"], "3", None, "9")[1] == "6"


def test_clearing_fires_even_when_the_count_is_unchanged():
    """同じ枚数を続けて渡しても発火すること (毎回 nonce が変わる)。"""
    first = persist._cleared_state()["so_restore_count"]
    second = persist._cleared_state()["so_restore_count"]
    assert first["count"] == second["count"]
    assert first != second
    _, value = _sync_card_count(["so-restore-count.data"], "3", second, "4")
    assert value == "6"
