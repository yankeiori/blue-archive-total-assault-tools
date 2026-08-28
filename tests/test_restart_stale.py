"""足切りライン最適化の「結果が古い」検知 (指紋) の検証。

run_restart は解析実行ボタンでしか発火しないため、実行後に並べ替え・パラメータ
変更をしても図/表/手動調整スライダーは古い前提のまま残る。restart-config に
焼き込んだ指紋と現在の入力の指紋を突き合わせて警告を出す仕組みを確認する。

正解基準:
  - 同じ入力なら指紋は一致 (誤検知しない)。特に run_restart 側は D を float 化
    するなど正規化を挟むので、生値から作った指紋と一致する必要がある。
  - 並び順・カードパラメータ・目標 D・チェックポイントの変更で指紋が変わる。
  - カード名 (メモ) は数値結果に影響しないので指紋に含めない。
  - 未実行 (config なし) では警告を出さない。
"""
import main  # noqa: F401 - コールバック登録のため
from app.frontend import callbacks as cb

_PARAMS = ["crit_min", "crit_max", "normal_min", "normal_max",
           "hits", "crit_rate", "evade_rate", "enemies", "hp_dep"]


def _card_states(index, hits):
    vals = [100000, 120000, 50000, 60000, hits, 60, 0, 1, []]
    return [({"type": "param", "param": p, "index": index}, v)
            for p, v in zip(_PARAMS, vals)]


def _inputs():
    pairs = _card_states(0, 5) + _card_states(1, 3)
    return [i for i, _ in pairs], [v for _, v in pairs]


_BASE = dict(D=400000, cp_store=[5],
             seg_times={"0": 1.0, "5": 1.0},
             seg_success={"0": 100.0, "5": 100.0},
             global_crit=60, global_evade=0, damage_mode="post_decay",
             hp_mode="off", hp_H=None, hp_H1=None, hp_R0=None, hp_R1=None)


def _fp(order, param_values, param_ids, **over):
    kw = {**_BASE, **over}
    return cb._restart_fingerprint(order, [0, 1], param_values, param_ids, **kw)


def _run(order, param_values, param_ids):
    """run_restart を実行し config を返す (指紋込み)。"""
    _fig, _summary, cfg, _sliders = cb.run_restart(
        1, _BASE["D"], order, [0, 1], param_values, param_ids,
        ["A", "B"], [{"index": 0}, {"index": 1}],
        _BASE["cp_store"], _BASE["seg_times"], _BASE["seg_success"],
        _BASE["global_crit"], _BASE["global_evade"], _BASE["damage_mode"],
        _BASE["hp_mode"], _BASE["hp_H"], _BASE["hp_H1"],
        _BASE["hp_R0"], _BASE["hp_R1"])
    return cfg


def _stale(cfg, order, param_values, param_ids, **over):
    kw = {**_BASE, **over}
    return cb.flag_restart_stale(
        cfg, kw["D"], order, [0, 1], param_values, kw["cp_store"],
        kw["seg_times"], kw["seg_success"], kw["global_crit"],
        kw["global_evade"], kw["damage_mode"], kw["hp_mode"],
        kw["hp_H"], kw["hp_H1"], kw["hp_R0"], kw["hp_R1"], param_ids)


def test_fingerprint_is_stable_and_order_sensitive():
    ids, vals = _inputs()
    assert _fp([0, 1], vals, ids) == _fp([0, 1], vals, ids)
    assert _fp([0, 1], vals, ids) != _fp([1, 0], vals, ids)


def test_fingerprint_tracks_the_inputs_that_matter():
    ids, vals = _inputs()
    base = _fp([0, 1], vals, ids)
    changed = list(vals)
    changed[0] = 130000                      # 会心ダメージ下限
    assert _fp([0, 1], changed, ids) != base
    assert _fp([0, 1], vals, ids, D=500000) != base
    assert _fp([0, 1], vals, ids, cp_store=[3]) != base
    assert _fp([0, 1], vals, ids, global_crit=70) != base


def test_hp_params_ignored_while_hp_mode_off():
    ids, vals = _inputs()
    assert _fp([0, 1], vals, ids, hp_H=1) == _fp([0, 1], vals, ids, hp_H=2)
    assert (_fp([0, 1], vals, ids, hp_mode="on", hp_H=1)
            != _fp([0, 1], vals, ids, hp_mode="on", hp_H=2))


def test_run_then_reorder_raises_the_stale_note():
    ids, vals = _inputs()
    cfg = _run([0, 1], vals, ids)
    assert cfg and cfg.get("fingerprint")

    # 実行直後は警告なし (run_restart の正規化を挟んでも指紋が一致する)
    assert _stale(cfg, [0, 1], vals, ids) == ("", "")

    # 並べ替え後は両方の注記が出る (結果側と手動調整側)
    note, note_interactive = _stale(cfg, [1, 0], vals, ids)
    assert note != "" and note_interactive != ""

    # 目標ダメージ変更でも出る
    assert _stale(cfg, [0, 1], vals, ids, D=500000)[0] != ""


def test_memo_change_does_not_flag_stale():
    """カード名は数値結果に影響しないので、変えても警告は出ない。"""
    ids, vals = _inputs()
    cfg = _run([0, 1], vals, ids)
    # flag_restart_stale はメモを受け取らない = メモだけ変えても指紋は不変
    assert _stale(cfg, [0, 1], vals, ids) == ("", "")


def test_no_note_before_the_first_run():
    ids, vals = _inputs()
    assert _stale(None, [0, 1], vals, ids) == ("", "")
    assert _stale({}, [0, 1], vals, ids) == ("", "")
