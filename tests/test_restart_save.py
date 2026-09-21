"""凸区切り(セーブポイント)付き 足切り最適化 restart_save の検証。

理論は docs/restart_save.md。正解基準:
  - セーブ点が無いとき、既存の Dinkelbach 実装 restart_cos と厳密に一致する
    (期待総時間 E[T] = 1/g*、関門も同じ)。これが §5 の同値性の数値的確認。
  - 「足切り無し」基準の期待時間は Σt·∏succ / 成功率 (幾何分布の期待値)。
  - 区分線形係数化は元の関数を再現する。
  - 足切りは到達可能下限を下回らない。セーブ地点の累積が大きいほど足切りは上がる。
  - ブロック別の期待時間の総和が全体の期待総時間に一致する。
  - 凸区切りを入れると期待総時間は減る (やり直しの損失が小さくなるため)。
  - 手動関門に最適関門を渡すと最適と一致し、0 を渡すと基準と一致する。
  - 所要時間 0 の凸は縮退 (無コストで引き直し放題) なので明示的に弾く。
"""
import math

import numpy as np
import pytest

from app.backend.cos import HPParams, build_hit_mixtures, build_sum_dist, y_mixture
from app.backend.mixed import (
    blocks_from_specs,
    hit_specs_from_cards,
    mixed_support,
    normalize_specs,
)
from app.backend import restart_cos, restart_mixed, restart_save
from app.backend.restart_cos import _CosEngine, _segs_sum, _split_bounds


def _card(cmin, cmax, nmin, nmax, hits, cr, ev=0):
    return {"crit_min": cmin, "crit_max": cmax, "normal_min": nmin,
            "normal_max": nmax, "hits": hits, "crit_rate": cr,
            "evade_rate": ev, "enemies": 1}


_CARDS = [
    _card(800_000, 1_000_000, 150_000, 190_000, 4, 55),
    _card(200_000, 270_000, 38_000, 49_000, 8, 55),
    _card(800_000, 1_000_000, 150_000, 190_000, 4, 55),
    _card(200_000, 270_000, 38_000, 49_000, 8, 55),
]
_CPS = [4, 12, 16]
_D = 4_500_000


def _setup():
    hits = build_hit_mixtures(_CARDS, 55, 0.0, "post_decay")
    return hits, [1.0] * len(hits)


# ---------------------------------------------------------------------------
# セーブ点なし = 既存実装との同値
# ---------------------------------------------------------------------------
def test_no_save_matches_restart_cos():
    hits, ht = _setup()
    a = restart_cos.analyze(hits, _CPS, ht, _D)
    b = restart_save.analyze(hits, _CPS, [], ht, _D)
    assert b["exp_time"] == pytest.approx(1.0 / a["g_star_dp"], rel=1e-6)
    for ra, rb in zip(a["rows"], b["rows"]):
        assert rb["gate"] == pytest.approx(ra["gate"], rel=1e-6, abs=1.0)
    assert b["has_save"] is False
    assert b["save_points"] == []


def test_no_save_with_segment_success_matches_restart_cos():
    hits, ht = _setup()
    succ = [1.0, 0.9, 0.95, 1.0]
    a = restart_cos.analyze(hits, _CPS, ht, _D, seg_success=succ)
    b = restart_save.analyze(hits, _CPS, [], ht, _D, seg_success=succ)
    assert b["exp_time"] == pytest.approx(1.0 / a["g_star_dp"], rel=1e-6)
    for ra, rb in zip(a["rows"], b["rows"]):
        assert rb["gate"] == pytest.approx(ra["gate"], rel=1e-6, abs=1.0)


def test_baseline_is_total_time_over_success():
    """セーブ点が無ければ基準 = 「全部回して失敗したらやり直す」= Σt/P。"""
    hits, ht = _setup()
    b = restart_save.analyze(hits, _CPS, [], ht, _D)
    P = 1.0 - float(build_sum_dist(hits).cdf(np.array([float(_D)]))[0])
    assert b["baseline"]["exp_time"] == pytest.approx(sum(ht) / P, rel=1e-3)


def test_product_no_save_matches_restart_cos():
    hits, ht = _setup()
    hp = HPParams(H=100_000_000.0, H1=100_000_000.0, R0=1.0, R1=2.0)
    ymix = [y_mixture(m, hp.beta) for m in hits]
    D = 4_000_000
    a = restart_cos.analyze_product(ymix, hp, _CPS, ht, D)
    b = restart_save.analyze_product(ymix, hp, _CPS, [], ht, D)
    assert b["exp_time"] == pytest.approx(1.0 / a["g_star_dp"], rel=1e-6)
    for ra, rb in zip(a["rows"], b["rows"]):
        assert rb["gate"] == pytest.approx(ra["gate"], rel=1e-6, abs=1.0)


# ---------------------------------------------------------------------------
# 区分線形の係数化
# ---------------------------------------------------------------------------
def test_pw_linear_coeffs_reproduces_function():
    hits, _ht = _setup()
    bounds, _cps = _split_bounds(len(hits), _CPS)
    eng = _CosEngine(_segs_sum(hits, bounds))
    xs = [1.0e6, 2.0e6, 3.0e6, 4.0e6, 4.0e6, 6.0e6]
    ys = [50.0, 30.0, 18.0, 12.0, 0.0, 0.0]          # x=4e6 で跳び
    c = restart_save._pw_linear_coeffs(eng, xs, ys)
    Z = np.zeros(eng.N)
    for x, y in [(1.0e6, 50.0), (2.0e6, 30.0), (3.0e6, 18.0), (6.0e6, 0.0)]:
        assert eng.eval(c, Z, x) == pytest.approx(y, abs=0.1)
    # 折れ点の間 = 線形補間、左は定数外挿、跳びの右は 0
    assert eng.eval(c, Z, 2.5e6) == pytest.approx(24.0, abs=0.1)
    assert eng.eval(c, Z, 3.5e6) == pytest.approx(15.0, abs=0.1)
    assert eng.eval(c, Z, 0.5e6) == pytest.approx(50.0, abs=0.1)
    # 跳びの近傍はギブス振動が乗る (指示関数終端と同じ性質)。数波長離れれば収まる。
    assert eng.eval(c, Z, 3.8e6) == pytest.approx(13.2, abs=0.5)
    assert eng.eval(c, Z, 4.2e6) == pytest.approx(0.0, abs=0.5)
    # 定数関数 (跳びも折れ点も無い) は k=0 の係数だけで表される
    flat = restart_save._pw_linear_coeffs(eng, [1.0e6, 5.0e6], [3.0, 3.0])
    assert np.allclose(flat[1:], 0.0, atol=1e-9)
    assert eng.eval(flat, Z, 1.234e6) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 凸区切りあり
# ---------------------------------------------------------------------------
def _saved():
    hits, ht = _setup()
    return hits, ht, restart_save.analyze(hits, _CPS, [12], ht, _D)


def test_save_point_marked_in_rows():
    _hits, _ht, res = _saved()
    assert res["has_save"] is True
    assert res["save_points"] == [12]
    assert [r["save"] for r in res["rows"]] == [False, True, False]


def test_save_point_reduces_expected_time():
    """凸区切りを入れると、リスタートで失う進捗が減るので期待総時間は下がる。"""
    hits, ht = _setup()
    plain = restart_save.analyze(hits, _CPS, [], ht, _D)
    saved = restart_save.analyze(hits, _CPS, [12], ht, _D)
    assert saved["exp_time"] < plain["exp_time"]


def test_block_times_sum_to_expected_time():
    _hits, _ht, res = _saved()
    assert sum(b["exp_time"] for b in res["blocks"]) == pytest.approx(
        res["exp_time"], rel=1e-9)
    assert len(res["blocks"]) == 2


def test_gates_respect_reachability_floor():
    """関門は「ここから最大値を引き続けても届かない」下限を下回らない。"""
    hits, ht = _setup()
    bounds, cps = _split_bounds(len(hits), _CPS)
    eng = _CosEngine(_segs_sum(hits, bounds))
    tail = np.cumsum([sg.s_hi for sg in eng.segs][::-1])[::-1]
    res = restart_save.analyze(hits, _CPS, [12], ht, _D)
    for k, r in enumerate(res["rows"], start=1):
        floor = max(0.0, _D - float(tail[k]))
        assert r["gate_lo"] >= floor - 1.0


def test_gate_band_increases_with_save_state():
    """セーブ地点の累積が大きいほど中断価値が上がり、足切りも上がる。

    1凸目の中 (セーブ地点は 0 固定) は幅ゼロ、2凸目の中は幅を持つ。
    """
    _hits, _ht, res = _saved()
    first_block = res["rows"][0]              # cp=4 (1凸目の中)
    assert first_block["gate_hi"] == pytest.approx(first_block["gate_lo"])
    later = res["rows"][2]                    # cp=16 (2凸目の中)
    assert later["gate_hi"] > later["gate_lo"]
    assert later["gate_lo"] <= later["gate"] <= later["gate_hi"]


def test_manual_gates_reproduce_optimal_and_baseline():
    hits, ht = _setup()
    opt = restart_save.analyze(hits, _CPS, [], ht, _D)
    same = restart_save.analyze(hits, _CPS, [], ht, _D,
                                manual_gates=[r["gate"] for r in opt["rows"]])
    assert same["exp_time"] == pytest.approx(opt["exp_time"], rel=1e-6)
    zero = restart_save.analyze(hits, _CPS, [], ht, _D,
                                manual_gates=[0.0] * len(_CPS))
    assert zero["exp_time"] == pytest.approx(opt["baseline"]["exp_time"], rel=1e-6)


def test_manual_gates_are_never_better_than_optimal():
    hits, ht = _setup()
    opt = restart_save.analyze(hits, _CPS, [12], ht, _D)
    worse = restart_save.analyze(hits, _CPS, [12], ht, _D,
                                 manual_gates=[0.0, 1.0e6, 0.0])
    assert worse["exp_time"] >= opt["exp_time"] - 1e-6


def test_zero_time_block_is_rejected():
    """所要時間 0 の凸は「無コストで引き直し放題」の縮退なので弾く。"""
    hits, _ht = _setup()
    ht = [0.0] * 12 + [1.0] * (len(hits) - 12)     # 先頭の凸 (0..12) が 0 時間
    with pytest.raises(ValueError, match="所要時間が 0"):
        restart_save.analyze(hits, _CPS, [12], ht, _D)


def test_save_point_outside_checkpoints_is_ignored():
    hits, ht = _setup()
    a = restart_save.analyze(hits, _CPS, [7], ht, _D)     # 7 は関門ではない
    b = restart_save.analyze(hits, _CPS, [], ht, _D)
    assert a["save_points"] == []
    assert a["exp_time"] == pytest.approx(b["exp_time"], rel=1e-9)


def test_product_with_save_point_runs_and_is_consistent():
    hits, ht = _setup()
    hp = HPParams(H=100_000_000.0, H1=100_000_000.0, R0=1.0, R1=2.0)
    ymix = [y_mixture(m, hp.beta) for m in hits]
    D = 4_000_000
    res = restart_save.analyze_product(ymix, hp, _CPS, [12], ht, D)
    assert res["has_save"] is True
    assert math.isfinite(res["exp_time"]) and res["exp_time"] > 0
    assert sum(b["exp_time"] for b in res["blocks"]) == pytest.approx(
        res["exp_time"], rel=1e-9)
    for r in res["rows"]:
        assert 0.0 <= r["gate"] <= D


# ---------------------------------------------------------------------------
# 混在モデル (HP依存 + 通常) — アフィンカーネルのグリッドバックエンド
# ---------------------------------------------------------------------------
_HP = HPParams(H=100_000_000.0, H1=100_000_000.0, R0=1.0, R1=2.0)


def _mixed_card(cmin, cmax, nmin, nmax, hits, cr, dep):
    c = _card(cmin, cmax, nmin, nmax, hits, cr)
    c["hp_dep"] = dep
    return c


_MIXED_CARDS = [
    _mixed_card(800_000, 1_000_000, 150_000, 190_000, 4, 55, True),
    _mixed_card(200_000, 270_000, 38_000, 49_000, 8, 55, False),
    _mixed_card(800_000, 1_000_000, 150_000, 190_000, 4, 55, True),
    _mixed_card(200_000, 270_000, 38_000, 49_000, 8, 55, False),
]
_MIXED_D = 6_000_000


def _mixed_setup():
    specs = hit_specs_from_cards(_MIXED_CARDS, 55, 0.0, "post_decay")
    return specs, [1.0] * len(specs)


def test_mixed_no_save_matches_restart_mixed():
    """セーブ点なしなら、既存の混在モデル Dinkelbach と同じ方策・同じ E[T]。"""
    specs, ht = _mixed_setup()
    a = restart_mixed.analyze_mixed(specs, _HP, _CPS, ht, _MIXED_D)
    b = restart_save.analyze_mixed(specs, _HP, _CPS, [], ht, _MIXED_D)
    assert b["exp_time"] == pytest.approx(1.0 / a["g_star_dp"], rel=1e-6)
    assert b["success"] == pytest.approx(a["success"], rel=1e-3)
    span = max(r["cum_max_at_cp"] for r in b["rows"])
    for ra, rb in zip(a["rows"], b["rows"]):
        assert rb["gate"] == pytest.approx(ra["gate"], abs=0.01 * span)


def test_mixed_save_point_reduces_expected_time():
    specs, ht = _mixed_setup()
    plain = restart_save.analyze_mixed(specs, _HP, _CPS, [], ht, _MIXED_D)
    saved = restart_save.analyze_mixed(specs, _HP, _CPS, [12], ht, _MIXED_D)
    assert saved["exp_time"] < plain["exp_time"]
    assert saved["has_save"] is True
    assert [r["save"] for r in saved["rows"]] == [False, True, False]


def test_mixed_block_times_sum_to_expected_time():
    specs, ht = _mixed_setup()
    res = restart_save.analyze_mixed(specs, _HP, _CPS, [12], ht, _MIXED_D)
    assert sum(b["exp_time"] for b in res["blocks"]) == pytest.approx(
        res["exp_time"], rel=1e-9)


def test_mixed_optimum_is_not_worse_than_its_own_constant_gates():
    """最適方策の値は、同じ関門を s 非依存に固定して評価した値を上回らない。

    W_b の折れ線表現が粗いと中断価値が上振れし、この不等号が破れる (凸区切りを
    入れたのに期待時間が増える)。適応細分 (docs/restart_save.md §3.3) の回帰試験。
    """
    specs, ht = _mixed_setup()
    opt = restart_save.analyze_mixed(specs, _HP, _CPS, [12], ht, _MIXED_D)
    fixed = restart_save.analyze_mixed(
        specs, _HP, _CPS, [12], ht, _MIXED_D,
        manual_gates=[r["gate"] for r in opt["rows"]])
    assert opt["exp_time"] <= fixed["exp_time"] * (1.0 + 1e-6)


def test_mixed_gate_floor_matches_reachability():
    """関門の到達可能下限は「そこから最大を引き続けて丁度 D」の点。"""
    specs, ht = _mixed_setup()
    specs = normalize_specs(specs, _HP)
    bounds, cps = restart_save._split_bounds(len(specs), _CPS)
    seg_blocks = [blocks_from_specs(specs[bounds[i]:bounds[i + 1]], _HP.beta)
                  for i in range(len(bounds) - 1)]
    s_hi = mixed_support(specs, _HP)[1]
    be = restart_save._GridBackend(seg_blocks, specs, bounds, _HP, float(s_hi))
    for cp in range(1, len(cps) + 2):
        floor = be.gate_floor(cp, float(_MIXED_D))
        if floor <= 0.0:
            assert be._max_final(cp, 0.0) >= _MIXED_D      # どこからでも届く
            continue
        assert be._max_final(cp, floor) == pytest.approx(_MIXED_D, rel=1e-6)
        assert be._max_final(cp, floor * 0.99) < _MIXED_D


def test_mixed_works_for_negative_beta():
    """瀕死特効型 (R0 > R1 ⇒ β<0, H̃₁<0) でも符号場合分けなしに動く。"""
    hp = HPParams(H=100_000_000.0, H1=100_000_000.0, R0=2.0, R1=1.0)
    assert hp.beta < 0 and hp.Htil < 0
    specs, ht = _mixed_setup()
    a = restart_mixed.analyze_mixed(specs, hp, _CPS, ht, _MIXED_D)
    plain = restart_save.analyze_mixed(specs, hp, _CPS, [], ht, _MIXED_D)
    saved = restart_save.analyze_mixed(specs, hp, _CPS, [12], ht, _MIXED_D)
    assert plain["exp_time"] == pytest.approx(1.0 / a["g_star_dp"], rel=1e-6)
    assert saved["exp_time"] < plain["exp_time"]
    assert saved["feasible"]


def test_block_starter_is_a_similarity_transform():
    """開始状態 s からの増分は、s=0 の分布の相似変換 (質量保存・平均が k 倍)。"""
    specs, _ht = _mixed_setup()
    specs = normalize_specs(specs, _HP)
    blocks = blocks_from_specs(specs[:4], _HP.beta)      # HP依存ブロック
    assert blocks[0][0] == "y"
    grid = np.linspace(0.0, mixed_support(specs, _HP)[1], 20001)
    st = restart_save._BlockStarter(blocks[0], _HP)
    f0 = st.density(grid, 0.0)
    assert np.trapezoid(f0, grid) == pytest.approx(1.0, abs=1e-3)
    s = 2_000_000.0
    fs = st.density(grid, s)
    assert np.trapezoid(fs, grid) == pytest.approx(1.0, abs=1e-3)
    k = (_HP.Htil - s) / _HP.Htil
    mean0 = np.trapezoid(f0 * grid, grid)
    means = np.trapezoid(fs * grid, grid)
    assert means - s == pytest.approx(k * mean0, rel=2e-3)
    # 通常ブロックは平行移動のみ (k=1)
    zb = blocks_from_specs(specs[4:12], _HP.beta)[0]
    assert zb[0] == "z"
    stz = restart_save._BlockStarter(zb, _HP)
    assert stz.scale_at(s) == 1.0


# ---------------------------------------------------------------------------
# UI 結線 (凸区切りのトグル → Store → 解析 → 表示)
# ---------------------------------------------------------------------------
import main  # noqa: E402,F401 - コールバック登録のため
from app.frontend import callbacks as cb  # noqa: E402

_UI_PARAMS = ["crit_min", "crit_max", "normal_min", "normal_max", "hits",
              "crit_rate", "evade_rate", "enemies", "hp_dep"]


def _ui_card(index, hits, cmin, cmax, nmin, nmax):
    vals = [cmin, cmax, nmin, nmax, hits, 60, 0, 1, []]
    return [({"type": "param", "param": p, "index": index}, v)
            for p, v in zip(_UI_PARAMS, vals)]


def _ui_inputs():
    pairs = (_ui_card(0, 4, 800_000, 1_000_000, 150_000, 190_000)
             + _ui_card(1, 8, 200_000, 270_000, 38_000, 49_000)
             + _ui_card(2, 4, 800_000, 1_000_000, 150_000, 190_000)
             + _ui_card(3, 8, 200_000, 270_000, 38_000, 49_000))
    return [i for i, _v in pairs], [v for _i, v in pairs]


def _ui_run(saves, D=6_000_000, hp_dep=None):
    """hp_dep: カードごとの HP依存フラグ (None なら全て通常 = 和モデル)。"""
    ids, vals = _ui_inputs()
    if hp_dep is not None:
        for i, dep in enumerate(hp_dep):
            for k, (pid, _v) in enumerate(zip(ids, vals)):
                if pid["index"] == i and pid["param"] == "hp_dep":
                    vals[k] = (["on"] if dep else [])
    keys = [0, 4, 12, 16]
    hp = ((100_000_000, 100_000_000, 1.0, 2.0) if hp_dep is not None
          else (None, None, None, None))
    return cb.run_restart(
        1, D, [0, 1, 2, 3], [0, 1, 2, 3], vals, ids,
        ["A", "B", "C", "D"], [{"index": i} for i in range(4)],
        _CPS, {str(k): 1.0 for k in keys}, {str(k): 100.0 for k in keys},
        saves, 60, 0, "post_decay",
        "on" if hp_dep is not None else "off", *hp,
        [], [])          # 蓄積スキルなし


def test_save_toggle_store_round_trip():
    ids = [{"type": "restart-seg-save", "index": m} for m in (4, 12, 16)]
    assert cb.update_restart_save([[], ["on"], []], ids) == [12]
    assert cb.update_restart_save([[], [], []], ids) == []


def test_segment_card_marks_the_save_boundary():
    plain = str(cb._seg_card(1, 3, 4, 12, 1.0, "カード2", 100.0, False))
    marked = str(cb._seg_card(1, 3, 4, 12, 1.0, "カード2", 100.0, True))
    assert "足切り2・🚩凸区切り" in marked and "足切り2・🚩凸区切り" not in plain
    assert "value=['on']" in marked and "value=['on']" not in plain
    assert "restart-seg-save" in plain          # トグル自体は常に出す
    # 最終区間の後ろには境界が無いのでトグルも出さない
    assert "restart-seg-save" not in str(
        cb._seg_card(2, 3, 12, 16, 1.0, "", 100.0, False))


def test_run_restart_with_save_point_reports_expected_total_time():
    _fig, summary, cfg, sliders = _ui_run([12])
    assert cfg["saves"] == [12]
    assert "期待総時間" in str(summary)
    assert len(sliders) == len(_CPS) + 1        # 各関門 + 「最適に戻す」ボタン
    # 凸区切りを外すと従来表示 (成功率/スループット) に戻る
    _f2, summary2, cfg2, _s2 = _ui_run([])
    assert cfg2["saves"] == []
    assert "スループット" in str(summary2)


def test_run_restart_save_point_beats_no_save_in_total_time():
    """同じ設定なら、凸区切りを入れたほうが目標到達までの延べ時間は短い。"""
    _f1, _s1, with_save, _sl1 = _ui_run([12])
    _f2, _s2, no_save, _sl2 = _ui_run([])
    total_no_save = 1.0 / no_save["opt_throughput"]
    assert with_save["opt_exp_time"] < total_no_save


def test_interactive_path_accepts_save_points():
    _fig, _summary, cfg, _sliders = _ui_run([12])
    slider_ids = [{"type": "restart-gate-slider", "index": m} for m in _CPS]
    values = [int(round(cfg["D"] - g)) for g in cfg["opt_gates"]]
    _f, s = cb.update_restart_interactive(values, slider_ids, cfg)
    assert "期待総時間" in str(s)


def test_ui_mixed_model_accepts_save_points():
    """HP依存カードと通常カードが混ざる = 混在モデルでも凸区切りが効く。"""
    dep = [True, False, True, False]
    _f0, summary0, plain, _s0 = _ui_run([], hp_dep=dep)
    assert "混在モデル" in str(summary0)
    _f1, summary1, cfg, sliders = _ui_run([12], hp_dep=dep)
    assert cfg["model"] == "mixed" and cfg["saves"] == [12]
    assert "期待総時間" in str(summary1)
    assert "未対応" not in str(summary1)
    # 凸区切りありのほうが、目標到達までの延べ時間は短い
    assert cfg["opt_exp_time"] < 1.0 / plain["opt_throughput"]
    # 手動調整パスも混在モデル + 凸区切りで動く
    slider_ids = [{"type": "restart-gate-slider", "index": m} for m in _CPS]
    values = [int(round(cfg["D"] - g)) for g in cfg["opt_gates"]]
    _f, s2 = cb.update_restart_interactive(values, slider_ids, cfg)
    assert "期待総時間" in str(s2)
    assert len(sliders) == len(_CPS) + 1


def test_fingerprint_changes_with_save_points():
    ids, vals = _ui_inputs()
    keys = [0, 4, 12, 16]
    common = dict(cp_store=_CPS, seg_times={str(k): 1.0 for k in keys},
                  seg_success={str(k): 100.0 for k in keys}, D=6_000_000,
                  global_crit=60, global_evade=0, damage_mode="post_decay",
                  hp_mode="off", hp_H=None, hp_H1=None, hp_R0=None, hp_R1=None)

    def fp(saves):
        return cb._restart_fingerprint(
            [0, 1, 2, 3], [0, 1, 2, 3], vals, ids, common["cp_store"],
            common["seg_times"], common["seg_success"], saves, common["D"],
            common["global_crit"], common["global_evade"],
            common["damage_mode"], common["hp_mode"], common["hp_H"],
            common["hp_H1"], common["hp_R0"], common["hp_R1"])

    assert fp([]) != fp([12])
    assert fp([12]) == fp([12])
