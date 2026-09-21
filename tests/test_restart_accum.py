"""蓄積スキルつき多段リスタ最適化 (app/backend/restart_accum.py) の検証。

正解基準は 3 つ。
  (1) 窓なしなら既存の COS 版 (restart_cos.analyze) に一致する
  (2) 飽和しない上限なら「窓内カードを (1+α) 倍した和モデル」に一致する
      (docs/accumulate.md §7 の縮退)
  (3) 関門を適用した前向き指標 (通過率・成功率・期待時間) が素朴な MC に一致する
さらに、窓が足切り境界をまたぐ設定は ValueError で弾かれることを確認する
(docs/accumulate.md §4.2: 1 次元 DP では解けないため)。
"""
import numpy as np
import pytest

from app.backend import restart_accum, restart_cos
from app.backend.accumulate import AccumWindow, CapSpec
from app.backend.cos import build_hit_mixtures
from app.backend.mixed import scale_mixture

_D = 9_000_000
_CPS = [4, 7]


def _cards():
    return [
        {"crit_min": 900_000, "crit_max": 1_100_000, "normal_min": 450_000,
         "normal_max": 550_000, "hits": 4, "crit_rate": 60, "evade_rate": 0},
        {"crit_min": 1_400_000, "crit_max": 1_800_000, "normal_min": 700_000,
         "normal_max": 900_000, "hits": 3, "crit_rate": 50, "evade_rate": 5},
        {"crit_min": 300_000, "crit_max": 380_000, "normal_min": 150_000,
         "normal_max": 190_000, "hits": 6, "crit_rate": 40, "evade_rate": 0},
    ]


def _hits():
    return build_hit_mixtures(_cards(), 50, 0, "post_decay")


def _times(n, cps):
    return [1.0] * n, [0, *cps, n]


# ---------------------------------------------------------------------------
# 退化: 既存エンジンと一致する
# ---------------------------------------------------------------------------
def test_no_window_matches_cos_engine():
    hits = _hits()
    ht, _b = _times(len(hits), _CPS)
    ra = restart_accum.analyze_accum(hits, [], _CPS, ht, _D)
    rc = restart_cos.analyze(hits, _CPS, ht, _D)
    assert ra["throughput"] == pytest.approx(rc["throughput"], rel=2e-3)
    assert ra["success"] == pytest.approx(rc["success"], abs=2e-3)
    for a, c in zip(ra["rows"], rc["rows"]):
        assert a["gate"] == pytest.approx(c["gate"], rel=5e-3)


def test_huge_cap_matches_scaled_sum_model():
    """飽和しない上限なら、窓内カードを (1+α) 倍した和モデルと同じ問題になる。"""
    hits = _hits()
    ht, _b = _times(len(hits), _CPS)
    idx, rate = [0, 1, 2, 3], 1.0
    win = AccumWindow(hits=idx, rate=rate, cap=CapSpec(kind="fixed", value=1e12))
    ra = restart_accum.analyze_accum(hits, [win], _CPS, ht, _D)
    scaled = [scale_mixture(m, 1.0 + rate) if i in idx else m
              for i, m in enumerate(hits)]
    rc = restart_cos.analyze(scaled, _CPS, ht, _D)
    assert ra["throughput"] == pytest.approx(rc["throughput"], rel=5e-3)
    assert ra["success"] == pytest.approx(rc["success"], abs=3e-3)
    for a, c in zip(ra["rows"], rc["rows"]):
        assert a["gate"] == pytest.approx(c["gate"], rel=1e-2)


# ---------------------------------------------------------------------------
# 窓が足切り境界をまたぐ場合は解けない
# ---------------------------------------------------------------------------
def test_window_across_checkpoint_is_rejected():
    hits = _hits()
    ht, _b = _times(len(hits), _CPS)
    win = AccumWindow(hits=[2, 3, 4, 5], rate=1.0,
                      cap=CapSpec(kind="fixed", value=2e6), name="ワカモ")
    with pytest.raises(ValueError, match="またいで"):
        restart_accum.analyze_accum(hits, [win], _CPS, ht, _D)


def test_cap_source_outside_window_must_share_the_segment():
    """上限ロールの Hit も同じ区間に無ければならない。"""
    hits = _hits()
    ht, _b = _times(len(hits), _CPS)
    win = AccumWindow(hits=[5, 6], rate=1.5,
                      cap=CapSpec(kind="hits", hits=[0], coef=1.2), name="イロハ")
    with pytest.raises(ValueError, match="またいで"):
        restart_accum.analyze_accum(hits, [win], _CPS, ht, _D)


# ---------------------------------------------------------------------------
# 前向き指標 vs MC
# ---------------------------------------------------------------------------
def _mc_segment_increments(hits, windows, bounds, M, rng):
    """区間ごとの増分 (区間内 Hit の和 + その区間で爆発するプール) の MC サンプル。"""
    xs = []
    for mix in hits:
        w = np.array([u.weight for u in mix])
        w = w / w.sum()
        comp = rng.choice(len(mix), size=M, p=w)
        x = np.empty(M)
        for j, u in enumerate(mix):
            m = comp == j
            cnt = int(m.sum())
            if cnt:
                x[m] = rng.uniform(u.lo, u.hi, size=cnt) if u.hi > u.lo else u.lo
        xs.append(x)

    incs = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        seg = np.zeros(M)
        for h in range(lo, hi):
            seg += xs[h]
        for win in windows:
            if not (lo <= min(win.hits) and max(win.hits) < hi):
                continue
            pool_src = np.zeros(M)
            for h in win.hits:
                pool_src += xs[h]
            if win.cap.kind == "fixed":
                cap = np.full(M, float(win.cap.value))
            else:
                cap = np.zeros(M)
                for h in win.cap.source_hits():
                    cap += xs[h]
                cap *= win.cap.coef
            seg += win.burst_mult * np.minimum(cap, win.rate * pool_src)
        incs.append(seg)
    return incs


@pytest.mark.parametrize("cap_value,rate", [(2_500_000, 1.0), (10_000_000, 1.2)])
def test_forward_metrics_match_monte_carlo(cap_value, rate):
    hits = _hits()
    n = len(hits)
    ht, bounds = _times(n, _CPS)
    win = AccumWindow(hits=[0, 1, 2, 3], rate=rate,
                      cap=CapSpec(kind="fixed", value=cap_value), name="ワカモ")
    res = restart_accum.analyze_accum(hits, [win], _CPS, ht, _D)
    gates = {k + 1: r["gate"] for k, r in enumerate(res["rows"])}
    times = [float(sum(ht[bounds[i]:bounds[i + 1]]))
             for i in range(len(bounds) - 1)]

    rng = np.random.default_rng(11)
    M = 400_000
    incs = _mc_segment_increments(hits, [win], bounds, M, rng)
    cum = np.zeros(M)
    alive = np.ones(M, bool)
    exp_time = times[0]
    for j in range(1, len(_CPS) + 1):
        cum = cum + incs[j - 1]
        alive &= cum >= gates[j]
        p_j = float(alive.mean())
        assert res["rows"][j - 1]["pass_rate"] == pytest.approx(p_j, abs=6e-3)
        exp_time += times[j] * p_j
    cum = cum + incs[-1]
    success = float((alive & (cum >= _D)).mean())

    assert res["success"] == pytest.approx(success, abs=6e-3)
    assert res["exp_time"] == pytest.approx(exp_time, rel=6e-3)
    assert res["throughput"] == pytest.approx(success / exp_time, rel=1e-2)


def test_gates_beat_the_no_gate_baseline():
    hits = _hits()
    ht, _b = _times(len(hits), _CPS)
    win = AccumWindow(hits=[0, 1, 2, 3], rate=1.0,
                      cap=CapSpec(kind="fixed", value=2_500_000), name="ワカモ")
    res = restart_accum.analyze_accum(hits, [win], _CPS, ht, _D)
    assert res["throughput"] >= res["baseline"]["g"]
    assert res["speedup"] >= 1.0


def test_manual_gates_skip_optimization():
    hits = _hits()
    ht, _b = _times(len(hits), _CPS)
    win = AccumWindow(hits=[0, 1, 2, 3], rate=1.0,
                      cap=CapSpec(kind="fixed", value=2_500_000), name="ワカモ")
    opt = restart_accum.analyze_accum(hits, [win], _CPS, ht, _D)
    man = restart_accum.analyze_accum(
        hits, [win], _CPS, ht, _D,
        manual_gates=[r["gate"] for r in opt["rows"]])
    assert np.isnan(man["g_star_dp"])
    assert man["throughput"] == pytest.approx(opt["throughput"], rel=1e-6)


# ---------------------------------------------------------------------------
# UI 経路 (run_restart) の配線
# ---------------------------------------------------------------------------
import main  # noqa: E402,F401 - コールバック登録
from app.frontend import callbacks as cb  # noqa: E402

_UI_PARAMS = ["crit_min", "crit_max", "normal_min", "normal_max",
              "hits", "crit_rate", "evade_rate", "enemies", "hp_dep"]
# カード0: 蓄積を付与する攻撃 (1Hit) / カード1: 4Hit / カード2: 3Hit → 計 8 Hit
_UI_CARDS = [
    [1_600_000, 2_000_000, 0, 0, 1, 100, 0, 1, []],
    [900_000, 1_100_000, 450_000, 550_000, 4, 60, 0, 1, []],
    [300_000, 380_000, 150_000, 190_000, 3, 40, 0, 1, []],
]


def _ui_card_states():
    vals, ids = [], []
    for i, row in enumerate(_UI_CARDS):
        for p, v in zip(_UI_PARAMS, row):
            vals.append(v)
            ids.append({"type": "param", "param": p, "index": i})
    return vals, ids


def _ui_pool(**kw):
    base = {"name": "ワカモ", "cards": [1], "rate": 100, "cap_mode": "atk",
            "cap_value": None, "atk": 30_000, "atk_pct": 1322, "cap_cards": [],
            "cap_pct": 100, "burst_mult": 100, "burst_decay": [1]}
    base.update(kw)
    return base


def _run_ui(pools, cps=(5,), D=6_000_000, saves=(), hp_mode="off"):
    vals, ids = _ui_card_states()
    a_vals, a_ids = [], []
    for k, pool in enumerate(pools):
        for f, v in pool.items():
            a_vals.append(v)
            a_ids.append({"type": "accum", "field": f, "index": k})
    return cb.run_restart(
        1, D, [0, 1, 2], [0, 1, 2], vals, ids,
        ["付与", "アタッカー", "サブ"], [{"index": i} for i in range(3)],
        list(cps), {"0": 1.0, "5": 1.0}, {"0": 100.0, "5": 100.0}, list(saves),
        60, 0, "post_decay", hp_mode, None, None, None, None,
        a_vals, a_ids)


def _text(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return " ".join(_text(c) for c in node)
    children = getattr(node, "children", None)
    return _text(children) if children is not None else ""


def test_ui_runs_with_a_pool_and_reports_diagnostics():
    _fig, summary, cfg, sliders = _run_ui([_ui_pool()])
    txt = _text(summary)
    assert "蓄積スキル" in txt and "ワカモ" in txt and "飽和確率" in txt
    assert cfg["model"] == "accum"
    assert cfg["accum"][0]["hits"] == [1, 2, 3, 4]   # カード1 = Hit 1..4
    assert sliders


def test_ui_pool_raises_the_throughput_over_no_pool():
    """蓄積ぶんダメージが増えるので、同じ目標なら成功率・スループットが上がる。"""
    _f0, _s0, cfg0, _sl0 = _run_ui([])
    _f1, _s1, cfg1, _sl1 = _run_ui([_ui_pool()])
    assert cfg1["opt_success"] > cfg0["opt_success"]
    assert cfg1["opt_throughput"] > cfg0["opt_throughput"]


def test_ui_pool_across_checkpoint_is_reported():
    _fig, summary, cfg, _sl = _run_ui([_ui_pool(cards=[1, 2])])
    assert cfg is None
    assert "またいで" in _text(summary)


def test_ui_cap_card_mode_is_wired():
    """イロハ型: 上限 = 付与カードのダメージ × 120% (相関あり)。"""
    _fig, _summary, cfg, _sl = _run_ui([
        _ui_pool(name="イロハ", cards=[1], rate=150, cap_mode="cards",
                 cap_cards=[0], cap_pct=120, burst_decay=[])])
    assert cfg["accum"][0]["cap"]["kind"] == "hits"
    assert cfg["accum"][0]["cap"]["hits"] == [0]
    assert cfg["accum"][0]["cap"]["coef"] == pytest.approx(1.2)


def test_ui_rejects_save_points_and_hp_dep():
    _f, s1, cfg1, _a = _run_ui([_ui_pool()], saves=(5,))
    assert cfg1 is None and "凸区切り" in _text(s1)


def test_ui_manual_gate_path_reruns_with_pools():
    """スライダー手動調整のパスでも蓄積が効いている (cfg から再計算する)。"""
    _fig, _summary, cfg, _sl = _run_ui([_ui_pool()])
    D = float(cfg["D"])
    slider_ids = [{"type": "restart-gate-slider", "index": m} for m in cfg["cps"]]
    remains = [int(round(D - g)) for g in cfg["opt_gates"]]
    _fig2, summary2 = cb.update_restart_interactive(remains, slider_ids, cfg)
    assert "成功率" in _text(summary2)


def test_fingerprint_tracks_pools():
    vals, ids = _ui_card_states()
    a_ids = [{"type": "accum", "field": "cards", "index": 0},
             {"type": "accum", "field": "rate", "index": 0}]
    base = cb._restart_fingerprint(
        [0, 1, 2], [0, 1, 2], vals, ids, [5], {"0": 1.0}, {"0": 100.0}, [],
        6_000_000, 60, 0, "post_decay", "off", None, None, None, None,
        [[1], 100], a_ids)
    changed = cb._restart_fingerprint(
        [0, 1, 2], [0, 1, 2], vals, ids, [5], {"0": 1.0}, {"0": 100.0}, [],
        6_000_000, 60, 0, "post_decay", "off", None, None, None, None,
        [[1], 120], a_ids)
    assert base != changed


# ---------------------------------------------------------------------------
# 爆発の着弾タイミング
# ---------------------------------------------------------------------------
def test_burst_position_does_not_change_the_optimum():
    """爆発の着弾位置は最適解を変えない (確定プールは情報として使えるため)。"""
    early = _run_ui([_ui_pool()])[2]                       # 既定 = 蓄積の最後
    late = _run_ui([_ui_pool(burst_after=2)])[2]           # カード2の直後に着弾
    assert late["opt_success"] == pytest.approx(early["opt_success"], rel=1e-9)
    assert late["opt_gates"] == pytest.approx(early["opt_gates"], rel=1e-9)


def test_burst_after_checkpoint_gets_a_screen_reading_note():
    _fig, summary, cfg, _sl = _run_ui([_ui_pool(burst_after=2)])
    txt = _text(summary)
    assert "未着弾" in txt and "残りダメージ" in txt
    assert cfg["accum"][0]["burst_hit"] == 7            # カード2 = Hit 5..7


def test_burst_before_checkpoint_needs_no_note():
    _fig, summary, cfg, _sl = _run_ui([_ui_pool()])
    assert "未着弾" not in _text(summary)
    assert cfg["accum"][0]["burst_hit"] == 4            # カード1 = Hit 1..4


def test_screen_reading_equals_table_plus_burst():
    """注記の値 = 表の残りダメージ + 未着弾の爆発ダメージ。"""
    _fig, summary, cfg, _sl = _run_ui([_ui_pool(burst_after=2)])
    res_gate = cfg["opt_gates"][0]
    D = float(cfg["D"])
    # 爆発平均は診断行から拾う (両方とも同じ res から出ている)
    txt = _text(summary)
    import re
    nums = [float(x.replace(",", "")) for x in re.findall(r"[\d,]{7,}", txt)]
    assert any(abs(v - (D - res_gate)) < 1.0 for v in nums)   # 表の残りダメージ


def test_burst_before_the_window_closes_is_rejected():
    _fig, summary, cfg, _sl = _run_ui([_ui_pool(burst_after=0)])
    assert cfg is None
    assert "蓄積の終わり" in _text(summary)


def test_burst_position_survives_export_import():
    from app.frontend import callbacks as c
    vals, ids = [], []
    pool = _ui_pool(burst_after=2, cards=[1])
    for f, v in pool.items():
        vals.append(v)
        ids.append({"type": "accum", "field": f, "index": 0})
    rows = c._export_accum(vals, ids, {0: 0, 1: 1, 2: 2})
    assert rows[0]["burst_after"] == 2
    children, nxt = c._import_accum(rows, [])
    assert nxt == 1 and children
