"""混在モデル (mixed.py / restart_mixed.py) の検証。

正解基準は (1) モーメント再帰による厳密平均・分散、(2) Monte Carlo。
docs/mixed.md の設定に倣い、HP依存ブロックと通常ブロックが交互に入る列で
分布・足切り最適化の両方を突き合わせる。
"""
import math

import numpy as np

from app.backend import restart_cos
from app.backend.cos import HPParams, build_hit_mixtures, y_mixture
from app.backend.mixed import (
    build_dist_auto,
    build_mixed_dist,
    card_is_hp_dep,
    hit_specs_from_cards,
    mc_mixed,
    mixed_moments,
    mixed_support,
)
from app.backend.restart_mixed import analyze_mixed


def _hp():
    return HPParams(H=1_000_000, H1=1_000_000, R0=1.0, R1=2.0)


def _mixed_cards():
    """通常(3) → HP依存(5) → 通常(2) → HP依存(4) の交互列。"""
    return [
        {"crit_min": 40000, "crit_max": 55000, "normal_min": 20000,
         "normal_max": 30000, "hits": 3, "crit_rate": 50, "evade_rate": 0,
         "hp_dep": []},
        {"crit_min": 30000, "crit_max": 40000, "normal_min": 15000,
         "normal_max": 20000, "hits": 5, "crit_rate": 60, "evade_rate": 0,
         "hp_dep": [1]},
        {"crit_min": 50000, "crit_max": 60000, "normal_min": 25000,
         "normal_max": 30000, "hits": 2, "crit_rate": 40, "evade_rate": 0,
         "hp_dep": []},
        {"crit_min": 35000, "crit_max": 45000, "normal_min": 18000,
         "normal_max": 24000, "hits": 4, "crit_rate": 60, "evade_rate": 0,
         "hp_dep": [1]},
    ]


def _specs(cards=None):
    return hit_specs_from_cards(cards or _mixed_cards(), 60, 0, "pre_decay")


def test_card_is_hp_dep_normalization():
    assert card_is_hp_dep({}) is True                     # 後方互換: 未指定は依存
    assert card_is_hp_dep({"hp_dep": None}) is True
    assert card_is_hp_dep({"hp_dep": [1]}) is True
    assert card_is_hp_dep({"hp_dep": []}) is False
    assert card_is_hp_dep({"hp_dep": 0}) is False
    assert card_is_hp_dep({"hp_dep": True}) is True


def test_mixed_moments_match_mc():
    specs, hp = _specs(), _hp()
    mean, var = mixed_moments(specs, hp)
    rng = np.random.default_rng(7)
    mc = mc_mixed(specs, hp, 1_000_000, rng)
    assert abs(mc.mean() - mean) / mean < 5e-3
    assert abs(mc.var() - var) / var < 2e-2


def test_mixed_cdf_matches_mc_quantiles():
    specs, hp = _specs(), _hp()
    dist = build_mixed_dist(specs, hp)
    rng = np.random.default_rng(11)
    mc = np.sort(mc_mixed(specs, hp, 1_000_000, rng))
    for p in (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98):
        q = mc[int(p * mc.size)]
        assert abs(float(dist.cdf(np.array([q]))[0]) - p) < 5e-3, p


def test_mixed_cdf_support_ends():
    specs, hp = _specs(), _hp()
    dist = build_mixed_dist(specs, hp)
    lo, hi = mixed_support(specs, hp)
    assert dist.cdf(np.array([lo - 1.0]))[0] == 0.0
    assert dist.cdf(np.array([hi + 1.0]))[0] == 1.0
    assert lo >= 0.0


def test_mixed_with_evade_atoms_matches_mc():
    cards = _mixed_cards()
    cards[1]["evade_rate"] = 15                 # HP依存ヒットに miss 原子
    cards[2]["evade_rate"] = 20                 # 通常ヒットにも miss 原子
    specs, hp = _specs(cards), _hp()
    dist = build_mixed_dist(specs, hp)
    rng = np.random.default_rng(23)
    mc = np.sort(mc_mixed(specs, hp, 1_000_000, rng))
    for p in (0.1, 0.5, 0.9):
        q = mc[int(p * mc.size)]
        assert abs(float(dist.cdf(np.array([q]))[0]) - p) < 7e-3, p


def test_mixed_negative_beta_matches_mc():
    """瀕死特効型 (R1 < R0, β < 0) でも分布が合う。"""
    hp = HPParams(H=1_000_000, H1=1_000_000, R0=2.0, R1=1.0)
    specs = _specs()
    mean, var = mixed_moments(specs, hp)
    dist = build_mixed_dist(specs, hp)
    rng = np.random.default_rng(31)
    mc = np.sort(mc_mixed(specs, hp, 500_000, rng))
    assert abs(mc.mean() - mean) / mean < 5e-3
    for p in (0.1, 0.5, 0.9):
        q = mc[int(p * mc.size)]
        assert abs(float(dist.cdf(np.array([q]))[0]) - p) < 7e-3, p


def test_negative_beta_all_hp_product_cos_matches_mc():
    """R1<R0 (β<0) でも ProductDist (COS) が CDF の向きを正しく扱い MC と一致する。"""
    hp = HPParams(H=1_000_000, H1=1_000_000, R0=2.0, R1=1.0)
    cards = [dict(c, hp_dep=[1]) for c in _mixed_cards()]
    specs = _specs(cards)
    dist = build_dist_auto(specs, hp)
    assert type(dist).__name__ == "ProductDist"
    rng = np.random.default_rng(53)
    mc = np.sort(mc_mixed(specs, hp, 500_000, rng))
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        q = mc[int(p * mc.size)]
        assert abs(float(dist.cdf(np.array([q]))[0]) - p) < 5e-3, p
    # 高ダメージ端 d_max は MC 最大値を上から押さえる
    assert dist.d_max >= mc[-1]
    assert float(dist.cdf(np.array([dist.d_max + 1.0]))[0]) == 1.0


def test_restart_product_negative_beta_matches_mc_and_grid():
    """R1<R0 の全 HP依存列でも restart_cos (COS) の前向き指標が MC・グリッド DP と一致。"""
    hp = HPParams(H=1_000_000, H1=1_000_000, R0=2.0, R1=1.0)
    cards = [dict(c, hp_dep=[1]) for c in _mixed_cards()]
    specs = _specs(cards)
    hits = build_hit_mixtures(cards, 60, 0, "pre_decay")
    ymix = [y_mixture(m, hp.beta) for m in hits]
    cps = [3, 10]
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.05
    manual = [mean * 0.12, mean * 0.55]
    res = restart_cos.analyze_product(ymix, hp, cps, 1.0, D, manual_gates=manual)
    res_dp = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=manual)
    rng = np.random.default_rng(59)
    mc_pass, mc_succ = _mc_policy(specs, hp, cps, manual, D, 300_000, rng)
    for k in range(len(cps)):
        assert abs(res["rows"][k]["pass_rate"] - mc_pass[k]) < 1.5e-2, k
        assert abs(res["rows"][k]["pass_rate"]
                   - res_dp["rows"][k]["pass_rate"]) < 1e-2, k
    assert abs(res["success"] - mc_succ) < 1.5e-2
    assert abs(res["success"] - res_dp["success"]) < 1e-2
    # 最適化パスも回り、関門での前向き指標が MC と整合する
    res_opt = restart_cos.analyze_product(ymix, hp, cps, 1.0, D)
    gates = [r["gate"] for r in res_opt["rows"]]
    mc_pass2, mc_succ2 = _mc_policy(specs, hp, cps, gates, D, 300_000, rng)
    assert abs(res_opt["success"] - mc_succ2) < 1.5e-2


def test_build_dist_auto_dispatch():
    hp = _hp()
    cards = _mixed_cards()
    all_z = [dict(c, hp_dep=[]) for c in cards]
    all_y = [dict(c, hp_dep=[1]) for c in cards]
    assert type(build_dist_auto(_specs(all_z), hp)).__name__ == "SumDist"
    assert type(build_dist_auto(_specs(all_y), hp)).__name__ == "ProductDist"
    assert type(build_dist_auto(_specs(cards), hp)).__name__ == "MixedDist"
    # β=0 は HP依存ヒットを R0 倍の通常ダメージへ退化 → 和モデル
    hp0 = HPParams(H=1_000_000, H1=1_000_000, R0=1.5, R1=1.5)
    d0 = build_dist_auto(_specs(cards), hp0)
    assert type(d0).__name__ == "SumDist"
    specs0 = [(False, m) if f else (False, m) for f, m in _specs(cards)]
    rng = np.random.default_rng(41)
    mc = mc_mixed(_specs(cards), hp0, 300_000, rng)
    assert abs(float(mc.mean()) - d0.mean) / d0.mean < 5e-3


def test_mixed_grid_dp_matches_sum_model_when_no_hp():
    """全ヒット通常でも build_mixed_dist (グリッドDP) 自体が和モデル COS と一致。"""
    hp = _hp()
    cards = [dict(c, hp_dep=[]) for c in _mixed_cards()]
    specs = _specs(cards)
    dist_dp = build_mixed_dist(specs, hp)
    dist_cos = build_dist_auto(specs, hp)      # SumDist
    for p in (0.1, 0.5, 0.9):
        xs = np.array([dist_dp.mean + (p - 0.5) * 4 * math.sqrt(dist_dp.var)])
        assert abs(float(dist_dp.cdf(xs)[0]) - float(dist_cos.cdf(xs)[0])) < 3e-3


# ---------------------------------------------------------------------------
# 足切りライン最適化 (restart_mixed)
# ---------------------------------------------------------------------------
def _mc_policy(specs, hp, cps, gates, D, n, rng):
    """関門固定のリスタ運用を MC で回し (段別通過率, 成功率) を返す。"""
    Hn = np.full(n, hp.H1, dtype=float)
    alive = np.ones(n, dtype=bool)
    pass_rates = []
    cp_set = {c: k for k, c in enumerate(cps)}
    for t, (is_hp, mix) in enumerate(specs, start=1):
        w = np.array([u.weight for u in mix])
        w = w / w.sum()
        comp = rng.choice(len(mix), size=n, p=w)
        x = np.empty(n)
        for j, u in enumerate(mix):
            m = comp == j
            if m.any():
                x[m] = rng.uniform(u.lo, u.hi, size=int(m.sum())) if u.hi > u.lo else u.lo
        if is_hp:
            Hn = Hn - (hp.beta * Hn + hp.R0) * x
        else:
            Hn = Hn - x
        if t in cp_set:
            k = cp_set[t]
            alive = alive & ((hp.H1 - Hn) >= gates[k])
            pass_rates.append(float(alive.mean()))
    success = float((alive & ((hp.H1 - Hn) >= D)).mean())
    return pass_rates, success


def test_restart_mixed_forward_matches_mc():
    specs, hp = _specs(), _hp()
    cps = [3, 10]                               # カード境界 (3=通常後, 10=通常2枚目後)
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.05
    manual = [mean * 0.12, mean * 0.55]         # 適当な手動関門 (累積ダメージ)
    res = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=manual)
    rng = np.random.default_rng(101)
    mc_pass, mc_succ = _mc_policy(specs, hp, cps, manual, D, 400_000, rng)
    for k in range(len(cps)):
        assert abs(res["rows"][k]["pass_rate"] - mc_pass[k]) < 1.5e-2, k
    assert abs(res["success"] - mc_succ) < 1.5e-2


def test_restart_mixed_optimize_beats_baseline_and_matches_mc():
    specs, hp = _specs(), _hp()
    cps = [3, 10]
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.1
    res = analyze_mixed(specs, hp, cps, 1.0, D)
    # 最適足切りはスループットを基準 (足切り無し) 以上にする
    assert res["throughput"] >= res["baseline"]["g"] * 0.999
    # 最適関門での前向き指標が MC と一致する
    gates = [r["gate"] for r in res["rows"]]
    rng = np.random.default_rng(103)
    mc_pass, mc_succ = _mc_policy(specs, hp, cps, gates, D, 400_000, rng)
    for k in range(len(cps)):
        assert abs(res["rows"][k]["pass_rate"] - mc_pass[k]) < 1.5e-2, k
    assert abs(res["success"] - mc_succ) < 1.5e-2


def test_restart_mixed_reduces_to_sum_model():
    """全ヒット通常なら restart_cos (和モデル COS) と一致する。"""
    cards = [dict(c, hp_dep=[]) for c in _mixed_cards()]
    specs, hp = _specs(cards), _hp()
    hits = build_hit_mixtures(cards, 60, 0, "pre_decay")
    cps = [3, 10]
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.05
    gates = [mean * 0.1, mean * 0.5]
    res_dp = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=gates)
    res_cos = restart_cos.analyze(hits, cps, 1.0, D, manual_gates=gates)
    assert abs(res_dp["success"] - res_cos["success"]) < 1e-2
    for k in range(2):
        assert abs(res_dp["rows"][k]["pass_rate"]
                   - res_cos["rows"][k]["pass_rate"]) < 1e-2


def test_restart_mixed_seg_success_scaling():
    """区間独立成功確率が通過率・成功率に正しく乗る。"""
    specs, hp = _specs(), _hp()
    cps = [3, 10]
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.05
    gates = [mean * 0.12, mean * 0.55]
    base = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=gates)
    scaled = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=gates,
                           seg_success=[0.5, 1.0, 0.8])
    assert abs(scaled["rows"][0]["pass_rate"]
               - 0.5 * base["rows"][0]["pass_rate"]) < 1e-9
    assert abs(scaled["success"] - 0.5 * 0.8 * base["success"]) < 1e-9
