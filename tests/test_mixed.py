"""混在モデル (mixed.py / restart_mixed.py) の検証。

正解基準は (1) モーメント再帰による厳密平均・分散、(2) Monte Carlo、
(3) 2 ブロック配置 (通常全先/全後) の準厳密リファレンス (docs/mixed.md §6.1 の
1 次元 COS 表現を Simpson 求積で合成、~1e-9)、(4) 高解像度 DP との自己収束。
裾確率 (カットオフ率) の絶対誤差 ~1e-5 を保証するのが目的。
docs/mixed.md の設定に倣い、HP依存ブロックと通常ブロックが交互に入る列で
分布・足切り最適化の両方を突き合わせる。
"""
import math

import numpy as np

from app.backend import restart_cos, restart_mixed
from app.backend.cos import (
    HPParams,
    _atom_step_cdf,
    build_hit_mixtures,
    build_product_dist,
    build_sum_dist,
    y_mixture,
)
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


# ---------------------------------------------------------------------------
# 裾確率の高精度検証 (準厳密リファレンス / 自己収束, 目標 ~1e-5)
# ---------------------------------------------------------------------------
def _specs_two_block(cards, order):
    """2 ブロック配置: normal_last = HP依存全部→通常全部、normal_first は逆。"""
    specs = hit_specs_from_cards(cards, 60, 0, "pre_decay")
    hp_hits = [s for s in specs if s[0]]
    z_hits = [s for s in specs if not s[0]]
    return hp_hits + z_hits if order == "normal_last" else z_hits + hp_hits


def _reference_tail(specs, hp, xs, order):
    """2 ブロック配置の P(D > x) 準厳密リファレンス (docs/mixed.md §6.1)。

    normal_last:  D = D_prod + Z (独立和)。 P(D<=x) = E_Z[F_prod(x - Z)]
    normal_first: D = H̃₁ - (H̃₁-Z)Π。      P(D<=x|z) = 1 - F_S(ln((H̃₁-x)/(H̃₁-z)))
    Z の連続部は生の COS 級数 (pdf の負リンギング 0 クリップは質量を ~1e-6 水増し
    するため使わない) を Simpson 求積、原子は厳密に足す。誤差 ~1e-9。"""
    z_mixes = [m for f, m in specs if not f]
    y_mixes = [y_mixture(m, hp.beta) for f, m in specs if f]
    sd = build_sum_dist(z_mixes)
    pd = build_product_dist(y_mixes, hp)
    Htil = hp.Htil

    z_lo = sum(min(u.lo for u in m) for m in z_mixes)
    z_hi = sum(max(u.hi for u in m) for m in z_mixes)
    nz = 20001
    zg = np.linspace(z_lo, z_hi, nz)
    Fk = sd.Fk.copy()
    Fk[0] *= 0.5
    fz = np.concatenate([
        (Fk[None, :] * np.cos(np.outer(zg[i:i + 4000] - sd.a, sd.u))).sum(axis=1)
        for i in range(0, nz, 4000)])
    w = np.ones(nz)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w *= (zg[1] - zg[0]) / 3.0

    def F_cond(x, zv):
        if order == "normal_last":
            return pd.cdf(np.asarray(x - zv, dtype=float))
        s = np.log(np.clip((Htil - x) / (Htil - zv), 1e-300, None))
        Fs = pd._cdf_S(s)
        if pd.av is not None:
            Fs = Fs + _atom_step_cdf(pd.av, pd.ap, s)
        return 1.0 - np.clip(Fs, 0.0, 1.0)

    out = []
    for x in xs:
        val = float((w * fz * F_cond(x, zg)).sum())
        if sd.av is not None:
            val += float(sum(p * F_cond(x, np.array([v]))[0]
                             for v, p in zip(sd.av, sd.ap)))
        out.append(1.0 - val)
    return np.array(out)


def test_mixed_tail_matches_semiexact_reference():
    """2 ブロック配置の裾確率が準厳密リファレンスと 5e-6 以内で一致する。"""
    hp = _hp()
    for evade in (0, 15):
        cards = _mixed_cards()
        for c in cards:
            c["evade_rate"] = evade
        for order in ("normal_last", "normal_first"):
            specs = _specs_two_block(cards, order)
            mean, var = mixed_moments(specs, hp)
            sd = math.sqrt(var)
            xs = [mean + k * sd for k in (1.0, 1.5, 2.0, 2.5, 3.0)]
            ref = _reference_tail(specs, hp, xs, order)
            dist = build_mixed_dist(specs, hp)
            dp = 1.0 - dist.cdf(np.array(xs))
            err = np.max(np.abs(dp - ref))
            assert err < 5e-6, (order, evade, err)


def test_mixed_tail_self_convergence():
    """交互配置 (原子込み) の裾確率が高解像度 DP と 5e-6 以内で一致する。"""
    hp = _hp()
    cards = _mixed_cards()
    for c in cards:
        c["evade_rate"] = 15
    specs = hit_specs_from_cards(cards, 60, 0, "pre_decay")
    mean, var = mixed_moments(specs, hp)
    sd = math.sqrt(var)
    xs = np.array([mean + k * sd for k in (1.0, 1.5, 2.0, 2.5, 3.0)])
    tail = 1.0 - build_mixed_dist(specs, hp).cdf(xs)
    tail_fine = 1.0 - build_mixed_dist(specs, hp, n_grid=24001).cdf(xs)
    assert np.max(np.abs(tail - tail_fine)) < 5e-6


def test_restart_mixed_forward_self_convergence():
    """関門固定の通過率・成功率が高解像度前向きグリッドと 8e-6 以内で一致する。"""
    hp = _hp()
    cards = _mixed_cards()
    for c in cards:
        c["evade_rate"] = 15
    specs = hit_specs_from_cards(cards, 60, 0, "pre_decay")
    cps = [3, 10]
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.05
    manual = [mean * 0.12, mean * 0.55]
    res = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=manual)
    old = (restart_mixed._N_GRID_FORWARD, restart_mixed._N_GRID_FORWARD_MAX)
    try:
        restart_mixed._N_GRID_FORWARD = restart_mixed._N_GRID_FORWARD_MAX = 36000
        ref = analyze_mixed(specs, hp, cps, 1.0, D, manual_gates=manual)
    finally:
        restart_mixed._N_GRID_FORWARD, restart_mixed._N_GRID_FORWARD_MAX = old
    for k in range(len(cps)):
        assert abs(res["rows"][k]["pass_rate"]
                   - ref["rows"][k]["pass_rate"]) < 8e-6, k
    assert abs(res["success"] - ref["success"]) < 8e-6


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


def test_restart_mixed_negative_beta_optimize_matches_mc():
    """β<0 (瀕死特効型) でも最適化パスが回り、最適関門の前向き指標が MC と一致。"""
    hp = HPParams(H=1_000_000, H1=1_000_000, R0=2.0, R1=1.0)
    specs = _specs()
    cps = [3, 10]
    mean, _ = mixed_moments(specs, hp)
    D = mean * 1.1
    res = analyze_mixed(specs, hp, cps, 1.0, D)
    assert res["throughput"] >= res["baseline"]["g"] * 0.999
    gates = [r["gate"] for r in res["rows"]]
    rng = np.random.default_rng(107)
    mc_pass, mc_succ = _mc_policy(specs, hp, cps, gates, D, 300_000, rng)
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
