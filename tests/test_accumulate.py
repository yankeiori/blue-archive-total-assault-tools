"""蓄積 (チャージ) 型スキル (app/backend/accumulate.py) の検証。

正解基準は 4 つ。(1) 窓なし・上限 0 / ∞ の退化で既存の和モデル (build_sum_dist) に
一致すること、(2) 単調写像の厳密解 1 - F_S(ψ^{-1}(D))、(3) COS の CDF を使う 1 次元
求積 (pass_prob_quad)、(4) Monte Carlo。さらに E[min(C, αS)] は生存関数積分
(docs/accumulate.md §6) と突き合わせる。
"""
import numpy as np
import pytest

from app.backend.accumulate import (
    AccumWindow,
    CapSpec,
    build_accum_dist,
    mc_accum,
    min_moments_survival,
    pass_prob_quad,
    psi,
    psi_inv,
)
from app.backend.cos import Uniform, build_hit_mixtures, build_sum_dist
from app.backend.mixed import scale_mixture

N_MC = 200_000


def _hm():
    """EX 4Hit + NS 3Hit + サブ 3Hit の攻撃列 (10 Hit)。"""
    cards = [
        {"crit_min": 900_000, "crit_max": 1_100_000, "normal_min": 450_000,
         "normal_max": 550_000, "hits": 4, "crit_rate": 60, "evade_rate": 0},
        {"crit_min": 1_400_000, "crit_max": 1_800_000, "normal_min": 700_000,
         "normal_max": 900_000, "hits": 3, "crit_rate": 50, "evade_rate": 5},
        {"crit_min": 300_000, "crit_max": 380_000, "normal_min": 150_000,
         "normal_max": 190_000, "hits": 3, "crit_rate": 40, "evade_rate": 0},
    ]
    return build_hit_mixtures(cards, 50, 0, "post_decay")


def _grid(hm, **kw):
    return np.linspace(*_span(hm, **kw), 41)


def _span(hm, **kw):
    d = build_accum_dist(hm, **kw)
    return d.support_lo, d.support_hi


# ---------------------------------------------------------------------------
# 退化ケース: 既存の和モデルに一致する
# ---------------------------------------------------------------------------
def test_no_window_matches_sum_model():
    hm = _hm()
    d = build_accum_dist(hm, [])
    s = build_sum_dist(hm)
    xs = np.linspace(s.support_lo, s.support_hi, 201)
    assert np.abs(d.cdf(xs) - s.cdf(xs)).max() < 1e-6
    assert d.mean == pytest.approx(s.mean, rel=1e-9)


def test_zero_cap_matches_sum_model():
    hm = _hm()
    win = AccumWindow(hits=[0, 1, 2, 3], rate=1.0, cap=CapSpec(kind="fixed", value=0.0))
    d = build_accum_dist(hm, [win])
    s = build_sum_dist(hm)
    xs = np.linspace(s.support_lo, s.support_hi, 201)
    assert np.abs(d.cdf(xs) - s.cdf(xs)).max() < 1e-6


def test_huge_cap_matches_scaled_sum_model():
    """上限が十分大きければ「窓内カードを (1+α) 倍した和モデル」に一致する。"""
    hm = _hm()
    idx, rate = [0, 1, 2, 3], 1.2
    win = AccumWindow(hits=idx, rate=rate, cap=CapSpec(kind="fixed", value=1e12))
    d = build_accum_dist(hm, [win])
    scaled = [scale_mixture(m, 1.0 + rate) if i in idx else m for i, m in enumerate(hm)]
    s = build_sum_dist(scaled)
    xs = np.linspace(s.support_lo, s.support_hi, 201)
    assert np.abs(d.cdf(xs) - s.cdf(xs)).max() < 1e-4
    assert d.window_stats[0].sat_prob == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# ψ とその逆写像
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("burst_decay", [False, True])
def test_psi_inv_is_exact(burst_decay):
    lo, hi, cap, rate = 0.0, 3e7, 8e6, 1.2
    s = np.linspace(lo, hi, 977)
    z = psi(s, cap, rate, burst_decay)
    assert np.all(np.diff(z) > 0)  # 狭義単調増加
    back = psi_inv(z, cap, rate, burst_decay, lo, hi)
    assert np.abs(back - s).max() < 1e-6 * hi


# ---------------------------------------------------------------------------
# 求積リファレンス・MC との一致
# ---------------------------------------------------------------------------
def test_fixed_cap_matches_quadrature_and_mc():
    hm = _hm()
    win = AccumWindow(hits=list(range(7)), rate=1.0,
                      cap=CapSpec(kind="fixed", value=5_000_000))
    d = build_accum_dist(hm, [win])
    mc = mc_accum(hm, [win], N_MC, np.random.default_rng(1))
    for p in (0.25, 0.5, 0.9, 0.99):
        D = float(np.quantile(mc, p))
        assert d.pass_prob(D) == pytest.approx(pass_prob_quad(hm, win, D), abs=1e-5)
        assert d.pass_prob(D) == pytest.approx(1.0 - p, abs=5e-3)
    assert d.mean == pytest.approx(mc.mean(), rel=2e-3)


def test_random_cap_matches_quadrature():
    hm = _hm()
    cap = [Uniform(0.5, 4_000_000, 5_000_000), Uniform(0.5, 7_000_000, 8_000_000)]
    win = AccumWindow(hits=list(range(7)), rate=1.2,
                      cap=CapSpec(kind="mixture", mixture=cap))
    d = build_accum_dist(hm, [win])
    mc = mc_accum(hm, [win], N_MC, np.random.default_rng(2))
    for p in (0.5, 0.9):
        D = float(np.quantile(mc, p))
        assert d.pass_prob(D) == pytest.approx(pass_prob_quad(hm, win, D), abs=1e-5)


@pytest.mark.parametrize("hits", [[0, 1, 2, 3], [1, 2, 3]])
def test_cap_from_own_roll_matches_mc(hits):
    """上限がスキル自身のダメージロール由来 (窓内 / 窓外の両方)。"""
    hm = _hm()
    win = AccumWindow(hits=hits, rate=1.0, cap=CapSpec(kind="hit", hit=0, coef=3.0))
    d = build_accum_dist(hm, [win])
    mc = mc_accum(hm, [win], N_MC, np.random.default_rng(3))
    xs = np.quantile(mc, [0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.abs(d.cdf(xs) - np.array([0.05, 0.25, 0.5, 0.75, 0.95])).max() < 5e-3
    assert d.mean == pytest.approx(mc.mean(), rel=2e-3)


def test_burst_decay_matches_monotone_exact():
    """窓が全 Hit を覆う場合、厳密解は 1 - F_S(ψ^{-1}(D)) になる。"""
    hm = _hm()
    win = AccumWindow(hits=list(range(len(hm))), rate=1.2,
                      cap=CapSpec(kind="fixed", value=25_000_000))
    d = build_accum_dist(hm, [win], burst_decay=True)
    S = build_sum_dist(hm)
    for p in (0.1, 0.5, 0.9):
        s_star = float(np.quantile(
            [S.support_lo + (S.support_hi - S.support_lo) * p], 0.5))
        D = float(psi(np.array([s_star]), 25_000_000, 1.2, True)[0])
        exact = 1.0 - float(S.cdf(np.array([s_star]))[0])
        assert d.pass_prob(D) == pytest.approx(exact, abs=2e-5)


# ---------------------------------------------------------------------------
# 複数回撃ち
# ---------------------------------------------------------------------------
def test_multi_window_matches_mc_and_mean_decomposition():
    hm = _hm()
    cap = CapSpec(kind="mixture", mixture=[Uniform(1.0, 1_800_000, 2_400_000)])
    wins = [
        AccumWindow(hits=[0, 1, 2, 3], rate=1.0, cap=cap),
        AccumWindow(hits=[4, 5, 6], rate=1.0, cap=cap),
        AccumWindow(hits=[7, 8, 9], rate=1.2, cap=cap),
    ]
    d = build_accum_dist(hm, wins)
    mc = mc_accum(hm, wins, N_MC, np.random.default_rng(4))
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.abs(d.cdf(np.quantile(mc, qs)) - qs).max() < 5e-3

    # 平均の分解: E[T] = Σ E[X_i] + Σ E[pool_k]
    base = build_sum_dist(hm).mean
    assert d.mean == pytest.approx(
        base + sum(s.pool_mean for s in d.window_stats), rel=1e-6)


def test_pool_mean_matches_survival_integral():
    """E[min(C, αS)] を生存関数積分 (MC 非依存) と突き合わせる。"""
    hm = _hm()
    for cap_spec, cap_arg in (
        (CapSpec(kind="fixed", value=5_000_000), 5_000_000.0),
        (CapSpec(kind="mixture", mixture=[Uniform(1.0, 4e6, 6e6)]), [Uniform(1.0, 4e6, 6e6)]),
    ):
        win = AccumWindow(hits=list(range(7)), rate=1.1, cap=cap_spec)
        d = build_accum_dist(hm, [win])
        S = build_sum_dist([hm[i] for i in win.hits])
        m1, _ = min_moments_survival(S, cap_arg, win.rate)
        assert d.window_stats[0].pool_mean == pytest.approx(m1, rel=1e-4)


def test_even_split_reduces_overflow():
    """同じ攻撃列でも窓への配分を均すほど溢れが小さい (docs/accumulate.md §3.2)。"""
    hm = _hm()
    cap = CapSpec(kind="fixed", value=2_000_000)

    def overflow(groups):
        wins = [AccumWindow(hits=g, rate=1.0, cap=cap) for g in groups]
        return sum(s.overflow_mean for s in build_accum_dist(hm, wins).window_stats)

    even = overflow([[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]])
    skew = overflow([[0], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    one = overflow([list(range(10))])
    assert even < skew < one


# ---------------------------------------------------------------------------
# 入力検証
# ---------------------------------------------------------------------------
def test_overlapping_windows_rejected():
    hm = _hm()
    cap = CapSpec(kind="fixed", value=1e6)
    wins = [AccumWindow(hits=[0, 1], rate=1.0, cap=cap),
            AccumWindow(hits=[1, 2], rate=1.0, cap=cap)]
    with pytest.raises(ValueError, match="重なっています"):
        build_accum_dist(hm, wins)


def test_non_emitting_window_is_plain_damage():
    """上書きでプールが消える窓は、素のダメージだけになる。"""
    hm = _hm()
    win = AccumWindow(hits=[0, 1], rate=1.0, cap=CapSpec(kind="fixed", value=1e6),
                      emit=False)
    d = build_accum_dist(hm, [win])
    s = build_sum_dist(hm)
    xs = np.linspace(s.support_lo, s.support_hi, 101)
    assert np.abs(d.cdf(xs) - s.cdf(xs)).max() < 1e-6
