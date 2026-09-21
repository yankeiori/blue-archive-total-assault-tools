"""蓄積 (チャージ) 型スキルの分布計算の検証と、TL 設計向けの診断レポート。

理論は docs/accumulate.md、実装は app/backend/accumulate.py。
3 つの独立した経路を突き合わせる:

    - グリッド合成 (build_accum_dist)   … セル質量 + FFT 畳み込み
    - 求積リファレンス (pass_prob_quad) … COS の CDF を直接使う 1 次元求積 (K=1 限定)
    - モンテカルロ (mc_accum)           … 正解基準
    - 生存関数積分 (min_moments_survival) … MC 非依存の E[min] 検証 (docs §6)

さらに docs/accumulate.md §3.2 の「窓への配分は均すほど溢れが小さい」を数値で確認する。

実行例:
    uv run python -m experiments.accum_compare
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.accumulate import (  # noqa: E402
    AccumWindow,
    CapSpec,
    build_accum_dist,
    mc_accum,
    min_moments_survival,
    pass_prob_quad,
)
from app.backend.cos import Uniform, build_hit_mixtures, build_sum_dist  # noqa: E402

N_MC = 400_000
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _cards() -> list[dict]:
    """EX 4Hit + NS 3Hit + サブアタッカー 6Hit 相当の適当な攻撃列。"""
    return [
        {"crit_min": 900_000, "crit_max": 1_100_000, "normal_min": 450_000,
         "normal_max": 550_000, "hits": 4, "crit_rate": 60, "evade_rate": 0},
        {"crit_min": 1_400_000, "crit_max": 1_800_000, "normal_min": 700_000,
         "normal_max": 900_000, "hits": 3, "crit_rate": 50, "evade_rate": 5},
        {"crit_min": 300_000, "crit_max": 380_000, "normal_min": 150_000,
         "normal_max": 190_000, "hits": 6, "crit_rate": 40, "evade_rate": 0},
    ]


def _compare(name: str, hm, windows, *, burst_decay=False, quad_win=None, seed=0) -> None:
    dist = build_accum_dist(hm, windows, burst_decay=burst_decay)
    rng = np.random.default_rng(seed)
    mc = mc_accum(hm, windows, N_MC, rng, burst_decay=burst_decay)

    print(f"\n=== {name} ===")
    print(f"  平均   グリッド {dist.mean:14,.0f}   MC {mc.mean():14,.0f}"
          f"   (MC 標準誤差 {mc.std() / np.sqrt(N_MC):,.0f})")
    print(f"  標準偏差 グリッド {np.sqrt(dist.var):12,.0f}   MC {mc.std():14,.0f}")
    xs = np.quantile(mc, QUANTILES)
    err = np.abs(dist.cdf(xs) - np.array(QUANTILES)).max()
    print(f"  分位点での |CDF 差| 最大 = {err:.2e}  (MC のばらつき ~{1 / np.sqrt(N_MC):.2e})")
    if quad_win is not None:
        for p in (0.5, 0.9, 0.99):
            D = float(np.quantile(mc, p))
            g = dist.pass_prob(D)
            q = pass_prob_quad(hm, quad_win, D, burst_decay=burst_decay)
            m = float((mc >= D).mean())
            print(f"  P(T>=D) @ D={D:12,.0f}:  グリッド {g:.6f}  求積 {q:.6f}  MC {m:.6f}"
                  f"   |グリッド-求積| = {abs(g - q):.2e}")
    for i, st in enumerate(dist.window_stats):
        print(f"  窓{i}: 率 {st.rate:.2f}  窓内ダメージ平均 {st.damage_mean:11,.0f}"
              f"  上限平均 {st.cap_mean:10,.0f}  飽和確率 {st.sat_prob:6.3f}"
              f"  プール平均 {st.pool_mean:10,.0f}  期待溢れ {st.overflow_mean:10,.0f}")


def main() -> None:
    hm = build_hit_mixtures(_cards(), 50, 0, "post_decay")
    n = len(hm)
    print(f"Hit 数 = {n}  (EX 0-3 / NS 4-6 / サブ 7-12)")

    # --- 0. 窓なし: 既存の和モデルと一致するか -------------------------------
    d0 = build_accum_dist(hm, [])
    s0 = build_sum_dist(hm)
    xs = np.linspace(s0.support_lo, s0.support_hi, 201)
    print(f"\n=== 0. 窓なし (退化) ===\n  build_sum_dist との |CDF 差| 最大 = "
          f"{np.abs(d0.cdf(xs) - s0.cdf(xs)).max():.2e}")

    # --- 1. 単発・固定上限 ---------------------------------------------------
    w1 = AccumWindow(hits=list(range(0, 7)), rate=1.0,
                     cap=CapSpec(kind="fixed", value=6_000_000))
    _compare("1. 単発・固定上限 6,000,000 / 蓄積率 100%", hm, [w1], quad_win=w1)
    S = build_sum_dist([hm[i] for i in w1.hits])
    m1, m2 = min_moments_survival(S, 6_000_000.0, 1.0)
    d = build_accum_dist(hm, [w1])
    print(f"  E[min] 生存関数積分 {m1:,.2f} / グリッド {d.window_stats[0].pool_mean:,.2f}"
          f"  相対差 {abs(m1 - d.window_stats[0].pool_mean) / m1:.2e}")

    # --- 2. 単発・乱数上限 (独立) -------------------------------------------
    cap_mix = [Uniform(0.5, 5_000_000, 6_000_000), Uniform(0.5, 8_000_000, 9_500_000)]
    w2 = AccumWindow(hits=list(range(0, 7)), rate=1.2,
                     cap=CapSpec(kind="mixture", mixture=cap_mix))
    _compare("2. 単発・乱数上限 (会心で上限が変わる型) / 蓄積率 120%", hm, [w2], quad_win=w2)

    # --- 3. 上限がスキル自身のダメージロール由来 (相関あり) ------------------
    w3in = AccumWindow(hits=[0, 1, 2, 3], rate=1.0,
                       cap=CapSpec(kind="hit", hit=0, coef=4.0))
    _compare("3a. 上限 = Hit0 のダメージ × 4 (その Hit も蓄積に寄与)", hm, [w3in])
    w3out = AccumWindow(hits=[1, 2, 3], rate=1.0,
                        cap=CapSpec(kind="hit", hit=0, coef=4.0))
    _compare("3b. 上限 = Hit0 のダメージ × 4 (その Hit は蓄積に寄与しない)", hm, [w3out])

    # --- 4. 複数回撃ち ------------------------------------------------------
    caps = CapSpec(kind="mixture", mixture=[Uniform(1.0, 2_000_000, 2_600_000)])
    w4 = [
        AccumWindow(hits=[0, 1, 2, 3], rate=1.0, cap=caps),
        AccumWindow(hits=[4, 5, 6], rate=1.0, cap=caps),
        AccumWindow(hits=[7, 8, 9, 10, 11, 12], rate=1.0, cap=caps),
    ]
    _compare("4. 蓄積スキルを 3 回撃つ (非重複窓)", hm, w4)

    # --- 5. 爆発に減衰を通す場合 --------------------------------------------
    w5 = AccumWindow(hits=list(range(0, 13)), rate=1.2,
                     cap=CapSpec(kind="fixed", value=30_000_000))
    _compare("5. 蓄積率 120% / 上限 30,000,000 / 爆発に減衰あり", hm, [w5],
             burst_decay=True, quad_win=w5)

    # --- 6. 配分の最適性 (docs/accumulate.md §3.2) ---------------------------
    print("\n=== 6. 同じ攻撃列を何回の窓に割るか (配分損失) ===")
    cap = CapSpec(kind="fixed", value=2_500_000)
    splits = {
        "1 窓にまとめ打ち": [list(range(0, 13))],
        "2 窓に分割": [list(range(0, 7)), list(range(7, 13))],
        "3 窓に分割 (均等寄り)": [[0, 1, 2, 3], [4, 5, 6], list(range(7, 13))],
        "3 窓に分割 (偏り)": [[0, 1, 2, 3, 4, 5], [6], list(range(7, 13))],
    }
    for label, groups in splits.items():
        wins = [AccumWindow(hits=g, rate=1.0, cap=cap) for g in groups]
        dd = build_accum_dist(hm, wins)
        pool = sum(s.pool_mean for s in dd.window_stats)
        over = sum(s.overflow_mean for s in dd.window_stats)
        print(f"  {label:22s} 窓数 {len(groups)}  総プール {pool:11,.0f}"
              f"  総溢れ {over:11,.0f}  合計ダメージ平均 {dd.mean:12,.0f}")
    print("  → 窓を増やすほど実効上限 ΣC_k が伸び、同じ窓数なら均等配分ほど溢れが小さい")


if __name__ == "__main__":
    main()
