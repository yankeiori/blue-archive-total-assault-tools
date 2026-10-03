"""イロハ(水着) × ハルカ: 合計ダメージの密度を「ハルカの会心数」で色分けする。

設定は experiments/accum_app.py のプリセット「イロハ(水着) × ハルカ」と同じ
(イロハ1 の 3Hit × 120% が上限、ハルカの 3Hit が 150% で蓄積、爆発に減衰なし)。

MC ではなく厳密に分ける。ハルカの 3Hit それぞれを「会心のみ」「非会心のみ」のカードに
差し替えた 2^3 = 8 通りで build_accum_dist を回し、会心数 k ごとに

    P(k) · f_{T | k}(x)

を積み上げる。8 通りの確率つき和が元の分布と一致することも確かめる。
上段は蓄積あり、下段は同じ攻撃列で蓄積なし (比較用)。

実行例:
    uv run python -m experiments.accum_iroha_crit_split
    # → experiments/output/accum_iroha_haruka_crit.png
"""
from __future__ import annotations

import itertools
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.accumulate import build_accum_dist  # noqa: E402
from experiments.accum_app import PRESETS, _build_windows  # noqa: E402
from experiments.edgeworth_animation import build_all_hits  # noqa: E402

OUTPUT = "experiments/output/accum_iroha_haruka_crit.png"
PRESET = "イロハ(水着) × ハルカ (上限 = イロハ1 の 3Hit × 120%)"
GLOBAL_CRIT, GLOBAL_EVADE, DAMAGE_MODE = 50.0, 0.0, "post_decay"

# 会心数 0→3 を明→暗の 1 色相 (序数ランプ。dataviz の validator で確認済み)
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "HackGen Console NF",
                               "DejaVu Sans"]


def _split_cards(cards: list[dict]) -> tuple[list[dict], list[int], float]:
    """ハルカの 2 枚 (2Hit + 1Hit) を 1Hit ずつのカード 3 枚に展開する。

    戻り値: (カード列, ハルカ Hit のカード位置, 会心率)。
    """
    base = [dict(c) for c in cards[:4]]
    haruka = []
    for c in cards[4:]:
        for _ in range(int(c["hits"])):
            haruka.append({**c, "hits": 1})
    rate = float(cards[4]["crit_rate"]) / 100.0
    return base + haruka, list(range(4, 4 + len(haruka))), rate


def main() -> None:
    preset = PRESETS[PRESET]
    cards, h_pos, p_crit = _split_cards(preset["cards"])
    # ハルカは 1Hit ずつに分けたのでカード番号は 5-7、上限はイロハ1 (1-2) のまま
    pool = {**preset["pools"][0], "cards": "5-7"}

    def dist_for(pattern: tuple[int, ...] | None, accum: bool):
        cs = [dict(c) for c in cards]
        if pattern is not None:
            for pos, crit in zip(h_pos, pattern):
                cs[pos]["crit_rate"] = 100.0 if crit else 0.0
        hm, bounds = build_all_hits(cs, GLOBAL_CRIT, GLOBAL_EVADE, DAMAGE_MODE)
        wins = _build_windows([pool], bounds) if accum else []
        return build_accum_dist(hm, wins)

    rows = []
    for accum in (True, False):
        full = dist_for(None, accum)
        parts = {k: [] for k in range(len(h_pos) + 1)}
        for pat in itertools.product((0, 1), repeat=len(h_pos)):
            k = sum(pat)
            prob = p_crit ** k * (1 - p_crit) ** (len(h_pos) - k)
            parts[k].append((prob, dist_for(pat, accum)))
        rows.append((accum, full, parts))

    lo = min(r[1].support_lo for r in rows)
    hi = max(r[1].support_hi for r in rows)
    xs = np.linspace(lo, hi, 2000)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True, facecolor=SURFACE)
    for ax, (accum, full, parts) in zip(axes, rows):
        ax.set_facecolor(SURFACE)
        layers, total_check = [], np.zeros_like(xs)
        print(f"\n=== {'蓄積あり' if accum else '蓄積なし'} ===")
        for k, items in parts.items():
            pk = sum(p for p, _ in items)
            dens = sum(p * d.pdf(xs) for p, d in items)       # P(k) · f_{T|k}
            mean_k = sum(p * d.mean for p, d in items) / pk
            sat_k = (sum(p * d.window_stats[0].sat_prob for p, d in items) / pk
                     if accum else float("nan"))
            layers.append((k, pk, dens, mean_k, sat_k))
            total_check += dens
            print(f"  会心 {k}: P = {pk:.3f}  平均 {mean_k:14,.0f}"
                  + (f"  飽和確率 {sat_k:.3f}" if accum else ""))
        err = np.abs(total_check - full.pdf(xs)).max() / full.pdf(xs).max()
        print(f"  8 通りの和 vs 元の分布: 密度の最大相対差 {err:.1e}")

        scale = 1e6  # 縦軸は 1/百万ダメージ あたりの密度
        cum = np.zeros_like(xs)
        for k, pk, dens, mean_k, sat_k in layers:
            ax.fill_between(xs / 1e6, cum * scale, (cum + dens) * scale, color=COLORS[k],
                            linewidth=0, label=f"会心 {k} 回 ({pk:.1%})")
            # 積み上げの境目に面の色で 2px の隙間
            ax.plot(xs / 1e6, (cum + dens) * scale, color=SURFACE, linewidth=1.2)
            cum = cum + dens
        ax.plot(xs / 1e6, full.pdf(xs) * scale, color=INK, linewidth=1.2, label="合計")

        # 会心数ごとの平均に目印と直接ラベル (選択的に: 平均の位置だけ)
        top = cum.max() * scale
        for k, pk, dens, mean_k, sat_k in layers:
            ax.axvline(mean_k / 1e6, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
            txt = f"会心{k}\n平均 {mean_k / 1e6:.1f}M"
            if accum:
                txt += f"\n飽和 {sat_k:.0%}"
            ax.text(mean_k / 1e6, top * 1.02, txt, ha="center", va="bottom",
                    fontsize=8, color=INK2)

        title = ("蓄積あり (イロハ1 の 3Hit × 120% が上限、ハルカ 3Hit を 150% で蓄積)"
                 if accum else "蓄積なし (同じ攻撃列)")
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=6)
        ax.set_ylabel("密度 (/百万ダメージ)", fontsize=9, color=INK2)
        ax.set_ylim(0, top * 1.22)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    axes[-1].set_xlabel("合計ダメージ (百万)", fontsize=9, color=INK2)
    fig.suptitle("イロハ(水着) × ハルカ: 合計ダメージの密度をハルカの会心数で分解",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=130, facecolor=SURFACE)
    print(f"\n→ {OUTPUT}")


if __name__ == "__main__":
    main()
