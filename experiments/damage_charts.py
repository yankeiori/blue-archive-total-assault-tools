"""ダメージ計算まとめ (Claude Docs「ブルアカ ダメージ計算と編成の考え方」) の図 5 枚。

    01 バフ 1 つあたりの伸び (枠ごとの加算・乗算の差)
    02 防御力ごとの防御デバフの効き (重ねるほど伸びる)
    03 総力戦ボス別の防御デバフの効き (上限 80%)
    04 ボス別の会心ダメージ (%) バフの効き
    05 ボス別の会心ダメージ抵抗デバフの効き (上限 80%)
    06 ドラム缶ガニの鋼鉄ハサミへの防御デバフと防御無視の比較
    07 防御無視・防御貫通があるときの防御デバフの伸び (ヒエロニムス / KAITEN 左腕活性中)

防御倍率は 1666.67 / (1666.67 + 防御)、会心倍率は SchaleDB の
max(会心ダメージ - 抵抗, 100%)。ボスの防御・会心ダメージ抵抗は SchaleDB 日本版
(raids / enemies, 2026-10-22 取得) の最高難易度。01 の値は隣接 repo の
damage_simulator (calc_total_damage_expected) で求めた期待値の伸び。

uv run python -m experiments.damage_charts
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "HackGen35 Console NF"

OUT_DIR = "shots/damage_charts"
K = 1666.67           # 防御倍率の定数
CD = 246.0            # 会心ダメージ初期値 (固定値バフ込み, %)
CAP = 80              # デバフ合計の下限 -80% (StatLimit の RatioMin)

ACCENT = "#2a6fdb"
GREY = "#9a9a9a"


def blues(n: int) -> list:
    """順序のある系列用に、薄い → 濃い青を n 色。"""
    return [plt.cm.Blues(v) for v in np.linspace(0.4, 0.95, n)]


def def_gain(d: float, x: np.ndarray) -> np.ndarray:
    """防御 d の敵に防御デバフ x% を入れたときの倍率 (デバフなし比)。"""
    return (K + d) / (K + d * (1 - x / 100))


def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)


def title(ax, main: str, sub: str | None = None) -> None:
    """左寄せのタイトルと、その下の小さな副題。"""
    ax.set_title(main, loc="left", pad=24 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, va="bottom",
                color="#555", fontsize=9)


def shade_cap(ax, top: float) -> None:
    """80% より先を網掛けし、上限の縦線を引く。"""
    ax.axvspan(CAP, 100, color="#f0f0f0", zorder=0)
    ax.axvline(CAP, color=GREY, ls="--", lw=1.2)
    ax.annotate("上限 80%", xy=(CAP, top), xytext=(5, -4), textcoords="offset points",
                ha="left", va="top", color="#555")


def label_ends(ax, items: list, x_end: float, fmt, min_gap: float) -> None:
    """系列の右端ラベルを、重ならないよう縦にずらして置く。"""
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for name, y, color in items:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(fmt(name), xy=(x_end, y), xytext=(6, 0),
                    textcoords="offset points", va="center", color=color)


def save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"保存: {path}")


# 01: バフ 1 つあたりの伸び -------------------------------------------------
def chart_buff_gain() -> None:
    rows = [  # (バフ, 期待値の伸び %)  ※装備で攻撃力 +50% 済みの生徒
        ("与ダメージ増加 +50%", 50.0),
        ("被ダメージ増加 +50%", 50.0),
        ("会心ダメージ +50%", 47.6),
        ("攻撃力 +50%", 33.3),
        ("属性特効 +50%", 23.8),
        ("会心値 +100%", 7.9),
        ("防御力 −50%(防御100の敵)", 2.9),
    ]
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [ACCENT if n.startswith("攻撃力") else "#c9c9c9" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, vals, color=colors)
    ax.axvline(50, color=GREY, ls="--", lw=1.2)
    ax.text(50, len(rows) - 0.4, "バフ量そのまま(+50%)", ha="center", color="#555")
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"+{v:.1f}%", va="center",
                color=ACCENT if colors[i] == ACCENT else "#333")
    ax.set_xlim(0, 58)
    title(ax, "攻撃力 +50% の伸びは +33% どまり、別枠のバフに届かない")
    style(ax, "ダメージ期待値の伸び (%)", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e5e5e5")
    save(fig, "01_buff_gain_per_type.png")


# 02: 防御力ごとの防御デバフ ---------------------------------------------------
def chart_def_by_defense() -> None:
    x = np.linspace(0, CAP, 161)
    defs = [100, 500, 1000, 2000, 5000]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ends = []
    for d, c in zip(defs, blues(len(defs))):
        y = def_gain(d, x)
        ax.plot(x, y, color=c, lw=2.4)
        ends.append((f"防御 {d:,}  ×{y[-1]:.2f}", y[-1], "#222"))
    atk = (1.5 + x / 100) / 1.5
    ax.plot(x, atk, color=GREY, ls="--", lw=1.6)
    ends.append(("攻撃力 +同じ%", atk[-1], "#555"))
    label_ends(ax, ends, CAP, lambda s: s, min_gap=0.08)
    ax.set_xlim(0, CAP)
    ax.set_ylim(1, 2.6)
    title(ax, "防御デバフは重ねるほど伸びるが、効きの大きさはボスの防御力で決まる",
          "点線は同じ%だけ攻撃力を上げた場合(装備 +50% の上)")
    style(ax, "防御デバフ合計 (%)", "ダメージ倍率(デバフなし = ×1.0)")
    save(fig, "02_def_debuff_by_defense.png")


# 03: ボス別の防御デバフ -----------------------------------------------------
def chart_def_by_boss() -> None:
    bosses = [  # 総力戦 Lunatic。ヒエロニムスは不浄なる光の -1,500 込み (素は 8,000)
        ("ホド", 100), ("グレゴリオ", 1000), ("ゲブラ", 2500),
        ("ヒエロニムス", 6500), ("クロカゲ", 8000),
    ]
    xa = np.linspace(0, CAP, 161)
    xb = np.linspace(CAP, 100, 41)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    shade_cap(ax, 6.0)
    for (name, d), c in zip(bosses, blues(len(bosses))):
        ax.plot(xa, def_gain(d, xa), color=c, lw=2.4)
        ax.plot(xb, def_gain(d, xb), color=c, lw=1.4, ls="--")
        y80 = float(def_gain(d, np.array(CAP)))
        ax.plot(CAP, y80, "o", color=c, ms=6)
        ax.annotate(f"{name}({d:,})  80% で ×{y80:.2f}",
                    xy=(100, float(def_gain(d, np.array(100)))), xytext=(6, 0),
                    textcoords="offset points", va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 6.0)
    title(ax, "防御デバフを上限 80% まで積むと、クロカゲは ×2.96、ホドは ×1.05")
    ax.text(0, -0.17, "ヒエロニムスは不浄なる光(聖遺物)の −1,500 を適用済み(素は 8,000)。"
            "括弧内は防御力。80% より先(点線)は上限で届かない",
            transform=ax.transAxes, color="#555", fontsize=9)
    style(ax, "防御デバフ合計 (%)", "ダメージ倍率(デバフなし = ×1.0)")
    save(fig, "03_def_debuff_by_boss.png")


# 04 / 05: 会心ダメージ ------------------------------------------------------
CRIT_BOSSES = [("ホバークラフト", 10), ("ホド", 50), ("ビナー", 80)]


def crit_mult(cd: np.ndarray, cdr: np.ndarray) -> np.ndarray:
    return np.maximum(cd - cdr, 100) / 100


def chart_crit_buff() -> None:
    x = np.linspace(0, 100, 201)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (name, cdr), c in zip(CRIT_BOSSES, blues(len(CRIT_BOSSES))):
        y = crit_mult(CD * (1 + x / 100), cdr) / crit_mult(np.array(CD), cdr)
        ax.plot(x, y, color=c, lw=2.4)
        ax.annotate(f"{name}({cdr}%)  ×{y[-1]:.2f}", xy=(100, y[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 3)
    title(ax, "会心ダメージ +100% は抵抗の高いボスほど伸びる",
          f"会心時のダメージ倍率。会心ダメージ初期値 {CD:.0f}%、括弧内はボスの会心ダメージ抵抗")
    style(ax, "会心ダメージ (%) バフ合計", "会心時の倍率(バフなし = ×1.0)")
    save(fig, "04_crit_dmg_buff_by_boss.png")


def chart_crit_resist_debuff() -> None:
    xa = np.linspace(0, CAP, 161)
    xb = np.linspace(CAP, 100, 41)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    shade_cap(ax, 1.6)
    for (name, cdr), c in zip(CRIT_BOSSES, blues(len(CRIT_BOSSES))):
        base = crit_mult(np.array(CD), cdr)
        f = lambda x: crit_mult(np.array(CD), cdr * (1 - x / 100)) / base
        ax.plot(xa, f(xa), color=c, lw=2.4)
        ax.plot(xb, f(xb), color=c, lw=1.4, ls="--")
        y80 = float(f(np.array(CAP)))
        ax.plot(CAP, y80, "o", color=c, ms=6)
        ax.annotate(f"{name}({cdr}%)  80% で ×{y80:.2f}",
                    xy=(100, float(f(np.array(100)))), xytext=(6, 0),
                    textcoords="offset points", va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 1.6)
    title(ax, "抵抗デバフは上限 80% でビナー ×1.39、ホバークラフトは ×1.03")
    ax.text(0, -0.17, f"会心時のダメージ倍率。会心ダメージ {CD:.0f}%。"
            "80% より先(点線)は上限で届かない",
            transform=ax.transAxes, color="#555", fontsize=9)
    style(ax, "会心ダメージ抵抗デバフ合計 (%)", "会心時の倍率(デバフなし = ×1.0)")
    save(fig, "05_crit_dmg_resist_debuff_by_boss.png")


# 06: ドラム缶ガニの鋼鉄ハサミ — 防御デバフと防御無視 ---------------------------
def pincer_gain(debuff: np.ndarray, ignore: np.ndarray, base: float = 2000,
                coef: float = 4.0) -> np.ndarray:
    """鋼鉄ハサミ (防御 2,000、自身に防御 +400%) への倍率 (どちらもなし比)。

    防御デバフは +400% と同じ Coefficient 枠で加算され (下限 -80%)、
    防御無視はその結果の防御に乗算で効く (SchaleDB の計算機と同じ)。
    """
    d = base * np.maximum(1 + coef - debuff / 100, 0.2)
    eff = d * (1 - ignore / 100)
    return (K + base * (1 + coef)) / (K + eff)


def chart_pincer() -> None:
    xa = np.linspace(0, CAP, 161)
    xi = np.linspace(0, 100, 201)
    zero = np.zeros_like
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axvline(CAP, color=GREY, ls="--", lw=1.2)
    ax.annotate("防御デバフの上限 80%", xy=(CAP, 7.0), xytext=(-5, -4),
                textcoords="offset points", ha="right", va="top", color="#555")

    ign = pincer_gain(zero(xi), xi)
    ax.plot(xi, ign, color=ACCENT, lw=2.6)
    deb = pincer_gain(xa, zero(xa))
    ax.plot(xa, deb, color=plt.cm.Blues(0.45), lw=2.6)
    normal = (K + 2000) / (K + 2000 * (1 - xi / 100))
    ax.plot(xi, normal, color=GREY, ls="--", lw=1.6)

    ax.annotate("鋼鉄ハサミ:防御無視",
                xy=(100, ign[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=ACCENT)
    ax.annotate(f"鋼鉄ハサミ:防御デバフ  上限で ×{deb[-1]:.2f}",
                xy=(CAP, deb[-1]), xytext=(6, 4), textcoords="offset points",
                va="bottom", color="#333")
    ax.annotate("参考:防御 2,000 の部位(デバフ = 無視)", xy=(100, normal[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", color="#555")

    # 防御無視を持つ爆発属性 (ハサミの弱点) の 2 人
    for x, who, off, ha in [(60, "トキ 60%", (-8, 6), "right"),
                            (68, "ハルカ(ドレス) 68%", (-8, 6), "right")]:
        y = float(pincer_gain(0, np.array(x)))
        ax.plot(x, y, "o", color=ACCENT, ms=6)
        ax.annotate(f"{who}  ×{y:.2f}", xy=(x, y), xytext=off, textcoords="offset points",
                    ha=ha, va="top" if ha == "left" else "baseline", color=ACCENT,
                    fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_ylim(1, 7.2)
    title(ax, "鋼鉄ハサミには防御無視が効き(ハルカ(ドレス) 68% で ×2.40)、防御デバフは上限でも ×1.16",
          "ドラム缶ガニ(Torment / Lunatic)フェーズ2。防御 2,000 × (1 + 400%) = 10,000。"
          "どちらもなし = ×1.0")
    ax.text(0, -0.17, "防御デバフはハサミ自身の +400% と同じ枠で加算されるため、"
            "−80% でも防御は 10,000 → 8,400 にしかならない",
            transform=ax.transAxes, color="#555", fontsize=9)
    style(ax, "防御デバフ合計 / 防御無視率 (%)", "ダメージ倍率")
    save(fig, "06_steel_pincer_debuff_vs_ignore.png")


# 07: 防御無視 (トキ)・防御貫通 (ヒナ(ドレス)・ノノミ(水着)) と防御デバフの伸び -------
PEN_CASES = [  # (名前, 防御貫通, 防御無視率, 色, 線種)
    ("素の状態", 0, 0.0, plt.cm.Blues(0.45), "-"),
    ("ヒナ(ドレス):防御貫通 357", 357, 0.0, "#8a8a8a", "--"),
    ("ノノミ(水着):防御貫通 684", 684, 0.0, "#333333", "-."),
    ("トキ:防御無視 60%", 0, 0.6, ACCENT, "-"),
]


def debuff_growth_panel(ax, base: float, heading: str, sub: str) -> None:
    """防御 base の敵で、各条件の「デバフ 0 = ×1.0」とした防御デバフの伸び。

    実効防御 = (デバフ後の防御 - 防御貫通) × (1 - 防御無視率)。
    """
    x = np.linspace(0, CAP, 321)
    d = np.round(base * (1 - x / 100))
    ends = []
    for name, pen, ign, c, ls in PEN_CASES:
        y = (K + base) / (K + np.maximum(d - pen, 0) * (1 - ign))
        y = y / y[0]
        ax.plot(x, y, color=c, lw=2.6, ls=ls)
        ax.plot(CAP, y[-1], "o", color=c, ms=6)
        ends.append((f"{name}  ×{y[-1]:.2f}", y[-1], c))
        if pen and base * (1 - CAP / 100) < pen:
            # 実効防御が 0 に届いた点。そこから先のデバフは伸びない
            x0 = (1 - pen / base) * 100
            y0 = float(y[np.searchsorted(x, x0)])
            ax.plot(x0, y0, "D", color=c, ms=6)
            ax.annotate(f"防御 0 に到達({x0:.0f}%)", xy=(x0, y0), xytext=(-8, 8),
                        textcoords="offset points", ha="right", color=c, fontsize=9)
    ymax = max(e[1] for e in ends)
    label_ends(ax, ends, CAP, lambda t: t, min_gap=(ymax - 1) * 0.07)
    ax.set_xlim(0, CAP)
    ax.set_ylim(1, 1 + (ymax - 1) * 1.08)
    title(ax, heading, sub)
    style(ax, "防御デバフ合計 (%、上限 80%)", "ダメージ倍率(デバフ 0 = ×1.0)")


def chart_ignore_pen_growth() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17, 6.6))
    # 右端ラベルが軸の外にはみ出すので、tight_layout ではなく手で余白を取る
    fig.subplots_adjust(left=0.05, right=0.86, top=0.80, bottom=0.13, wspace=0.6)
    debuff_growth_panel(ax_l, 8000 - 1500,
                        "ヒエロニムス:ノノミ(水着) ×3.28、ヒナ(ドレス) ×2.99、トキ ×1.95",
                        "Lunatic、不浄なる光込みで防御 6,500")
    debuff_growth_panel(ax_r, 420 + 1000,
                        "KAITEN(左腕活性中):ヒナ(ドレス) ×1.64、ノノミ(水着) ×1.44、トキ ×1.26",
                        "Lunatic、左腕(グリーン)の防御 +1,000 込みで防御 1,420")
    fig.suptitle("防御無視・防御貫通があるときの防御デバフの伸び(各条件のデバフ 0 = ×1.0)",
                 x=0.01, y=0.98, ha="left", fontsize=14)
    fig.text(0.01, 0.01, "防御無視は残りの防御を割合で削るため、併用すると防御デバフの伸びが小さくなる。"
             "防御貫通は固定量を引くため伸びが大きくなるが、実効防御が 0 に届くとそれ以上は伸びない。"
             "トキの防御無視はサブスキルのダメージのみ、ノノミ(水着)の 684 は固有武器★2 以上のパッシブ",
             color="#555", fontsize=9)
    path = os.path.join(OUT_DIR, "07_debuff_growth_with_ignore_penetration.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"保存: {path}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_buff_gain()
    chart_def_by_defense()
    chart_def_by_boss()
    chart_crit_buff()
    chart_crit_resist_debuff()
    chart_pincer()
    chart_ignore_pen_growth()
