"""COS 法 + 原子分離 (DP) によるダメージ分布の準厳密計算 (参照実装)。

このモジュールはクライアント JS (`assets/cos.js`) の **正解基準** であり、
`experiments/cos_compare.py`(和モデル)・`experiments/product_cos.py`(積モデル =
HP依存)・`experiments/edgeworth_animation.py`(カード→一様混合の減衰分割)から
matplotlib 依存を除いて抽出・整理したもの。理論は docs/saddlepoint.md・
docs/discrete.md・docs/product.md を参照。

2 つのモデルを提供する:

- **和モデル** (HP非依存): 各 Hit のダメージは一様分布の混合 + miss 点質量。
  合計 S_n = Σ X_i の特性関数は積 φ_S = Π φ_hit。これを COS 反転して CDF/PDF を得る。
  純原子部 (全 Hit が点質量成分を取る組合せ) は DP で厳密分離し Gibbs を回避する。

- **積モデル** (HP依存, ミカ型): 累積ダメージ D = H̃_1(1 - Π Y_n), Y_n = 1 - β x_n。
  S = ln P = Σ ln Y_n の特性関数を COS 反転し、D = H̃_1(1 - e^S) で写す。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.backend.simulation import (
    DAMAGE_CAP,
    DAMAGE_FUNC,
    raw_damage_bounds,
    stability_min_ratio,
)

# 再エクスポート (JS 移植の検証で cos.* から参照するため)
__all__ = ["DAMAGE_CAP", "raw_damage_bounds", "stability_min_ratio"]

# COS 法の設定 (cos_compare.py / product_cos.py と同値)
_COS_L = 12.0                  # キュムラント基準窓の幅 (σ 単位)
_COS_TERMS_PER_SIGMA = 256
_COS_N_MIN = 2048
_COS_N_MAX = 1 << 18
_ATOM_MERGE_TOL = 1e-6
_ATOM_MAX = 5_000_000


# =============================================================================
# 一様分布混合の表現
# =============================================================================

@dataclass
class Uniform:
    """1 つの一様分布成分。退化 (lo == hi) は 1 点分布 (点質量) を表す。"""
    weight: float
    lo: float
    hi: float

    @property
    def center(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)


# =============================================================================
# カード → 1Hit の一様混合 (減衰関数の区分線形分割)
#   edgeworth_animation.split_uniform_through_decay / expand_card_to_hit_mixture 由来。
# =============================================================================

def _decay_scalar(x: float) -> float:
    for (x_lo, x_hi), (a, b) in DAMAGE_FUNC:
        if x_lo <= x < x_hi:
            return a * x + b
    return DAMAGE_FUNC[-1][1][1]


def split_uniform_through_decay(a: float, b: float) -> list[tuple[float, float, float]]:
    """生ダメージ上の U(a, b) を decay を通して post-decay scale の混合へ分解する。
    各 decay セグメントに該当する部分が post-decay 域で 1 つの (退化しうる) 一様になる。
    Returns: list of (weight, post_lo, post_hi), 重み合計 1.0。"""
    if b <= a:
        y = _decay_scalar(a)
        return [(1.0, y, y)]
    parts: list[tuple[float, float, float]] = []
    total = b - a
    for (x_lo, x_hi), (a_d, b_d) in DAMAGE_FUNC:
        lo = max(a, x_lo)
        hi = min(b, x_hi)
        if hi <= lo:
            continue
        w = (hi - lo) / total
        y_lo = a_d * lo + b_d
        y_hi = a_d * hi + b_d
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        parts.append((w, y_lo, y_hi))
    return parts or [(1.0, a, b)]


def expand_card_to_hit_mixture(
    card: dict, global_crit: float, global_evade: float, damage_mode: str,
) -> list[Uniform]:
    """1 枚のカードから、各 Hit (同分布) に対応する 1 つの一様混合を作る。
    会心/非会心の一様を減衰分割し、miss は点質量 (lo=hi=0) として加える。"""
    crit_min = float(card.get("crit_min") or 0)
    crit_max = float(card.get("crit_max") or 0)
    normal_min = float(card.get("normal_min") or 0)
    normal_max = float(card.get("normal_max") or 0)
    cr_raw = card.get("crit_rate")
    er_raw = card.get("evade_rate")
    cr = float(cr_raw if cr_raw is not None else (global_crit or 0)) / 100.0
    er = float(er_raw if er_raw is not None else (global_evade or 0)) / 100.0

    stab = card.get("stability")
    stab = None if (stab is None or stab == "") else float(stab)
    raw_crit_lo, raw_crit_hi = raw_damage_bounds(crit_min, crit_max, stab, damage_mode)
    raw_norm_lo, raw_norm_hi = raw_damage_bounds(normal_min, normal_max, stab, damage_mode)

    crit_parts = split_uniform_through_decay(raw_crit_lo, raw_crit_hi)
    norm_parts = split_uniform_through_decay(raw_norm_lo, raw_norm_hi)

    mix: list[Uniform] = []
    if (1 - er) * cr > 0:
        for w, lo, hi in crit_parts:
            mix.append(Uniform((1 - er) * cr * w, lo, hi))
    if (1 - er) * (1 - cr) > 0:
        for w, lo, hi in norm_parts:
            mix.append(Uniform((1 - er) * (1 - cr) * w, lo, hi))
    if er > 0:
        mix.append(Uniform(er, 0.0, 0.0))

    total = sum(u.weight for u in mix)
    if total > 0:
        for u in mix:
            u.weight /= total
    return mix


def build_hit_mixtures(
    cards: list[dict], global_crit: float, global_evade: float, damage_mode: str,
) -> list[list[Uniform]]:
    """全カードを (各 Hit の一様混合のリスト, 長さ = 総 Hit 数) に展開する。"""
    hit_mixtures: list[list[Uniform]] = []
    for c in cards:
        mix = expand_card_to_hit_mixture(c, global_crit, global_evade, damage_mode)
        hits = int(c.get("hits") or 1)
        hit_mixtures.extend([mix] * hits)
    return hit_mixtures


# =============================================================================
# 和モデル: キュムラント・台
# =============================================================================

def hit_moments(mix: list[Uniform]) -> tuple[float, float, float]:
    """1Hit 混合の (平均, 分散 M2, 4次キュムラント κ4)。COS 区間幅の決定に使う。"""
    if not mix:
        return 0.0, 0.0, 0.0
    w = np.array([u.weight for u in mix])
    c = np.array([u.center for u in mix])
    h = np.array([u.half_width for u in mix])
    mu = float(np.sum(w * c))
    d = c - mu
    h2 = h ** 2
    M2 = float(np.sum(w * (h2 / 3.0 + d ** 2)))
    M4 = float(np.sum(w * (h2 ** 2 / 5.0 + 2.0 * d ** 2 * h2 + d ** 4)))
    kappa4 = M4 - 3.0 * M2 ** 2
    return mu, M2, kappa4


def support_bounds(hit_mixtures: list[list[Uniform]]) -> tuple[float, float]:
    """S_n の取りうる値の下限・上限 (各 Hit の min/max を加算)。"""
    lo = hi = 0.0
    for mix in hit_mixtures:
        lo += min(u.lo for u in mix)
        hi += max(u.hi for u in mix)
    return lo, hi


def _cos_interval(hit_mixtures: list[list[Uniform]]) -> tuple[float, float, int]:
    """COS 積分区間 [a, b] と項数 n_terms。キュムラント基準窓をサポートでクリップ。"""
    mean = var = k4 = 0.0
    for mix in hit_mixtures:
        m, v, c4 = hit_moments(mix)
        mean += m
        var += v
        k4 += c4
    std = math.sqrt(var) if var > 0 else 0.0
    s_lo, s_hi = support_bounds(hit_mixtures)
    if std > 0:
        half = _COS_L * math.sqrt(var + math.sqrt(abs(k4)))
        a = max(s_lo, mean - half)
        b = min(s_hi, mean + half)
        width_sigma = (b - a) / std
    else:
        a, b = s_lo, s_hi
        width_sigma = 1.0
    n_terms = int(math.ceil(_COS_TERMS_PER_SIGMA * width_sigma))
    n_terms = max(_COS_N_MIN, min(_COS_N_MAX, n_terms))
    return a, b, n_terms


# =============================================================================
# 和モデル: 特性関数と原子分離 (DP)
# =============================================================================

def mixture_cf(mix: list[Uniform], u: np.ndarray) -> np.ndarray:
    """1Hit 混合の特性関数 φ(u) = Σ_j w_j e^{i u c_j} sinc(u h_j)。"""
    phi = np.zeros_like(u, dtype=complex)
    for c in mix:
        if c.half_width == 0.0:
            phi += c.weight * np.exp(1j * u * c.center)
        else:
            phi += c.weight * np.exp(1j * u * c.center) * np.sinc((u * c.half_width) / np.pi)
    return phi


def sum_cf(hit_mixtures: list[list[Uniform]], u: np.ndarray) -> np.ndarray:
    """合計 S_n の特性関数 φ_S(u) = Π_hit φ_hit(u)。"""
    phi = np.ones_like(u, dtype=complex)
    for mix in hit_mixtures:
        phi = phi * mixture_cf(mix, u)
    return phi


def _atom_points(mix: list[Uniform]) -> list[tuple[float, float]] | None:
    """1Hit の点質量成分 [(weight, center), ...]。無ければ None。"""
    pts = [(c.weight, c.center) for c in mix if c.half_width == 0.0]
    return pts or None


def atom_cf(hit_mixtures: list[list[Uniform]], u: np.ndarray) -> np.ndarray:
    """純原子部の特性関数 φ_S^atom(u) = Π_hit (Σ_pts w e^{iuc})。
    どれか 1 Hit でも点質量成分を持たなければ純原子部は無く 0。"""
    phi = np.ones_like(u, dtype=complex)
    for mix in hit_mixtures:
        pts = _atom_points(mix)
        if pts is None:
            return np.zeros_like(u, dtype=complex)
        g = np.zeros_like(u, dtype=complex)
        for w, c in pts:
            g = g + w * np.exp(1j * u * c)
        phi = phi * g
    return phi


def atom_part_distribution(hit_mixtures: list[list[Uniform]]) -> tuple[np.ndarray, np.ndarray]:
    """純原子部の (値, 確率) を疎な逐次畳み込みで返す。純原子部が無い、または
    列挙数が _ATOM_MAX を超える (準連続) なら空配列 (→ 通常 COS)。"""
    per_hit: list[list[tuple[float, float]]] = []
    for mix in hit_mixtures:
        pts = _atom_points(mix)
        if pts is None:
            return np.empty(0), np.empty(0)
        per_hit.append(pts)

    vals = np.array([0.0])
    probs = np.array([1.0])
    for pts in per_hit:
        pw = np.array([w for w, _ in pts])
        pc = np.array([c for _, c in pts])
        if vals.size * pc.size > _ATOM_MAX:
            return np.empty(0), np.empty(0)
        v = (vals[:, None] + pc[None, :]).ravel()
        p = (probs[:, None] * pw[None, :]).ravel()
        order = np.argsort(v)
        v, p = v[order], p[order]
        keep = np.concatenate([[True], np.diff(v) > _ATOM_MERGE_TOL])
        grp = np.cumsum(keep) - 1
        vals = v[keep]
        probs = np.zeros(vals.size)
        np.add.at(probs, grp, p)
    return vals, probs


def _atom_step_cdf(av: np.ndarray, ap: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """純原子部の厳密な階段 CDF Σ_{a_k ≤ x} p_k を xs 上で返す。"""
    order = np.argsort(av, kind="mergesort")
    cum = np.concatenate([[0.0], np.cumsum(ap[order])])
    idx = np.searchsorted(av[order], np.asarray(xs, dtype=float), side="right")
    return cum[idx]


# =============================================================================
# 和モデル: COS 反転 (CDF / PDF) と公開 API
# =============================================================================

@dataclass
class SumDist:
    """和モデルの COS 反転結果。CDF/PDF を任意の x 上で評価できる。"""
    a: float
    b: float
    u: np.ndarray
    Fk: np.ndarray
    av: np.ndarray | None
    ap: np.ndarray | None
    support_lo: float
    support_hi: float
    mean: float
    var: float

    def cdf(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        dx = xs - self.a
        cdf = 0.5 * self.Fk[0] * dx
        if self.u.size > 1:
            arg = np.outer(dx, self.u[1:])
            cdf += (self.Fk[1:][None, :] * np.sin(arg) / self.u[1:][None, :]).sum(axis=1)
        if self.av is not None:
            cdf = cdf + _atom_step_cdf(self.av, self.ap, xs)
        # サポート外を 0 / 1 にクランプ
        cdf = np.where(xs < self.support_lo, 0.0, cdf)
        cdf = np.where(xs >= self.support_hi, 1.0, cdf)
        return np.clip(cdf, 0.0, 1.0)

    def pdf(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        Fk = self.Fk.copy()
        Fk[0] *= 0.5
        arg = np.outer(xs - self.a, self.u)
        f = (Fk[None, :] * np.cos(arg)).sum(axis=1)
        f = np.where((xs < self.support_lo) | (xs > self.support_hi), 0.0, f)
        return np.clip(f, 0.0, None)


def build_sum_dist(hit_mixtures: list[list[Uniform]], *, dp: bool = True) -> SumDist:
    """和モデルの COS 係数を構築して SumDist を返す。"""
    a, b, n_terms = _cos_interval(hit_mixtures)
    L = b - a
    u = np.arange(n_terms) * np.pi / L
    phi = sum_cf(hit_mixtures, u)
    av = ap = None
    if dp:
        av, ap = atom_part_distribution(hit_mixtures)
        if av.size == 0:
            av = ap = None
        else:
            phi = phi - atom_cf(hit_mixtures, u)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))

    mean = var = 0.0
    for mix in hit_mixtures:
        m, v, _ = hit_moments(mix)
        mean += m
        var += v
    s_lo, s_hi = support_bounds(hit_mixtures)
    return SumDist(a, b, u, Fk, av, ap, s_lo, s_hi, mean, var)


# =============================================================================
# 積モデル (HP依存): y = 1 - β x の混合と ln Y の特性関数
#   product_cos.py 由来。
# =============================================================================

@dataclass
class HPParams:
    H: float       # 敵の最大 HP (倍率の基準)
    H1: float      # 攻撃開始時の敵 HP
    R0: float      # HP=0 時の倍率
    R1: float      # HP満タン時の倍率

    @property
    def dR(self) -> float:
        return self.R1 - self.R0

    @property
    def beta(self) -> float:
        return self.dR / self.H

    @property
    def Htil(self) -> float:
        return self.H1 + self.R0 / self.beta


def y_mixture(base: list[Uniform], beta: float) -> list[Uniform]:
    """基礎ダメージ x の一様混合を Y = 1 - β x の混合へ写す (アフィン、向き反転)。"""
    out: list[Uniform] = []
    for u in base:
        y1 = 1.0 - beta * u.lo
        y2 = 1.0 - beta * u.hi
        lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)
        if not (lo > 0.0):
            raise ValueError(f"β x ≥ 1 となり Y≤0 (x_hi={u.hi}, β={beta})。HPが負になる設定。")
        out.append(Uniform(u.weight, lo, hi))
    return out


def logmix_cf(ymix: list[Uniform], u: np.ndarray) -> np.ndarray:
    """1Hit の ln Y の特性関数 φ(u) = Σ_j w_j (対数一様 = 切断指数)。"""
    phi = np.zeros_like(u, dtype=complex)
    for c in ymix:
        w, p, q = c.weight, c.lo, c.hi
        if q <= p:
            phi += w * np.exp(1j * u * math.log(p))
        else:
            lp, lq = math.log(p), math.log(q)
            num = q * np.exp(1j * u * lq) - p * np.exp(1j * u * lp)
            phi += w * num / ((q - p) * (1.0 + 1j * u))
    return phi


def cf_S_hits(ymix_per_hit: list[list[Uniform]], u: np.ndarray) -> np.ndarray:
    """S = Σ ln Y_n の特性関数 φ_S = Π_n φ_{ln Y_n}(u)。"""
    phi = np.ones_like(u, dtype=complex)
    for ymix in ymix_per_hit:
        phi = phi * logmix_cf(ymix, u)
    return phi


def support_bounds_hits(ymix_per_hit: list[list[Uniform]]) -> tuple[float, float]:
    """S = Σ ln Y の厳密な台 [A, B]。"""
    lo = sum(min(math.log(c.lo) for c in ymix) for ymix in ymix_per_hit)
    hi = sum(max(math.log(c.hi) for c in ymix) for ymix in ymix_per_hit)
    return lo, hi


def logmix_moments(ymix: list[Uniform]) -> tuple[float, float]:
    """1Hit の ln Y の (平均, 分散)。"""
    mean = ex2 = 0.0
    for c in ymix:
        w, p, q = c.weight, c.lo, c.hi
        if q <= p:
            m1 = math.log(p)
            e2 = m1 * m1
        else:
            lp, lq = math.log(p), math.log(q)
            m1 = (q * lq - p * lp) / (q - p) - 1.0
            anti = lambda y, ly: y * (ly * ly - 2.0 * ly + 2.0)
            e2 = (anti(q, lq) - anti(p, lp)) / (q - p)
        mean += w * m1
        ex2 += w * e2
    return mean, ex2 - mean * mean


def moments_hits(ymix_per_hit: list[list[Uniform]]) -> tuple[float, float]:
    mean = var = 0.0
    for ymix in ymix_per_hit:
        m, v = logmix_moments(ymix)
        mean += m
        var += v
    return mean, var


def y_raw_moment(ymix: list[Uniform], k: int) -> float:
    """1Hit の Y の k 次積率 E[Y^k] = Σ_j w_j E[Y_j^k]。"""
    m = 0.0
    for c in ymix:
        w, p, q = c.weight, c.lo, c.hi
        if q <= p:
            m += w * p ** k
        else:
            m += w * (q ** (k + 1) - p ** (k + 1)) / ((k + 1) * (q - p))
    return m


def damage_moments(ymix_per_hit: list[list[Uniform]], Htil: float) -> tuple[float, float]:
    """累積ダメージ D = H̃_1(1 - Π Y_n) の厳密な (平均, 分散) (MC 非依存の検証基準)。"""
    EP = EP2 = 1.0
    for ymix in ymix_per_hit:
        EP *= y_raw_moment(ymix, 1)
        EP2 *= y_raw_moment(ymix, 2)
    mean = Htil * (1.0 - EP)
    var = Htil * Htil * (EP2 - EP * EP)
    return mean, var


# --- 積モデルの純原子部 (ln スケール) ---

def _atom_points_ln(ymix: list[Uniform]) -> list[tuple[float, float]] | None:
    pts = [(c.weight, math.log(c.lo)) for c in ymix if c.hi <= c.lo]
    return pts or None


def atom_cf_hits(ymix_per_hit: list[list[Uniform]], u: np.ndarray) -> np.ndarray:
    phi = np.ones_like(u, dtype=complex)
    for ymix in ymix_per_hit:
        pts = _atom_points_ln(ymix)
        if pts is None:
            return np.zeros_like(u, dtype=complex)
        g = np.zeros_like(u, dtype=complex)
        for w, s in pts:
            g = g + w * np.exp(1j * u * s)
        phi = phi * g
    return phi


def atom_part_distribution_ln(ymix_per_hit: list[list[Uniform]]) -> tuple[np.ndarray, np.ndarray]:
    per_hit: list[list[tuple[float, float]]] = []
    for ymix in ymix_per_hit:
        pts = _atom_points_ln(ymix)
        if pts is None:
            return np.empty(0), np.empty(0)
        per_hit.append(pts)

    vals = np.array([0.0])
    probs = np.array([1.0])
    for pts in per_hit:
        pw = np.array([w for w, _ in pts])
        ps = np.array([s for _, s in pts])
        if vals.size * ps.size > _ATOM_MAX:
            return np.empty(0), np.empty(0)
        v = (vals[:, None] + ps[None, :]).ravel()
        p = (probs[:, None] * pw[None, :]).ravel()
        order = np.argsort(v)
        v, p = v[order], p[order]
        keep = np.concatenate([[True], np.diff(v) > 1e-9])
        grp = np.cumsum(keep) - 1
        vals = v[keep]
        probs = np.zeros(vals.size)
        np.add.at(probs, grp, p)
    return vals, probs


@dataclass
class ProductDist:
    """積モデルの COS 反転結果。D = H̃_1(1 - e^S) の CDF/PDF を評価できる。"""
    a: float
    b: float
    u: np.ndarray
    Fk: np.ndarray
    av: np.ndarray | None
    ap: np.ndarray | None
    Htil: float
    d_max: float          # 高ダメージ端 (H̃>0: S=a、H̃<0: S=b で実現)

    def _cdf_S(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        ds = s - self.a
        cdf = 0.5 * self.Fk[0] * ds
        if self.u.size > 1:
            arg = np.outer(ds, self.u[1:])
            cdf += (self.Fk[1:][None, :] * np.sin(arg) / self.u[1:][None, :]).sum(axis=1)
        if self.av is not None:
            cdf = cdf + _atom_step_cdf(self.av, self.ap, s)
        return np.clip(cdf, 0.0, 1.0)

    def _pdf_S(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        Fk = self.Fk.copy()
        Fk[0] *= 0.5
        arg = np.outer(s - self.a, self.u)
        return (Fk[None, :] * np.cos(arg)).sum(axis=1)

    def cdf(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        arg = 1.0 - d / self.Htil
        s_of_d = np.full_like(d, np.nan)
        pos = (d >= 0.0) & (arg > 0.0)
        s_of_d[pos] = np.log1p(-d[pos] / self.Htil)
        # D = H̃(1-e^S) は H̃>0 (β>0) で S の減少関数、H̃<0 (β<0, 瀕死特効型) で
        # 増加関数。向きに応じて F_S を反転する。
        inc = self.Htil < 0
        F = np.zeros_like(d)
        in_range = pos & (s_of_d >= self.a) & (s_of_d <= self.b)
        if in_range.any():
            Fs = self._cdf_S(s_of_d[in_range])
            F[in_range] = Fs if inc else 1.0 - Fs
        if inc:
            F[pos & (s_of_d > self.b)] = 1.0        # 最大ダメージ超
        else:
            F[pos & (s_of_d < self.a)] = 1.0        # 最大ダメージ超
            F[(d >= 0.0) & ~(arg > 0.0)] = 1.0      # d >= H̃ (台の右外)
        return np.clip(F, 0.0, 1.0)

    def pdf(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        arg = 1.0 - d / self.Htil
        s_of_d = np.full_like(d, np.nan)
        pos = (d >= 0.0) & (arg > 0.0)
        s_of_d[pos] = np.log1p(-d[pos] / self.Htil)
        f = np.zeros_like(d)
        in_range = pos & (s_of_d >= self.a) & (s_of_d <= self.b)
        if in_range.any():
            f[in_range] = self._pdf_S(s_of_d[in_range]) / np.abs(self.Htil - d[in_range])
        return np.clip(f, 0.0, None)


def build_product_dist(ymix_per_hit: list[list[Uniform]], hp: HPParams,
                       *, dp: bool = True) -> ProductDist:
    """積モデル (HP依存) の COS 係数を構築して ProductDist を返す。"""
    a, b = support_bounds_hits(ymix_per_hit)
    mean_S, var_S = moments_hits(ymix_per_hit)
    std_S = math.sqrt(var_S) if var_S > 0 else (b - a)
    width_sigma = (b - a) / std_S if std_S > 0 else 1.0
    n_terms = max(1024, min(_COS_N_MAX, int(math.ceil(_COS_TERMS_PER_SIGMA * width_sigma))))

    L = b - a
    u = np.arange(n_terms) * np.pi / L
    phi = cf_S_hits(ymix_per_hit, u)
    av = ap = None
    if dp:
        av, ap = atom_part_distribution_ln(ymix_per_hit)
        if av.size == 0:
            av = ap = None
        else:
            phi = phi - atom_cf_hits(ymix_per_hit, u)
    Fk = (2.0 / L) * np.real(phi * np.exp(-1j * u * a))
    # 高ダメージ端: H̃>0 (β>0) は S=a (最小)、H̃<0 (β<0) は S=b (最大) で実現。
    d_max = -hp.Htil * math.expm1(b if hp.Htil < 0 else a)
    return ProductDist(a, b, u, Fk, av, ap, hp.Htil, d_max)


def build_product_from_cards(
    cards: list[dict], global_crit: float, global_evade: float, damage_mode: str,
    hp: HPParams, *, dp: bool = True,
) -> ProductDist:
    """カード列 + HP パラメータから積モデルの ProductDist を構築する。
    各 Hit の基礎ダメージ x 混合 (= 減衰分割後) を Y = 1 - β x へ写す。"""
    base_per_hit = build_hit_mixtures(cards, global_crit, global_evade, damage_mode)
    ymix_per_hit = [y_mixture(base, hp.beta) for base in base_per_hit]
    return build_product_dist(ymix_per_hit, hp, dp=dp)


# =============================================================================
# Monte Carlo (検証用)
# =============================================================================

def mc_sum(hit_mixtures: list[list[Uniform]], n: int, rng: np.random.Generator) -> np.ndarray:
    """和モデルの合計ダメージ MC サンプル。"""
    total = np.zeros(n)
    for mix in hit_mixtures:
        w = np.array([c.weight for c in mix])
        w = w / w.sum()
        comp = rng.choice(len(mix), size=n, p=w)
        x = np.empty(n)
        for j, c in enumerate(mix):
            m = comp == j
            cnt = int(m.sum())
            if cnt == 0:
                continue
            x[m] = rng.uniform(c.lo, c.hi, size=cnt) if c.hi > c.lo else c.lo
        total += x
    return total


def mc_product(base_per_hit: list[list[Uniform]], hp: HPParams,
               n: int, rng: np.random.Generator) -> np.ndarray:
    """積モデル (HP依存) の累積ダメージ MC サンプル。漸化式 H_{n+1}=H_n-(βH_n+R0)x を直接回す。"""
    Hn = np.full(n, hp.H1, dtype=float)
    for base in base_per_hit:
        w = np.array([c.weight for c in base])
        w = w / w.sum()
        comp = rng.choice(len(base), size=n, p=w)
        x = np.empty(n)
        for j, c in enumerate(base):
            m = comp == j
            cnt = int(m.sum())
            if cnt == 0:
                continue
            x[m] = rng.uniform(c.lo, c.hi, size=cnt) if c.hi > c.lo else c.lo
        Hn = Hn - (hp.beta * Hn + hp.R0) * x
    return hp.H1 - Hn
