"""混在モデル (HP依存ダメージ + 通常ダメージが同一ターゲットに混在) の分布計算。

理論は docs/mixed.md。1 Hit の作用は HP のアフィン写像
    HP依存: H̃ → H̃·Y   (Y = 1 - β x, シフト座標 H̃ = H + R0/β)
    通常:   H̃ → H̃ - z
であり、累積ダメージ D = H̃₁ - H̃_fin は 1 次元 Markov 状態 s = D_t を持つ。
平行移動とスカラー倍は非可換 (アフィン群) なので 1 次元の COS では閉じず
(docs/mixed.md §5)、グリッド DP に落とす (§6.2)。ただしヒット単位ではなく
**同型連続ブロック単位** で更新する (§6.1(c) の効率化):

  - 通常ブロック: 増分分布 = 和モデル (build_sum_dist, 特性関数は厳密)
  - HP依存ブロック: 状態写像 s' = π s + H̃₁(1 - π), π = ΠY はブロック積
    (ln π の分布は積モデルの COS で厳密)

各ブロックの厳密 CDF をグリッド解像度に合わせた等間隔セルで離散化した
アフィンカーネル {(a_i, c_i, w_i): s' = a_i s + c_i} を密度/価値関数に適用する。
更新回数はヒット数ではなくブロック数で済み、補間誤差の蓄積も同様に減る。

平均・分散はモーメント再帰 m_p(t+1) = Σ_j C(p,j) E[a^j b^(p-j)] m_j(t) で
厳密に計算できる (§6.3 の効率化、検証基準)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.backend.cos import (
    HPParams,
    Uniform,
    build_product_dist,
    build_sum_dist,
    expand_card_to_hit_mixture,
    support_bounds_hits,
    y_mixture,
    y_raw_moment,
)

# グリッド解像度の既定値
_GRID_N_MIN = 2001
_GRID_N_MAX = 24001
_NODES_MIN = 33
_NODES_MAX = 1024
_WINDOW_SIGMA = 13.0          # 分布ウィンドウ [mean ± 13σ] (COS の L=12 と同水準)


# ---------------------------------------------------------------------------
# カード → ヒット仕様 (HP依存フラグ + 混合)
# ---------------------------------------------------------------------------
def card_is_hp_dep(card: dict) -> bool:
    """カードの「HP依存」フラグを正規化する。

    Checklist の value (list)・bool・int のいずれでも受け、キーが無い場合は
    True (従来の hp_mode=on = 全カード HP依存 と後方互換)。"""
    v = card.get("hp_dep", True)
    if v is None:
        return True
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    return bool(v)


def hit_specs_from_cards(
    cards: list[dict], global_crit: float, global_evade: float, damage_mode: str,
) -> list[tuple[bool, list[Uniform]]]:
    """全カードを (HP依存フラグ, 1Hit の一様混合) の列に展開する。
    build_hit_mixtures と同じ展開 (敵の数は掛けない) + フラグ付き。"""
    specs: list[tuple[bool, list[Uniform]]] = []
    for c in cards:
        mix = expand_card_to_hit_mixture(c, global_crit, global_evade, damage_mode)
        hits = int(c.get("hits") or 1)
        specs.extend([(card_is_hp_dep(c), mix)] * hits)
    return specs


def scale_mixture(mix: list[Uniform], k: float) -> list[Uniform]:
    """混合を k 倍にスケールする (β=0 の退化: HP依存ヒット → 通常ダメージ R0·x)。"""
    return [Uniform(u.weight, u.lo * k, u.hi * k) for u in mix]


# ---------------------------------------------------------------------------
# ブロック分割
# ---------------------------------------------------------------------------
def blocks_from_specs(
    specs: list[tuple[bool, list[Uniform]]], beta: float,
) -> list[tuple[str, list[list[Uniform]]]]:
    """ヒット列を同型連続ブロックへ。("z", [damage 混合...]) / ("y", [Y 混合...])。"""
    blocks: list[tuple[str, list[list[Uniform]]]] = []
    for is_hp, mix in specs:
        kind = "y" if is_hp else "z"
        m = y_mixture(mix, beta) if is_hp else mix
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(m)
        else:
            blocks.append((kind, [m]))
    return blocks


# ---------------------------------------------------------------------------
# 厳密モーメント (再帰) と台 (区間演算)
# ---------------------------------------------------------------------------
def _mix_moments12(mix: list[Uniform]) -> tuple[float, float]:
    """混合の (E[X], E[X²])。"""
    m1 = m2 = 0.0
    for u in mix:
        c, h = u.center, u.half_width
        m1 += u.weight * c
        m2 += u.weight * (c * c + h * h / 3.0)
    return m1, m2


def mixed_moments(specs, hp: HPParams) -> tuple[float, float]:
    """累積ダメージ D の厳密な (平均, 分散)。H̃ のモーメント再帰による。"""
    Htil = hp.Htil
    m1, m2 = Htil, Htil * Htil
    for is_hp, mix in specs:
        if is_hp:
            ym = y_mixture(mix, hp.beta)
            m1 *= y_raw_moment(ym, 1)
            m2 *= y_raw_moment(ym, 2)
        else:
            z1, z2 = _mix_moments12(mix)
            m2 = m2 - 2.0 * z1 * m1 + z2
            m1 = m1 - z1
    mean = Htil - m1
    var = max(m2 - m1 * m1, 0.0)
    return mean, var


def mixed_support(specs, hp: HPParams) -> tuple[float, float]:
    """累積ダメージ D の厳密な台 [lo, hi] (H̃ の区間演算)。β の符号によらない。"""
    h_lo = h_hi = hp.Htil
    for is_hp, mix in specs:
        if is_hp:
            ym = y_mixture(mix, hp.beta)
            y_lo = min(u.lo for u in ym)
            y_hi = max(u.hi for u in ym)
            cands = (h_lo * y_lo, h_lo * y_hi, h_hi * y_lo, h_hi * y_hi)
            h_lo, h_hi = min(cands), max(cands)
        else:
            z_lo = min(u.lo for u in mix)
            z_hi = max(u.hi for u in mix)
            h_lo, h_hi = h_lo - z_hi, h_hi - z_lo
    return hp.Htil - h_hi, hp.Htil - h_lo


# ---------------------------------------------------------------------------
# ブロックのアフィンカーネル {(a_i, c_i, w_i)} : s' = a_i s + c_i
# ---------------------------------------------------------------------------
def _cdf_nodes(edge_cdf: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """セル境界の CDF から (セル中点, セル質量) のノード列 (質量 0 のセルは除去)。"""
    F = np.clip(np.asarray(edge_cdf, dtype=float), 0.0, 1.0)
    F = np.maximum.accumulate(F)
    F[0], F[-1] = 0.0, 1.0
    w = np.diff(F)
    mids = 0.5 * (edges[:-1] + edges[1:])
    keep = w > 1e-14
    w = w[keep]
    if w.size == 0:
        return np.array([float(edges[len(edges) // 2])]), np.array([1.0])
    return mids[keep], w / w.sum()


def _n_nodes(feature_width: float, grid_step: float) -> int:
    """ブロック範囲 (ダメージ規模) をグリッド刻みの ~2 倍で解像するノード数。"""
    if not (feature_width > 0) or not (grid_step > 0):
        return _NODES_MIN
    return int(np.clip(math.ceil(feature_width / (2.0 * grid_step)),
                       _NODES_MIN, _NODES_MAX))


def block_kernel(block, hp: HPParams, grid_step: float):
    """ブロックのアフィンカーネル (a, c, w) を返す。

    z ブロック: 増分 T の厳密 CDF (SumDist, 原子込み) をセル離散化。a=1, c=T ノード。
    y ブロック: ブロック積 π = ΠY。ln π の厳密 CDF (積モデル COS, 原子込み) を
                セル離散化し、s' = π s + H̃₁(1-π) より a=π, c=H̃₁(1-π)。
    原子はセルに吸収される (位置誤差 <= セル幅 ~ 2×グリッド刻み)。"""
    kind, mixes = block
    Htil = hp.Htil if hp is not None else 0.0
    if kind == "z":
        lo = sum(min(u.lo for u in m) for m in mixes)
        hi = sum(max(u.hi for u in m) for m in mixes)
        if hi - lo <= 1e-9:
            return (np.array([1.0]), np.array([0.5 * (lo + hi)]), np.array([1.0]))
        dist = build_sum_dist(mixes)
        n = _n_nodes(hi - lo, grid_step)
        edges = np.linspace(lo, hi, n + 1)
        mids, w = _cdf_nodes(dist.cdf(edges), edges)
        return (np.ones_like(mids), mids, w)
    # y ブロック
    A, B = support_bounds_hits(mixes)
    dmg_scale = abs(Htil) * abs(math.exp(B) - math.exp(A))
    if dmg_scale <= 1e-9 or B - A <= 1e-12:
        pi = math.exp(0.5 * (A + B))
        return (np.array([pi]), np.array([Htil * (1.0 - pi)]), np.array([1.0]))
    pd = build_product_dist(mixes, hp)
    n = _n_nodes(dmg_scale, grid_step)
    edges = np.linspace(A, B, n + 1)
    mids, w = _cdf_nodes(pd._cdf_S(edges), edges)
    pi = np.exp(mids)
    return (pi, Htil * (1.0 - pi), w)


# ---------------------------------------------------------------------------
# カーネル適用 (前向き密度 / 後ろ向き期待値)
# ---------------------------------------------------------------------------
def kernel_forward(f: np.ndarray, grid: np.ndarray, kernel) -> np.ndarray:
    """密度 f にブロックカーネルを畳み込む: f'(t) = Σ_i (w_i/a_i) f((t-c_i)/a_i)。"""
    a, c, w = kernel
    out = np.zeros_like(f)
    for ai, ci, wi in zip(a, c, w):
        out += (wi / ai) * np.interp((grid - ci) / ai, grid, f, left=0.0, right=0.0)
    return out


def kernel_backward(V: np.ndarray, grid: np.ndarray, kernel,
                    positions: np.ndarray | None = None) -> np.ndarray:
    """価値関数の期待値 E[V(s')] (s) = Σ_i w_i V(a_i s + c_i)。

    positions は kernel_positions で前計算した評価点 (Dinkelbach 反復での再利用)。
    グリッド外は端値に飽和 (V は単調で外側は定数)。"""
    a, c, w = kernel
    if positions is None:
        positions = kernel_positions(grid, kernel)
    vals = np.interp(positions.ravel(), grid, V).reshape(positions.shape)
    return w @ vals


def kernel_positions(grid: np.ndarray, kernel) -> np.ndarray:
    """後ろ向き E-step の評価点 a_i·s + c_i を (ノード, グリッド) 行列で前計算する。"""
    a, c, _w = kernel
    return a[:, None] * grid[None, :] + c[:, None]


# ---------------------------------------------------------------------------
# 前向き密度の初期化 (最初のブロックは厳密 pdf + 原子で立ち上げる)
# ---------------------------------------------------------------------------
def _deposit(f: np.ndarray, grid: np.ndarray, pos: float, mass: float) -> None:
    """点質量を隣接 2 格子点へ線形分配して密度へ足す (台形積分で質量保存)。"""
    n = grid.size
    step = grid[1] - grid[0]
    if pos <= grid[0]:
        f[0] += mass / (0.5 * step)
        return
    if pos >= grid[-1]:
        f[-1] += mass / (0.5 * step)
        return
    x = (pos - grid[0]) / step
    i = min(int(x), n - 2)
    t = x - i
    for j, share in ((i, mass * (1.0 - t)), (i + 1, mass * t)):
        weight = 0.5 * step if j in (0, n - 1) else step
        f[j] += share / weight
    return


def first_block_density(block, hp: HPParams, grid: np.ndarray) -> np.ndarray:
    """状態 0 から最初のブロックを適用した後の密度 (厳密 pdf + 原子デポジット)。"""
    kind, mixes = block
    f = np.zeros_like(grid)
    if kind == "z":
        lo = sum(min(u.lo for u in m) for m in mixes)
        hi = sum(max(u.hi for u in m) for m in mixes)
        if hi - lo <= 1e-9:
            _deposit(f, grid, 0.5 * (lo + hi), 1.0)
            return f
        dist = build_sum_dist(mixes)
        f += dist.pdf(grid)
        if dist.av is not None:
            for v, p in zip(dist.av, dist.ap):
                _deposit(f, grid, float(v), float(p))
        return f
    Htil = hp.Htil
    A, B = support_bounds_hits(mixes)
    if abs(Htil) * abs(math.exp(B) - math.exp(A)) <= 1e-9 or B - A <= 1e-12:
        pi = math.exp(0.5 * (A + B))
        _deposit(f, grid, Htil * (1.0 - pi), 1.0)
        return f
    pd = build_product_dist(mixes, hp)
    f += pd.pdf(grid)                     # D = H̃₁(1 - e^S) の密度 (ヤコビアン込み)
    if pd.av is not None:
        for s, p in zip(pd.av, pd.ap):
            _deposit(f, grid, Htil * (1.0 - math.exp(float(s))), float(p))
    return f


# ---------------------------------------------------------------------------
# 分布オブジェクト
# ---------------------------------------------------------------------------
@dataclass
class MixedDist:
    """混在モデルの累積ダメージ分布 (グリッド DP)。SumDist/ProductDist 互換 API。"""
    grid: np.ndarray
    dens: np.ndarray
    cum: np.ndarray               # 正規化済み累積 (cum[-1] == 1)
    support_lo: float
    support_hi: float
    mean: float                   # 厳密 (モーメント再帰)
    var: float                    # 厳密 (モーメント再帰)

    def cdf(self, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        F = np.interp(xs, self.grid, self.cum, left=0.0, right=1.0)
        F = np.where(xs < self.support_lo, 0.0, F)
        F = np.where(xs >= self.support_hi, 1.0, F)
        return np.clip(F, 0.0, 1.0)

    def pdf(self, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        f = np.interp(xs, self.grid, self.dens, left=0.0, right=0.0)
        f = np.where((xs < self.support_lo) | (xs > self.support_hi), 0.0, f)
        return np.clip(f, 0.0, None)


def _cumtrapz(f: np.ndarray, step: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * step)
    return out


def build_mixed_dist(specs, hp: HPParams, n_grid: int | None = None) -> MixedDist:
    """混在モデルの累積ダメージ分布をブロック集約グリッド DP で構築する。"""
    if not specs:
        raise ValueError("ヒットがありません。")
    blocks = blocks_from_specs(specs, hp.beta)
    mean, var = mixed_moments(specs, hp)
    sup_lo, sup_hi = mixed_support(specs, hp)
    std = math.sqrt(var)
    # 下端はブロック伝播の中間状態 (累積ダメージ 0 から立ち上がる) を覆う必要が
    # あるため 0 (病的な負ダメージ設定では sup_lo)。上端のみ ±13σ 窓でクリップ。
    a = min(0.0, sup_lo)
    b = min(sup_hi, mean + _WINDOW_SIGMA * std) if std > 0 else sup_hi
    if b - a <= 0:
        b = a + max(1.0, abs(a) * 1e-9)

    if n_grid is None:
        # 最も狭い連続成分 (ダメージ規模) を ~8 点で解像する
        w_min = math.inf
        for is_hp, mix in specs:
            for u in mix:
                if u.hi > u.lo:
                    w = (u.hi - u.lo) * (abs(hp.Htil) * abs(hp.beta) if is_hp else 1.0)
                    if w > 0:
                        w_min = min(w_min, w)
        if math.isfinite(w_min) and w_min > 0:
            n_grid = int(np.clip(math.ceil((b - a) / (w_min / 8.0)) + 1,
                                 _GRID_N_MIN, _GRID_N_MAX))
        else:
            n_grid = _GRID_N_MIN
    grid = np.linspace(a, b, n_grid)
    step = grid[1] - grid[0]

    f = first_block_density(blocks[0], hp, grid)
    for block in blocks[1:]:
        f = kernel_forward(f, grid, block_kernel(block, hp, step))

    cum = _cumtrapz(f, step)
    total = cum[-1]
    if total > 0:
        cum = cum / total
        f = f / total
    return MixedDist(grid, f, cum, sup_lo, sup_hi, mean, var)


# ---------------------------------------------------------------------------
# ディスパッチ: 仕様に応じて 和 / 積 / 混在 の分布を返す
# ---------------------------------------------------------------------------
def normalize_specs(specs, hp: HPParams | None):
    """β=0 (R1=R0) の退化を処理する: HP依存ヒットは倍率一定 R0 の通常ダメージ
    R0·x になるため、混合を R0 倍して通常ヒットに変換する。"""
    if hp is None or hp.beta != 0.0:
        return specs
    return [(False, scale_mixture(m, hp.R0)) if is_hp else (is_hp, m)
            for is_hp, m in specs]


def build_dist_auto(specs, hp: HPParams | None):
    """混在状況に応じて SumDist / ProductDist / MixedDist を構築する。
    ProductDist は Htil の符号で CDF の向きを切り替えるため β<0 (R1<R0) もそのまま。"""
    specs = normalize_specs(specs, hp)
    any_hp = any(is_hp for is_hp, _ in specs)
    all_hp = all(is_hp for is_hp, _ in specs) and bool(specs)
    if not any_hp or hp is None:
        return build_sum_dist([m for _f, m in specs])
    if all_hp:
        return build_product_dist([y_mixture(m, hp.beta) for _f, m in specs], hp)
    return build_mixed_dist(specs, hp)


# ---------------------------------------------------------------------------
# Monte Carlo (検証用)
# ---------------------------------------------------------------------------
def mc_mixed(specs, hp: HPParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """混在モデルの累積ダメージ MC。漸化式を直接回す (正解基準)。"""
    Hn = np.full(n, hp.H1, dtype=float)
    for is_hp, mix in specs:
        w = np.array([u.weight for u in mix])
        w = w / w.sum()
        comp = rng.choice(len(mix), size=n, p=w)
        x = np.empty(n)
        for j, u in enumerate(mix):
            m = comp == j
            cnt = int(m.sum())
            if cnt == 0:
                continue
            x[m] = rng.uniform(u.lo, u.hi, size=cnt) if u.hi > u.lo else u.lo
        if is_hp:
            Hn = Hn - (hp.beta * Hn + hp.R0) * x
        else:
            Hn = Hn - x
    return hp.H1 - Hn
