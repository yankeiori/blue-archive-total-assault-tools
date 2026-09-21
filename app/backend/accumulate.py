"""蓄積 (チャージ) 型スキル: 上限つきプールと多段爆発のダメージ分布。

理論は docs/accumulate.md。和モデル (HP 非依存) を対象に

    A_k = g_k(mult_k * min(C_k, α_k Σ_{i∈W_k} X_i)),   T = Σ_i X_i + Σ_k A_k

を計算する。W_k (蓄積窓) は互いに素と仮定する (docs/accumulate.md §3.1 の主定理)。
このとき窓の寄与 Z_k = S_k + A_k と窓外合計 V は相互独立なので、各成分の分布を
1 次元で作ってから畳み込めばよい。

実装は全体を「原点 0・刻み h の等間隔セル質量」で統一する。

    1. Hit ごとの一様混合を厳密な重なり積分でセル質量にする (質量・平均とも保存)
    2. 同一カードの Hit は rfft を count 乗して一括 (特性関数の積 = 畳み込み)
    3. 窓の基礎分布 S_k は逆変換で実空間に戻し、ψ を通して押し出し、再び rfft
    4. 全成分を周波数領域で掛けて 1 回だけ逆変換する

セル質量は常に非負・総和 1 なので、COS 反転と違い Gibbs も負密度も出ず、裾でも
単調性が保たれる。誤差はセル内の位置ずれのみで、各 Hit で平均を保存しているため
合計の平均は厳密、分散への影響も h^2·n/12 と無視できる。この方式は assets/accumulate.js
にそのまま移植してある (本ファイルが正本)。

`pass_prob_quad` は単発・独立上限に限った求積リファレンス (COS の CDF を直接使う)。
`mc_accum` は MC。3 者の一致で検証する (tests/test_accumulate.py)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.backend.cos import SumDist, Uniform, build_sum_dist, support_bounds
from app.backend.simulation import DAMAGE_FUNC, decay

# 合成グリッドのセル数 (2 の冪。台全体を覆う)。位置誤差は台幅/この値。
_N_CELLS = 1 << 17
# 上限分布を離散化するノード数 (等質量バケット)。
_N_CAP_NODES = 64
# CDF をまとめて評価するときのチャンク幅 (np.outer のメモリ抑制)。
_CDF_CHUNK = 2048
# セル質量を保持する下限 (これ未満の裾は切り詰める)。
_TRIM_EPS = 1e-15


# =============================================================================
# 上限・窓の指定
# =============================================================================

@dataclass
class CapSpec:
    """蓄積上限の指定。

    - kind="fixed"   : 固定値 value (攻撃力 × 倍率% を外で計算して入れる)
    - kind="mixture" : 独立な乱数上限 (Uniform 混合。ダメージ列とは無相関)
    - kind="hits"    : Hit 集合 `hits` のダメージ合計 x に対し 上限 = coef * x。
                       その Hit が窓内なら S_k と相関し、窓外なら「その Hit の
                       ダメージ + 窓の寄与」をひとかたまりの成分として扱う。
    """
    kind: str = "fixed"
    value: float = 0.0
    mixture: list[Uniform] | None = None
    hits: list[int] | None = None
    coef: float = 1.0
    hit: int | None = None          # 後方互換 (単一 Hit 指定)

    def source_hits(self) -> list[int]:
        if self.kind not in ("hits", "hit"):
            return []
        if self.hits:
            return list(self.hits)
        return [] if self.hit is None else [self.hit]


@dataclass
class AccumWindow:
    """1 回分の蓄積 (プール)。

    hits        : このプールに寄与する Hit 番号 (蓄積率 rate は窓内共通)
    name        : 表示用の名前 (計算には影響しない)
    burst_hit   : 爆発が着弾する Hit 番号 (この Hit の直後に入る)。None なら窓の最後。
                  **合計ダメージ分布には影響しない** (和モデルなので順序不問)。
                  多段リスタで「その関門では画面にまだ乗っていない」を出すために使う
                  (app/backend/restart_accum.py)。
    rate        : 蓄積率 α (与ダメージの 100% なら 1.0、10% なら 0.1)
    cap         : 蓄積上限
    burst_mult  : 爆発時に蓄積値へ掛かる倍率 (属性特効・バフ等をまとめた係数)
    burst_decay : 爆発ダメージに減衰関数を通すか (None なら全体設定に従う)
    emit        : False なら爆発せずプールが消える (上書きリセット運用)
    """
    hits: list[int]
    rate: float
    cap: CapSpec
    burst_mult: float = 1.0
    burst_decay: bool | None = None
    emit: bool = True
    name: str = ""
    burst_hit: int | None = None


@dataclass
class WindowStats:
    """窓ごとの診断量 (docs/accumulate.md §3.2)。"""
    name: str
    hits: list[int]
    rate: float
    sat_prob: float          # 飽和確率 P(α S_k > C_k)
    pool_mean: float         # E[min(C_k, α S_k)]
    overflow_mean: float     # E[(α S_k - C_k)^+] = 配分損失
    burst_mean: float        # E[g(mult · min(C_k, α S_k))] = 実際に入る爆発ダメージ
    cap_mean: float
    damage_mean: float       # E[S_k]
    burst_lo: float = 0.0    # 爆発ダメージの 10% 点
    burst_hi: float = 0.0    # 爆発ダメージの 90% 点


# =============================================================================
# 区分線形写像 ψ (docs/accumulate.md §1.1)
# =============================================================================

def pool_amount(s, cap: float, rate: float) -> np.ndarray:
    """プール量 min(cap, rate * s)。"""
    return np.minimum(cap, rate * np.asarray(s, dtype=float))


def burst_amount(s, cap: float, rate: float, burst_decay: bool,
                 mult: float = 1.0) -> np.ndarray:
    """爆発量 g(mult * min(cap, rate * s))。burst_decay なら減衰を通す。"""
    p = mult * pool_amount(s, cap, rate)
    return decay(np.asarray(p, dtype=float)) if burst_decay else p


def psi(s, cap: float, rate: float, burst_decay: bool, mult: float = 1.0) -> np.ndarray:
    """窓の寄与 ψ(s) = s + g(mult · min(cap, α s))。s について狭義単調増加。"""
    s = np.asarray(s, dtype=float)
    return s + burst_amount(s, cap, rate, burst_decay, mult)


def psi_knots(cap: float, rate: float, burst_decay: bool,
              lo: float, hi: float, mult: float = 1.0) -> list[float]:
    """ψ の折れ点 (引数空間) のうち (lo, hi) に入るもの。"""
    if rate <= 0:
        return []
    ks = [cap / rate]
    if burst_decay and mult > 0:
        for (x_lo, _x_hi), _ab in DAMAGE_FUNC[1:]:
            if x_lo < mult * cap:
                ks.append(x_lo / (mult * rate))
    return sorted({k for k in ks if lo < k < hi})


def psi_inv(z, cap: float, rate: float, burst_decay: bool,
            lo: float, hi: float, mult: float = 1.0) -> np.ndarray:
    """[lo, hi] 上での ψ の逆写像。ψ は区分線形なので、折れ点を節点にとれば
    線形補間が厳密解になる。"""
    s_k = np.array([lo, *psi_knots(cap, rate, burst_decay, lo, hi, mult), hi])
    return np.interp(np.asarray(z, dtype=float),
                     psi(s_k, cap, rate, burst_decay, mult), s_k)


# =============================================================================
# セル質量 (原点 0・刻み step の等間隔グリッド)
# =============================================================================

def _deposit(out: np.ndarray, pos, mass, step: float) -> None:
    """質量を隣接 2 セルへ線形分配する (総質量と平均を保存)。"""
    n = out.size
    x = np.clip(np.asarray(pos, dtype=float) / step, 0.0, n - 1.0)
    i = np.minimum(x.astype(np.int64), n - 2)
    t = x - i
    m = np.asarray(mass, dtype=float)
    np.add.at(out, i, m * (1.0 - t))
    np.add.at(out, i + 1, m * t)


def mixture_cells(mix: list[Uniform], step: float, n: int) -> np.ndarray:
    """1 Hit の一様混合を厳密なセル質量へ。各セルの質量と条件付き平均を保つ。"""
    out = np.zeros(n)
    for u in mix:
        if u.half_width == 0.0:
            _deposit(out, np.array([u.center]), np.array([u.weight]), step)
            continue
        i0 = max(0, int(math.floor(u.lo / step + 0.5)))
        i1 = min(n - 1, int(math.floor(u.hi / step + 0.5)))
        idx = np.arange(i0, i1 + 1)
        e_lo = np.maximum(u.lo, (idx - 0.5) * step)
        e_hi = np.minimum(u.hi, (idx + 0.5) * step)
        seg = np.maximum(e_hi - e_lo, 0.0)
        m = u.weight * seg / (u.hi - u.lo)
        _deposit(out, 0.5 * (e_lo + e_hi), m, step)
    return out


def _cell_quantile(cells: np.ndarray, step: float, p: float) -> float:
    """セル質量の p 分位点。"""
    total = cells.sum()
    if total <= 0:
        return 0.0
    i = int(np.searchsorted(np.cumsum(cells), p * total))
    return float(min(i, cells.size - 1) * step)


def _quantize(values: np.ndarray, mass: np.ndarray, n_nodes: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """セル質量を等質量バケットへまとめ、(代表値 = 条件付き平均, 質量) を返す。"""
    keep = mass > _TRIM_EPS
    v, m = values[keep], mass[keep]
    if v.size <= n_nodes:
        return v, m
    cum = np.cumsum(m)
    total = cum[-1]
    edges = np.searchsorted(cum, total * np.arange(1, n_nodes) / n_nodes)
    bounds = [0, *np.unique(np.clip(edges + 1, 1, v.size - 1)), v.size]
    nv, nm = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = m[a:b].sum()
        if w <= 0:
            continue
        nv.append(float((m[a:b] * v[a:b]).sum() / w))
        nm.append(float(w))
    return np.array(nv), np.array(nm)


def _cap_nodes_from(cap: CapSpec, src_cells: np.ndarray | None, step: float,
                    n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """上限ノード (cap 値, 付随するロール値 x, 確率)。
    kind="hits" のとき x はその Hit 集合のダメージ合計、それ以外は x = 0。"""
    if cap.kind == "fixed":
        return np.array([float(cap.value)]), np.array([0.0]), np.array([1.0])
    if cap.kind == "mixture":
        mix = cap.mixture or []
        span = max((u.hi for u in mix), default=0.0)
        m = mixture_cells(mix, step, max(2, int(span / step) + 3))
        v, w = _quantize(np.arange(m.size) * step, m, n_nodes)
        return v, np.zeros_like(v), w
    if cap.kind in ("hits", "hit"):
        if src_cells is None:
            raise ValueError("上限ロールの Hit 指定が不正です")
        v, w = _quantize(np.arange(src_cells.size) * step, src_cells, n_nodes)
        return cap.coef * v, v, w
    raise ValueError(f"未知の CapSpec.kind: {cap.kind}")


# =============================================================================
# 窓・成分の構成
# =============================================================================

def _normalize_windows(hit_mixtures, windows: list[AccumWindow]) -> list[AccumWindow]:
    """寄与しない窓 (率 0 / 爆発なし / 空) を落とし、重複と範囲を検証する。"""
    n = len(hit_mixtures)
    live: list[AccumWindow] = []
    seen: set[int] = set()
    for w in windows:
        for i in w.hits:
            if not (0 <= i < n):
                raise ValueError(f"蓄積窓の Hit 番号が範囲外です: {i}")
        if len(set(w.hits)) != len(w.hits):
            raise ValueError(f"蓄積窓に重複した Hit があります: {w.hits}")
        if not w.emit or w.rate <= 0 or not w.hits or w.burst_mult <= 0:
            continue  # プールが消える / 溜まらない窓は素の damage 扱い
        dup = seen & set(w.hits)
        if dup:
            raise ValueError(
                f"蓄積窓が重なっています (Hit {sorted(dup)})。"
                "重なりは docs/accumulate.md §3.5 のプール DP が必要で、未対応です。")
        seen |= set(w.hits)
        live.append(w)

    owners: dict[int, int] = {}
    for k, w in enumerate(live):
        src = w.cap.source_hits()
        for h in src:
            if not (0 <= h < n):
                raise ValueError(f"上限ロールの Hit 番号が範囲外です: {h}")
            if h in owners:
                raise ValueError(f"Hit {h} を複数の窓が上限ロールとして共有しています (未対応)")
            owners[h] = k
        if src and not set(src) <= set(w.hits):
            if set(src) & set(w.hits):
                raise ValueError("上限ロールの Hit は窓内・窓外のどちらかに揃えてください")
            for o_i, o in enumerate(live):
                if o_i != k and set(src) & set(o.hits):
                    raise ValueError(f"上限ロール Hit {src} が別の窓に属しています (未対応)")
    return live


def _roles(n: int, live: list[AccumWindow]) -> list[tuple]:
    """Hit ごとの役割 ('free',) / ('win', k) / ('cap', k) を返す。"""
    roles: list[tuple] = [("free",)] * n
    for k, w in enumerate(live):
        for i in w.hits:
            roles[i] = ("win", k)
    for k, w in enumerate(live):
        for i in w.cap.source_hits():
            roles[i] = ("cap", k)
    return roles


def _grouped_cf(hit_mixtures, idx: list[int], step: float, n: int) -> np.ndarray | None:
    """Hit 集合の合計の特性関数 (rfft)。同一混合オブジェクトの連続 Hit はまとめる。"""
    if not idx:
        return None
    cf = None
    i = 0
    while i < len(idx):
        mix = hit_mixtures[idx[i]]
        j = i + 1
        while j < len(idx) and hit_mixtures[idx[j]] is mix:
            j += 1
        g = np.fft.rfft(mixture_cells(mix, step, n))
        g = g ** (j - i)
        cf = g if cf is None else cf * g
        i = j
    return cf


def _real_cells(cf: np.ndarray | None, n: int) -> np.ndarray:
    """特性関数 (rfft) を実空間のセル質量へ戻す。"""
    if cf is None:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    return np.clip(np.fft.irfft(cf, n), 0.0, None)


# =============================================================================
# 分布オブジェクト
# =============================================================================

@dataclass
class AccumDist:
    """蓄積つき合計ダメージ T の分布 (セル質量表現)。"""
    lo: float
    step: float
    mass: np.ndarray
    mean: float
    var: float
    support_lo: float
    support_hi: float
    window_stats: list[WindowStats] = field(default_factory=list)

    @property
    def centers(self) -> np.ndarray:
        return self.lo + self.step * np.arange(self.mass.size)

    def cdf(self, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        edges = self.centers + 0.5 * self.step
        cum = np.clip(np.cumsum(self.mass), 0.0, 1.0)
        return np.interp(xs, edges, cum, left=0.0, right=1.0)

    def sf(self, xs) -> np.ndarray:
        return 1.0 - self.cdf(xs)

    def pdf(self, xs) -> np.ndarray:
        return np.interp(np.asarray(xs, dtype=float), self.centers,
                         self.mass / self.step, left=0.0, right=0.0)

    def pass_prob(self, D: float) -> float:
        return float(self.sf(np.array([float(D)]))[0])

    def quad_nodes(self, n_nodes: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """等質量バケットの (代表値, 確率)。合計 1。

        app/backend/restart.py の後ろ向き帰納が増分を離散化するのに使う。
        密度を等間隔に標本化するより、セル質量をそのまままとめる方が厳密
        (質量と平均が保たれる)。
        """
        return _quantize(self.centers, self.mass, int(n_nodes))


def build_accum_dist(hit_mixtures: list[list[Uniform]], windows: list[AccumWindow], *,
                     burst_decay: bool = False, n_cells: int = _N_CELLS,
                     n_cap_nodes: int = _N_CAP_NODES, **_legacy) -> AccumDist:
    """蓄積つき合計ダメージ T の分布を構築する。

    windows が空なら素の和モデル (build_sum_dist) と一致する。
    """
    live = _normalize_windows(hit_mixtures, windows)
    n_hits = len(hit_mixtures)

    # --- グリッド (原点 0、台全体を覆う刻み) -------------------------------
    _lo_all, hi_all = support_bounds(hit_mixtures)
    hi_t = hi_all
    for w in live:
        _wl, w_hi = support_bounds([hit_mixtures[i] for i in w.hits])
        cap_hi = _cap_upper(w, hit_mixtures)
        hi_t += float(burst_amount(np.array([w_hi]), cap_hi, w.rate,
                                   _bd(w, burst_decay), w.burst_mult)[0])
    n = int(n_cells)
    step = max(hi_t / (n - 2), 1e-9)

    roles = _roles(n_hits, live)
    free_idx = [i for i in range(n_hits) if roles[i] == ("free",)]
    cf_total = _grouped_cf(hit_mixtures, free_idx, step, n)

    stats: list[WindowStats] = []
    for k, w in enumerate(live):
        base_idx = [i for i in w.hits if roles[i] == ("win", k)]
        src_idx = [i for i in range(n_hits) if roles[i] == ("cap", k)]
        inside = bool(src_idx) and set(src_idx) <= set(w.hits)
        base_cells = _real_cells(_grouped_cf(hit_mixtures, base_idx, step, n), n)
        src_cells = (_real_cells(_grouped_cf(hit_mixtures, src_idx, step, n), n)
                     if src_idx else None)
        z_cells, st = _window_cells(w, base_cells, src_cells, inside, step, n,
                                    _bd(w, burst_decay), n_cap_nodes)
        cf_z = np.fft.rfft(z_cells)
        cf_total = cf_z if cf_total is None else cf_total * cf_z
        st.hits = list(w.hits)
        stats.append(st)

    mass = _real_cells(cf_total, n)
    total = mass.sum()
    if total > 0:
        mass = mass / total

    nz = np.nonzero(mass > _TRIM_EPS)[0]
    i0, i1 = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
    mass = mass[i0:i1 + 1]
    lo = i0 * step
    centers = lo + step * np.arange(mass.size)
    mean = float((mass * centers).sum())
    var = float((mass * (centers - mean) ** 2).sum())
    return AccumDist(lo=lo, step=step, mass=mass, mean=mean, var=var,
                     support_lo=lo, support_hi=lo + step * (mass.size - 1),
                     window_stats=stats)


def _bd(w: AccumWindow, default: bool) -> bool:
    return default if w.burst_decay is None else bool(w.burst_decay)


def _cap_upper(w: AccumWindow, hit_mixtures) -> float:
    """上限の取りうる最大値 (グリッド幅の見積り用)。"""
    if w.cap.kind == "fixed":
        return float(w.cap.value)
    if w.cap.kind == "mixture":
        return max((u.hi for u in (w.cap.mixture or [])), default=0.0)
    src = w.cap.source_hits()
    _lo, hi = support_bounds([hit_mixtures[i] for i in src]) if src else (0.0, 0.0)
    return w.cap.coef * hi


def _window_cells(w: AccumWindow, base_cells: np.ndarray,
                  src_cells: np.ndarray | None, inside: bool, step: float, n: int,
                  burst_decay: bool, n_cap_nodes: int
                  ) -> tuple[np.ndarray, WindowStats]:
    """窓の寄与 Z_k = S_k + g(mult·min(C_k, α S_k)) のセル質量と診断量。

    上限が窓内 Hit 由来なら ψ(x + s)、窓外 Hit 由来なら x + ψ(s) を押し出す
    (docs/accumulate.md §2.3)。どちらも s について単調なので、基礎分布のセル質量を
    そのまま写して落とせばよい。
    """
    cap_v, cap_x, cap_w = _cap_nodes_from(w.cap, src_cells, step, n_cap_nodes)
    keep = base_cells > _TRIM_EPS
    s_vals = np.nonzero(keep)[0] * step
    s_mass = base_cells[keep]

    out = np.zeros(n)
    # 爆発ダメージ自体の分布 (足切り表示で「まだ画面に乗っていない量」の幅を出す)
    b_arg = float(s_vals[-1] + (cap_x.max() if inside else 0.0))
    b_top = float(burst_amount(np.array([b_arg]), cap_v.max(), w.rate,
                               burst_decay, w.burst_mult)[0])
    b_cells = np.zeros(max(2, int(math.ceil(b_top / step)) + 2))
    sat = pool = over = burst = 0.0
    for c, x, wt in zip(cap_v, cap_x, cap_w):
        arg = s_vals + x if inside else s_vals
        z = psi(arg, c, w.rate, burst_decay, w.burst_mult)
        if not inside:
            z = z + x
        _deposit(out, z, wt * s_mass, step)
        raw = w.rate * arg
        b_vals = burst_amount(arg, c, w.rate, burst_decay, w.burst_mult)
        _deposit(b_cells, b_vals, wt * s_mass, step)
        sat += wt * float(s_mass[raw > c].sum())
        pool += wt * float((s_mass * np.minimum(c, raw)).sum())
        over += wt * float((s_mass * np.maximum(raw - c, 0.0)).sum())
        burst += wt * float((s_mass * b_vals).sum())

    dmg = float((s_mass * s_vals).sum()) + (float((cap_w * cap_x).sum()) if inside else 0.0)
    stats = WindowStats(name=w.name, hits=[], rate=w.rate, sat_prob=sat, pool_mean=pool,
                        overflow_mean=over, burst_mean=burst,
                        cap_mean=float((cap_w * cap_v).sum()), damage_mean=dmg,
                        burst_lo=_cell_quantile(b_cells, step, 0.10),
                        burst_hi=_cell_quantile(b_cells, step, 0.90))
    return out, stats


# =============================================================================
# リファレンス: 単発・独立上限の求積 (COS の CDF を直接使う)
# =============================================================================

def _cdf_chunked(dist: SumDist, xs: np.ndarray) -> np.ndarray:
    """SumDist.cdf をチャンク評価する (np.outer のメモリ爆発を避ける)。"""
    out = np.empty(xs.size)
    for i in range(0, xs.size, _CDF_CHUNK):
        out[i:i + _CDF_CHUNK] = dist.cdf(xs[i:i + _CDF_CHUNK])
    return out


def pass_prob_quad(hit_mixtures: list[list[Uniform]], win: AccumWindow, D: float, *,
                   burst_decay: bool = False, n_cap_nodes: int = 16,
                   n_panel: int = 400) -> float:
    """P(T >= D) を docs/accumulate.md §2.1 の 1 次元求積で計算する (K=1 限定)。

    上限は fixed / mixture のみ (ダメージ列と無相関)。グリッド合成と独立な経路なので、
    裾確率の相互検証に使う。
    """
    if win.cap.kind in ("hits", "hit"):
        raise ValueError("pass_prob_quad は独立な上限 (fixed/mixture) のみ対応します")
    bd = _bd(win, burst_decay)
    mult = win.burst_mult
    idx = set(win.hits)
    free = [m for i, m in enumerate(hit_mixtures) if i not in idx]
    S = build_sum_dist([hit_mixtures[i] for i in win.hits])
    V = build_sum_dist(free) if free else None
    lo, hi = S.support_lo, S.support_hi

    # 上限ノード (混合成分ごとに Gauss-Legendre)
    if win.cap.kind == "fixed":
        cap_v, cap_w = np.array([float(win.cap.value)]), np.array([1.0])
    else:
        vs, ws = [], []
        mix = win.cap.mixture or []
        tot = sum(u.weight for u in mix) or 1.0
        for u in mix:
            if u.half_width == 0.0:
                vs.append(np.array([u.center]))
                ws.append(np.array([u.weight / tot]))
            else:
                gx, gw = np.polynomial.legendre.leggauss(n_cap_nodes)
                vs.append(0.5 * (u.lo + u.hi) + 0.5 * (u.hi - u.lo) * gx)
                ws.append(gw / 2.0 * (u.weight / tot))
        cap_v, cap_w = np.concatenate(vs), np.concatenate(ws)

    if V is None:
        # 窓外が空なら求積は不要。ψ が単調なので P(ψ(S) >= D) = 1 - F_S(ψ^{-1}(D))。
        total = 0.0
        for c, w in zip(cap_v, cap_w):
            s_star = float(psi_inv(D, c, win.rate, bd, lo, hi, mult))
            total += w * (1.0 - float(S.cdf(np.array([s_star]))[0]))
        return total

    def tail(z: np.ndarray) -> np.ndarray:
        return 1.0 - _cdf_chunked(V, D - z)

    total = 0.0
    for c, w in zip(cap_v, cap_w):
        acc = 0.0
        if S.av is not None:
            acc += float((S.ap * tail(psi(S.av, c, win.rate, bd, mult))).sum())
        # ψ の折れ点に加え、tail が 0/1 に飽和する点でも被積分関数が滑らかでなくなる。
        cuts = [D - V.support_lo, D - V.support_hi]
        extra = [float(psi_inv(t, c, win.rate, bd, lo, hi, mult)) for t in cuts]
        bounds = sorted({lo, hi, *psi_knots(c, win.rate, bd, lo, hi, mult),
                         *[e for e in extra if lo < e < hi]})
        for p, q in zip(bounds[:-1], bounds[1:]):
            if q <= p:
                continue
            m = n_panel if n_panel % 2 == 0 else n_panel + 1
            s = np.linspace(p, q, m + 1)
            f = S.pdf(s) * tail(psi(s, c, win.rate, bd, mult))
            wts = np.ones(m + 1)
            wts[1:-1:2] = 4.0
            wts[2:-1:2] = 2.0
            acc += float((wts * f).sum()) * (q - p) / (3.0 * m)
        total += w * acc
    return total


# =============================================================================
# MC (正解基準)
# =============================================================================

def _sample_mixture(mix: list[Uniform], n: int, rng: np.random.Generator) -> np.ndarray:
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
    return x


def mc_accum(hit_mixtures: list[list[Uniform]], windows: list[AccumWindow],
             n: int, rng: np.random.Generator, *,
             burst_decay: bool = False) -> np.ndarray:
    """蓄積つき合計ダメージの MC サンプル。"""
    xs = [_sample_mixture(mix, n, rng) for mix in hit_mixtures]
    total = np.zeros(n)
    for x in xs:
        total += x
    for w in windows:
        if not w.emit or w.rate <= 0 or not w.hits or w.burst_mult <= 0:
            continue
        s = np.zeros(n)
        for i in w.hits:
            s += xs[i]
        if w.cap.kind == "fixed":
            cap = np.full(n, float(w.cap.value))
        elif w.cap.kind == "mixture":
            cap = _sample_mixture(w.cap.mixture or [], n, rng)
        else:
            cap = np.zeros(n)
            for i in w.cap.source_hits():
                cap += xs[i]
            cap = w.cap.coef * cap
        pool = w.burst_mult * np.minimum(cap, w.rate * s)
        total += decay(pool) if _bd(w, burst_decay) else pool
    return total


# =============================================================================
# MC 非依存のモーメント検証 (docs/accumulate.md §6)
# =============================================================================

def min_moments_survival(S: SumDist, cap_mix: list[Uniform] | float, rate: float,
                         n_grid: int = 20001) -> tuple[float, float]:
    """独立な C に対する (E[min(C, rate*S)], E[min(C, rate*S)^2])。

    min の生存関数が積になることから
        E[min] = ∫ P(C>t) P(rate*S>t) dt,  E[min^2] = ∫ 2t P(C>t) P(rate*S>t) dt。
    押し出しとは独立な経路なので検証基準に使える。
    """
    if isinstance(cap_mix, (int, float)):
        c_hi = float(cap_mix)

        def sf_c(t: np.ndarray) -> np.ndarray:
            # 上限に原子があると積分区間の端 t = c で不連続になる。求積では左極限
            # (P(C >= t)) を使うのが正しく、連続な上限では P(C > t) と一致する。
            return (t <= c_hi).astype(float)
    else:
        nodes = np.concatenate([np.linspace(u.lo, u.hi, 257) if u.half_width > 0
                                else np.array([u.center]) for u in cap_mix])
        wts = np.concatenate([np.full(257, u.weight / 257) if u.half_width > 0
                              else np.array([u.weight]) for u in cap_mix])
        wts = wts / wts.sum()
        c_hi = float(max(u.hi for u in cap_mix))

        def sf_c(t: np.ndarray) -> np.ndarray:
            return (wts[None, :] * (nodes[None, :] >= t[:, None])).sum(axis=1)

    hi = min(c_hi, rate * S.support_hi)
    m = n_grid - 1 if (n_grid - 1) % 2 == 0 else n_grid
    t = np.linspace(0.0, hi, m + 1)
    sf_s = 1.0 - _cdf_chunked(S, t / rate) if rate > 0 else np.zeros_like(t)
    g = sf_c(t) * sf_s
    # 被積分関数は区分的に滑らかなので Simpson (trapezoid だと O(h^2) で収束が遅い)
    wq = np.ones(m + 1)
    wq[1:-1:2] = 4.0
    wq[2:-1:2] = 2.0
    scale = hi / (3.0 * m)
    return float((wq * g).sum()) * scale, float((wq * 2.0 * t * g).sum()) * scale
