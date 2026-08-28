"""凸区切り(セーブポイント)付き 足切りライン最適化。

理論は docs/restart_save.md。既存の restart_cos.py / restart_mixed.py は
「リスタート = 累積ダメージ 0 に戻る」前提の更新報酬最大化 (Dinkelbach) だが、
凸の区切りを跨ぐとダメージは確定し、リスタートしても直前の凸区切りまでしか戻らない。
そこで

  * 目的関数を「目標 D 到達までの期待総時間 E[T] の最小化」に置き換える
    (セーブ点が無ければ E[T] = 1/g* で既存と厳密に同値。docs/restart_save.md §5)
  * 中断価値をスカラー 0 ではなく W_b(s) (直前セーブ地点の累積ダメージ s の関数) にする

の 2 点で拡張する。W_b は不動点 W_b(s) = Φ_{W_b(s)}(s) で定まる。後ろ向き帰納は中断価値
R にしか依存せず s に依存しないので、R を掃引して Φ_R(s) = R の根 s*(R) を追うと、後ろ向き
帰納 O(N_R) 回で W_b の曲線と足切り曲線 d_j(s) が同時に得られる。

DP 本体 (_SaveDP) はモデルに依存せず、価値関数・密度の演算を **バックエンド** に委ねる:

  * `_CosBackend`  : 係数空間 (COS/F&O)。和モデル・積モデル。増分が状態に依らないので
                     畳み込み = 特性関数の積、打ち切り = C/M 行列で求積なし。
  * `_GridBackend` : アフィンカーネルのグリッド DP。混在モデル (HP依存 + 通常) は
                     増分が状態に依存する (s' = a s + c) ため係数空間で閉じない。

エントリは analyze (和) / analyze_product (積) / analyze_mixed (混在)。
"""
from __future__ import annotations

import math

import numpy as np

from app.backend.cos import (
    HPParams,
    build_product_dist,
    build_sum_dist,
    support_bounds_hits,
    y_mixture,
)
from app.backend.mixed import (
    _deposit,
    _eval_chunked,
    block_kernel,
    blocks_from_specs,
    kernel_forward,
    kernel_positions,
    kernel_backward,
    mixed_support,
    normalize_specs,
)
from app.backend.restart_cos import (
    _CosEngine,
    _seg_success,
    _seg_times,
    _segs_product,
    _segs_sum,
    _split_bounds,
)
from app.backend.restart_mixed import _mass_above, _truncate_below

# 掃引・二分法のパラメータ
_N_SWEEP = 32           # W_b(s) 曲線の初期節点数 (= 最適方策での後ろ向き帰納の回数)
_N_REFINE = 40          # s のギャップを埋める適応細分の追加点数 (1 点 = 後ろ向き 1 回)
_N_EVAL_KNOTS = 96      # 固定方策 (方策評価) での W_b(s) の標本点数
_FIX_ITERS = 40         # 不動点二分法の反復数 (相対精度 ~1e-12)
_RANGE_ITERS = 16       # 掃引範囲の端点だけを決める粗い不動点 (精度は範囲決めに十分)
_BRACKET_DOUBLES = 40   # 上界を倍々に広げる回数の上限
_R_CAP_FACTOR = 1.0e4   # 到達不能領域の値の頭打ち (基準時間スケールの倍数)

# 混在モデル (グリッド) のグリッド解像度。restart_mixed と同じ方針。
_GRID_N_BACKWARD = 3000
_GRID_N_FORWARD = 8000
_GRID_N_FORWARD_MAX = 24000
_CELL_FACTOR_BACKWARD = 2.0


def _clamp01(p) -> float:
    return min(1.0, max(0.0, float(p)))


# ---------------------------------------------------------------------------
# COS 係数ヘルパー (restart_cos._CosEngine のチルダ規約: C0 = A0/2, Ck = Ak)
# ---------------------------------------------------------------------------
def _shifted_density(eng: _CosEngine, seg_index: int, s: float) -> np.ndarray:
    """s + T_seg の密度の余弦係数。特性関数に e^{ius} を掛けるだけ。"""
    phi = eng.phi[seg_index] * np.exp(1j * eng.u * float(s))
    c = (2.0 / eng.L) * np.real(phi * np.exp(-1j * eng.u * eng.a))
    c[0] = 1.0 / eng.L
    return c


def _int_cos(eng: _CosEngine, x1: float, x2: float) -> np.ndarray:
    """∫_{x1}^{x2} cos(u_k (x-a)) dx (k=0 は長さ)。"""
    out = np.empty(eng.N)
    out[0] = x2 - x1
    un = eng.u[1:]
    out[1:] = (np.sin(un * (x2 - eng.a)) - np.sin(un * (x1 - eng.a))) / un
    return out


def _int_ramp_cos(eng: _CosEngine, x1: float, x2: float) -> np.ndarray:
    """∫_{x1}^{x2} (x-x1) cos(u_k (x-a)) dx。部分積分の閉形式。"""
    out = np.empty(eng.N)
    out[0] = 0.5 * (x2 - x1) ** 2
    un = eng.u[1:]
    out[1:] = ((x2 - x1) * np.sin(un * (x2 - eng.a)) / un
               + (np.cos(un * (x2 - eng.a)) - np.cos(un * (x1 - eng.a))) / un ** 2)
    return out


def _pw_linear_coeffs(eng: _CosEngine, xs, ys) -> np.ndarray:
    """区分線形関数の半区間余弦係数。xs の外側は端の値で定数外挿する。

    xs は非減少。同じ x が 2 つ並ぶ区間は長さ 0 として飛ばすので、そこが跳びになる
    (W_b の x = D での「討伐 ⇒ 残り時間 0」の跳びを表現する)。
    """
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    acc = np.zeros(eng.N)
    if xs[0] > eng.a:                                   # 左の定数部 [a, xs[0]]
        acc += ys[0] * _int_cos(eng, eng.a, xs[0])
    for i in range(len(xs) - 1):                        # 線形部
        x1, x2 = xs[i], xs[i + 1]
        if x2 <= x1:
            continue
        x1c, x2c = max(x1, eng.a), min(x2, eng.b_pad)
        if x2c <= x1c:
            continue
        m = (ys[i + 1] - ys[i]) / (x2 - x1)
        y1c = ys[i] + m * (x1c - x1)
        acc += y1c * _int_cos(eng, x1c, x2c) + m * _int_ramp_cos(eng, x1c, x2c)
    if xs[-1] < eng.b_pad:                              # 右の定数部
        acc += ys[-1] * _int_cos(eng, max(xs[-1], eng.a), eng.b_pad)
    out = (2.0 / eng.L) * acc
    out[0] = acc[0] / eng.L
    return out


# ---------------------------------------------------------------------------
# バックエンド: 価値関数と密度の演算
#
# _SaveDP はこの API しか使わない。価値 V と密度 F の内部表現はバックエンド任意
# (COS は (余弦係数, 正弦係数) の対、グリッドは配列)。expect / push は「打ち切り
# 直後の表現」にだけ適用される (COS の正弦成分が 0 であることを前提にできる)。
# ---------------------------------------------------------------------------
class _CosBackend:
    """係数空間 (COS/F&O) バックエンド。増分が状態に依らないモデル (和/積) 用。"""

    n_particles = 33

    def __init__(self, segs):
        self.eng = _CosEngine(segs)
        self.seg_lo = [float(s.s_lo) for s in segs]
        self.seg_hi = [float(s.s_hi) for s in segs]
        self.n_segs = len(segs)
        self._cum_lo = np.cumsum(self.seg_lo)
        self._cum_hi = np.cumsum(self.seg_hi)
        # tail_hi[p] = Σ_{i>=p} s_hi (増分が加法的なので単純な後ろ向き累和)
        rev = np.cumsum(self.seg_hi[::-1])[::-1]
        self._tail_hi = np.concatenate([rev, [0.0]])
        self.clip_lo, self.clip_hi = float(self.eng.a), float(self.eng.b)
        self._zero = np.zeros(self.eng.N)

    # ---- 状態の範囲 ----
    def cum_lo(self, cp: int) -> float:
        return float(self._cum_lo[cp - 1])

    def cum_hi(self, cp: int) -> float:
        return float(self._cum_hi[cp - 1])

    def gate_floor(self, cp: int, D: float) -> float:
        """関門 cp で、これ未満だと以後どう転んでも目標に届かない累積ダメージ。"""
        return max(0.0, D - float(self._tail_hi[cp]))

    # ---- 価値関数 ----
    def indicator_ge(self, D: float):
        return (self.eng.indicator_coeffs(D), self._zero)

    def pw_linear(self, xs, ys):
        return (_pw_linear_coeffs(self.eng, xs, ys), self._zero)

    def add_const(self, V, c: float):
        C = V[0].copy()
        C[0] += c
        return (C, V[1])

    def scale(self, V, a: float):
        return (a * V[0], a * V[1])

    def expect(self, V, j: int):
        return self.eng.expect(V[0], j)      # V は打ち切り直後 = 余弦のみ

    def truncate_ge(self, V, d: float):
        return (self.eng.truncate(V[0], V[1], d), self._zero)

    def eval(self, V, x: float) -> float:
        return self.eng.eval(V[0], V[1], x)

    # ---- 密度 ----
    def start_density(self, j: int, s: float):
        return (_shifted_density(self.eng, j, s), self._zero)

    def push(self, F, j: int):
        return self.eng.convolve(F[0], j)    # F は打ち切り直後 = 余弦のみ

    def mass_ge(self, F, d: float) -> float:
        return float(self.eng.integrate(F[0], F[1], d))

    def truncate_density_ge(self, F, d: float):
        return self.truncate_ge(F, d)

    def zero_density(self):
        return (np.zeros(self.eng.N), self._zero)

    def axpy(self, acc, F, w: float):
        return (acc[0] + w * F[0], acc[1] + w * F[1])


class _BlockStarter:
    """セグメント先頭ブロックを「状態 s から」適用したときの増分密度。

    s=0 で 1 度だけ厳密 pdf (+ 純原子) を作り、s>0 へは相似変換で移す。

      通常 (z) ブロック: s' = s + T なので増分分布は s に依らない (k=1)。
      HP依存 (y) ブロック: s' = s + (H̃₁ - s)(1 - π) で、π の分布は s に依らない。
        よって増分は s=0 のときの k = (H̃₁ - s)/H̃₁ 倍の相似変換になる
        (docs/cutoff.md §3.1 の「現在 HP からの再スタート」表現)。

    グリッドに点質量を落とすのではなく厳密 pdf から立ち上げるのは、
    mixed.first_block_density と同じ理由 (点質量の格子量子化を避ける)。
    """

    def __init__(self, block, hp: HPParams, n_u: int = 4001):
        kind, mixes = block
        self.kind = kind
        self.Htil = hp.Htil if hp is not None else 0.0
        if kind == "z":
            lo = sum(min(u.lo for u in m) for m in mixes)
            hi = sum(max(u.hi for u in m) for m in mixes)
            if hi - lo <= 1e-9:
                self.u = None
                self.atoms = [(0.5 * (lo + hi), 1.0)]
                return
            dist = build_sum_dist(mixes)
            self.u = np.linspace(lo, hi, n_u)
            self.pdf = np.maximum(_eval_chunked(dist.pdf, self.u), 0.0)
            self.atoms = ([] if dist.av is None
                          else [(float(v), float(p))
                                for v, p in zip(dist.av, dist.ap)])
            return
        A, B = support_bounds_hits(mixes)
        if abs(self.Htil) * abs(math.exp(B) - math.exp(A)) <= 1e-9 or B - A <= 1e-12:
            pi = math.exp(0.5 * (A + B))
            self.u = None
            self.atoms = [(self.Htil * (1.0 - pi), 1.0)]
            return
        pd = build_product_dist(mixes, hp)
        d1 = self.Htil * (1.0 - math.exp(A))
        d2 = self.Htil * (1.0 - math.exp(B))
        self.u = np.linspace(min(d1, d2), max(d1, d2), n_u)
        self.pdf = np.maximum(_eval_chunked(pd.pdf, self.u), 0.0)
        self.atoms = ([] if pd.av is None
                      else [(self.Htil * (1.0 - math.exp(float(v))), float(p))
                            for v, p in zip(pd.av, pd.ap)])

    def scale_at(self, s: float) -> float:
        """状態 s での相似比 k。z ブロックは 1、y ブロックは (H̃₁ - s)/H̃₁。"""
        if self.kind == "z" or self.Htil == 0.0:
            return 1.0
        return max((self.Htil - s) / self.Htil, 1e-12)

    def density(self, grid: np.ndarray, s: float) -> np.ndarray:
        k = self.scale_at(s)
        f = np.zeros_like(grid)
        if self.u is not None:
            f += np.interp((grid - s) / k, self.u, self.pdf,
                           left=0.0, right=0.0) / k
        for v, p in self.atoms:
            _deposit(f, grid, s + k * v, p)
        return f


class _GridBackend:
    """アフィンカーネルのグリッドバックエンド。混在モデル (HP依存 + 通常) 用。

    増分が状態に依存する (s' = a s + c) ため係数空間では閉じない (docs/mixed.md §5)。
    価値関数は粗いグリッド (方策の決定にしか使わない)、密度は細かいグリッド
    (裾質量の精度が通過率に効く) と、restart_mixed と同じ二段構えにする。
    """

    n_particles = 13          # 前向きはカーネル適用が O(ノード数×格子数) で重い

    def __init__(self, seg_blocks, specs, bounds, hp: HPParams, s_hi: float):
        self.hp = hp
        self.seg_blocks = seg_blocks
        self.n_segs = len(seg_blocks)
        self.Htil = hp.Htil
        self.grid_b = np.linspace(0.0, s_hi, _GRID_N_BACKWARD)
        step_b = self.grid_b[1] - self.grid_b[0]
        self.kern_b = [[block_kernel(b, hp, step_b, _CELL_FACTOR_BACKWARD)
                        for b in blks] for blks in seg_blocks]
        self.pos_b = [[kernel_positions(self.grid_b, k) for k in ks]
                      for ks in self.kern_b]
        # 前向きグリッド: 最小ブロック幅を ~40 点で解像する (restart_mixed の半分。
        # 粒子ごとに 1 本ずつ流すので刻みを欲張ると実用時間に収まらない)
        widths = [w for blks in seg_blocks for w in
                  (_block_width(b, hp) for b in blks) if w > 0]
        n_f = _GRID_N_FORWARD
        if widths and s_hi > 0:
            n_f = int(np.clip(math.ceil(s_hi / (min(widths) / 40.0)) + 1,
                              _GRID_N_FORWARD, _GRID_N_FORWARD_MAX))
        self.grid_f = np.linspace(0.0, s_hi, n_f)
        step_f = self.grid_f[1] - self.grid_f[0]
        self.kern_f = [[block_kernel(b, hp, step_f) for b in blks]
                       for blks in seg_blocks]
        self.starters = [_BlockStarter(blks[0], hp) for blks in seg_blocks]
        # 状態の範囲: 累積の厳密な台 (H̃ の区間演算)。増分は状態依存なので
        # 単純な累和では出せない。
        self._cum = [mixed_support(specs[:e], hp) for e in bounds]
        self._specs = specs
        self._bounds = bounds
        self.clip_lo, self.clip_hi = 0.0, float(s_hi)

    # ---- 状態の範囲 ----
    def cum_lo(self, cp: int) -> float:
        return float(self._cum[cp][0])

    def cum_hi(self, cp: int) -> float:
        return float(self._cum[cp][1])

    def _max_final(self, cp: int, x: float) -> float:
        """関門 cp で累積 x のとき、以後どう転んでも超えられない最終ダメージ。

        シフト HP h = H̃₁ - x を残りヒットで最小化する (各段は h について単調増加
        なので貪欲で最適)。h·y は y の端 2 通りの小さいほうを取れば符号に依らない。
        """
        h = self.Htil - x
        for is_hp, mix in self._specs[self._bounds[cp]:]:
            if is_hp:
                ym = y_mixture(mix, self.hp.beta)
                y_lo = min(u.lo for u in ym)
                y_hi = max(u.hi for u in ym)
                h = min(h * y_lo, h * y_hi)
            else:
                h -= max(u.hi for u in mix)
        return self.Htil - h

    def gate_floor(self, cp: int, D: float) -> float:
        """max_final(cp, ·) は x について単調増加なので二分法で逆に解く。"""
        hi = self.cum_hi(cp)
        if self._max_final(cp, 0.0) >= D:
            return 0.0
        if self._max_final(cp, hi) < D:
            return hi                    # ここから先はどうやっても到達不能
        lo = 0.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if self._max_final(cp, m) >= D:
                hi = m
            else:
                lo = m
        return 0.5 * (lo + hi)

    # ---- 価値関数 (粗グリッド) ----
    def indicator_ge(self, D: float):
        return (self.grid_b >= D).astype(float)

    def pw_linear(self, xs, ys):
        return np.interp(self.grid_b, np.asarray(xs, dtype=float),
                         np.asarray(ys, dtype=float))

    def add_const(self, V, c: float):
        return V + c

    def scale(self, V, a: float):
        return a * V

    def expect(self, V, j: int):
        W = V
        for k, pos in zip(reversed(self.kern_b[j]), reversed(self.pos_b[j])):
            W = kernel_backward(W, self.grid_b, k, pos)
        return W

    def truncate_ge(self, V, d: float):
        return np.where(self.grid_b >= d, V, 0.0)

    def eval(self, V, x: float) -> float:
        return float(np.interp(x, self.grid_b, V))

    # ---- 密度 (細グリッド) ----
    def start_density(self, j: int, s: float):
        f = self.starters[j].density(self.grid_f, s)
        for k in self.kern_f[j][1:]:
            f = kernel_forward(f, self.grid_f, k)
        return f

    def push(self, F, j: int):
        for k in self.kern_f[j]:
            F = kernel_forward(F, self.grid_f, k)
        return F

    def mass_ge(self, F, d: float) -> float:
        return _mass_above(F, self.grid_f, d)

    def truncate_density_ge(self, F, d: float):
        return _truncate_below(F, self.grid_f, d)

    def zero_density(self):
        return np.zeros_like(self.grid_f)

    def axpy(self, acc, F, w: float):
        return acc + w * F


def _block_width(block, hp: HPParams) -> float:
    """ブロック増分のダメージ規模 (前向きグリッドの解像度決定用)。"""
    kind, mixes = block
    if kind == "z":
        lo = sum(min(u.lo for u in m) for m in mixes)
        hi = sum(max(u.hi for u in m) for m in mixes)
        return hi - lo
    lo = sum(math.log(min(u.lo for u in m)) for m in mixes)
    hi = sum(math.log(max(u.hi for u in m)) for m in mixes)
    return abs(hp.Htil) * abs(math.exp(hi) - math.exp(lo))


# ---------------------------------------------------------------------------
# ブロック方策 (足切り曲線 d_j(s))
# ---------------------------------------------------------------------------
class _Policy:
    """1 凸ブロックの足切り曲線。セーブ地点の累積ダメージ s から関門値を引く。

    s_knots は昇順、gates[j] は同じ長さの配列。ブロック 1 は s=0 固定なので節点 1 個。
    """

    def __init__(self, s_knots, gates):
        self.s = np.asarray(s_knots, dtype=float)
        self.gates = {int(j): np.asarray(v, dtype=float) for j, v in gates.items()}

    def at(self, s: float) -> dict:
        if self.s.size == 1:
            return {j: float(v[0]) for j, v in self.gates.items()}
        return {j: float(np.interp(float(s), self.s, v)) for j, v in self.gates.items()}

    def stats(self, samples, weights) -> dict:
        """粒子 (s, w) 上での関門の重み付き平均・最小・最大。"""
        out = {}
        w = np.asarray(weights, dtype=float)
        tot = w.sum()
        for j in self.gates:
            vals = np.array([self.at(s)[j] for s in samples], dtype=float)
            mean = float(vals @ w / tot) if tot > 0 else float(vals[0])
            out[j] = (mean, float(vals.min()), float(vals.max()))
        return out


def _const_policy(gate_map: dict) -> _Policy:
    return _Policy([0.0], {j: [g] for j, g in gate_map.items()})


# ---------------------------------------------------------------------------
# セーブ点付き DP 本体 (バックエンド非依存)
# ---------------------------------------------------------------------------
class _SaveDP:
    """凸区切り付きの最小期待時間 DP。

    save_cps はセーブ点になる関門番号 (1..K) の昇順リスト。ブロック b は区間
    p_b..q_b を持ち、p_1 = 0、p_{b+1} = save_cps[b-1]。
    """

    def __init__(self, be, times, succ, D: float, save_cps):
        self.be = be
        self.times = list(times)
        self.succ = list(succ)
        self.D = float(D)
        self.K = be.n_segs - 1
        ps = [0, *[int(c) for c in save_cps]]
        self.blocks = [(ps[i], (ps[i + 1] - 1 if i + 1 < len(ps) else self.K))
                       for i in range(len(ps))]
        # 関門 cp の到達可能下限 (cp = 1..K+1、K+1 は最終の目標判定)
        self.floor = [0.0] + [be.gate_floor(cp, self.D)
                              for cp in range(1, self.K + 2)]
        # 到達不能領域の頭打ち値。基準時間スケール (全区間 1 周 / 独立成功確率) の倍数。
        qq = max(float(np.prod(self.succ)), 1e-6)
        self.r_cap = _R_CAP_FACTOR * max(sum(self.times), 1e-9) / qq
        # 所要時間 0 の凸ブロックは「タダで何度でも引き直せる」縮退を招き、
        # 不動点 Φ_R(s)=R が R=0 に潰れる (中断が無コストなので最良の目を引くまで
        # 引き直すのが最適になる)。凸区切りを入れる以上、各凸には正の時間が要る。
        for b, (p, q) in enumerate(self.blocks):
            if sum(self.times[p:q + 1]) <= 0.0:
                raise ValueError(
                    f"{b + 1}凸目の所要時間が 0 です。凸区切りで区切られた各凸には"
                    "正の所要時間を設定してください。")

    def gate_floor(self, cp: int) -> float:
        return self.floor[cp]

    # ---- 後ろ向き帰納 (中断価値 R を固定) ----
    def block_backward(self, b: int, R: float, w_next, fixed: dict | None = None):
        """ブロック b を中断価値 R で解く。(gates, phi_V, phi_const) を返す。

        Φ_R(s) = phi_const + succ[p] · eval(phi_V, s)。

        fixed に {関門: 足切り値} を渡すと最適化せずその関門で評価する(方策評価)。
        漸化式は同じで、無差別点を探す代わりに与えられた値で打ち切るだけ。
        """
        be, (p, q) = self.be, self.blocks[b]
        gates = {}
        if b == len(self.blocks) - 1:
            # 終端: 最終区間まで回して目標未達なら最後の凸をやり直す V = R·1{x<D}
            V = be.add_const(be.scale(be.indicator_ge(self.D), -R), R)
        else:
            # セーブ点 q+1 の関門: min(R, W_{b+1}(x))。W_{b+1} は x について非増加。
            H = be.add_const(w_next, -R)
            d = self._gate(q + 1, H, fixed, floor=True)
            gates[q + 1] = d
            V = be.add_const(be.truncate_ge(H, d), R)
        for j in range(q, p, -1):
            H = be.scale(be.expect(V, j), self.succ[j])
            H = be.add_const(H, self.times[j] - self.succ[j] * R)  # H_j = cont_j - R
            d = self._gate(j, H, fixed)
            gates[j] = d
            V = be.add_const(be.truncate_ge(H, d), R)
        phi_V = be.expect(V, p)
        phi_const = self.times[p] + (1.0 - self.succ[p]) * R
        return gates, phi_V, phi_const

    def _gate(self, cp: int, V, fixed, floor: bool = False) -> float:
        """関門 cp の足切り。fixed があればその値(セーブ点だけ到達可能下限でクリップ)。"""
        if fixed is None or cp not in fixed:
            return self._first_below(cp, V)
        d = float(fixed[cp])
        if floor:
            d = max(d, self.gate_floor(cp))   # 到達不能な確定は期待時間が発散する
        return float(np.clip(d, self.be.clip_lo, self.be.clip_hi))

    def _first_below(self, cp: int, V) -> float:
        """関門 cp の足切り = 減少関数 f(x)=eval(V,x) が初めて <=0 になる x。

        探索範囲は [到達可能下限, 物理上限]。下限未満では続行しても永久に完走でき
        ない(価値 +∞)ので、これは厳密な制約であると同時に、W_b の発散域を数値表現で
        なぞる必要をなくして誤判定を構造的に排除する。
        """
        lo = self.gate_floor(cp)
        hi = self.be.cum_hi(cp)
        ev = self.be.eval
        if hi <= lo:
            return lo
        if ev(V, lo) <= 0.0:
            return lo
        if ev(V, hi) > 0.0:
            return hi                     # どこでも続行不利 (事実上到達不能)
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if ev(V, m) <= 0.0:
                hi = m
            else:
                lo = m
        return 0.5 * (lo + hi)

    def _phi(self, b: int, R: float, s: float, w_next, fixed=None) -> float:
        _g, V, k = self.block_backward(b, R, w_next, fixed)
        return k + self.succ[self.blocks[b][0]] * self.be.eval(V, s)

    # ---- 固定方策なら Φ は R について線形 → 後ろ向き 2 回で W(s) が全域出る ----
    def _affine_value(self, b: int, w_next, fixed: dict):
        """(gates, W) を返す。W は s -> W_b(s) の関数。

        関門を固定すると中断確率が R に依らないので Φ_R(s) = A(s) + ρ(s)·R。
        不動点は A/(1-ρ) で、掃引も二分法も要らない。
        """
        g0, V0, k0 = self.block_backward(b, 0.0, w_next, fixed)
        _g1, V1, k1 = self.block_backward(b, 1.0, w_next, fixed)
        sp = self.succ[self.blocks[b][0]]
        ev = self.be.eval

        def W(s: float) -> float:
            p0 = k0 + sp * ev(V0, s)
            rho = (k1 + sp * ev(V1, s)) - p0
            if rho >= 1.0 - 1e-12:
                return self.r_cap
            return float(min(max(p0 / (1.0 - rho), 0.0), self.r_cap))

        return g0, W

    # ---- 不動点 Φ_R(s) = R ----
    def fixed_point(self, b: int, s: float, w_next, fixed=None,
                    iters: int = _FIX_ITERS) -> float:
        """与えた s での W_b(s)。到達不能なら r_cap を返す。"""
        if fixed is not None:
            return self._affine_value(b, w_next, fixed)[1](s)
        lo = 0.0
        hi = max(self._phi(b, 0.0, s, w_next), 1e-12) * 2.0
        for _ in range(_BRACKET_DOUBLES):
            if self._phi(b, hi, s, w_next) <= hi:
                break
            lo = hi
            hi *= 2.0
            if hi >= self.r_cap:
                return self.r_cap
        else:
            return self.r_cap
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self._phi(b, mid, s, w_next) > mid:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ---- ブロック b (b>=1) の W_b 曲線と足切り曲線 ----
    def _s_range(self, b: int):
        p, _q = self.blocks[b]
        eps = 1e-9 * max(self.D, 1.0)
        s_lo_raw = self.be.cum_lo(p)
        s_hi_raw = self.be.cum_hi(p)
        s_doom = self.gate_floor(p)                   # これ未満は到達不能
        lo = max(s_lo_raw, s_doom + eps)
        hi = min(s_hi_raw, self.D - eps)
        if hi <= lo:
            hi = lo + eps
        return s_lo_raw, s_hi_raw, s_doom, lo, hi, eps

    def solve_block(self, b: int, w_next, fixed=None):
        s_lo_raw, _s_hi_raw, s_doom, lo, hi, eps = self._s_range(b)
        search_lo = max(min(s_lo_raw, lo), s_doom)
        if fixed is not None:
            gates, W = self._affine_value(b, w_next, fixed)
            xs = list(np.linspace(search_lo, hi, _N_EVAL_KNOTS))
            ys = list(np.minimum.accumulate([W(float(x)) for x in xs]))
            gs = {j: [g] for j, g in gates.items()}
            return self._w_repr(xs, ys, eps), _Policy([0.0], gs)

        # 端点は掃引範囲を決めるだけなので粗くてよい (細分が本体を担う)
        r_hi = self.fixed_point(b, lo, w_next, iters=_RANGE_ITERS)
        r_lo = self.fixed_point(b, hi, w_next, iters=_RANGE_ITERS)
        r_lo = max(r_lo, 1e-12)
        r_hi = max(r_hi, r_lo * (1.0 + 1e-9))
        pts = [self._sweep_point(b, float(R), w_next, search_lo, hi)
               for R in np.geomspace(r_lo, r_hi, _N_SWEEP)]
        pts.sort(key=lambda t: t[0])
        # 適応細分。W_b が s について平らな領域では R → s が悪条件で節点が飛び、
        # 凸な W を粗い折れ線で結ぶと系統的に上振れする (中断価値が過大に見積もら
        # れ、外側のブロックの期待時間まで押し上げる)。s のギャップが大きいところを
        # R の幾何中点で埋める。s(R) は単調なので新しい根は必ずギャップの内側に入り、
        # 1 点あたり後ろ向き帰納 1 回で済む (不動点の解き直しは不要)。
        for _ in range(_N_REFINE):
            span = pts[-1][0] - pts[0][0]
            if span <= 0:
                break
            widths = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
            i = int(np.argmax(widths))
            if widths[i] <= 1.5 * span / len(pts):
                break
            r_mid = math.sqrt(max(pts[i][1] * pts[i + 1][1], 1e-300))
            new_pt = self._sweep_point(b, r_mid, w_next, search_lo, hi)
            if not (pts[i][0] < new_pt[0] < pts[i + 1][0]):
                break                      # 悪条件で内側に入らない → これ以上刻めない
            pts.insert(i + 1, new_pt)
        xs, ys, gs = [], [], {j: [] for j in pts[0][2]}
        for sv, rv, g in pts:
            if xs and sv <= xs[-1] + 1e-12:
                continue
            xs.append(sv)
            ys.append(float(rv))
            for j in gs:
                gs[j].append(g[j])
        if len(xs) < 2:                       # 退化 (掃引が 1 点に潰れた)
            base_x = xs[0] if xs else lo
            base_y = ys[0] if ys else r_hi
            xs = [base_x, base_x + eps]
            ys = [base_y, base_y]
            for j in gs:
                gs[j] = [gs[j][0] if gs[j] else 0.0] * 2
        return self._w_repr(xs, ys, eps), _Policy(xs, gs)

    def _sweep_point(self, b: int, R: float, w_next, search_lo: float, hi: float):
        """中断価値 R での後ろ向き 1 回。(Φ_R(s)=R の根 s, R, 関門) を返す。

        Φ_R(s) - R は s について減少なので根は一意。両端で符号が変わらないときは
        端に張り付ける (その R は探索範囲の外側に対応する)。
        """
        g, V, k = self.block_backward(b, R, w_next)
        sp = self.succ[self.blocks[b][0]]

        def gap(x):
            return k + sp * self.be.eval(V, x) - R

        if gap(search_lo) <= 0.0:
            return (search_lo, R, g)
        if gap(hi) >= 0.0:
            return (hi, R, g)
        a_, b_ = search_lo, hi
        for _ in range(_FIX_ITERS):
            m = 0.5 * (a_ + b_)
            if gap(m) > 0.0:
                a_ = m
            else:
                b_ = m
        return (0.5 * (a_ + b_), R, g)

    def _w_repr(self, xs, ys, eps):
        """W_b の節点をバックエンドの価値表現へ。x >= D は討伐済みで 0 (跳び)。"""
        w_xs = list(xs) + [max(xs[-1], self.D - eps), self.D]
        w_ys = list(ys) + [ys[-1], 0.0]
        return self.be.pw_linear(w_xs, w_ys)

    # ---- 全ブロックを後ろから解く / 方策を評価する ----
    def evaluate(self, fixed: dict | None = None):
        """(期待総時間 W_1(0), 各ブロックの方策, 各ブロックの W_b) を返す。

        fixed=None なら最適方策。fixed に {関門: 足切り値} を渡すとその方策の
        期待総時間を厳密に評価する(「足切り無し」基準・手動調整の評価に使う)。
        """
        M = len(self.blocks)
        policies = [None] * M
        w_reps = [None] * M
        w_next = None
        for b in range(M - 1, 0, -1):
            w_next, policies[b] = self.solve_block(b, w_next, fixed)
            w_reps[b] = w_next
        # ブロック 1 は s = 0 固定のスカラー不動点
        r0 = self.fixed_point(0, 0.0, w_next, fixed)
        g0, _V, _k = self.block_backward(0, r0, w_next, fixed)
        policies[0] = _const_policy(g0)
        return r0, policies, w_reps

    # ---- 「足切り無し」基準の関門 ----
    def baseline_gates(self, gate_stat: dict) -> dict:
        """「足切り無し」基準の関門。凸の中では一切足切りしない (しきい値 0)。

        凸区切りがある場合、区切りの関門まで 0 にすると基準の期待時間が発散する
        (到達不能な累積ダメージを確定させると二度と目標に届かず、しかもその近傍の
        質量 × 1/完走確率 の積分が実際に発散する)。そこで区切りの関門だけは評価
        対象の方策の値を残し、「凸の中での足切りをやめたら何倍かかるか」を測る。
        区切りが無ければ全関門 0 = 従来の「足切り無し」(期待時間 Σt/P) に一致する。
        """
        out = {}
        n_blocks = len(self.blocks)
        for b, (p, q) in enumerate(self.blocks):
            for j in range(p + 1, q + 1):
                out[j] = 0.0
            if b < n_blocks - 1:
                out[q + 1] = float(gate_stat.get(q + 1, (0.0, 0.0, 0.0))[0])
        return out

    # ---- 前向きパス (通過率・セーブ地点分布・ブロック別内訳) ----
    def forward(self, policies, w_reps=None, dp_time=None, n_particles=None):
        be, D = self.be, self.D
        n_particles = n_particles or be.n_particles
        parts = [(0.0, 1.0)]
        exp_time = 0.0
        pass_rate = {}            # cp -> 重み付き累積通過率 (ブロック内)
        gate_stat = {}            # cp -> (mean, lo, hi)
        blocks_info = []
        enter_value = []          # ブロック b 開始時点の期待残り総時間 E_b
        dead_w = 0.0              # 完走確率 0 の状態に居る確率質量
        for b, (p, q) in enumerate(self.blocks):
            is_last = (b == len(self.blocks) - 1)
            end_cp = q + 1
            active = [(s, w) for (s, w) in parts if s < D]
            w_act = sum(w for _s, w in active)
            if w_act <= 0:
                blocks_info.append({"block": b + 1, "segments": [p, q],
                                    "exp_time": 0.0, "attempts": 0.0,
                                    "completion": 1.0, "weight": 0.0})
                enter_value.append(0.0)
                continue
            gate_stat.update(policies[b].stats([s for s, _w in active],
                                               [w for _s, w in active]))
            if b == 0:
                enter_value.append(dp_time)
            elif w_reps is not None and w_reps[b] is not None:
                enter_value.append(sum(w * max(be.eval(w_reps[b], s), 0.0)
                                       for s, w in parts))
            else:
                enter_value.append(None)
            mix = be.zero_density()
            lo_end = float("inf")     # このブロックで実際に使った終端関門の最小値
            acc_time = acc_comp = nxt_w = 0.0
            for (s, w) in active:
                g = policies[b].at(s)
                F = be.start_density(p, s)
                t1 = self.times[p]
                qc = self.succ[p]
                for j in range(p + 1, q + 1):
                    pdmg = _clamp01(be.mass_ge(F, g[j]))
                    pass_rate[j] = pass_rate.get(j, 0.0) + w * _clamp01(qc * pdmg)
                    t1 += self.times[j] * _clamp01(qc * pdmg)
                    F = be.push(be.truncate_density_ge(F, g[j]), j)
                    qc *= self.succ[j]
                d_end = D if is_last else g[end_cp]
                lo_end = min(lo_end, d_end)
                pdmg = _clamp01(be.mass_ge(F, d_end))
                comp = _clamp01(qc * pdmg)
                pass_rate[end_cp] = pass_rate.get(end_cp, 0.0) + w * comp
                if comp <= 1e-15:
                    dead_w += w          # この状態からは目標に届かない
                    tau = 0.0
                else:
                    tau = t1 / comp
                acc_time += w * tau
                acc_comp += w * comp
                if not is_last and pdmg > 1e-15:
                    mix = be.axpy(mix, be.truncate_density_ge(F, d_end), w / pdmg)
                    nxt_w += w
            exp_time += acc_time
            comp_avg = acc_comp / w_act if w_act > 0 else 0.0
            blocks_info.append({
                "block": b + 1, "segments": [p, q],
                "exp_time": acc_time, "weight": w_act,
                "completion": comp_avg,
                "attempts": (1.0 / comp_avg) if comp_avg > 0 else float("inf"),
            })
            for j in list(range(p + 1, q + 1)) + [end_cp]:   # 稼働重みで正規化
                if j in pass_rate:
                    pass_rate[j] = pass_rate[j] / w_act
            if is_last:
                break
            done = [(s, w) for (s, w) in parts if s >= D]
            parts = done + self._resample(mix, nxt_w, end_cp, n_particles, lo_end)
        # ブロック別の期待時間は「入口の期待残り総時間の差」で出す。粒子上の
        # Σ w·τ_b(s) は τ_b = (1試行の時間)/(完走確率) で完走確率が小さい領域が
        # 急峻になり、粗い求積では大きく外れる。E_b は有界で滑らかなので安定で、
        # かつ E_1 = W_1(0)・E_{M+1} = 0 を端に固定すれば総和が厳密に一致する。
        if dp_time is not None and len(enter_value) == len(blocks_info):
            ev = list(enter_value) + [0.0]
            ev[0] = dp_time
            for i, info in enumerate(blocks_info):
                if ev[i] is not None and ev[i + 1] is not None:
                    info["exp_time"] = max(ev[i] - ev[i + 1], 0.0)
        return {"fwd_exp_time": exp_time, "pass_rate": pass_rate,
                "gate_stat": gate_stat, "blocks": blocks_info,
                "dead_weight": dead_w, "feasible": dead_w <= 1e-6}

    def _resample(self, mix, weight: float, cp: int, n: int,
                  lo_end: float = float("-inf")):
        """セーブ点 (関門 cp) の累積ダメージ密度を n 粒子に落とす。

        台の左端は実際に使った終端関門。ここより下はサブ密度が厳密に 0 で、数値
        表現の振動だけが乗る (そこは完走確率も 0 なので粒子を置くと「到達不能な
        質量」を偽って作ってしまう)。重みも点評価の pdf ではなくセル質量で取る。
        """
        if weight <= 0:
            return []
        lo = max(self.be.cum_lo(cp), self.gate_floor(cp), float(lo_end))
        hi = self.be.cum_hi(cp)
        if hi <= lo:
            return [(hi, weight)]
        edges = np.linspace(lo, hi, n + 1)
        above = np.array([self.be.mass_ge(mix, float(x)) for x in edges])
        w = np.maximum(above[:-1] - above[1:], 0.0)
        tot = w.sum()
        if tot <= 0:
            return [(0.5 * (lo + hi), weight)]
        keep = w >= 1e-9 * tot
        if not keep.any():
            return [(0.5 * (lo + hi), weight)]
        xs = 0.5 * (edges[:-1] + edges[1:])
        w = w[keep] / w[keep].sum() * weight
        return [(float(x), float(wi)) for x, wi in zip(xs[keep], w)]


# ---------------------------------------------------------------------------
# 結果 dict の組み立てと共通ドライバ
# ---------------------------------------------------------------------------
def _build_result(n, cps, sps, D, dp_time, base_time, fwd, base_fwd,
                  cum_max_vals, to_dmg=None):
    """restart._result と互換の形 (+ 凸区切り用のキー) を返す。

    期待時間は前向きパスではなく DP の不動点 W_1(0) を正とする。前向きは通過率と
    足切りの代表値・幅を出すためのもので、セーブ地点の粒子離散化を含むため、
    完走確率が小さい領域 (1/完走確率 が急峻) では期待時間の求積として粗い。
    to_dmg: 作業座標 (積モデルの G) からダメージへの逆写像。None なら恒等。
    """
    conv = to_dmg or (lambda x: float(x))
    save_set = {int(s) for s in sps}
    rows = []
    for k, m in enumerate(cps, start=1):
        mean, glo, ghi = fwd["gate_stat"].get(k, (0.0, 0.0, 0.0))
        rows.append({
            "checkpoint": m,
            "gate": conv(mean),
            "gate_lo": conv(glo),
            "gate_hi": conv(ghi),
            "save": m in save_set,
            "pass_rate": _clamp01(fwd["pass_rate"].get(k, 0.0)),
            "cum_max_at_cp": cum_max_vals[k - 1],
        })
    speedup = (base_time / dp_time) if dp_time > 0 else float("nan")
    final_cp = len(cps) + 1
    success = _clamp01(fwd["pass_rate"].get(final_cp, 0.0))
    return {
        "n_hits": n,
        "checkpoints": list(cps),
        "save_points": sorted(save_set),
        "D": D,
        "rows": rows,
        "throughput": (1.0 / dp_time) if dp_time > 0 else 0.0,
        "success": success,
        "exp_time": dp_time,
        "baseline": {
            "success": _clamp01(base_fwd["pass_rate"].get(final_cp, 0.0)),
            "exp_time": base_time,
            "g": (1.0 / base_time) if base_time > 0 else 0.0,
        },
        "speedup": speedup,
        "g_star_dp": (1.0 / dp_time) if dp_time > 0 else float("nan"),
        "fwd_exp_time": fwd["fwd_exp_time"],
        "blocks": fwd["blocks"],
        "feasible": fwd["feasible"],
        "has_save": bool(save_set),
    }


def _cp_index(cps, save_points):
    """セーブ点のヒット数を関門番号 (1..K) に変換する。cps に無いものは無視。"""
    pos = {int(m): i + 1 for i, m in enumerate(cps)}
    out = sorted({pos[int(s)] for s in (save_points or []) if int(s) in pos})
    return [j for j in out if j <= len(cps)]


def _run(dp: _SaveDP, manual_work_gates):
    """最適 (または手動固定) 方策を解き、前向き指標と基準を揃えて返す。"""
    fixed = (None if manual_work_gates is None
             else {j: float(g) for j, g in enumerate(manual_work_gates, start=1)})
    dp_time, policies, w_reps = dp.evaluate(fixed)
    fwd = dp.forward(policies, w_reps, dp_time)
    base_fixed = dp.baseline_gates(fwd["gate_stat"])
    base_time, base_pols, base_w = dp.evaluate(base_fixed)
    base_fwd = dp.forward(base_pols, base_w, base_time)
    return dp_time, base_time, fwd, base_fwd


# ---------------------------------------------------------------------------
# エントリ: 和モデル
# ---------------------------------------------------------------------------
def analyze(hit_mixtures, checkpoints, save_points, hit_times, D,
            manual_gates=None, seg_success=None):
    """和モデルの凸区切り付き足切り最適化。

    save_points : セーブ点になるチェックポイントの累積ヒット数 (checkpoints の部分集合)
    manual_gates: 各チェックポイントの累積ダメージしきい値 (長さ K)。渡すと最適化せず
                  その方策で評価する (手動調整用)。
    """
    n = len(hit_mixtures)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    bounds, cps = _split_bounds(n, checkpoints)
    times = _seg_times(hit_times, bounds)
    succ = _seg_success(seg_success, len(bounds) - 1)
    be = _CosBackend(_segs_sum(hit_mixtures, bounds))
    sps = _cp_index(cps, save_points)
    dp = _SaveDP(be, times, succ, float(D), sps)

    mg = None if manual_gates is None else [float(g) for g in manual_gates]
    dp_time, base_time, fwd, base_fwd = _run(dp, mg)
    save_hits = [cps[j - 1] for j in sps]
    return _build_result(n, cps, save_hits, float(D), dp_time, base_time,
                         fwd, base_fwd,
                         [be.cum_hi(k) for k in range(1, len(cps) + 1)])


# ---------------------------------------------------------------------------
# エントリ: 積モデル (HP依存のみ)
# ---------------------------------------------------------------------------
def analyze_product(ymix_per_hit, hp: HPParams, checkpoints, save_points,
                    hit_times, D, manual_gates=None, seg_success=None):
    """積モデル(HP依存)の凸区切り付き足切り最適化。G = -Σ ln Y 座標で実行する。"""
    n = len(ymix_per_hit)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    Htil = hp.Htil
    inc = Htil < 0
    bounds, cps = _split_bounds(n, checkpoints)
    times = _seg_times(hit_times, bounds)
    succ = _seg_success(seg_success, len(bounds) - 1)
    be = _CosBackend(_segs_product(ymix_per_hit, bounds, inc))
    sps = _cp_index(cps, save_points)

    if not inc and D >= Htil:
        raise ValueError("目標 D が到達不能 (D >= H̃₁)")
    s_thr = math.log1p(-D / Htil)
    D_thr = s_thr if inc else -s_thr

    def dmg_to_g(dmg):
        if dmg <= 0:
            return 0.0
        if 1.0 - dmg / Htil <= 0.0:
            return float(be.clip_hi)
        v = math.log1p(-dmg / Htil)
        return v if inc else -v

    def g_to_dmg(gv):
        if not math.isfinite(gv):
            return float(Htil)
        v = gv if inc else -gv
        return float(-Htil * math.expm1(v))

    dp = _SaveDP(be, times, succ, float(D_thr), sps)
    mg = (None if manual_gates is None
          else [dmg_to_g(float(g)) for g in manual_gates])
    dp_time, base_time, fwd, base_fwd = _run(dp, mg)
    cum_max_vals = [g_to_dmg(be.cum_hi(k)) for k in range(1, len(cps) + 1)]
    save_hits = [cps[j - 1] for j in sps]
    return _build_result(n, cps, save_hits, float(D), dp_time, base_time,
                         fwd, base_fwd, cum_max_vals, to_dmg=g_to_dmg)


# ---------------------------------------------------------------------------
# エントリ: 混在モデル (HP依存 + 通常)
# ---------------------------------------------------------------------------
def analyze_mixed(hit_specs, hp: HPParams, checkpoints, save_points, hit_times,
                  D, manual_gates=None, seg_success=None):
    """混在モデルの凸区切り付き足切り最適化 (アフィンカーネルのグリッド DP)。

    hit_specs : [(HP依存フラグ, 1Hit 混合), ...] (mixed.hit_specs_from_cards)
    増分が状態に依存するので係数空間では閉じず、価値関数・密度をグリッドで持つ
    (docs/mixed.md §6.2)。DP の構造そのものは和/積モデルと同じ。
    """
    specs = normalize_specs(hit_specs, hp)
    n = len(specs)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    bounds, cps = _split_bounds(n, checkpoints)
    times = _seg_times(hit_times, bounds)
    succ = _seg_success(seg_success, len(bounds) - 1)
    seg_blocks = [blocks_from_specs(specs[bounds[i]:bounds[i + 1]], hp.beta)
                  for i in range(len(bounds) - 1)]
    s_hi = mixed_support(specs, hp)[1]
    if not (s_hi > 0):
        s_hi = max(float(D), 1.0)
    if D >= s_hi:
        raise ValueError(f"目標 D が到達不能 (最大 {s_hi:,.0f})")
    be = _GridBackend(seg_blocks, specs, bounds, hp, float(s_hi))
    sps = _cp_index(cps, save_points)
    dp = _SaveDP(be, times, succ, float(D), sps)

    mg = None if manual_gates is None else [float(g) for g in manual_gates]
    dp_time, base_time, fwd, base_fwd = _run(dp, mg)
    save_hits = [cps[j - 1] for j in sps]
    return _build_result(n, cps, save_hits, float(D), dp_time, base_time,
                         fwd, base_fwd,
                         [be.cum_hi(k) for k in range(1, len(cps) + 1)])
