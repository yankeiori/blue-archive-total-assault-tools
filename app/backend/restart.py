"""多段リスタ(複数足切り関門)のスループット最適化。

リセット運用: チェックポイント m_1<...<m_K で累積ダメージが関門 d_j 未満なら即リセット、
全関門通過したら最後まで回して合計 >= D を狙う。各セグメントの所要時間を t_i とし、
更新報酬定理で長期成功率 g = E[成功]/E[時間] を最大化する関門 {d_j} を求める。

増分独立性(和モデル)/対数増分独立性(積モデル)で各段の状態は1次元の累積ダメージ
だけ(マルコフ性)なので、Dinkelbach 変換 + 後ろ向き帰納(Bermudan 型)で解ける:

    固定 g に対し U_{K+1}(s)=1{達成}, U_j(s)=max(0, -g t_j + E[U_{j+1}(s+増分)])
    開始 V0(g) = -g t_0 + E[U_1(増分_0)],  V0(g*)=0 を二分法で解くと g* が最適長期率。

各段の最適関門 d_j* は「続行価値 >= 0」のしきい値(= 足切りライン)として出る。

和モデル(analyze)と積モデル/HP依存(analyze_product)の両対応。積モデルは
G = -Σ ln Y(= ダメージ増加方向)座標に反転すると和モデルと同型になるため、
同じ DP(optimize / forward_metrics)をそのまま流用する(_GDist ラッパー)。
"""
from __future__ import annotations

import math

import numpy as np

from app.backend.cos import build_product_dist, build_sum_dist


# ---------------------------------------------------------------------------
# セグメント分割
# ---------------------------------------------------------------------------
def split_segments(hit_mixtures, checkpoints):
    """ヒット列をチェックポイントで K+1 セグメントに分割し、各セグメントの
    増分分布(SumDist)と区切り位置を返す。checkpoints は 1..n-1 のヒット数。"""
    n = len(hit_mixtures)
    cps = sorted({int(c) for c in checkpoints if 0 < int(c) < n})
    bounds = [0, *cps, n]
    seg_dists = [build_sum_dist(hit_mixtures[bounds[i]:bounds[i + 1]])
                 for i in range(len(bounds) - 1)]
    return seg_dists, bounds, cps


# ---------------------------------------------------------------------------
# 期待値演算 (増分との畳み込み)
# ---------------------------------------------------------------------------
def _segment_nodes(seg, n_nodes=512):
    """セグメント増分の数値積分ノード(offset, weight)。weight は確率(合計1に正規化)。"""
    o = np.linspace(seg.support_lo, seg.support_hi, n_nodes)
    w = seg.pdf(o)
    s = w.sum()
    w = w / s if s > 0 else np.full_like(o, 1.0 / n_nodes)
    return o, w


def _expect_shift(U, grid, nodes):
    """E[U(s + T)] を grid 上の全 s について返す。U はグリッド外で端値に飽和。"""
    o, w = nodes
    out = np.zeros_like(grid)
    for oi, wi in zip(o, w):
        out += wi * np.interp(grid + oi, grid, U, left=U[0], right=U[-1])
    return out


# ---------------------------------------------------------------------------
# 後ろ向き帰納 (固定 g) と Dinkelbach
# ---------------------------------------------------------------------------
def _backward(g, seg_dists, seg_nodes, times, D, grid, succ):
    """固定 g での後ろ向き帰納。V0(g) と各段の続行価値 cont_j(s) を返す。

    succ[j] は区間 j のダメージと独立な成功確率。続行すると時間 t_j を消費し、確率
    succ[j] で次段へ、1-succ[j] でリスタート(価値 0)なので、続行価値の期待値項に
    succ[j] が掛かる: cont_j(s) = -g t_j + succ[j]·E[V_{j+1}(s+T_j)]。
    """
    K = len(seg_dists) - 1                       # 関門の数
    # 終端の1つ手前 = 最後のチェックポイント K: 続行で最終セグメントを回す
    last = seg_dists[K]
    cont = {}
    # cont_K(s) = -g t_K + succ_K · P(最終増分 >= D - s)
    contK = -g * times[K] + succ[K] * (1.0 - last.cdf(D - grid))
    cont[K] = contK
    U = np.maximum(0.0, contK)
    # j = K-1 .. 1
    for j in range(K - 1, 0, -1):
        EU = _expect_shift(U, grid, seg_nodes[j])
        contj = -g * times[j] + succ[j] * EU
        cont[j] = contj
        U = np.maximum(0.0, contj)
    # 開始: 累積 0 から増分 seg0 を足すので E[U_1(seg0)] = U_1 を seg0 分布で平均
    o0, w0 = seg_nodes[0]
    EU0 = float(np.sum(w0 * np.interp(o0, grid, U, left=U[0], right=U[-1])))
    V0 = -g * times[0] + succ[0] * EU0
    return V0, cont


def _gate_from_cont(cont_j, grid):
    """続行価値 cont_j(s) が初めて 0 以上になる s = 最適関門 d_j*。"""
    pos = np.where(cont_j >= 0.0)[0]
    if pos.size == 0:
        return grid[-1]            # どこでも続行不利 → 事実上到達不能
    return float(grid[pos[0]])


def _norm_succ(succ, n_segs):
    """区間ごとのダメージ独立成功確率を長さ n_segs の list に正規化 ([0,1] クランプ)。"""
    if succ is None:
        return [1.0] * n_segs
    out = [1.0] * n_segs
    for i in range(n_segs):
        try:
            out[i] = min(1.0, max(0.0, float(succ[i])))
        except (TypeError, ValueError, IndexError):
            out[i] = 1.0
    return out


def optimize(seg_dists, times, D, n_grid=3000, n_nodes=512, iters=80, succ=None):
    """Dinkelbach + 後ろ向き帰納で最適スループット g* と各段の関門 d_j* を求める。"""
    K = len(seg_dists) - 1
    succ = _norm_succ(succ, K + 1)
    s_hi = sum(d.support_hi for d in seg_dists)
    grid = np.linspace(0.0, s_hi, n_grid)
    seg_nodes = [_segment_nodes(d, n_nodes) for d in seg_dists]

    # g の上界: 1 試行は最低 times[0] かかり成功は <=1 なので g <= 1/times[0]
    g_lo, g_hi = 0.0, 1.0 / max(times[0], 1e-9)
    for _ in range(iters):
        g_mid = 0.5 * (g_lo + g_hi)
        V0, _ = _backward(g_mid, seg_dists, seg_nodes, times, D, grid, succ)
        if V0 > 0:                 # この g は達成可能 → もっと上げられる
            g_lo = g_mid
        else:
            g_hi = g_mid
    g_star = 0.5 * (g_lo + g_hi)
    _, cont = _backward(g_star, seg_dists, seg_nodes, times, D, grid, succ)
    gates = {j: _gate_from_cont(cont[j], grid) for j in range(1, K + 1)}
    return g_star, gates, grid


# ---------------------------------------------------------------------------
# 前向きパス: 与えた関門でのスループット・通過率・達成率(検証兼表示用)
# ---------------------------------------------------------------------------
def forward_metrics(seg_dists, times, D, gates, n_grid=4000, n_nodes=512, succ=None):
    """関門 {j: d_j} を適用したときの (成功率, 平均時間, 長期率 g, 段別通過率) を返す。

    通過率・成功率は **区間増分の CDF(生存関数 1-F)** で計算する。各関門の通過
    質量は「直前チェックポイントの生存サブ密度 a(s) × P(増分 >= d_j - s)」の積分で、
    初段は CDF の直接評価。台形で密度そのものの質量を取る旧実装は、序盤の幅の狭い
    セグメント(例: 数 Hit)を全体台 [0, s_hi] の粗いグリッドで覆うと積分が膨らみ
    通過率が 100% を超えていた。グリッドは最小セグメント幅を解像するよう自動調整する。
    """
    K = len(seg_dists) - 1
    succ = _norm_succ(succ, K + 1)
    s_hi = sum(d.support_hi for d in seg_dists)
    # 全体台は数千万規模でも序盤セグメントの台幅は数万のことがある。最も狭い
    # セグメントを ~80 点以上で覆えるようグリッド点数を底上げ(前向きは一発計算)。
    widths = [d.support_hi - d.support_lo
              for d in seg_dists if d.support_hi > d.support_lo]
    if widths and s_hi > 0:
        n_need = int(math.ceil(s_hi / (min(widths) / 80.0))) + 1
        n_grid = min(80000, max(n_grid, n_need))
    grid = np.linspace(0.0, s_hi, n_grid)
    seg_nodes = [_segment_nodes(d, n_nodes) for d in seg_dists]

    def surv(dist, x):
        """生存関数 P(dist >= x)。x はスカラー/配列。台外は 0/1 に飽和。"""
        return 1.0 - dist.cdf(np.asarray(x, dtype=float))

    def conv_density(dens, nodes):
        """密度 dens(累積) に増分を畳み込む → 次チェックポイントの累積密度。"""
        o, w = nodes
        out = np.zeros_like(grid)
        for oi, wi in zip(o, w):
            out += wi * np.interp(grid - oi, grid, dens, left=0.0, right=0.0)
        return out

    clamp = lambda p: min(1.0, max(0.0, float(p)))
    seg0 = seg_dists[0]
    pass_rates = []
    exp_time = times[0]            # seg0 は必ず回す
    q_cum = succ[0]               # ∏_{i<j} succ[i] (j=1 で succ[0])
    # a(s): 直前の関門まで通過した経路の、現チェックポイント累積の生存サブ密度。
    a = None
    for j in range(1, K + 1):
        if j == 1:
            # cp1 = seg0 後の累積。通過率は CDF で厳密に。
            p_dmg = clamp(surv(seg0, np.array([gates[1]]))[0])
            dens_cp = np.where((grid >= seg0.support_lo) & (grid <= seg0.support_hi),
                               seg0.pdf(grid), 0.0)
        else:
            # p_dmg = ∫ a(s)·P(直前セグメント増分 >= d_j - s) ds (生存関数で計算)
            prev = seg_dists[j - 1]
            p_dmg = clamp(np.trapezoid(a * surv(prev, gates[j] - grid), grid))
            # 次チェックポイントの累積密度 = a を直前セグメントで畳み込み
            dens_cp = conv_density(a, seg_nodes[j - 1])
        p_j = clamp(q_cum * p_dmg)        # 独立成功確率込みの真の累積通過率
        pass_rates.append(p_j)
        exp_time += times[j] * p_j        # seg j は gate j 到達時のみ回す
        # 関門 d_j で打ち切った生存サブ密度を次段へ
        a = np.where(grid >= gates[j], dens_cp, 0.0)
        q_cum *= succ[j]                  # ∏_{i<=j} succ[i] (次段用)
    # 最終: gate K 通過後 seg K を回して合計 >= D。成功も生存関数で。
    last = seg_dists[K]
    success = clamp(q_cum * clamp(np.trapezoid(a * surv(last, D - grid), grid)))
    g = success / exp_time if exp_time > 0 else 0.0
    return {"success": success, "exp_time": exp_time, "g": g, "pass_rates": pass_rates}


def baseline_nogate(full, times, D, succ=None):
    """足切り無し(全部回す)の成功率・時間・長期率。full は全体の SumDist。

    succ は区間 (= len(times) 個) ごとのダメージと独立な成功確率。足切り無しでも
    独立成功判定の失敗ではリスタートが起きる(時間は消費済み)ため、成功率は全区間の
    独立成功積 ∏succ を掛け、期待時間は区間 j に到達する確率 ∏_{i<j} succ[i] で重み
    づける: T = Σ_j t_j·∏_{i<j} succ[i]。succ=None (全 1) なら従来どおり P, Σt。
    """
    P = 1.0 - float(full.cdf(np.array([D]))[0])
    if succ is None:
        succ = [1.0] * len(times)
    T = 0.0
    q = 1.0
    for t, s in zip(times, succ):
        T += t * q                 # 区間に到達する確率ぶん時間を課金
        q *= s                     # 次区間到達確率 (ループ後 q = ∏ succ)
    P *= q                          # 全区間の独立成功を満たして初めて達成
    return {"success": P, "exp_time": T, "g": P / T if T > 0 else 0.0}


# ---------------------------------------------------------------------------
# エントリ: 多段リスタ解析
# ---------------------------------------------------------------------------
def analyze(hit_mixtures, checkpoints, hit_times, D, seg_success=None):
    """和モデルの多段リスタ最適化。

    hit_mixtures : 各ヒットの一様混合 (build_hit_mixtures の出力)
    checkpoints  : チェックポイントのヒット数 m_j (1..n-1) のリスト
    hit_times    : 各ヒットの所要時間 (長さ n) または スカラー (全ヒット同一)
    D            : 目標ダメージ
    seg_success  : 区間 (= K+1 個) ごとのダメージと独立な成功確率 (None なら全 1)
    返り値: dict(関門 d_j*, 段別通過率, スループット, 足切り無し比, セグメント情報 ...)
    """
    n = len(hit_mixtures)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    seg_dists, bounds, cps = split_segments(hit_mixtures, checkpoints)
    # セグメント所要時間 = そのセグメントに含まれるヒットの時間和
    times = [float(sum(hit_times[bounds[i]:bounds[i + 1]]))
             for i in range(len(bounds) - 1)]
    succ = _norm_succ(seg_success, len(bounds) - 1)
    full = build_sum_dist(hit_mixtures)

    base = baseline_nogate(full, times, D, succ)
    g_star, gates, _grid = optimize(seg_dists, times, D, succ=succ)
    fwd = forward_metrics(seg_dists, times, D, gates, succ=succ)

    cum_max = np.cumsum([d.support_hi for d in seg_dists])  # 各cp到達時の累積上限(参考)
    gates_dmg = {k: gates[k] for k in gates}
    return _result(n, cps, D, gates_dmg, fwd, base, g_star,
                   [float(cum_max[k - 1]) for k in range(1, len(cps) + 1)])


def _result(n, cps, D, gates_dmg, fwd, base, g_star, cum_max_vals):
    rows = []
    for k, m in enumerate(cps, start=1):
        rows.append({
            "checkpoint": m,
            "gate": gates_dmg[k],
            "pass_rate": fwd["pass_rates"][k - 1],
            "cum_max_at_cp": cum_max_vals[k - 1],
        })
    speedup = fwd["g"] / base["g"] if base["g"] > 0 else float("nan")
    return {
        "n_hits": n,
        "checkpoints": cps,
        "D": D,
        "rows": rows,
        "throughput": fwd["g"],
        "success": fwd["success"],
        "exp_time": fwd["exp_time"],
        "baseline": base,
        "speedup": speedup,
        "g_star_dp": g_star,
    }


# ---------------------------------------------------------------------------
# 積モデル (HP依存): 対数増分 G = -Σ ln Y (ダメージ増加方向) で和モデルに帰着
# ---------------------------------------------------------------------------
class _GDist:
    """ProductDist の S=Σln Y 分布を「ダメージ増加方向」の座標 G に写し、SumDist
    互換 (support_lo/hi, cdf(P(G<=g)), pdf) で見せるラッパー。H̃>0 (β>0) では
    S<=0 で D は S の減少関数なので G=-S、H̃<0 (β<0, 瀕死特効型) では S>=0 の
    まま増加関数なので G=S。いずれも G が大きいほど高ダメージ。"""

    def __init__(self, pd):
        self.pd = pd
        self.inc = pd.Htil < 0
        if self.inc:
            self.support_lo, self.support_hi = pd.a, pd.b
        else:
            self.support_lo, self.support_hi = -pd.b, -pd.a

    def _cdf_S_clamped(self, s):
        s = np.asarray(s, dtype=float)
        out = np.empty_like(s)
        below, above = s < self.pd.a, s > self.pd.b
        mid = ~(below | above)
        out[below], out[above] = 0.0, 1.0
        if mid.any():
            out[mid] = self.pd._cdf_S(s[mid])
        return out

    def cdf(self, g):
        g = np.asarray(g, dtype=float)
        if self.inc:
            return self._cdf_S_clamped(g)
        return 1.0 - self._cdf_S_clamped(-g)

    def pdf(self, g):
        g = np.asarray(g, dtype=float)
        out = np.zeros_like(g)
        m = (g >= self.support_lo) & (g <= self.support_hi)
        if m.any():
            s = g[m] if self.inc else -g[m]
            out[m] = np.maximum(self.pd._pdf_S(s), 0.0)
        return out


def split_segments_product(ymix_per_hit, hp, checkpoints):
    """積モデル: ymix をチェックポイントで分割し、各セグメントの G 座標分布を返す。"""
    n = len(ymix_per_hit)
    cps = sorted({int(c) for c in checkpoints if 0 < int(c) < n})
    bounds = [0, *cps, n]
    seg = [_GDist(build_product_dist(ymix_per_hit[bounds[i]:bounds[i + 1]], hp))
           for i in range(len(bounds) - 1)]
    return seg, bounds, cps


def analyze_product(ymix_per_hit, hp, checkpoints, hit_times, D, seg_success=None):
    """積モデル(HP依存)の多段リスタ最適化。和モデルと同じ DP を G=-Σln Y 座標で実行。

    ymix_per_hit : 各ヒットの Y=1-βx 混合 (y_mixture の出力)
    hp           : HPParams
    seg_success  : 区間 (= K+1 個) ごとのダメージと独立な成功確率 (None なら全 1)
    返り値は analyze と同形式 (gate はダメージ値に逆変換済み)。
    """
    n = len(ymix_per_hit)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    Htil = hp.Htil
    inc = Htil < 0                     # β<0: G=S (ln Y >= 0)、β>0: G=-S
    seg_dists, bounds, cps = split_segments_product(ymix_per_hit, hp, checkpoints)
    times = [float(sum(hit_times[bounds[i]:bounds[i + 1]]))
             for i in range(len(bounds) - 1)]
    succ = _norm_succ(seg_success, len(bounds) - 1)
    full_pd = build_product_dist(ymix_per_hit, hp)

    # 達成しきい値: D_n >= D ⟺ G_n >= D_thr (D_thr = ±ln(1 - D/Htil))
    if not inc and D >= Htil:
        D_thr = float("inf")
    else:
        s_thr = math.log1p(-D / Htil)
        D_thr = s_thr if inc else -s_thr
    base = baseline_nogate(full_pd, times, D, succ)   # ProductDist.cdf はダメージ CDF
    g_star, gates_G, _grid = optimize(seg_dists, times, D_thr, succ=succ)
    fwd = forward_metrics(seg_dists, times, D_thr, gates_G, succ=succ)

    def g_to_dmg(g):
        if not math.isfinite(g):
            return Htil
        s = g if inc else -g
        return float(-Htil * math.expm1(s))
    gates_dmg = {k: g_to_dmg(gates_G[k]) for k in gates_G}
    cumG = np.cumsum([d.support_hi for d in seg_dists])
    cum_max_vals = [g_to_dmg(float(cumG[k - 1])) for k in range(1, len(cps) + 1)]
    return _result(n, cps, D, gates_dmg, fwd, base, g_star, cum_max_vals)
