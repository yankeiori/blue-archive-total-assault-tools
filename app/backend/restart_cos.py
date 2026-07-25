"""多段リスタ最適化の係数空間 (COS / Fang–Oosterlee) 実装。グリッド・求積なし。

restart.py のグリッドDP(np.interp 畳み込み + np.trapezoid 求積)を、COS 係数空間の
後ろ向き帰納(Bermudan 型)に置き換えたもの。畳み込み=特性関数の積、打ち切り=
F&O の C/M 行列(閉形式)、通過率・成功率=余弦係数と ψ 係数の内積で、すべて
解析的に計算する。誤差は級数打ち切りのみ(求積誤差ゼロ)。

理論と式の対応は docs/restart_cos.md(正本は docs/cutoff.md 付録 A.6 / A.7)。
出力 dict の形は restart.py と同一(_result を共用)。和モデル analyze / 積モデル
analyze_product の両対応。
"""
from __future__ import annotations

import math

import numpy as np

from app.backend.cos import (
    HPParams,
    build_product_dist,
    build_sum_dist,
    cf_S_hits,
    support_bounds,
    support_bounds_hits,
    sum_cf,
)
from app.backend.restart import _result, baseline_nogate


# ---------------------------------------------------------------------------
# セグメント仕様: 増分の特性関数 cf(u) と台 [s_lo, s_hi](作業座標)
# ---------------------------------------------------------------------------
class _Seg:
    def __init__(self, cf, s_lo, s_hi):
        self.cf = cf          # 関数 u(ndarray) -> complex ndarray
        self.s_lo = s_lo
        self.s_hi = s_hi


def _choose_n_terms(b: float, widths: list[float]) -> int:
    """最も狭いセグメント増分の台幅を ~12 点で解像するよう項数を決める。"""
    w = min((w for w in widths if w > 0), default=b)
    if not (b > 0 and w > 0):
        return 1024
    n = int(math.ceil(b / (w / 12.0)))
    return max(1024, min(16384, n))


# ---------------------------------------------------------------------------
# FFT 相互相関 (Toeplitz/Hankel 行列ベクトル積の素): r[m] = Σ_k K[m+k] x[k]
# ---------------------------------------------------------------------------
def _corr_valid(K: np.ndarray, x: np.ndarray) -> np.ndarray:
    """np.correlate(K, x, 'valid') と同値 (r[m]=Σ_k K[m+k]x[k]) を FFT で。
    返り値長 = len(K)-len(x)+1。len(K) >= len(x) を仮定。"""
    nK, nx = len(K), len(x)
    P = 1 << int(math.ceil(math.log2(nK + nx)))
    full = np.fft.irfft(np.fft.rfft(K, P) * np.fft.rfft(x[::-1], P), P)
    return full[nx - 1: nx - 1 + (nK - nx + 1)]


# ---------------------------------------------------------------------------
# 係数空間エンジン (共通区間 [0, b], u_k = kπ/b, k=0..N-1)
# ---------------------------------------------------------------------------
class _CosEngine:
    def __init__(self, segs: list[_Seg]):
        self.segs = segs
        self.b = float(sum(s.s_hi for s in segs))           # 真の累積ダメージ上限
        max_seg = max((s.s_hi for s in segs), default=0.0)
        # 左端マージン: 半区間余弦の偶対称周期拡張による畳み込みエイリアシング
        # (ミラー像が台へ折り込む) を防ぐため、a を負側へ最大セグメント増分ぶん
        # 開ける。これで各畳み込みでミラー+増分が [a,b_pad] の外(左)に収まる。
        self.a = -1.05 * max_seg
        # 右端マージン: 後ろ向き期待値 E[V(s+T)] は引数 s+T が真の上限 b を右に
        # 超える (s<=b, T<=max_seg)。半区間余弦は基底上端について反射する周期拡張な
        # ので、右にマージンを取らないと s+T>b の領域の値 (指示関数なら 1) が
        # [a,b] へ折り返り、本来単調増加の続行価値 cont_j(s) が高 s 側で崩れて
        # find_gate が誤って到達不能 (gate=b) を返す。基底上端を b_pad=b+max_seg と
        # し、評価域 [0,b] では s+T<b_pad に収めて反射を排除する。
        self.b_pad = self.b + 1.05 * max_seg
        self.L = self.b_pad - self.a
        widths = [s.s_hi - s.s_lo for s in segs]
        self.N = _choose_n_terms(self.L, widths)
        self.u = np.arange(self.N) * np.pi / self.L          # u_k
        self.pref = np.full(self.N, 2.0 / self.L)
        self.pref[0] = 1.0 / self.L
        # 各セグメント増分の CF を u_k 上で前計算 (g に依らない)
        self.phi = [s.cf(self.u) for s in segs]
        # 各チェックポイント j (1..K) で取りうる累積ダメージの物理上限
        # = Σ_{i<j} s_hi。cum_hi[j-1] が cp j の上限 (find_gate の探索上限に使う)。
        self.cum_hi = np.cumsum([s.s_hi for s in segs])

    # ---- 余弦係数 <-> 値 ----
    def density_coeffs(self, seg_index: int) -> np.ndarray:
        """セグメント増分の密度の余弦係数 (チルダ規約, 正弦成分は 0)。"""
        phi = self.phi[seg_index]
        c = (2.0 / self.L) * np.real(phi * np.exp(-1j * self.u * self.a))
        c[0] = 1.0 / self.L
        return c

    def eval(self, C: np.ndarray, S: np.ndarray, s: float) -> float:
        ang = self.u * (s - self.a)
        return float(np.sum(C * np.cos(ang) + S * np.sin(ang)))

    # ---- [d, b] 上の ∫cos, ∫sin (= SC_k, SS_k), θ=s-a ----
    def _sc_ss(self, d: float, length: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.arange(length)
        ur = r * np.pi / self.L
        da = d - self.a
        sc = np.empty(length)
        ss = np.empty(length)
        sc[0] = self.b_pad - d        # 基底上端まで積分 (真の上限超の台は密度0)
        ss[0] = 0.0
        urn = ur[1:]
        sc[1:] = -np.sin(urn * da) / urn
        ss[1:] = (np.cos(urn * da) - ((-1.0) ** r[1:])) / urn
        return sc, ss

    def integrate(self, C: np.ndarray, S: np.ndarray, d: float) -> float:
        """∫_d^b g(s) ds, g = Σ C_k cos(u_k s) + S_k sin(u_k s)。"""
        sc, ss = self._sc_ss(d, self.N)
        return float(np.sum(C * sc + S * ss))

    def truncate(self, C: np.ndarray, S: np.ndarray, d: float) -> np.ndarray:
        """g·1{s>=d} の半区間余弦係数 (チルダ規約)。C は余弦, S は正弦係数。"""
        N = self.N
        sc, ss = self._sc_ss(d, 2 * N - 1)        # 核 (index 0..2N-2)
        # 対称 Toeplitz 核 (偶, SC) と 反対称 Toeplitz 核 (奇, SS)
        idx = np.arange(2 * N - 1) - (N - 1)       # -(N-1)..(N-1)
        ksym = sc[np.abs(idx)]
        kodd = np.sign(idx) * ss[np.abs(idx)]
        # TE(x)_m = Σ_k x_k SC_{|k-m|}; HC(x)_m = Σ_k x_k SC_{k+m}
        te_C = _corr_valid(ksym, C)[::-1]
        hc_C = _corr_valid(sc[:2 * N - 1], C)
        # TO(x)_m = Σ_k x_k sgn(k-m)SS_{|k-m|}; HS(x)_m = Σ_k x_k SS_{k+m}
        to_S = _corr_valid(kodd, S)[::-1]
        hs_S = _corr_valid(ss[:2 * N - 1], S)
        icc = 0.5 * (te_C + hc_C)        # Σ_k C_k I^cc_{m,k}
        isc = 0.5 * (hs_S + to_S)        # Σ_k S_k I^sc_{m,k}
        return self.pref * (icc + isc)

    # ---- 畳み込み(前向き密度)/ 期待値(後ろ向き価値)----
    def convolve(self, c: np.ndarray, seg_index: int) -> tuple[np.ndarray, np.ndarray]:
        """密度 (余弦のみ c) に増分 T_seg を畳み込む -> (余弦, 正弦)。"""
        phi = self.phi[seg_index]
        return c * np.real(phi), c * np.imag(phi)

    def expect(self, A: np.ndarray, seg_index: int) -> tuple[np.ndarray, np.ndarray]:
        """E[V(s+T_seg)] の (余弦, 正弦) 係数。V は余弦のみ A。"""
        phi = self.phi[seg_index]
        return A * np.real(phi), -A * np.imag(phi)

    def indicator_coeffs(self, D: float) -> np.ndarray:
        """1{s>=D} の半区間余弦係数 (チルダ規約)。"""
        c = np.empty(self.N)
        c[0] = (self.b_pad - D) / self.L      # 指示関数は [D, b_pad] で 1
        un = self.u[1:]
        c[1:] = -(2.0 / self.L) * np.sin(un * (D - self.a)) / un
        return c

    def find_gate(self, C: np.ndarray, S: np.ndarray, hi: float | None = None) -> float:
        """cont(s)=eval(C,S,s) が初めて >=0 になる s (単調増加を仮定, 二分法)。

        hi = そのチェックポイントの累積ダメージ物理上限 (既定 self.b)。探索上限を
        物理上限に絞ることで (1) s+T が真の上限 b を超えず反射域 [hi, b] を評価せず、
        (2) 続行不利時の sentinel が hi になり 残り D-gate >= D-hi が構造的に非負に
        収まる (足切りが物理上限を超える=負の残り、という退化を防ぐ)。下限は 0 のまま
        にして真のゼロ交点で打ち切り、価値関数 V=max(0,cont) をそこで連続に保つ
        (下限を物理下限へ持ち上げると不連続が生じリンギングが増える)。"""
        lo = 0.0
        if hi is None:
            hi = self.b
        if self.eval(C, S, lo) >= 0.0:
            return lo
        if self.eval(C, S, hi) < 0.0:
            return hi                  # 物理上限でも続行不利 (事実上到達不能)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.eval(C, S, mid) >= 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 後ろ向き帰納 (固定 g) と Dinkelbach
# ---------------------------------------------------------------------------
def _backward(eng: _CosEngine, g: float, times, D: float, want_gates: bool,
              succ):
    """固定 g で V0(g) を返す。want_gates なら各段の関門 d_j* も返す。

    succ[j] は区間 j のダメージと独立な成功確率 (区間を回しきってリスタートせず
    次へ進める確率)。区間 j を続行すると時間 t_j を消費し、確率 succ[j] で次段へ、
    1-succ[j] でリスタート (価値 0) なので、続行価値の期待値項に succ[j] が掛かる:
        cont_j(s) = -g t_j + succ[j]·E[V_{j+1}(s+T_j)]。
    succ が全て 1 なら独立確率なしの素の DP に一致する。
    """
    K = len(eng.segs) - 1
    A = eng.indicator_coeffs(D)                 # V_{K+1} = 1{s>=D}
    gates = {}
    for j in range(K, 0, -1):
        cC, cS = eng.expect(A, j)               # E[V_{j+1}(s+T_j)]
        cC = succ[j] * cC                        # 独立成功確率で割引 (新規配列)
        cS = succ[j] * cS
        cC[0] -= g * times[j]                   # -g t_j (DC シフト)
        d_j = eng.find_gate(cC, cS, hi=float(eng.cum_hi[j - 1]))  # cp j の物理上限まで
        if want_gates:
            gates[j] = d_j
        A = eng.truncate(cC, cS, d_j)           # V_j = max(0, cont_j)
    cC, cS = eng.expect(A, 0)                    # E[V_1(0 + T_0)]
    V0 = -g * times[0] + succ[0] * eng.eval(cC, cS, 0.0)
    return (V0, gates) if want_gates else V0


def _optimize(eng: _CosEngine, times, D: float, succ, iters: int = 60):
    """Dinkelbach + 後ろ向き帰納で最適 g* と関門 d_j* を求める。"""
    g_lo, g_hi = 0.0, 1.0 / max(times[0], 1e-9)
    for _ in range(iters):
        g_mid = 0.5 * (g_lo + g_hi)
        if _backward(eng, g_mid, times, D, want_gates=False, succ=succ) > 0:
            g_lo = g_mid
        else:
            g_hi = g_mid
    g_star = 0.5 * (g_lo + g_hi)
    _, gates = _backward(eng, g_star, times, D, want_gates=True, succ=succ)
    return g_star, gates


# ---------------------------------------------------------------------------
# 前向きパス: 通過率・成功率・期待時間(係数空間の積分)
# ---------------------------------------------------------------------------
def forward_metrics(eng: _CosEngine, times, D: float, gates, succ=None):
    """関門固定で通過率・成功率・期待時間を算出。

    succ[j] は区間 j の独立成功確率。区間 j を回しきって次へ進むには、ダメージ関門に
    加えて区間 0..j-1 の独立成功 (∏_{i<j} succ[i]) が必要。区間 j の所要時間は、その
    区間に到達して回す確率 = (∏_{i<j} succ[i])·(ダメージ関門の累積通過率) で課金される
    (失敗時も区間を回しきってからリスタートするため時間は消費済み)。成功 (完走) には
    最終区間 K の独立成功も要るので ∏_{i<=K} succ[i] が掛かる。pass_rates は表示用に
    独立確率を含めた真の累積通過率を返す。
    """
    K = len(eng.segs) - 1
    succ = _seg_success(succ, K + 1)
    clamp = lambda p: min(1.0, max(0.0, float(p)))
    C = eng.density_coeffs(0)                    # cp1 の累積 = seg0 増分の密度
    S = np.zeros_like(C)
    pass_rates = []
    exp_time = times[0]                          # seg0 は必ず回す
    q_cum = succ[0]                              # ∏_{i<j} succ[i] (j=1 で succ[0])
    for j in range(1, K + 1):
        p_dmg = clamp(eng.integrate(C, S, gates[j]))   # ダメージ関門のみの累積通過率
        p_j = clamp(q_cum * p_dmg)                     # 独立確率込みの真の累積通過率
        pass_rates.append(p_j)
        exp_time += times[j] * p_j               # 区間 j に到達して回す確率ぶん課金
        c_tr = eng.truncate(C, S, gates[j])      # 関門通過後の生存サブ密度 (余弦のみ)
        C, S = eng.convolve(c_tr, j)             # 次チェックポイントへ
        q_cum *= succ[j]                         # ∏_{i<=j} succ[i] (次段 j+1 用)
    success = clamp(q_cum * clamp(eng.integrate(C, S, D)))  # 完走 = ∏_{i<=K} succ·達成
    g = success / exp_time if exp_time > 0 else 0.0
    return {"success": success, "exp_time": exp_time, "g": g, "pass_rates": pass_rates}


# ---------------------------------------------------------------------------
# セグメント仕様の構築 (和モデル / 積モデル)
# ---------------------------------------------------------------------------
def _segs_sum(hit_mixtures, bounds):
    segs = []
    for i in range(len(bounds) - 1):
        sub = hit_mixtures[bounds[i]:bounds[i + 1]]
        lo, hi = support_bounds(sub)
        segs.append(_Seg(cf=lambda u, s=sub: sum_cf(s, u), s_lo=lo, s_hi=hi))
    return segs


def _segs_product(ymix_per_hit, bounds, inc):
    """積モデル: ダメージ増加方向の座標 G での増分 CF と台。

    β>0 (Htil>0): S=Σ ln Y <= 0 で D は S の減少関数 → G=-S、CF は conj(φ_S)、
    台 [-B, -A] (>=0)。β<0 (Htil<0, inc=True): S >= 0 のまま増加関数 → G=S、
    CF は φ_S、台 [A, B] (>=0)。"""
    segs = []
    for i in range(len(bounds) - 1):
        sub = ymix_per_hit[bounds[i]:bounds[i + 1]]
        A, B = support_bounds_hits(sub)          # S=Σ ln Y の台
        if inc:
            segs.append(_Seg(cf=lambda u, s=sub: cf_S_hits(s, u),
                             s_lo=A, s_hi=B))
        else:
            segs.append(_Seg(cf=lambda u, s=sub: np.conj(cf_S_hits(s, u)),
                             s_lo=-B, s_hi=-A))
    return segs


def _split_bounds(n, checkpoints):
    cps = sorted({int(c) for c in checkpoints if 0 < int(c) < n})
    return [0, *cps, n], cps


def _seg_times(hit_times, bounds):
    return [float(sum(hit_times[bounds[i]:bounds[i + 1]]))
            for i in range(len(bounds) - 1)]


def _seg_success(seg_success, n_segs):
    """区間ごとのダメージ独立成功確率を長さ n_segs (= K+1) の list に正規化する。
    None なら全 1.0 (独立確率なし = 素の DP)。各値は [0, 1] にクランプ。"""
    if seg_success is None:
        return [1.0] * n_segs
    out = [1.0] * n_segs
    for i in range(n_segs):
        try:
            out[i] = min(1.0, max(0.0, float(seg_success[i])))
        except (TypeError, ValueError, IndexError):
            out[i] = 1.0
    return out


# ---------------------------------------------------------------------------
# エントリ
# ---------------------------------------------------------------------------
def analyze(hit_mixtures, checkpoints, hit_times, D, manual_gates=None,
            seg_success=None):
    """和モデルの多段リスタ最適化(係数空間版)。restart.analyze と同形式を返す。

    manual_gates を渡すと最適化せず、その関門 (各 cp の累積ダメージしきい値) で
    前向き評価する。リスタライン手動調整 (インタラクティブ表示) 用。
    seg_success は区間 (= K+1 個) ごとのダメージと独立な成功確率の list (None なら全 1)。
    """
    n = len(hit_mixtures)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    bounds, cps = _split_bounds(n, checkpoints)
    times = _seg_times(hit_times, bounds)
    succ = _seg_success(seg_success, len(bounds) - 1)
    eng = _CosEngine(_segs_sum(hit_mixtures, bounds))

    full = build_sum_dist(hit_mixtures)
    base = baseline_nogate(full, times, D, succ)
    if manual_gates is None:
        g_star, gates = _optimize(eng, times, D, succ)
    else:
        gates = {k: float(np.clip(manual_gates[k - 1], eng.a, eng.b))
                 for k in range(1, len(cps) + 1)}
        g_star = float("nan")
    fwd = forward_metrics(eng, times, D, gates, succ)

    cum_max = np.cumsum([s.s_hi for s in eng.segs])
    gates_dmg = {k: gates[k] for k in gates}
    return _result(n, cps, D, gates_dmg, fwd, base, g_star,
                   [float(cum_max[k - 1]) for k in range(1, len(cps) + 1)])


def analyze_product(ymix_per_hit, hp: HPParams, checkpoints, hit_times, D,
                    manual_gates=None, seg_success=None):
    """積モデル(HP依存)の多段リスタ最適化(係数空間版)。G=-Σ ln Y 座標で実行。

    manual_gates (各 cp の累積ダメージしきい値) を渡すと最適化せず前向き評価する。
    seg_success は区間 (= K+1 個) ごとのダメージと独立な成功確率の list (None なら全 1)。
    """
    n = len(ymix_per_hit)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    Htil = hp.Htil
    inc = Htil < 0                     # β<0: G=S、β>0: G=-S (ダメージ増加方向)
    bounds, cps = _split_bounds(n, checkpoints)
    times = _seg_times(hit_times, bounds)
    succ = _seg_success(seg_success, len(bounds) - 1)
    eng = _CosEngine(_segs_product(ymix_per_hit, bounds, inc))

    # 達成しきい値: D_n >= D ⟺ G_n >= D_thr (D_thr = ±ln(1 - D/Htil))
    if not inc and D >= Htil:
        D_thr = float("inf")
    else:
        s_thr = math.log1p(-D / Htil)
        D_thr = s_thr if inc else -s_thr
    full_pd = build_product_dist(ymix_per_hit, hp)
    base = baseline_nogate(full_pd, times, D, succ)

    def dmg_to_g(dmg):
        if dmg <= 0:
            return 0.0
        arg = 1.0 - dmg / Htil
        if arg <= 0.0:
            return float("inf")        # β>0 で dmg >= Htil (到達不能)
        s = math.log1p(-dmg / Htil)
        return s if inc else -s

    if manual_gates is None:
        g_star, gates_G = _optimize(eng, times, D_thr, succ)
    else:
        gates_G = {k: float(np.clip(dmg_to_g(float(manual_gates[k - 1])),
                                    eng.a, eng.b))
                   for k in range(1, len(cps) + 1)}
        g_star = float("nan")
    fwd = forward_metrics(eng, times, D_thr, gates_G, succ)

    def g_to_dmg(gv):
        if not math.isfinite(gv):
            return Htil
        s = gv if inc else -gv
        return float(-Htil * math.expm1(s))
    gates_dmg = {k: g_to_dmg(gates_G[k]) for k in gates_G}
    cumG = np.cumsum([s.s_hi for s in eng.segs])
    cum_max_vals = [g_to_dmg(float(cumG[k - 1])) for k in range(1, len(cps) + 1)]
    return _result(n, cps, D, gates_dmg, fwd, base, g_star, cum_max_vals)
