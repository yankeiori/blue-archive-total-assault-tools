"""混在モデル (HP依存 + 通常ダメージ) の多段リスタ最適化。

理論は docs/mixed.md §6.2 + docs/cutoff.md。混在モデルでも過程は累積ダメージ
s = D_t を状態とする 1 次元 Markov 過程のままなので、restart.py と同じ
Dinkelbach + 後ろ向き帰納 (Bermudan 型) が成立する。ただしセグメント増分は
状態に依存する (アフィン: s' = a s + c) ため、シフト畳み込みではなく
mixed.py のブロック集約アフィンカーネルで期待値・密度を伝播する:

    後ろ向き: E[V(s')] (s) = Σ_i w_i V(a_i s + c_i)   (評価点は前計算して再利用)
    前向き:   f'(t) = Σ_i (w_i/a_i) f((t - c_i)/a_i)

和モデルは a≡1 の特殊例としてそのまま含まれる。出力 dict は restart.py の
_result と同形式 (callbacks から restart_cos と差し替え可能)。
"""
from __future__ import annotations

import math

import numpy as np

from app.backend.cos import HPParams
from app.backend.mixed import (
    _deposit,
    block_kernel,
    blocks_from_specs,
    build_dist_auto,
    first_block_density,
    kernel_backward,
    kernel_forward,
    kernel_positions,
    mixed_support,
    normalize_specs,
)
from app.backend.restart import (
    _gate_from_cont,
    _norm_succ,
    _result,
    baseline_nogate,
)

_N_GRID_BACKWARD = 3000
_N_GRID_FORWARD = 12000
_N_GRID_FORWARD_MAX = 48000
# 二分法の g 相対精度は 2^-32 ≈ 2e-10 で、関門位置の分解能 (グリッド) より
# 十分細かい。真の Dinkelbach 反復 (成功率/時間の分離追跡) は複雑さに見合わず不採用。
_DINKELBACH_ITERS = 32
# 後ろ向きは関門位置 (方策) の決定のみでよく、前向き評価が方策の指標を厳密に
# 出すので、カーネルのセルは粗め (2×刻み) で足りる。メモリも抑えられる。
_CELL_FACTOR_BACKWARD = 2.0


# ---------------------------------------------------------------------------
# セグメント分割 (チェックポイントで切ってからブロック集約)
# ---------------------------------------------------------------------------
def _split_bounds(n: int, checkpoints) -> tuple[list[int], list[int]]:
    cps = sorted({int(c) for c in checkpoints if 0 < int(c) < n})
    return [0, *cps, n], cps


def _segment_blocks(specs, bounds, beta):
    return [blocks_from_specs(specs[bounds[i]:bounds[i + 1]], beta)
            for i in range(len(bounds) - 1)]


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
# 後ろ向き帰納 (固定 g) と Dinkelbach
# ---------------------------------------------------------------------------
def _backward(g, seg_kernels, seg_pos, grid, times, D, succ):
    """固定 g での後ろ向き帰納。V0(g) と各段の続行価値 cont_j(s) を返す。"""
    K = len(seg_kernels) - 1
    U = (grid >= D).astype(float)
    cont = {}
    for j in range(K, 0, -1):
        W = U
        for kern, pos in zip(reversed(seg_kernels[j]), reversed(seg_pos[j])):
            W = kernel_backward(W, grid, kern, pos)
        cont[j] = -g * times[j] + succ[j] * W
        U = np.maximum(0.0, cont[j])
    W = U
    for kern, pos in zip(reversed(seg_kernels[0]), reversed(seg_pos[0])):
        W = kernel_backward(W, grid, kern, pos)
    V0 = -g * times[0] + succ[0] * float(W[0])       # 開始状態 s=0 = grid[0]
    return V0, cont


def _optimize(seg_kernels, seg_pos, grid, times, D, succ):
    g_lo, g_hi = 0.0, 1.0 / max(times[0], 1e-9)
    for _ in range(_DINKELBACH_ITERS):
        g_mid = 0.5 * (g_lo + g_hi)
        V0, _ = _backward(g_mid, seg_kernels, seg_pos, grid, times, D, succ)
        if V0 > 0:
            g_lo = g_mid
        else:
            g_hi = g_mid
    g_star = 0.5 * (g_lo + g_hi)
    _, cont = _backward(g_star, seg_kernels, seg_pos, grid, times, D, succ)
    gates = {j: _gate_from_cont(cont[j], grid) for j in cont}
    return g_star, gates


# ---------------------------------------------------------------------------
# 前向きパス: 通過率・成功率・期待時間
# ---------------------------------------------------------------------------
def _mass_above(f: np.ndarray, grid: np.ndarray, x: float) -> float:
    """∫_x^∞ f。区分線形密度のもとで厳密 (境界セルは台形の部分積分)。"""
    step = grid[1] - grid[0]
    cum = np.zeros_like(f)
    cum[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * step)
    total = float(cum[-1])
    if x <= grid[0]:
        return total
    if x >= grid[-1]:
        return 0.0
    i = min(int((x - grid[0]) / step), grid.size - 2)
    fx = f[i] + (f[i + 1] - f[i]) * (x - grid[i]) / step
    part = 0.5 * (fx + f[i + 1]) * (grid[i + 1] - x)
    return max(0.0, total - float(cum[i + 1]) + float(part))


def _truncate_below(f: np.ndarray, grid: np.ndarray, x: float) -> np.ndarray:
    """密度 f を x 未満で 0 に打ち切る。x を含むセルの生存部分質量は、格子点
    打ち切り表現で失われる分を差し引きし、部分区間の重心へデポジットして保存する
    (関門での部分セル処理: 裾確率 1e-5 精度のため)。"""
    out = np.where(grid >= x, f, 0.0)
    if not (grid[0] < x < grid[-1]):
        return out
    step = grid[1] - grid[0]
    i = min(int((x - grid[0]) / step), grid.size - 2)
    x1 = grid[i + 1]
    fx = f[i] + (f[i + 1] - f[i]) * (x - grid[i]) / step
    true_mass = 0.5 * (fx + f[i + 1]) * (x1 - x)          # ∫_x^{x1} f (線形)
    kept_mass = 0.5 * f[i + 1] * step                     # 打ち切り表現でのセル質量
    delta = true_mass - kept_mass
    if abs(delta) > 0.0:
        denom = fx + f[i + 1]
        centroid = (0.5 * (x + x1) if denom <= 0.0
                    else (x * (2 * fx + f[i + 1]) + x1 * (fx + 2 * f[i + 1]))
                    / (3.0 * denom))
        _deposit(out, grid, min(max(centroid, x), x1), delta)
    return out


def forward_metrics(seg_blocks, hp, grid, times, D, gates, succ):
    """関門固定で通過率・成功率・期待時間を算出 (密度のブロック伝播)。"""
    K = len(seg_blocks) - 1
    step = grid[1] - grid[0]
    clamp = lambda p: min(1.0, max(0.0, float(p)))

    # seg0: 最初のブロックは厳密 pdf + 原子で立ち上げ、残りをカーネル伝播
    blocks0 = seg_blocks[0]
    f = first_block_density(blocks0[0], hp, grid)
    for block in blocks0[1:]:
        f = kernel_forward(f, grid, block_kernel(block, hp, step))
    m0 = _mass_above(f, grid, grid[0] - 1.0)
    if m0 > 0:
        f = f / m0                              # cp1 到達時点の密度は全確率 1

    pass_rates = []
    exp_time = times[0]
    q_cum = succ[0]
    for j in range(1, K + 1):
        p_dmg = clamp(_mass_above(f, grid, gates[j]))
        p_j = clamp(q_cum * p_dmg)
        pass_rates.append(p_j)
        exp_time += times[j] * p_j
        f = _truncate_below(f, grid, gates[j])   # 関門で打ち切った生存サブ密度
        for block in seg_blocks[j]:
            f = kernel_forward(f, grid, block_kernel(block, hp, step))
        q_cum *= succ[j]
    success = clamp(q_cum * clamp(_mass_above(f, grid, D)))
    g = success / exp_time if exp_time > 0 else 0.0
    return {"success": success, "exp_time": exp_time, "g": g,
            "pass_rates": pass_rates}


# ---------------------------------------------------------------------------
# エントリ
# ---------------------------------------------------------------------------
def analyze_mixed(hit_specs, hp: HPParams, checkpoints, hit_times, D,
                  manual_gates=None, seg_success=None):
    """混在モデルの多段リスタ最適化。restart.analyze と同形式の dict を返す。

    hit_specs : [(HP依存フラグ, 1Hit 混合), ...] (mixed.hit_specs_from_cards)
    manual_gates を渡すと最適化せず、その関門で前向き評価する (手動調整用)。
    seg_success は区間 (= K+1 個) ごとのダメージ独立成功確率 (None なら全 1)。
    """
    specs = normalize_specs(hit_specs, hp)
    n = len(specs)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    bounds, cps = _split_bounds(n, checkpoints)
    times = [float(sum(hit_times[bounds[i]:bounds[i + 1]]))
             for i in range(len(bounds) - 1)]
    succ = _norm_succ(seg_success, len(bounds) - 1)
    seg_blocks = _segment_blocks(specs, bounds, hp.beta)

    full = build_dist_auto(specs, hp)
    base = baseline_nogate(full, times, D, succ)

    s_hi = mixed_support(specs, hp)[1]
    if not (s_hi > 0):
        s_hi = max(D, 1.0)

    # 後ろ向き: 粗めの共通グリッド (評価点を前計算して Dinkelbach で再利用)
    grid_b = np.linspace(0.0, s_hi, _N_GRID_BACKWARD)
    step_b = grid_b[1] - grid_b[0]
    if manual_gates is None:
        seg_kernels = [[block_kernel(b, hp, step_b, _CELL_FACTOR_BACKWARD)
                        for b in blks] for blks in seg_blocks]
        seg_pos = [[kernel_positions(grid_b, k) for k in ks]
                   for ks in seg_kernels]
        g_star, gates = _optimize(seg_kernels, seg_pos, grid_b, times, D, succ)
    else:
        gates = {k: float(np.clip(manual_gates[k - 1], 0.0, s_hi))
                 for k in range(1, len(cps) + 1)}
        g_star = float("nan")

    # 前向き: 最小ブロック幅を ~80 点で解像するグリッド
    widths = [w for blks in seg_blocks for w in
              (_block_width(b, hp) for b in blks) if w > 0]
    n_f = _N_GRID_FORWARD
    if widths:
        n_f = int(np.clip(math.ceil(s_hi / (min(widths) / 80.0)) + 1,
                          _N_GRID_FORWARD, _N_GRID_FORWARD_MAX))
    grid_f = np.linspace(0.0, s_hi, n_f)
    fwd = forward_metrics(seg_blocks, hp, grid_f, times, D, gates, succ)

    cum_max_vals = [mixed_support(specs[:bounds[k]], hp)[1]
                    for k in range(1, len(cps) + 1)]
    gates_dmg = {k: gates[k] for k in gates}
    return _result(n, cps, D, gates_dmg, fwd, base, g_star, cum_max_vals)
