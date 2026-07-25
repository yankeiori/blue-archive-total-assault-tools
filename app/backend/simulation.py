import os

import numpy as np
import plotly.graph_objects as go

N_SAMPLES = int(os.environ.get("N_SAMPLES", 200_000))
_CHUNK_SIZE = 50_000

# ((x_min, x_max), (a, b)) -> y = a * x + b if x_min <= x < x_max
DAMAGE_FUNC = [
    ((0, 4000000), (1.0, 0)),
    ((4000000, 6248000), (0.8, 4000000 - 0.8 * 4000000)),
    ((6248000, 8496000), (0.65, 5798400 - 0.65 * 6248000)),
    ((8496000, 10744000), (0.5, 7259600 - 0.5 * 8496000)),
    ((10744000, 12992000), (0.4, 8383600 - 0.4 * 10744000)),
    ((12992000, 15240000), (0.3, 9282800 - 0.3 * 12992000)),
    ((15240000, 17488000), (0.225, 9957200 - 0.225 * 15240000)),
    ((17488000, 19736000), (0.15, 10463000 - 0.15 * 17488000)),
    ((19736000, 22000000), (0.075, 10800200 - 0.075 * 19736000)),
    ((22000000, 10**20), (0.0, 10966999)),
]

# 逆変換テーブル: 減衰後の境界値 y = a * x_min + b を事前計算
_INVERSE_TABLE: list[tuple[tuple[float, float], float, float]] = []
for (x_lo, x_hi), (a, b) in DAMAGE_FUNC:
    y_lo = a * x_lo + b
    y_hi = a * x_hi + b
    _INVERSE_TABLE.append(((y_lo, y_hi), a, b))


def decay(x: np.ndarray) -> np.ndarray:
    """減衰関数: 生ダメージ x → 減衰後ダメージ y"""
    y = np.empty_like(x)
    for (x_lo, x_hi), (a, b) in DAMAGE_FUNC:
        mask = (x >= x_lo) & (x < x_hi)
        y[mask] = a * x[mask] + b
    return y


def inverse_decay(y: float) -> float:
    """逆変換: 減衰後ダメージ y → 生ダメージ x (スカラー)"""
    for (y_lo, y_hi), a, b in _INVERSE_TABLE:
        lo, hi = min(y_lo, y_hi), max(y_lo, y_hi)
        if lo <= y <= hi:
            if a == 0.0:
                return DAMAGE_FUNC[-1][0][0]  # 上限キャップ
            return (y - b) / a
    # テーブル範囲外は恒等
    return y


# 減衰後ダメージの上限キャップ (DAMAGE_FUNC 末尾の定数項)。これ以上の減衰後値は
# 生ダメージへ一意に逆変換できない (情報が潰れる) ため、安定値から最大を逆算する。
DAMAGE_CAP = DAMAGE_FUNC[-1][1][1]


def stability_min_ratio(stability: float) -> float:
    """安定値 x に対する「減衰前最小 / 減衰前最大」比率。
    減衰前最小 = 減衰前最大 × (1 − 1/(1+0.001x) + 0.2)。"""
    return 1.0 - 1.0 / (1.0 + 0.001 * stability) + 0.2


def raw_damage_bounds(
    post_min: float, post_max: float, stability, damage_mode: str,
) -> tuple[float, float]:
    """ダメージ型 (会心/非会心) 1 つの (減衰前下限, 減衰前上限) を返す。

    減衰考慮前モードでは入力値をそのまま使う。減衰考慮済みモードでは逆変換するが、
    最大が減衰上限 (DAMAGE_CAP) に達していて安定値が与えられている場合は、最大を
    逆変換すると情報が潰れる (キャップに張り付く) ため、減衰前最小から
    減衰前最大 = 減衰前最小 / stability_min_ratio(x) で逆算する。"""
    if damage_mode != "post_decay":
        return post_min, max(post_max, post_min)
    raw_lo = inverse_decay(post_min)
    if stability is not None and post_max >= DAMAGE_CAP:
        ratio = stability_min_ratio(float(stability))
        raw_hi = raw_lo / ratio if ratio > 0 else raw_lo
    else:
        raw_hi = inverse_decay(post_max)
    return raw_lo, max(raw_hi, raw_lo)


def _extract_hit_params(
    indices: list[int],
    params: dict[int, dict],
    global_crit: float,
    global_evade: float,
    damage_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """カードリストからヒットごとのパラメータを展開し、6本の配列として返す。"""
    crit_lows: list[float] = []
    crit_highs: list[float] = []
    normal_lows: list[float] = []
    normal_highs: list[float] = []
    crit_rates: list[float] = []
    evade_rates: list[float] = []

    for idx in indices:
        p = params.get(idx)
        if p is None:
            continue

        crit_min = float(p.get("crit_min") or 0)
        crit_max = float(p.get("crit_max") or 0)
        normal_min = float(p.get("normal_min") or 0)
        normal_max = float(p.get("normal_max") or 0)
        hits = int(p.get("hits") or 1)
        cr = p.get("crit_rate")
        er = p.get("evade_rate")
        cr = float(cr if cr is not None else global_crit or 0) / 100.0
        er = float(er if er is not None else global_evade or 0) / 100.0
        stab = p.get("stability")
        stab = None if (stab is None or stab == "") else float(stab)

        raw_crit_lo, raw_crit_hi = raw_damage_bounds(crit_min, crit_max, stab, damage_mode)
        raw_norm_lo, raw_norm_hi = raw_damage_bounds(normal_min, normal_max, stab, damage_mode)

        for _ in range(hits):
            crit_lows.append(raw_crit_lo)
            crit_highs.append(raw_crit_hi)
            normal_lows.append(raw_norm_lo)
            normal_highs.append(raw_norm_hi)
            crit_rates.append(cr)
            evade_rates.append(er)

    return (
        np.asarray(crit_lows),
        np.asarray(crit_highs),
        np.asarray(normal_lows),
        np.asarray(normal_highs),
        np.asarray(crit_rates),
        np.asarray(evade_rates),
    )


def _simulate_chunk(
    rng: np.random.Generator,
    crit_lows: np.ndarray,
    crit_highs: np.ndarray,
    normal_lows: np.ndarray,
    normal_highs: np.ndarray,
    crit_rates: np.ndarray,
    evade_rates: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """チャンク単位でシミュレーションし、合計ダメージを返す。"""
    n_hits = len(crit_lows)

    hit_mask = rng.random((n_hits, n_samples)) >= evade_rates[:, None]
    is_crit = rng.random((n_hits, n_samples)) < crit_rates[:, None]

    u = rng.random((n_hits, n_samples))
    crit_raw = u * (crit_highs - crit_lows)[:, None] + crit_lows[:, None]

    u = rng.random((n_hits, n_samples))
    norm_raw = u * (normal_highs - normal_lows)[:, None] + normal_lows[:, None]

    raw_samples = np.where(is_crit, crit_raw, norm_raw)
    del crit_raw, norm_raw, is_crit

    dmg = decay(raw_samples.ravel()).reshape(n_hits, n_samples)
    del raw_samples

    return np.sum(dmg * hit_mask, axis=0)


def _simulate_vectorized(
    rng: np.random.Generator,
    crit_lows: np.ndarray,
    crit_highs: np.ndarray,
    normal_lows: np.ndarray,
    normal_highs: np.ndarray,
    crit_rates: np.ndarray,
    evade_rates: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """全ヒットをまとめてベクトル演算でシミュレーションし、合計ダメージを返す。

    メモリ使用量を抑えるため、チャンク分割して処理する。
    """
    n_hits = len(crit_lows)
    if n_hits == 0:
        return np.zeros(n_samples)

    if n_samples <= _CHUNK_SIZE:
        return _simulate_chunk(
            rng, crit_lows, crit_highs, normal_lows, normal_highs,
            crit_rates, evade_rates, n_samples,
        )

    parts = []
    remaining = n_samples
    while remaining > 0:
        chunk = min(remaining, _CHUNK_SIZE)
        parts.append(_simulate_chunk(
            rng, crit_lows, crit_highs, normal_lows, normal_highs,
            crit_rates, evade_rates, chunk,
        ))
        remaining -= chunk
    return np.concatenate(parts)


def run_simulation(
    indices: list[int],
    params: dict[int, dict],
    global_crit: float,
    global_evade: float,
    target_damage: float,
    damage_mode: str = "post_decay",
) -> tuple[go.Figure, str]:
    """モンテカルロ法でダメージ分布をシミュレーションし、Figureと通過率テキストを返す。"""
    rng = np.random.default_rng()

    hit_params = _extract_hit_params(indices, params, global_crit, global_evade, damage_mode)
    total_damage = _simulate_vectorized(rng, *hit_params, N_SAMPLES)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=total_damage, nbinsx=200, name="ダメージ分布"))
    fig.update_layout(
        title="合計ダメージ分布",
        xaxis_title="合計ダメージ",
        yaxis_title="頻度",
        bargap=0.05,
    )

    mean_val = np.mean(total_damage)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"期待値: {mean_val:,.0f}")

    target = float(target_damage or 0)
    pass_rate = float(np.mean(total_damage >= target) * 100) if target > 0 else None
    pass_text = ""
    if target > 0:
        fig.add_vline(x=target, line_dash="solid", line_color="green", annotation_text=f"目標: {target:,.0f}")
        pass_text = f"目標ダメージ {target:,.0f} の通過確率: {pass_rate:.2f}%"

    return fig, pass_text

