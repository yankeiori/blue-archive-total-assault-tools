"""蓄積 (チャージ) 型スキルつきの多段リスタ最適化。

理論は docs/accumulate.md §4。要点は 1 つだけ:

> チェックポイントを蓄積窓の**境界**にだけ置く限り、窓を丸ごと含む区間の増分
>     Z_k = S_k + g(mult · min(C_k, α S_k))
> は区間開始時の累積ダメージに依存しない独立確率変数である。

したがって状態は実効ダメージ 1 次元のままで、restart.py の Dinkelbach + 後ろ向き
帰納 (Bermudan 型) がそのまま成立する。実装としては「その区間の増分分布を
build_accum_dist の結果に差し替える」だけでよい。

逆に窓が足切り境界をまたぐと、十分統計量が (実効ダメージ, 残り上限) の 2 次元に
なり 1 次元 DP では解けない (docs/accumulate.md §4.1-4.2)。その場合は ValueError を
投げて呼び出し側に知らせる — 黙って無視すると蓄積ぶんを取り違えた足切り線が出る。

**爆発の着弾位置は最適解を変えない。** プールが確定した時点で、その分は確保済みの
確定ダメージであり (上限も蓄積量も画面から分かる)、十分統計量は「実ダメージ + 確定
プール」= 実効ダメージだからである。着弾が後ろにズレても最適方策は同じで、変わるのは
「その関門で画面に爆発分が乗っているかどうか」= 読み方だけ。そこで AccumWindow.burst_hit
を受け取り、関門ごとに未着弾かどうかと未着弾ぶんの目安を accum_stats に載せる。

出力 dict は restart.analyze と同形式 (callbacks から差し替え可能)。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.backend.accumulate import AccumWindow, CapSpec, build_accum_dist
from app.backend.restart import (
    _norm_succ,
    _result,
    baseline_nogate,
    forward_metrics,
    optimize,
)

# Dinkelbach の二分回数。g の相対精度 2^-40 で、関門位置の分解能 (グリッド) より
# 十分細かい。restart.py の既定 80 は過剰で、蓄積ありは 1 段が重いので減らす。
_DINKELBACH_ITERS = 40


# ---------------------------------------------------------------------------
# 窓を区間へ割り当てる
# ---------------------------------------------------------------------------
def window_span(win: AccumWindow) -> tuple[int, int]:
    """窓が触れる Hit 番号の範囲 [最小, 最大]。上限ロールの Hit も含む。"""
    hits = [*win.hits, *win.cap.source_hits()]
    return min(hits), max(hits)


def burst_hit_of(win: AccumWindow) -> int:
    """爆発が着弾する Hit 番号 (未指定なら窓の最後の Hit)。"""
    if win.burst_hit is None:
        return max(win.hits) if win.hits else 0
    return int(win.burst_hit)


def validate_burst_positions(windows) -> None:
    """爆発の着弾位置が蓄積の終わりより前なら ValueError。"""
    for win in windows:
        if not win.hits:
            continue
        last = max([*win.hits, *win.cap.source_hits()])
        if burst_hit_of(win) < last:
            name = win.name or "蓄積スキル"
            raise ValueError(
                f"「{name}」の爆発カードが蓄積の終わり (Hit {last + 1}) より前です。"
                "爆発は蓄積が止まったあとに起きるので、蓄積対象の最後のカード以降を"
                "指定してください。")


def split_windows(windows, bounds) -> list[list[AccumWindow]]:
    """各窓を、それを丸ごと含む区間へ割り当てる (Hit 番号は区間内へ付け替える)。

    境界をまたぐ窓があれば ValueError。
    """
    per_seg: list[list[AccumWindow]] = [[] for _ in range(len(bounds) - 1)]
    for win in windows:
        if not win.hits:
            continue
        lo, hi = window_span(win)
        seg = next(i for i in range(len(bounds) - 1)
                   if bounds[i] <= lo < bounds[i + 1])
        if hi >= bounds[seg + 1]:
            cp = bounds[seg + 1]
            name = win.name or "蓄積スキル"
            raise ValueError(
                f"「{name}」の蓄積範囲 (Hit {lo + 1}〜{hi + 1}) が足切り"
                f"(チェックポイント {cp} Hit) をまたいでいます。"
                "蓄積の途中で足切りを判断するには累積ダメージだけでは足りず"
                "(残り上限も要る) 1 次元の最適化では解けません。"
                "チェックポイントを蓄積の切れ目に置くか、対象カードを見直してください。")
        off = bounds[seg]
        per_seg[seg].append(replace(
            win,
            hits=[h - off for h in win.hits],
            cap=replace(win.cap, hits=[h - off for h in win.cap.source_hits()],
                        hit=None) if win.cap.source_hits() else win.cap,
        ))
    return per_seg


# ---------------------------------------------------------------------------
# エントリ
# ---------------------------------------------------------------------------
def analyze_accum(hit_mixtures, windows, checkpoints, hit_times, D,
                  manual_gates=None, seg_success=None, burst_decay=False):
    """蓄積スキルつき和モデルの多段リスタ最適化。

    hit_mixtures : 各 Hit の一様混合 (build_hit_mixtures の出力)
    windows      : AccumWindow のリスト (Hit 番号は hit_mixtures 全体での通し番号)
    checkpoints  : チェックポイントのヒット数 m_j (1..n-1)
    manual_gates : 渡すと最適化せずその関門で前向き評価する (手動調整用)
    返り値: restart.analyze と同形式 + "accum_stats" (窓ごとの診断)
    """
    n = len(hit_mixtures)
    if isinstance(hit_times, (int, float)):
        hit_times = [float(hit_times)] * n
    cps = sorted({int(c) for c in checkpoints if 0 < int(c) < n})
    bounds = [0, *cps, n]
    times = [float(sum(hit_times[bounds[i]:bounds[i + 1]]))
             for i in range(len(bounds) - 1)]
    succ = _norm_succ(seg_success, len(bounds) - 1)

    validate_burst_positions(windows)
    per_seg = split_windows(windows, bounds)
    seg_dists = [
        build_accum_dist(hit_mixtures[bounds[i]:bounds[i + 1]], per_seg[i],
                         burst_decay=burst_decay)
        for i in range(len(bounds) - 1)
    ]
    full = build_accum_dist(hit_mixtures, windows, burst_decay=burst_decay)

    base = baseline_nogate(full, times, D, succ)
    if manual_gates is None:
        g_star, gates, _grid = optimize(seg_dists, times, D, succ=succ,
                                        iters=_DINKELBACH_ITERS)
    else:
        s_hi = sum(d.support_hi for d in seg_dists)
        gates = {k: float(np.clip(manual_gates[k - 1], 0.0, s_hi))
                 for k in range(1, len(cps) + 1)}
        g_star = float("nan")
    fwd = forward_metrics(seg_dists, times, D, gates, succ=succ)

    cum_max = np.cumsum([d.support_hi for d in seg_dists])
    res = _result(n, cps, D, dict(gates), fwd, base, g_star,
                  [float(cum_max[k - 1]) for k in range(1, len(cps) + 1)])
    live = [w for w in windows if w.hits and w.rate > 0 and w.burst_mult > 0
            and w.emit]
    res["accum_stats"] = [
        {"name": st.name, "rate": st.rate, "sat_prob": st.sat_prob,
         "pool_mean": st.pool_mean, "overflow_mean": st.overflow_mean,
         "burst_mean": st.burst_mean, "burst_lo": st.burst_lo,
         "burst_hi": st.burst_hi, "cap_mean": st.cap_mean,
         "damage_mean": st.damage_mean, "burst_hit": burst_hit_of(win)}
        for st, win in zip(full.window_stats, live)
    ]
    return res





# ---------------------------------------------------------------------------
# dcc.Store 用のシリアライズ (リスタライン手動調整で再計算するため)
# ---------------------------------------------------------------------------
def windows_to_json(windows) -> list[dict]:
    """AccumWindow を JSON 化する。Hit 番号で持つのでカード構成に依存しない。"""
    return [{"hits": list(w.hits), "rate": w.rate, "burst_mult": w.burst_mult,
             "burst_decay": w.burst_decay, "name": w.name,
             "burst_hit": burst_hit_of(w),
             "cap": {"kind": w.cap.kind, "value": w.cap.value,
                     "hits": w.cap.source_hits(), "coef": w.cap.coef}}
            for w in windows]


def windows_from_json(rows) -> list[AccumWindow]:
    """windows_to_json の逆。"""
    out = []
    for r in rows or []:
        c = r.get("cap") or {}
        out.append(AccumWindow(
            hits=[int(h) for h in (r.get("hits") or [])],
            rate=float(r.get("rate") or 0.0),
            cap=CapSpec(kind=c.get("kind") or "fixed",
                        value=float(c.get("value") or 0.0),
                        hits=[int(h) for h in (c.get("hits") or [])],
                        coef=float(c.get("coef") or 1.0)),
            burst_mult=float(r.get("burst_mult") or 1.0),
            burst_decay=bool(r.get("burst_decay")),
            name=r.get("name") or "",
            burst_hit=(None if r.get("burst_hit") is None
                       else int(r["burst_hit"]))))
    return out


__all__ = ["AccumWindow", "CapSpec", "analyze_accum", "burst_hit_of",
           "split_windows", "validate_burst_positions", "window_span",
           "windows_to_json", "windows_from_json"]
