# -*- coding: utf-8 -*-
"""スキル順探索 (app.backend.skill_order) のベンチマーク。

docs/skill_order_theory.md の実測表を再現するスクリプト。

    python experiments/skill_order_bench.py          # 素朴法との比較も行う
    python experiments/skill_order_bench.py --fast   # 素朴法(10! 全列挙)を省く

素朴法は tests/test_skill_order.py の参照実装 _ref_solve をそのまま使う
(初期配置を全列挙して前向きにシミュレートする、正しいが遅い実装)。

ノード数は探索の内部関数 _dfs / _record を包んで数える。ライブラリ側に
計測コードを持ち込まないための計装なので、本番経路には影響しない。
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backend import skill_order as so           # noqa: E402
from app.backend import tl_parse                    # noqa: E402
from tests.test_skill_order import _ref_solve       # noqa: E402

NO_BUDGET = 10 ** 12        # 打ち切りたくないので実質無制限にする
ROOT = Path(__file__).resolve().parent.parent


# --- 計装 -------------------------------------------------------------------
_orig_dfs = so._dfs
_orig_record = so._record
COUNT = {"nodes": 0, "leaves": 0, "dead": 0}
CTX = [None]


def _dfs_counting(hand, deck, i, assign, used, nocop, trace, ctx):
    COUNT["nodes"] += 1
    CTX[0] = ctx                # budget の残りを後から読むため
    before = COUNT["nodes"]
    _orig_dfs(hand, deck, i, assign, used, nocop, trace, ctx)
    if COUNT["nodes"] == before and i < len(ctx.plan):
        COUNT["dead"] += 1      # 子ノードを1つも生まなかった = 死枝の先端


def _record_counting(assign, used, nocop, trace, ctx):
    COUNT["leaves"] += 1
    _orig_record(assign, used, nocop, trace, ctx)


so._dfs = _dfs_counting
so._record = _record_counting


# --- 計測 -------------------------------------------------------------------
def lazy(n, copiers, plan, hand_size, max_results=None, constraints=()):
    """遅延確定 DFS を1回走らせて統計を返す。"""
    COUNT.update(nodes=0, leaves=0, dead=0)
    CTX[0] = None
    stats = {}
    t = time.perf_counter()
    res, trunc = so.solve(n, copiers, plan, constraints, hand_size=hand_size,
                          max_results=max_results, node_budget=NO_BUDGET,
                          stats=stats)
    dt = time.perf_counter() - t
    spent = NO_BUDGET + (max_results or 0) * len(plan) - CTX[0].budget[0]
    return {"N": len(res), "total": so.total_layouts(res), "trunc": trunc,
            "nodes": COUNT["nodes"], "leaves": COUNT["leaves"],
            "dead": COUNT["dead"], "budget": spent,
            "depth": stats["max_depth"], "time": dt}


def naive(n, copiers, plan, hand_size):
    t = time.perf_counter()
    out = _ref_solve(n, copiers, plan, hand_size)
    return {"out": len(out), "time": time.perf_counter() - t}


def cyc(n, m):
    """0,1,...,n-1 を循環させる m 手の手順(枠指定なし)。"""
    return [so.Step(i % n) for i in range(m)]


def show(label, r):
    print(f"{label:<24} N={r['N']:>7,} Σcount={r['total']:>9,} "
          f"nodes={r['nodes']:>10,} budget={r['budget']:>10,} "
          f"dead={r['dead']:>7,} t={r['time']:.3f}s")


def tl_plan():
    """サンプル TL を手順に変換する。"""
    names = ["マリー", "ウイ", "リオ", "レイジョ", "マリナ",
             "キサキ", "シズコ", "臨戦", "ネル", "ナギサ"]
    text = Path(ROOT, "tl_sample/貫通コクマー_123用調整.txt").read_text(
        encoding="utf-8")
    steps, warns = tl_parse.parse_timeline(text, names, copiers={2})
    plan = [so.Step(s.skill,
                    use_copy=(s.kind == tl_parse.KIND_COPY),
                    copy_target=s.target,
                    draw=s.draw,
                    retreat=(s.kind == tl_parse.KIND_RETREAT))
            for s in steps]
    return names, plan, warns


def branch_bound(n, h, plan, copiers):
    """定理 5.7 の上界 B。手順を前からなぞって確定済みスキル数 i を追う。

      - 手札だけを見る手 (通常使用・変化) で新規確定 : 分岐 <= min(h, n-i)
      - 場全体を見る手 (撤退・ドロー付きコピー使用)   : 分岐 <= n-i
      - コピー使用そのもの: 同じ対象のコピーカードは最大 |Cop| 枚
    """
    b, done = 1, set()
    ncop = max(1, len(copiers))
    for st in plan:
        if st.skill is None:
            return None                     # ワイルドカードは対象外
        i = len(done)
        if st.retreat:
            if st.skill not in done:
                b *= n - i
                done.add(st.skill)
        elif st.use_copy:
            b *= ncop
            if st.draw and st.skill not in done:
                b *= n - i
                done.add(st.skill)
        else:
            if st.skill not in done:
                b *= min(h, n - i)
                done.add(st.skill)
            if st.draw and st.skill not in done:
                b *= n - len(done)
    return b


def random_plan(n, h, copiers, m, rng, with_slots):
    """ワイルドカードを含まないランダム手順。"""
    plan = []
    others = [x for x in range(n) if x not in copiers]
    for _ in range(m):
        k = rng.random()
        slot = (rng.randrange(1, h + 1)
                if with_slots and rng.random() < 0.3 else None)
        if k < 0.08:
            plan.append(so.Step(rng.randrange(n), retreat=True))
        elif k < 0.2 and copiers:
            plan.append(so.Step(rng.choice(sorted(copiers)),
                                copy_target=rng.choice(others), slot=slot))
        elif k < 0.3 and copiers:
            plan.append(so.Step(rng.choice(others), use_copy=True, slot=slot,
                                draw=rng.random() < 0.5))
        else:
            s = rng.randrange(n)
            if s in copiers:
                plan.append(so.Step(s, copy_target=rng.choice(others),
                                    slot=slot))
            else:
                plan.append(so.Step(s, slot=slot, draw=rng.random() < 0.35))
    return plan


def check_bound(trials=400, seed=20260818):
    """定理 5.7 (N <= B, nodes <= (m+1)B) を乱択手順で検証する。"""
    rng = random.Random(seed)
    worst_n = worst_nodes = 0.0
    solvable = 0
    for _ in range(trials):
        n = rng.choice([6, 7, 10])
        h = 3 if n == 6 else 5
        copiers = set(rng.sample(range(n), rng.choice([0, 0, 1])))
        m = rng.randrange(2, 16)
        plan = random_plan(n, h, copiers, m, rng, with_slots=rng.random() < .5)
        r = lazy(n, copiers, plan, h)
        b = branch_bound(n, h, plan, copiers)
        assert r["nodes"] <= (m + 1) * b, (n, h, r["nodes"], (m + 1) * b, plan)
        assert r["N"] <= b, (n, h, r["N"], b, plan)
        worst_nodes = max(worst_nodes, r["nodes"] / ((m + 1) * b))
        if r["N"]:
            solvable += 1
            worst_n = max(worst_n, r["N"] / b)
    return trials, solvable, worst_n, worst_nodes


def main():
    with_naive = "--fast" not in sys.argv

    print("== A. 素朴法との比較 (循環手順 Step(i mod n)、枠指定なし) ==")
    for label, n, h, m in [("6枚/手札3・12手", 6, 3, 12),
                           ("10枚/手札5・4手", 10, 5, 4),
                           ("10枚/手札5・14手", 10, 5, 14),
                           ("10枚/手札5・18手", 10, 5, 18),
                           ("7枚/手札5・6手", 7, 5, 6)]:
        plan = cyc(n, m)
        r = lazy(n, set(), plan, h)
        show(label, r)
        if with_naive:
            q = naive(n, set(), plan, h)
            print(f"{'':<24} 素朴: 出力={q['out']:>9,} t={q['time']:.2f}s "
                  f"(圧縮 {q['out'] / max(r['N'], 1):.0f}x, "
                  f"高速化 {q['time'] / max(r['time'], 1e-9):.0f}x)")

    print("\n== B. 長い手順 (10枚/手札5、循環98手) ==")
    show("98手・枠指定なし", lazy(10, set(), cyc(10, 98), 5))
    p = cyc(10, 98)
    p[0] = so.Step(0, slot=1)
    show("98手・1手目を枠固定", lazy(10, set(), p, 5))

    print("\n== C. 枠指定の効果 (10枚/手札5・18手、先頭 k 手の枠を固定) ==")
    base = cyc(10, 18)
    for k in range(6):
        p = [so.Step(s.skill, slot=(i + 1 if i < k else None))
             for i, s in enumerate(base)]
        show(f"固定 {k} 手", lazy(10, set(), p, 5))

    print("\n== D. 複製 + ドロー (10枚/手札5、複製=0) ==")
    cop = [so.Step(0, copy_target=1),
           so.Step(1, use_copy=True, draw=True),
           so.Step(1), so.Step(2), so.Step(3), so.Step(4),
           so.Step(5, draw=True), so.Step(5)]
    r = lazy(10, {0}, cop, 5)
    show("複製+ドロー8手", r)
    if with_naive:
        q = naive(10, {0}, cop, 5)
        print(f"{'':<24} 素朴: 出力={q['out']:>9,} t={q['time']:.2f}s")

    print("\n== E. ワイルドカードのみ (10枚/手札5、複製=0) ==")
    for k in range(1, 5):
        show(f"ワイルド{k}手", lazy(10, {0}, [so.Step(None)] * k, 5))

    print("\n== F. max_results の効果 (10枚/手札5・18手) ==")
    for mr in [50, 200, 500, 1000, 5000, 20000, None]:
        r = lazy(10, set(), cyc(10, 18), 5, max_results=mr)
        show(f"max_results={mr}", r)

    print("\n== G. 実 TL (tl_sample/貫通コクマー_123用調整.txt) ==")
    names, plan, warns = tl_plan()
    print(f"手順長 m={len(plan)}、読み取り警告 {len(warns)} 件")
    r = lazy(10, {2}, plan, 5)
    show("TL 全体", r)
    d = r["depth"]
    print(f"  到達できた手順数 max_depth={d} "
          f"→ {d + 1}手目「{names[plan[d].skill]}」で不成立")
    r2 = lazy(10, {2}, plan[:d], 5)
    show(f"TL 先頭{d}手", r2)

    print("\n== I. ワイルドカード無しの上界 (定理 5.7) の検証 ==")
    trials, solvable, wn, wnodes = check_bound()
    print(f"乱択 {trials} 手順 (うち解あり {solvable}) すべてで "
          f"N <= B かつ nodes <= (m+1)B")
    print(f"  最大比 N/B = {wn:.3f},  nodes/((m+1)B) = {wnodes:.3f}")
    for label, n, h, m in [("6枚/手札3・12手", 6, 3, 12),
                           ("10枚/手札5・14手", 10, 5, 14)]:
        plan = cyc(n, m)
        r = lazy(n, set(), plan, h)
        b = branch_bound(n, h, plan, set())
        print(f"  巡回 {label}: N={r['N']:,} B={b:,} (等号) "
              f"nodes={r['nodes']:,} <= {(m + 1) * b:,}")

    print("\n== H. 初期配置の種類数 ==")
    for label, n, h, m in [("6枚/手札3・12手", 6, 3, 12),
                           ("10枚/手札5・4手", 10, 5, 4)]:
        res, _ = so.solve(n, set(), cyc(n, m), hand_size=h,
                          node_budget=NO_BUDGET)
        t = time.perf_counter()
        cnt, exact = so.distinct_layouts(res)
        print(f"{label:<24} 解={len(res):,} Σcount={so.total_layouts(res):,} "
              f"distinct={cnt:,}{'' if exact else '+'} "
              f"t={time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
