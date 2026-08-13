# -*- coding: utf-8 -*-
"""ブルーアーカイブ スキル順(開始スキル設定)探索スクリプト

Web UI と同じ探索ロジック (app/backend/skill_order.py) をコマンドラインから
使うためのスクリプト。下部の設定を書き換えて実行する。

    python docs/ba_skill_order.py

モデル:
  - カードは全 CARDS 枚。うち先頭 HAND_SIZE 枚が手札(スロット1..HAND_SIZE)、
    残りが山札(上から順)。通常戦は 6枚/手札3、制約解除決戦は 10枚/手札5。
  - 初期配置(全カードの並び)が決定変数。
  - 手札のカードのみ使用可能。
  - スロット i のカードを使うと、そのカードは山札の一番下へ行き、
    山札の一番上のカードがスロット i にドローされる。
  - カード枚数が手札枚数に満たない場合、余った手札スロットは空欄になる。

手順に書ける要素:
  use(name, slot=None, draw=False)   通常のスキル使用
  wild(slot=None, draw=False)        何でもいい繋ぎの1枚
  copy_use(name, ...)                「name(コピー)」を使用する
  retreat(name)                      name のキャラがそこで撤退する

複製スキル(リオなど)は COPIERS に名前を書き、use(複製キャラ, target=対象) の
ように複製対象を指定する。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backend import skill_order as so   # noqa: E402


# --- 設定 -------------------------------------------------------------------
# カード名。CARDS 枚ぶん先頭から使う。
SKILLS = ["ハレ", "ヒカリ", "ノゾミ", "キサキ", "カヨコ", "ナギサ"]

CARDS = 6                          # カード枚数 (1〜10)
HAND_SIZE = so.HAND_SIZE_NORMAL    # 通常戦=3 / 制約解除決戦=so.HAND_SIZE_DECISIVE
COPIERS = []                       # 複製スキル持ちのカード名


# --- 手順の書き方ヘルパー ---------------------------------------------------
def _idx(name):
    try:
        return SKILLS.index(name)
    except ValueError:
        raise SystemExit(f"SKILLS に '{name}' がありません")


def use(name, slot=None, draw=False, target=None):
    """通常のスキル使用。複製キャラなら target に複製対象名を指定する。"""
    return so.Step(_idx(name), slot=slot, draw=draw,
                   copy_target=None if target is None else _idx(target))


def copy_use(name, slot=None, draw=False):
    """「name(コピー)」を使用する。"""
    return so.Step(_idx(name), use_copy=True, slot=slot, draw=draw)


def wild(slot=None, draw=False):
    """何でもいい繋ぎの1枚。"""
    return so.Step(None, slot=slot, draw=draw)


def retreat(name):
    """name のキャラがそこで撤退する。"""
    return so.Step(_idx(name), retreat=True)


# 手順間の制約 (添字は PLAN の 0 始まり)
different_slots = so.different_slots
same_slot = so.same_slot


# --- 出力 -------------------------------------------------------------------
def report(results, truncated, plan, names, hand_size, limit=None):
    plan_str = " → ".join(_step_desc(s, names, hand_size) for s in plan)
    labels = so.slot_labels(hand_size)
    n_layouts, exact = so.distinct_layouts(results)
    print(f"手順: {plan_str}")
    print(f"解の数: {len(results)}{'+' if truncated else ''}"
          f"  (初期配置 {n_layouts} 通り"
          f"{'' if exact and not truncated else '以上'})")
    print(f"スロット: {' / '.join(labels)}")
    print()
    shown = results if limit is None else results[:limit]
    for sol in shown:
        cards = "  ".join(
            f"{i + 1}:{names[s] if s is not None else '任意'}"
            for i, s in enumerate(sol.layout))
        seq = "  ".join(so.trace_entry_label(e, names, hand_size)
                        for e in sol.trace)
        suffix = f"  ({sol.count}通り)" if sol.count > 1 else ""
        print(f"初期配置: {cards}{suffix}")
        print(f"    使用順: {seq}")
    if limit is not None and len(results) > limit:
        print(f"... 他 {len(results) - limit} 件")


def _step_desc(step, names, hand_size):
    if step.retreat:
        return f"↩{names[step.skill]}撤退"
    if step.skill is None:
        s = "＊"
    elif step.use_copy:
        s = f"{names[step.skill]}(コピー)"
    else:
        s = names[step.skill]
        if step.copy_target is not None:
            s += f"→{names[step.copy_target]}(コピー)"
    if step.slot is not None:
        s += f"@{so.slot_labels(hand_size)[step.slot - 1]}"
    if step.draw:
        s += "+ドロー"
    return s


if __name__ == "__main__":
    # 例: 6枚・手札3。1周してから最後にノゾミを左で使いたい。
    PLAN = [
        use("ハレ"),
        use("ヒカリ"),
        use("ノゾミ"),
        use("キサキ"),
        use("カヨコ"),
        use("ハレ"),
        use("ナギサ"),
        use("ヒカリ"),
        use("カヨコ"),
        use("ノゾミ"),
        use("キサキ"),
        use("ヒカリ"),
        use("ハレ"),
        use("ノゾミ"),
        use("ナギサ"),
        use("ヒカリ"),
        use("キサキ"),
        use("ノゾミ", slot=1),
    ]

    # 手順間の制約(添字は PLAN の 0 始まり)
    CONSTRAINTS = [
        different_slots(2, 3),   # 3手目ノゾミ と 4手目キサキ を違うスロットに
    ]

    names = [SKILLS[i] if i < len(SKILLS) else f"カード{i + 1}"
             for i in range(CARDS)]
    results, truncated = so.solve(
        CARDS, {_idx(c) for c in COPIERS}, PLAN, CONSTRAINTS,
        hand_size=HAND_SIZE, max_results=20_000)
    report(results, truncated, PLAN, names, HAND_SIZE, limit=60)
