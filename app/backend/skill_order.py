# -*- coding: utf-8 -*-
"""ブルーアーカイブ スキル順(開始スキル設定)探索ロジック。

モデル:
  - カードは全 n 枚。うち先頭 hand_size 枚が手札(スロット1..hand_size)、
    残りが山札(上から順)。通常戦は 6枚/手札3、制約解除決戦は 10枚/手札5。
  - 初期配置(n 枚の並び)が決定変数。
  - 手札のカードのみ使用可能。
  - 通常カードをスロット i で使うと、そのカードは山札の一番下へ行き、
    山札の一番上のカードがスロット i にドローされる。
  - カード枚数が手札枚数に満たない場合、余った手札スロットは空欄になる。
    空欄スロットは以後も補充されない。

拡張:
  - 複製スキル: 複製キャラ A が対象 B を指定して撃つと、カード A が
    その場で「B(コピー)」に変化する(山札への移動・ドローは発生しない)。
    「B(コピー)」を使用するとカードは A に戻って山札の一番下へ行く。
  - ドローフラグ: 手順ステップにドローフラグが付いている場合、
    使用カードが山札の下へ行った後、「使用したスキルの元カード」が
    山札にあるかチェックし、あれば元々カードがあったスロットへ
    そのカードがドローされる(山札の途中からでも引き抜く)。
    無ければ通常どおり山札の一番上をドローする。
  - 撤退ステップ: そのキャラのカードを場(手札・山札)から完全に除外する。
    手札にあった場合はそのスロットへ山札の一番上をドローする
    (山札が空なら空欄になる)。複製キャラが撤退した場合、その場に出ている
    「◯◯(コピー)」もそのキャラのカードなので一緒に除外される。

カード表現:
  - 通常カード  : ("N", i)      i はスキル添字
  - コピーカード: ("C", a, b)   a=複製キャラ添字, b=複製対象添字
  - 未確定カード: ("P", p)      p は初期配置の位置。まだ正体が決まっていない
  - 空欄        : None

探索方式:
  初期配置を全列挙すると 10 枚で 10! = 3,628,800 通りになり実用に耐えない。
  そこで全カードを「未確定カード」として置き、手順がそのカードを要求した
  時点で初めて「この位置はこのスキル」と確定させる(遅延束縛)。手順に
  一度も現れないカードは未確定のまま残り、その並べ替えは解を展開せずに
  通り数として数え上げる。
"""

from math import factorial

HAND_SIZE_NORMAL = 3        # 通常戦
HAND_SIZE_DECISIVE = 5      # 制約解除決戦

SLOT_LABELS = ("左", "中", "右")
SLOT_LABELS_5 = ("1番目", "2番目", "3番目", "4番目", "5番目")


def slot_labels(hand_size=HAND_SIZE_NORMAL):
    """手札枚数に応じたスロット表示名。"""
    if hand_size == 3:
        return SLOT_LABELS
    if hand_size == 5:
        return SLOT_LABELS_5
    return tuple(str(i + 1) for i in range(hand_size))


class SearchBudgetExceeded(Exception):
    """探索ノード数が上限を超えた(ワイルドカード過多などで組合せ爆発)。"""


class _StopSearch(Exception):
    """十分な数の解が集まったので探索を打ち切る(内部用)。"""


class Step:
    """手順の1ステップ。

    skill      : 使う(または撤退する)スキルの添字。None なら
                 「何でもいい(繋ぎ)」。ただしワイルドカードは複製キャラの
                 元カードを使わない(複製対象を決められないため)。
    use_copy   : True ならスキル skill のコピーカード「skill(コピー)」を使う。
    copy_target: skill が複製キャラのとき必須。複製する対象スキルの添字。
    slot       : 使ってほしいスロット番号(1..hand_size)。None なら任意。
    draw       : ドローフラグ。
    retreat    : True なら「スキル skill のキャラが撤退する」イベント。
                 カード使用ではないので slot / draw は無視される。
    """

    def __init__(self, skill=None, *, use_copy=False, copy_target=None,
                 slot=None, draw=False, retreat=False):
        self.skill = skill
        self.use_copy = use_copy
        self.copy_target = copy_target
        self.slot = slot
        self.draw = draw
        self.retreat = retreat

    def __repr__(self):
        if self.retreat:
            return f"retreat({self.skill})"
        s = "*" if self.skill is None else (
            f"copy({self.skill})" if self.use_copy else str(self.skill))
        p = self.slot if self.slot is not None else "*"
        d = "+draw" if self.draw else ""
        return f"{s}@{p}{d}"


class Solution(tuple):
    """1つの解。(初期配置, trace) のタプルとして振る舞う。

    layout   : 長さ n のタプル。各要素はスキル添字、または None(=任意のカード
               でよい未確定位置)。
    trace    : [(使用スロット, 使用カード, action, step), ...]
               action は "use" / "transform" / "retreat"。
               撤退ステップで山札のカードが撤退した場合スロットは 0。
    count    : この解が表す具体的な初期配置の通り数。
    free     : 未確定位置のタプル。
    nocopier : 「複製キャラであってはならない」未確定位置の集合
               (ワイルドカードで使われた位置)。
    """

    def __new__(cls, layout, trace, count, free, nocopier, n, copiers):
        obj = super().__new__(cls, (layout, trace))
        obj.layout = layout
        obj.trace = trace
        obj.count = count
        obj.free = free
        obj.nocopier = nocopier
        obj._n = n
        obj._copiers = copiers
        return obj

    def expand(self, limit=None):
        """未確定位置に残りスキルを割り当てた具体的な初期配置を列挙する。"""
        assigned = {s for s in self.layout if s is not None}
        rest = [s for s in range(self._n) if s not in assigned]
        out = []

        def rec(i, cur, left):
            if limit is not None and len(out) >= limit:
                return
            if i == len(self.free):
                out.append(tuple(cur))
                return
            pos = self.free[i]
            for s in list(left):
                if pos in self.nocopier and s in self._copiers:
                    continue
                cur[pos] = s
                rec(i + 1, cur, left - {s})
                cur[pos] = None
                if limit is not None and len(out) >= limit:
                    return

        rec(0, list(self.layout), set(rest))
        return out


def card_label(card, names):
    """カードの表示名。"""
    if card is None:
        return "(空)"
    if card[0] == "N":
        return names[card[1]]
    if card[0] == "P":
        return "任意"
    return f"{names[card[2]]}(コピー)"


def trace_entry_label(entry, names, hand_size=HAND_SIZE_NORMAL):
    """トレース1要素の表示文字列 ([左/中/右]表記)。"""
    slot, card, action, step = entry
    labels = slot_labels(hand_size)
    if action == "retreat":
        where = f"[{labels[slot - 1]}]" if slot else "[山札]"
        return f"{where}{names[step.skill]}撤退"
    pos = f"[{labels[slot - 1]}]"
    if action == "transform":
        return f"{pos}{names[card[1]]}→{names[step.copy_target]}(コピー)"
    label = card_label(card, names)
    if step.draw:
        label += "(ドロー)"
    return f"{pos}{label}"


# --- 手順間の制約 -----------------------------------------------------------
def different_slots(*indices):
    """指定した手順どうしを全て違うスロットにする制約。添字は0始まり。"""
    def check(trace):
        slots = [trace[i][0] for i in indices]
        return len(set(slots)) == len(slots)
    return check


def same_slot(*indices):
    """指定した手順どうしを全て同じスロットにする制約。添字は0始まり。"""
    def check(trace):
        slots = [trace[i][0] for i in indices]
        return len(set(slots)) == 1
    return check


# --- 探索本体 ---------------------------------------------------------------
class _Ctx:
    __slots__ = ("n", "copiers", "plan", "hand_size", "constraints",
                 "max_results", "budget", "results", "truncated", "max_depth")


def _perm(u, k):
    """順列数 P(u, k)。"""
    if k > u:
        return 0
    r = 1
    for i in range(k):
        r *= (u - i)
    return r


def _locate(hand, deck, skill, assign, used, nocop, ctx):
    """スキル skill のカードの居場所を確定させる。

    yield (hand, deck, assign, used, where)
      where = ("hand", i) / ("deck", i) / None(撤退済みで場に無い)
    skill が未確定なら「どの未確定位置が skill だったか」で分岐する。
    """
    if skill in used:
        for i, c in enumerate(hand):
            if c is not None and c[0] in ("N", "C") and c[1] == skill:
                yield hand, deck, assign, used, ("hand", i)
                return
        for i, c in enumerate(deck):
            if c[0] in ("N", "C") and c[1] == skill:
                yield hand, deck, assign, used, ("deck", i)
                return
        yield hand, deck, assign, used, None
        return

    # 未確定 → 生きている未確定カードのどれかが skill である
    blocked = skill in ctx.copiers
    for i, c in enumerate(hand):
        if c is not None and c[0] == "P" and not (blocked and c[1] in nocop):
            nh = list(hand)
            nh[i] = ("N", skill)
            na = dict(assign)
            na[c[1]] = skill
            yield nh, deck, na, used | {skill}, ("hand", i)
    for i, c in enumerate(deck):
        if c[0] == "P" and not (blocked and c[1] in nocop):
            nd = list(deck)
            nd[i] = ("N", skill)
            na = dict(assign)
            na[c[1]] = skill
            yield hand, nd, na, used | {skill}, ("deck", i)


def _use(hand, deck, idx, card, step, assign, used, nocop, ctx):
    """スロット idx のカード card を使用した後の状態を yield する。

    yield (hand, deck, assign, used, action)
    """
    # 複製キャラの元カード → その場でコピーカードに変化(山札は動かない)
    if card[0] == "N" and card[1] in ctx.copiers:
        nh = list(hand)
        nh[idx] = ("C", card[1], step.copy_target)
        yield nh, list(deck), assign, used, "transform"
        return

    nh, nd = list(hand), list(deck)
    if card[0] == "C":
        # コピー使用: 複製キャラのカードに戻って山札の底へ
        nd.append(("N", card[1]))
        origin = ("N", card[2])   # ドローフラグで探す「元カード」
    else:
        nd.append(card)
        origin = card
    nh[idx] = None                # 使ったカードは山札へ移った

    if step.draw:
        if origin[0] == "P":
            # 未確定カードは直前に自分で山札の底へ置いたので必ず山札にいる
            nd2 = list(nd)
            nh2 = list(nh)
            nh2[idx] = nd2.pop(nd2.index(origin))
            yield nh2, nd2, assign, used, "use"
            return
        for h2, d2, a2, u2, where in _locate(nh, nd, origin[1],
                                             assign, used, nocop, ctx):
            h3, d3 = list(h2), list(d2)
            if where is not None and where[0] == "deck":
                h3[idx] = d3.pop(where[1])      # 山札の途中からでも引き抜く
            else:
                h3[idx] = d3.pop(0) if d3 else None
            yield h3, d3, a2, u2, "use"
        return

    nh[idx] = nd.pop(0) if nd else None
    yield nh, nd, assign, used, "use"


def _retreat(hand, deck, step, assign, used, nocop, ctx):
    """撤退イベント後の状態を yield する。

    yield (hand, deck, assign, used, slot)   slot は 0 なら山札からの撤退。
    """
    for h2, d2, a2, u2, where in _locate(hand, deck, step.skill,
                                         assign, used, nocop, ctx):
        if where is None:
            continue                      # 既に場に無い(二重撤退)
        nh, nd = list(h2), list(d2)
        if where[0] == "deck":
            nd.pop(where[1])
            yield nh, nd, a2, u2, 0
        else:
            i = where[1]
            nh[i] = nd.pop(0) if nd else None   # 山札が空なら空欄になる
            yield nh, nd, a2, u2, i + 1


def _record(assign, used, nocop, trace, ctx):
    """全手順を消化した状態を解として登録する。"""
    for c in ctx.constraints:
        if not c(trace):
            return
    free = tuple(p for p in range(ctx.n) if p not in assign)
    rem_cop = len([s for s in ctx.copiers if s not in used])
    unconstrained = len([p for p in free if p not in nocop])
    count = _perm(unconstrained, rem_cop) * factorial(len(free) - rem_cop)
    if not count:
        return
    layout = tuple(assign.get(p) for p in range(ctx.n))
    ctx.results.append(Solution(layout, list(trace), count, free,
                                frozenset(nocop), ctx.n, ctx.copiers))
    if ctx.max_results is not None and len(ctx.results) >= ctx.max_results:
        raise _StopSearch


def _dfs(hand, deck, i, assign, used, nocop, trace, ctx):
    if i > ctx.max_depth:
        ctx.max_depth = i          # どの手順まで到達できたか(0件時の診断用)
    if i == len(ctx.plan):
        _record(assign, used, nocop, trace, ctx)
        return

    step = ctx.plan[i]

    if step.retreat:
        ctx.budget[0] -= 1
        if ctx.budget[0] < 0:
            raise SearchBudgetExceeded
        for nh, nd, na, nu, slot in _retreat(hand, deck, step, assign,
                                             used, nocop, ctx):
            trace.append((slot, ("N", step.skill), "retreat", step))
            _dfs(nh, nd, i + 1, na, nu, nocop, trace, ctx)
            trace.pop()
        return

    slots = ([step.slot - 1] if step.slot is not None
             else range(ctx.hand_size))
    for idx in slots:
        if not (0 <= idx < ctx.hand_size):
            continue
        card = hand[idx]
        if card is None:
            continue
        nassign, nused, nnocop = assign, used, nocop

        if card[0] == "P":
            # --- 未確定カード: ここで正体を確定させる ---
            p = card[1]
            if step.skill is None:
                # ワイルドカードは複製キャラの元カードを使わない
                if ctx.copiers:
                    nnocop = nocop | {p}
                    room = len([q for q in range(ctx.n)
                                if q not in assign and q not in nnocop])
                    if room < len([s for s in ctx.copiers if s not in used]):
                        continue
                real = card
            elif step.use_copy:
                continue                      # コピー札は未確定にならない
            else:
                s = step.skill
                if s in used:
                    continue                  # s は別の位置に確定済み
                if s in ctx.copiers and p in nocop:
                    continue
                nassign = dict(assign)
                nassign[p] = s
                nused = used | {s}
                real = ("N", s)
        else:
            # --- 確定カード ---
            if step.skill is None:
                if card[0] == "N" and card[1] in ctx.copiers:
                    continue
            elif step.use_copy:
                if not (card[0] == "C" and card[2] == step.skill):
                    continue
            else:
                if not (card[0] == "N" and card[1] == step.skill):
                    continue
            real = card

        ctx.budget[0] -= 1
        if ctx.budget[0] < 0:
            raise SearchBudgetExceeded
        for nh, nd, na, nu, action in _use(hand, deck, idx, real, step,
                                           nassign, nused, nnocop, ctx):
            trace.append((idx + 1, real, action, step))
            _dfs(nh, nd, i + 1, na, nu, nnocop, trace, ctx)
            trace.pop()


NODE_BUDGET_BASE = 2_000_000        # 手順が短いときの下限
NODE_BUDGET_PER_STEP = 200_000      # 手順1手あたりに許す無駄な探索


def default_node_budget(plan):
    """手順長に応じた探索ノード数の既定上限。

    手順が長いほど、解に届くまでに辿るノードも死に枝も増える。絶対値で
    固定すると長い TL がそれだけで打ち切られるので、1手あたりで見る。
    """
    return max(NODE_BUDGET_BASE, NODE_BUDGET_PER_STEP * len(plan))


def solve(n_skills, copiers, plan, constraints=(), *,
          hand_size=HAND_SIZE_NORMAL, max_results=None,
          node_budget=None, stats=None):
    """条件を満たす初期配置を列挙する。

    n_skills   : カード枚数(通常戦6 / 制約解除決戦10)。
    copiers    : 複製キャラの添字集合。
    plan       : Step のリスト。
    constraints: trace を受け取り bool を返す関数のリスト。
    hand_size  : 手札スロット数(通常3 / 決戦5)。
    max_results: 解がこの数に達したら探索を打ち切る(None なら無制限)。
    node_budget: 探索ノード数の上限。超えたら SearchBudgetExceeded。
                 None なら default_node_budget()(手順長に比例)。
                 解を max_results 件そろえるには最低でも
                 max_results × 手順長 ノードを辿る必要があるため、
                 その分は上限とは別枠で加算する(手順が長いだけで
                 打ち切られないようにする)。
    stats      : dict を渡すと "max_depth"(探索が到達できた最大の手順数)を
                 書き込む。解が0件のとき、どの手順で成立しなくなるかが分かる。

    戻り値: (results, truncated)
    results   = [Solution, ...]  (layout, trace) としても展開できる
    truncated = max_results により打ち切った場合 True
    """
    ctx = _Ctx()
    ctx.n = n_skills
    ctx.copiers = frozenset(copiers)
    ctx.plan = plan
    ctx.hand_size = hand_size
    ctx.constraints = tuple(constraints)
    ctx.max_results = max_results
    if node_budget is None:
        node_budget = default_node_budget(plan)
    ctx.budget = [node_budget + (max_results or 0) * len(plan)]
    ctx.results = []
    ctx.truncated = False
    ctx.max_depth = 0

    hand = [("P", p) for p in range(min(n_skills, hand_size))]
    hand += [None] * max(0, hand_size - n_skills)
    deck = [("P", p) for p in range(hand_size, n_skills)]

    try:
        _dfs(hand, deck, 0, {}, frozenset(), frozenset(), [], ctx)
    except _StopSearch:
        ctx.truncated = True
    finally:
        if stats is not None:
            stats["max_depth"] = ctx.max_depth
    return ctx.results, ctx.truncated


def total_layouts(results):
    """解が表す具体的な初期配置の延べ数(同じ初期配置の重複を含む)。

    1つの初期配置が複数の使用順(trace)を持ちうるため、初期配置の種類数が
    欲しい場合は distinct_layouts() を使うこと。
    """
    return sum(r.count for r in results)


def distinct_layouts(results, cap=200_000):
    """解が表す初期配置の種類数を返す。

    戻り値: (件数, exact)  exact=False なら「これ以上ある」ことを示す。
    """
    seen = set()
    for sol in results:
        for lay in sol.expand(limit=cap + 1 - len(seen)):
            seen.add(lay)
        if len(seen) > cap:
            return cap, False
    return len(seen), True
