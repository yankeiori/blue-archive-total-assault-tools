# -*- coding: utf-8 -*-
"""総力戦のタイムライン(TL)テキストから、スキル順探索の手順を読み取る。

対応している書き方:
  - 発動タイミングの注記は読み飛ばす
    例: 即 / オート / auto / コストカット後 / ◯◯NS後 / 11 / 10.5 / 3:03.000
  - `/` `//` `※` `【` 以降は行末までコメント扱い
  - `目標…` の行、`〆` は無視
  - `c◯◯` `C◯◯` `◯◯(コピー)` はコピーカードの使用
  - `◯◯(△△)` の括弧は対象指定。複製キャラなら複製対象、
    それ以外(緑壺など生徒でないものを含む)はメモとして残す
  - `◯◯撤退` はその時点での撤退
  - `イロハナギサイブキ` のように名前が連結していても分割する

読み取れなかった語は警告として返すので、呼び出し側で提示すること。
"""

import re

KIND_USE = "use"          # 通常のカード使用(複製キャラなら複製の発動)
KIND_COPY = "copy"        # 「◯◯(コピー)」の使用
KIND_RETREAT = "retreat"  # 撤退


class TLStep:
    """TL から読み取った1手。"""

    def __init__(self, kind, skill, target=None, memo="", draw=False,
                 source=""):
        self.kind = kind
        self.skill = skill          # スキル添字
        self.target = target        # 複製対象のスキル添字 (複製キャラのみ)
        self.memo = memo            # 括弧内が生徒でない場合などの注記
        self.draw = draw
        self.source = source        # 元の行(参考表示用)

    def __repr__(self):
        return (f"TLStep({self.kind}, {self.skill}, target={self.target},"
                f" memo={self.memo!r}, draw={self.draw})")


# 行頭・語中に現れる発動タイミングの注記(カード使用ではない)
_NOISE = [
    r"[0-9]+[:.][0-9]+[:.][0-9]+",      # 3:03.000 / 1.19.1 / 2:22:500
    r"[0-9]+:[0-9]+",                   # 2:58 / 0:30
    r"[0-9]+(?:\.[0-9]+)?\s*[c憶億万]?",  # コスト・残HPなどの数値
    r"コストカット後?", r"指定できるようになったら", r"オート解除",
    r"オート", r"ｵｰﾄ", r"auto", r"Auto", r"AUTO",
    r"即", r"指定", r"付近", r"くらい", r"ぐらい", r"残", r"未満", r"以下",
    r"以上", r"適当", r"移行後", r"歯車", r"予告", r"直前", r"秒後",
    r"本体", r"諸説", r"単遅延", r"爆発後", r"HP", r"NS", r"ns", r"後",
]
_NOISE_RE = re.compile("|".join(_NOISE))
_PUNCT_RE = re.compile(r"[\s()（）?？、。,．・/｜|【】\[\]]+")


_ALIAS_SEP = re.compile(r"[/／,、|｜]")


def display_name(name):
    """カード名から表示用の名前(先頭の別名)を取り出す。"""
    return _ALIAS_SEP.split(name or "")[0].strip()


def aliases(name):
    """カード名に書かれた別名をすべて返す。「ドアル/アル」→ [ドアル, アル]"""
    return [a.strip() for a in _ALIAS_SEP.split(name or "") if a.strip()]


def parse_timeline(text, names, copiers=()):
    """TL テキストを TLStep のリストに変換する。

    names   : カード名のリスト(添字がスキル添字)。空文字の要素は無視。
              「ドアル/アル」のように書くと別名としても照合する。
    copiers : 複製スキル持ちのスキル添字集合。

    戻り値: (steps, warnings)
    """
    copiers = set(copiers)
    lookup = {}
    for i, nm in enumerate(names):
        for alias in aliases(nm):
            lookup.setdefault(alias, i)
    if not lookup:
        return [], ["カード設定にキャラ名が入力されていません。"]
    ordered = sorted(lookup, key=len, reverse=True)
    name_re = re.compile("|".join(re.escape(n) for n in ordered))

    steps, warnings = [], []
    separated = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"/.*", "", line)           # / も // 以降も注釈
        line = re.sub(r"【.*", "", line)          # 【254】以降は注釈
        line = re.sub(r"※.*", "", line)          # ※以降も注釈
        line = line.replace("（", "(").replace("）", ")")
        line = line.replace("ｃ", "c").replace("Ｃ", "C")
        if not line.strip():
            continue
        if line.startswith("目標") or line.strip() == "〆":
            # 区切りより後にもカード使用が続くなら、TL が複数入っている
            separated = bool(steps)
            continue
        # 「◯◯NS後」「◯◯NS」「NS直前」はタイミングの注記なのでカード使用ではない
        line = re.sub(
            rf"(?:{name_re.pattern})?[NnＮｎ][SsＳｓ](?:後|直前)?", " ", line)

        used, skipped = _scan_line(line, name_re, lookup, copiers, raw)
        rest = _PUNCT_RE.sub(" ", _NOISE_RE.sub(" ", skipped)).strip()
        if rest:
            warnings.append(f"「{rest}」を解釈できませんでした（{raw.strip()}）")
        if separated and used:
            warnings.append(
                "「目標…」や「〆」より後にも手順が続いています。"
                "1回の戦闘ぶんずつ貼り付けてください。")
            separated = False
        steps.extend(used)

    _infer_copy_targets(steps, copiers, names, warnings)
    _infer_draws(steps, names, warnings)
    return steps, warnings


def _scan_line(line, name_re, lookup, copiers, raw):
    """1行を走査して TLStep 群と、読み飛ばした文字列を返す。"""
    steps, skipped = [], []
    pos = 0
    while pos < len(line):
        ch = line[pos]

        if ch == "(":                                   # 対象指定
            end = line.find(")", pos)
            end = len(line) if end < 0 else end
            inner = line[pos + 1:end].strip()
            # 対象指定は同じ行のカードにのみ結び付ける
            target = steps[-1] if steps else None
            if target is not None:
                if inner in ("コピー", "copy"):
                    target.kind = KIND_COPY
                else:
                    target.memo = inner
            else:
                skipped.append(inner)
            pos = end + 1
            continue

        if ch in "cC":                                  # cマリー = コピー
            m = name_re.match(line, pos + 1)
            if m:
                steps.append(TLStep(KIND_COPY, lookup[m.group(0)],
                                    source=raw.strip()))
                pos = m.end()
                continue

        m = name_re.match(line, pos)
        if m:
            pos = m.end()
            nxt = line[pos:].lstrip()
            if nxt.startswith("撤退"):
                steps.append(TLStep(KIND_RETREAT, lookup[m.group(0)],
                                    source=raw.strip()))
                pos = line.index("撤退", pos) + 2
            else:
                steps.append(TLStep(KIND_USE, lookup[m.group(0)],
                                    source=raw.strip()))
            continue

        skipped.append(ch)
        pos += 1
    return steps, "".join(skipped)


def _infer_copy_targets(steps, copiers, names, warnings):
    """複製対象が省略された複製キャラの手は、直後のコピー使用から補う。"""
    for i, st in enumerate(steps):
        if st.kind != KIND_USE or st.skill not in copiers:
            continue
        if st.memo:
            idx = _name_index(st.memo, names)
            if idx is not None and idx != st.skill:
                st.target = idx
                st.memo = ""
                continue
        for later in steps[i + 1:]:
            if later.kind == KIND_COPY:
                st.target = later.skill
                warnings.append(
                    f"{i + 1}手目 {display_name(names[st.skill])} の複製対象が"
                    f"書かれていないため、後の"
                    f"「{display_name(names[later.skill])}(コピー)」から"
                    f"{display_name(names[later.skill])}と判断しました。")
                break
        else:
            warnings.append(
                f"{i + 1}手目 {display_name(names[st.skill])} の"
                "複製対象が判別できません。"
                "手順で指定してください。")


def _infer_draws(steps, names, warnings):
    """同じカードを続けて使う手は「ドロー」が無いと成立しないので印を付ける。"""
    copied = {s.skill for s in steps if s.kind == KIND_COPY}
    for i in range(len(steps) - 1):
        cur, nxt = steps[i], steps[i + 1]
        if nxt.kind != KIND_USE or cur.kind == KIND_RETREAT:
            continue
        if cur.skill == nxt.skill and not cur.draw:
            cur.draw = True
            label = (f"{display_name(names[cur.skill])}(コピー)"
                     if cur.kind == KIND_COPY
                     else display_name(names[cur.skill]))
            msg = (f"{i + 1}手目 {label} は直後に同じカードを使うため"
                   "「ドロー」を付けました。")
            if cur.kind == KIND_USE and cur.skill in copied:
                msg += (f"（{i + 2}手目が "
                        f"{display_name(names[cur.skill])}(コピー) の"
                        "使用である可能性もあります。解が0件ならこちらを"
                        "疑ってください）")
            warnings.append(msg)


def _name_index(text, names):
    for i, nm in enumerate(names):
        if text in aliases(nm):
            return i
    return None
