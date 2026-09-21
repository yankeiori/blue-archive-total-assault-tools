"""コールバックグラフの不変条件。

Dash の「Maximum update depth exceeded」は 2 通りの原因で出る。

1. **コンテナの children を Input にしつつ、その中のコンポーネントの prop を Output**
   する。children が変わったとみなされて再発火が止まらない。中に置く初期 prop は
   コールバックではなく生成時 (make_damage_card / make_accum_card の引数) に渡すこと。
2. **コールバックグラフの循環**。双方向同期のように意図的な循環は「同値なら
   no_update を返す」で止めるしかなく、その判定が甘いと往復し続ける。

どちらも実際に踏んだので、グラフの形をテストで縛る。
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import main  # noqa: F401 - クライアントサイドコールバックの登録
from dash._callback import GLOBAL_CALLBACK_MAP

from dash.dependencies import ALL, ALLSMALLER, MATCH

from app import app as application
from app.frontend.layout import create_layout, make_accum_card, make_damage_card

_ROOT = Path(__file__).resolve().parent.parent
_WILDCARDS = (ALL, MATCH, ALLSMALLER)

# 意図的に残している循環 (双方向同期)。値を正規化して同値なら no_update を返すので
# 1 往復で必ず止まる (main.py の目標ダメージ同期を参照)。増やすときは、止まる根拠を
# コメントに書いてからにすること。
ALLOWED_CYCLES = {frozenset({"target-damage.value", "restart-D.value"})}


def _all_callbacks():
    out = {}
    out.update(GLOBAL_CALLBACK_MAP)
    out.update(application.callback_map)
    return out


def _key(cid, prop: str) -> str:
    """(component id, prop) をグラフの節点名へ。

    ワイルドカード id は type でまとめる (ALL/MATCH の違いはグラフの形に関係ない)。
    id は dict のことも、JSON 文字列のこともある。
    """
    if isinstance(cid, dict):
        return "{" + str(cid.get("type", "?")) + "}." + prop
    if isinstance(cid, str) and cid.startswith("{"):
        try:
            return "{" + str(json.loads(cid).get("type", "?")) + "}." + prop
        except ValueError:
            pass
    return f"{cid}.{prop}"


def _inputs(cb) -> list:
    return [_key(i["id"], i["property"]) for i in cb.get("inputs", [])]


def _outputs(cb, spec: str) -> list:
    """コールバックの出力を節点名のリストへ。

    Output オブジェクトが取れるならそれを使う。文字列指定 (..a.b...c.d.. 形式) から
    読むときは allow_duplicate の @<hash> を落とすこと — 忘れると辺が丸ごと欠けて
    循環を見逃す (実際に見逃した)。
    """
    outs = cb.get("output")
    if isinstance(outs, (list, tuple)) and outs:
        return [_key(o.component_id, o.component_property) for o in outs]
    text = outs if isinstance(outs, str) else spec
    res = []
    for part in (text.split("...") if "..." in text else [text]):
        part = part.strip(".").split("@")[0]
        if not part:
            continue
        m = re.match(r"^(.*)\.([A-Za-z_]+)$", part)
        assert m, f"出力指定をパースできません: {part!r}"
        res.append(_key(m.group(1), m.group(2)))
    return res


def _edges() -> dict:
    edges: dict = {}
    for key, cb in _all_callbacks().items():
        dst = _outputs(cb, key)
        for src in _inputs(cb):
            edges.setdefault(src, set()).update(dst)
    return edges


def _cycles(edges: dict | None = None) -> list:
    edges = _edges() if edges is None else edges
    seen: set = set()
    stack: list = []
    found: list = []

    def dfs(node):
        if node in stack:
            found.append(stack[stack.index(node):])
            return
        if node in seen:
            return
        seen.add(node)
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            dfs(nxt)
        stack.pop()

    for node in sorted(edges):
        dfs(node)
    return found


def test_cycle_detector_itself_works():
    """検出器の自己テスト。

    allow_duplicate の @<hash> を落とし忘れて辺が欠け、循環を見逃したことがある。
    """
    planted = {"a.x": {"b.y"}, "b.y": {"a.x"}, "c.z": {"a.x"}}
    assert [set(c) for c in _cycles(planted)] == [{"a.x", "b.y"}]
    assert _outputs({}, "..a.x@deadbeef...b.y..") == ["a.x", "b.y"]
    assert _outputs({}, "..{\"index\":[\"ALL\"],\"type\":\"memo\"}.value..") == ["{memo}.value"]


def test_no_container_children_is_used_as_input():
    offenders = [
        (key, inp["id"])
        for key, cb in _all_callbacks().items()
        for inp in cb.get("inputs", [])
        if isinstance(inp.get("id"), str)
        and inp["id"].endswith("-container")
        and inp.get("property") == "children"
    ]
    assert not offenders, (
        "コンテナの children を Input にしています。中の prop を Output すると "
        f"無限ループになります: {offenders}")


def test_only_known_cycles_exist():
    """許可した組以外に循環が無いこと。

    自己ループ (出力が自分の入力でもある) も検出されるので、許可は
    「その組に収まっているか」で判定する。
    """
    unexpected = [c for c in _cycles()
                  if not any(set(c) <= allowed for allowed in ALLOWED_CYCLES)]
    assert not unexpected, (
        "コールバックグラフに想定外の循環があります (無限ループの原因): "
        + "; ".join(" → ".join(c) for c in unexpected))


def test_the_known_cycle_is_still_there():
    """許可リストが腐っていないことの確認 (循環が消えたら許可リストも消す)。"""
    nodes = {n for c in _cycles() for n in c}
    assert nodes == set().union(*ALLOWED_CYCLES)


def test_accum_container_children_is_output_or_state_only():
    for key, cb in _all_callbacks().items():
        for inp in cb.get("inputs", []):
            assert inp.get("id") != "accum-container", key


# ---------------------------------------------------------------------------
# 許可した循環が本当に止まるか (JS ごと動かす)
# ---------------------------------------------------------------------------
def _sync_js() -> str:
    """main.py に埋め込んだ「目標ダメージの双方向同期」の JS を取り出す。"""
    src = (_ROOT / "main.py").read_text(encoding="utf-8")
    i = src.index("# --- 目標ダメージの双方向同期")
    body = src[i:src.index('Output("target-damage"', i)]
    return body[body.index('"""') + 3:body.rindex('"""')].strip().rstrip(",")


@pytest.mark.skipif(shutil.which("node") is None, reason="node が無いので JS 検証をスキップ")
def test_the_known_cycle_settles():
    """双方向同期が必ず 1 往復以内で止まること。

    これが止まらないとブラウザで "Maximum update depth exceeded" になる。
    表現の食い違い ('' と null、数値と文字列) でも止まることを確認する。
    """
    script = f"""
    global.window = {{dash_clientside: {{no_update: 'NU',
                     callback_context: {{triggered: []}}}}}};
    var sync = {_sync_js()};
    function settle(sim, rest, trig) {{
      for (var i = 0; i < 50; i++) {{
        window.dash_clientside.callback_context.triggered = [{{prop_id: trig}}];
        var r = sync(sim, rest);
        if (r[0] === 'NU' && r[1] === 'NU') return i;
        if (r[0] !== 'NU') {{ sim = r[0]; trig = 'target-damage.value'; }}
        if (r[1] !== 'NU') {{ rest = r[1]; trig = 'restart-D.value'; }}
      }}
      return -1;                      // 収束しない = 無限ループ
    }}
    var cases = [
      ['', null, 'target-damage.value'],
      [null, '', 'restart-D.value'],
      [1000000, '1000000', 'target-damage.value'],
      [1000000, 1000000, 'target-damage.value'],
      [2000000, 1000000, 'target-damage.value'],
      [2000000, 1000000, 'restart-D.value'],
      [undefined, null, 'target-damage.value'],
      [NaN, 5, 'restart-D.value'],
    ];
    console.log(JSON.stringify(cases.map(function (c) {{
      return settle(c[0], c[1], c[2]);
    }})));
    """
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    steps = json.loads(out.stdout)
    assert all(0 <= n <= 1 for n in steps), f"収束しない/往復が多い: {steps}"


# ---------------------------------------------------------------------------
# ワイルドカード入力が、その prop を持たないコンポーネントを巻き込んでいないか
# ---------------------------------------------------------------------------
def _components():
    """レイアウトと、動的に作られるカードの全コンポーネント。"""
    out = []

    def walk(node):
        out.append(node)
        ch = getattr(node, "children", None)
        if ch is None or isinstance(ch, str):
            return
        for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
            walk(c)

    for root in (create_layout(), make_damage_card(0), make_accum_card(0)):
        walk(root)
    return out


def _as_pattern(raw):
    """コールバックの入力 id をパターン dict へ。

    コールバックマップにはワイルドカード id が JSON 文字列で入っており、
    ALL/MATCH は ["ALL"] のようなリストになっている (dict のまま来ることもある)。
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.startswith("{"):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return None


def _is_wildcard(v) -> bool:
    return v in _WILDCARDS or (isinstance(v, list) and len(v) == 1
                               and v[0] in ("ALL", "MATCH", "ALLSMALLER"))


def _matches(pattern: dict, cid) -> bool:
    if not isinstance(cid, dict) or set(cid) != set(pattern):
        return False
    return all(_is_wildcard(v) or cid.get(k) == v for k, v in pattern.items())


def test_wildcard_inputs_only_match_components_that_have_the_prop():
    """ALL/MATCH の入力が、その prop を持たない要素まで拾っていないこと。

    例えば {"type": "accum", "field": ALL} の value を入力にすると、同じ type の
    表示専用 html.Div (value を持たない) まで一致してしまう。その値は undefined に
    なり、自動保存のスナップショット配列に undefined が混ざる。localStorage を
    往復すると undefined は null になるので値が一致せず、dcc.Store が設定し直しを
    繰り返して「Maximum update depth exceeded」で画面が固まる (実際に踏んだ)。
    表示専用の要素は id の type を分けること。
    """
    comps = _components()
    bad = []
    for key, cb in _all_callbacks().items():
        for inp in cb.get("inputs", []):
            pat = _as_pattern(inp["id"])
            if pat is None:
                continue
            prop = inp["property"]
            for comp in comps:
                cid = getattr(comp, "id", None)
                if _matches(pat, cid) and prop not in getattr(comp, "_prop_names", ()):
                    bad.append(f"{type(comp).__name__}(id={cid}) は {prop} を"
                               f"持たないのに {key[:40]} の入力に一致します")
    assert not bad, "\n".join(sorted(set(bad)))


def test_snapshot_is_json_round_trippable():
    """自動保存が localStorage と往復しても同じ値になる形で返していること。"""
    js = (_ROOT / "assets" / "autosave.js").read_text(encoding="utf-8")
    assert "JSON.parse(JSON.stringify(snap))" in js, (
        "スナップショットを JSON 正規化せずに返すと、配列内の undefined が "
        "localStorage 往復で null になり dcc.Store が無限ループする")
