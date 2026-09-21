"""assets/cos_accumulate.js (蓄積スキルの JS 移植) を Python 参照実装と突き合わせる。

app/backend/accumulate.py が正本。同じアルゴリズム (セル質量 + FFT) の移植なので、
CDF は 1e-6 程度で一致するはず。Node が無い環境では skip する。
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from app.backend.accumulate import AccumWindow, CapSpec, build_accum_dist
from app.backend.cos import build_hit_mixtures

HARNESS = "tests/js/accum_check.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node が無いので JS 移植の検証をスキップ")

CARDS = [
    {"crit_min": 900_000, "crit_max": 1_100_000, "normal_min": 450_000,
     "normal_max": 550_000, "hits": 4, "crit_rate": 60, "evade_rate": 0},
    {"crit_min": 1_400_000, "crit_max": 1_800_000, "normal_min": 700_000,
     "normal_max": 900_000, "hits": 3, "crit_rate": 50, "evade_rate": 5},
    {"crit_min": 300_000, "crit_max": 380_000, "normal_min": 150_000,
     "normal_max": 190_000, "hits": 3, "crit_rate": 40, "evade_rate": 0},
]
# カード → Hit 番号 (build_hit_mixtures はカードの Hit 数だけ混合を並べる)
CARD_HITS = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}


def _js(pools, xs):
    req = {"cards": CARDS, "pools": pools, "globalCrit": 50, "globalEvade": 0,
           "globalStability": None, "damageMode": "post_decay", "xs": list(xs)}
    out = subprocess.run(["node", HARNESS], input=json.dumps(req),
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _hits_of(cards):
    return [h for c in cards for h in CARD_HITS[c]]


def _py(windows, xs):
    hm = build_hit_mixtures(CARDS, 50, 0, "post_decay")
    d = build_accum_dist(hm, windows)
    return d, d.cdf(np.asarray(xs, dtype=float))


def _xs(d):
    return np.linspace(d.mean - 2.5 * np.sqrt(d.var), d.mean + 2.5 * np.sqrt(d.var), 21)


def test_js_matches_python_no_pool():
    d, _ = _py([], [0.0])
    xs = _xs(d)
    js = _js([], xs)
    assert js["mean"] == pytest.approx(d.mean, rel=1e-9)
    assert np.abs(np.array(js["cdf"]) - d.cdf(xs)).max() < 1e-9


def test_js_matches_python_fixed_cap():
    """ワカモ/カンナ型: 上限は固定値 (攻撃力×倍率)、爆発倍率つき。"""
    pools = [{"name": "ワカモ", "cards": [0, 1], "rate": 1.0, "capKind": "fixed",
              "capValue": 5_000_000, "burstMult": 1.3, "burstDecay": True}]
    wins = [AccumWindow(hits=_hits_of([0, 1]), rate=1.0,
                        cap=CapSpec(kind="fixed", value=5_000_000),
                        burst_mult=1.3, burst_decay=True)]
    d, _ = _py(wins, [0.0])
    xs = _xs(d)
    js = _js(pools, xs)
    assert js["mean"] == pytest.approx(d.mean, rel=1e-6)
    assert np.abs(np.array(js["cdf"]) - d.cdf(xs)).max() < 1e-6
    st_js, st_py = js["windowStats"][0], d.window_stats[0]
    assert st_js["poolMean"] == pytest.approx(st_py.pool_mean, rel=1e-6)
    assert st_js["satProb"] == pytest.approx(st_py.sat_prob, abs=1e-6)
    assert st_js["burstMean"] == pytest.approx(st_py.burst_mean, rel=1e-6)


def test_js_matches_python_card_cap_outside():
    """イロハ(水着)型: 上限 = 付与した攻撃のダメージ × 120% (その攻撃は蓄積対象外)。"""
    pools = [{"name": "イロハ", "cards": [1, 2], "rate": 1.5, "capKind": "cards",
              "capCards": [0], "capCoef": 1.2, "burstMult": 1.0, "burstDecay": False}]
    wins = [AccumWindow(hits=_hits_of([1, 2]), rate=1.5,
                        cap=CapSpec(kind="hits", hits=_hits_of([0]), coef=1.2),
                        burst_mult=1.0, burst_decay=False)]
    d, _ = _py(wins, [0.0])
    xs = _xs(d)
    js = _js(pools, xs)
    assert js["mean"] == pytest.approx(d.mean, rel=1e-5)
    assert np.abs(np.array(js["cdf"]) - d.cdf(xs)).max() < 5e-5


def test_js_matches_python_multi_pool():
    """蓄積スキルを 2 回撃つ (非重複窓)。"""
    pools = [
        {"name": "1回目", "cards": [0], "rate": 1.0, "capKind": "fixed",
         "capValue": 2_500_000, "burstMult": 1.0, "burstDecay": True},
        {"name": "2回目", "cards": [1, 2], "rate": 0.1, "capKind": "fixed",
         "capValue": 400_000, "burstMult": 1.0, "burstDecay": False},
    ]
    wins = [
        AccumWindow(hits=_hits_of([0]), rate=1.0,
                    cap=CapSpec(kind="fixed", value=2_500_000), burst_decay=True),
        AccumWindow(hits=_hits_of([1, 2]), rate=0.1,
                    cap=CapSpec(kind="fixed", value=400_000), burst_decay=False),
    ]
    d, _ = _py(wins, [0.0])
    xs = _xs(d)
    js = _js(pools, xs)
    assert js["mean"] == pytest.approx(d.mean, rel=1e-6)
    assert np.abs(np.array(js["cdf"]) - d.cdf(xs)).max() < 1e-6


def test_js_reports_overlap_error():
    pools = [
        {"name": "A", "cards": [0, 1], "rate": 1.0, "capKind": "fixed",
         "capValue": 1e6, "burstMult": 1.0, "burstDecay": False},
        {"name": "B", "cards": [1], "rate": 1.0, "capKind": "fixed",
         "capValue": 1e6, "burstMult": 1.0, "burstDecay": False},
    ]
    assert "重なって" in _js(pools, [0.0])["error"]
