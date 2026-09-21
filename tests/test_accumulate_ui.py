"""蓄積スキルの UI 配線 (入力欄 → buildPools → cos_accumulate.js) の検証。

assets/simulation.js の ns.sim.runSimulation を Node で丸ごと実行し、
・蓄積分だけ通過確率が上がること
・ワカモ/カンナ・ケイ・イロハ(水着) の各型が設定どおり計算されること
・不正な設定 (重なり / HP依存併用) がエラー文言で返ること
を確認する。数値そのものの正しさは tests/test_accumulate_js.py が見る。
"""
import json
import shutil
import subprocess

import pytest

HARNESS = "tests/js/sim_check.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node が無いので UI 配線の検証をスキップ")

CARDS = [
    # 0: 蓄積を付与する攻撃 (確定会心・1Hit)
    {"crit_min": 1_600_000, "crit_max": 2_000_000, "normal_min": 0,
     "normal_max": 0, "hits": 1, "crit_rate": 100, "evade_rate": 0, "enemies": 1},
    # 1: 味方アタッカー
    {"crit_min": 900_000, "crit_max": 1_100_000, "normal_min": 450_000,
     "normal_max": 550_000, "hits": 4, "crit_rate": 60, "evade_rate": 0, "enemies": 1},
    # 2: 味方サブ
    {"crit_min": 300_000, "crit_max": 380_000, "normal_min": 150_000,
     "normal_max": 190_000, "hits": 6, "crit_rate": 40, "evade_rate": 0, "enemies": 1},
]


def _run(accum, **kw):
    req = {"cards": CARDS, "accum": accum, "globalCrit": 50, "globalEvade": 0,
           "globalStability": None, "target": kw.get("target", 8_000_000),
           "method": kw.get("method", "cos"), "hpMode": kw.get("hpMode", "off")}
    out = subprocess.run(["node", HARNESS], input=json.dumps(req),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _pct(text):
    return float(text.split(": ")[1].split("%")[0])


def _pool(**kw):
    """蓄積スキル入力欄の既定値 (make_accum_card の初期値に対応)。"""
    base = {"name": "", "cards": [], "rate": 100, "cap_mode": "atk",
            "cap_value": None, "atk": None, "atk_pct": 1322, "cap_cards": [],
            "cap_pct": 100, "burst_mult": 100, "burst_decay": [1]}
    base.update(kw)
    return base


def test_no_pool_matches_plain():
    res = _run([])
    assert res["summary"] in ("", [])
    assert "通過確率" in res["passText"]


def test_wakamo_preset_increases_pass_rate():
    """ワカモ/カンナ型: 上限 = 攻撃力 × 1322%、味方全員のダメージが 100% 蓄積。"""
    plain = _pct(_run([])["passText"])
    res = _run([_pool(name="ワカモ", cards=[0, 1, 2], atk=30_000, atk_pct=1322)])
    assert _pct(res["passText"]) > plain
    assert "ワカモ" in res["summary"][0]
    assert "飽和確率" in res["summary"][0]


def test_kei_preset_rate_10_percent():
    """ケイ型: 自身を除く味方の与ダメージの 10%、上限 = 基本攻撃力 × 5000%。"""
    res = _run([_pool(name="ケイ", cards=[1, 2], rate=10, atk=25_000, atk_pct=5000)])
    assert "ケイ" in res["summary"][0]
    full = _run([_pool(name="ケイ", cards=[1, 2], rate=100, atk=25_000, atk_pct=5000)])
    # 蓄積率が 10 倍なら爆発ダメージも増える (上限に当たるまで)
    assert _pct(full["passText"]) > _pct(res["passText"])


def test_iroha_cap_from_card_damage():
    """イロハ(水着)型: 上限 = 付与した攻撃のダメージ × 120% (相関あり)。"""
    res = _run([_pool(name="イロハ", cards=[1, 2], rate=150, cap_mode="cards",
                      cap_cards=[0], cap_pct=120, burst_decay=[])])
    assert "イロハ" in res["summary"][0]
    # 上限は約 1.8M × 1.2 ≒ 2.16M。蓄積は 150% なのでほぼ確実に飽和する。
    assert "飽和確率 100.0%" in res["summary"][0]


def test_multiple_casts():
    """蓄積スキルを 2 回撃つ (対象カードを分ける)。"""
    one = _run([_pool(name="1回目", cards=[1], atk=20_000, atk_pct=500)])
    two = _run([_pool(name="1回目", cards=[1], atk=20_000, atk_pct=500),
                _pool(name="2回目", cards=[2], atk=20_000, atk_pct=500)])
    assert len(two["summary"]) == 3      # 2 行 + 注記
    assert _pct(two["passText"]) > _pct(one["passText"])


def test_overlap_is_rejected():
    res = _run([_pool(name="A", cards=[0, 1], atk=20_000, atk_pct=500),
                _pool(name="B", cards=[1, 2], atk=20_000, atk_pct=500)])
    assert "重なって" in res["passText"]


def test_hp_dep_combination_is_rejected():
    res = _run([_pool(name="A", cards=[1], atk=20_000, atk_pct=500)], hpMode="on")
    assert "HP依存" in res["passText"]


def test_monte_carlo_path_agrees_with_cos():
    pools = [_pool(name="ワカモ", cards=[0, 1, 2], atk=30_000, atk_pct=1322)]
    cos = _pct(_run(pools)["passText"])
    mc = _pct(_run(pools, method="mc")["passText"])
    assert abs(cos - mc) < 1.0          # MC 10万サンプルのばらつき内
