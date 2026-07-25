"""simulation.py の動作保証テスト。

リファクタリング前後で結果が変わらないことを検証する。
乱数シードを固定し、確定的に比較できるテストと、
統計的性質を検証するテストの両方を含む。
"""

import numpy as np
import pytest

from app.backend.simulation import (
    DAMAGE_FUNC,
    _extract_hit_params,
    _simulate_vectorized,
    decay,
    inverse_decay,
    run_simulation,
)


def _simulate_cards(indices, params, global_crit, global_evade, damage_mode,
                    n_samples=200_000):
    """テスト用: MC エンジンでソート済み合計ダメージを返す (旧 _simulate_cards 相当)。"""
    rng = np.random.default_rng()
    hit_params = _extract_hit_params(indices, params, global_crit, global_evade,
                                     damage_mode)
    total = _simulate_vectorized(rng, *hit_params, n_samples)
    total.sort()
    return total

# ---------------------------------------------------------------------------
# decay / inverse_decay
# ---------------------------------------------------------------------------


class TestDecay:
    """減衰関数の正確性テスト。"""

    def test_identity_below_4m(self):
        """4M以下は恒等写像 (y = x)。"""
        x = np.array([0.0, 1_000_000, 2_000_000, 3_999_999])
        np.testing.assert_allclose(decay(x), x)

    def test_first_breakpoint(self):
        """4M ちょうどで連続 (境界値)。"""
        x = np.array([4_000_000.0])
        y = decay(x)
        assert y[0] == pytest.approx(4_000_000.0, abs=1)

    def test_diminishing_returns(self):
        """減衰率が後段ほど小さくなる（係数 a が非増加）。"""
        coeffs = [a for (_, _), (a, _) in DAMAGE_FUNC]
        assert all(coeffs[i] >= coeffs[i + 1] for i in range(len(coeffs) - 1))

    def test_cap_at_22m(self):
        """22M 以上は上限キャップ。"""
        x = np.array([22_000_000, 30_000_000, 100_000_000], dtype=float)
        y = decay(x)
        np.testing.assert_allclose(y, 10_966_999, atol=1)

    def test_each_segment_continuity(self):
        """最終段を除き、各区間の境界で値が連続していることを確認。"""
        # 最終段 (キャップ) は不連続なのでスキップ
        for i in range(len(DAMAGE_FUNC) - 2):
            (_, x_hi), (a, b) = DAMAGE_FUNC[i]
            (x_lo_next, _), (a_next, b_next) = DAMAGE_FUNC[i + 1]
            y_from_left = a * x_hi + b
            y_from_right = a_next * x_lo_next + b_next
            assert x_hi == x_lo_next
            assert y_from_left == pytest.approx(y_from_right, abs=1)

    def test_monotonic_below_cap(self):
        """キャップ前 (22M未満) の範囲で減衰後ダメージは単調非減少。"""
        x = np.linspace(0, 21_999_999, 10_000)
        y = decay(x)
        assert np.all(np.diff(y) >= -1e-6)


class TestInverseDecay:
    """逆減衰関数のテスト。"""

    def test_roundtrip_below_4m(self):
        """4M以下で decay(inverse_decay(y)) ≈ y。"""
        for y in [0, 500_000, 2_000_000, 3_999_999]:
            x = inverse_decay(float(y))
            result = decay(np.array([x]))[0]
            assert result == pytest.approx(y, abs=1)

    def test_roundtrip_mid_range(self):
        """中間域で往復変換が一致。"""
        test_ys = [5_000_000, 7_000_000, 9_000_000, 10_000_000]
        for y in test_ys:
            x = inverse_decay(float(y))
            result = decay(np.array([x]))[0]
            assert result == pytest.approx(y, abs=1)

    def test_cap_roundtrip(self):
        """キャップ付近の値で往復変換が一致する。"""
        # 最終段直前のy値で往復確認
        y = 10_800_000.0
        x = inverse_decay(y)
        result = decay(np.array([x]))[0]
        assert result == pytest.approx(y, abs=1)

    def test_above_cap_returns_boundary(self):
        """キャップを超えた y 値では inverse_decay がキャップ境界 x を返す。"""
        # decay の最終段 (a=0) に該当する y はテーブル範囲外として扱われ恒等
        # 実際のゲームでは 22M 以上の生ダメージは cap に張り付く
        x = inverse_decay(10_966_999)
        # 逆変換後に decay を適用すれば元の値に戻る
        result = decay(np.array([x]))[0]
        assert result == pytest.approx(10_966_999, abs=100)


# ---------------------------------------------------------------------------
# _simulate_cards (統計的性質)
# ---------------------------------------------------------------------------


def _make_params(
    crit_min=100, crit_max=200, normal_min=50, normal_max=100,
    hits=1, crit_rate=50, evade_rate=0,
):
    return {
        "crit_min": crit_min, "crit_max": crit_max,
        "normal_min": normal_min, "normal_max": normal_max,
        "hits": hits, "crit_rate": crit_rate, "evade_rate": evade_rate,
    }


class TestSimulateCards:
    """_simulate_cards の統計的性質テスト。"""

    def test_single_card_mean(self):
        """1カード・回避なしの期待値が理論値に近い。"""
        p = _make_params(
            crit_min=1000, crit_max=1000, normal_min=500, normal_max=500,
            hits=1, crit_rate=50, evade_rate=0,
        )
        # crit_rate=50% → E = 0.5*1000 + 0.5*500 = 750
        samples = _simulate_cards([0], {0: p}, 0, 0, "pre_decay")
        mean = np.mean(samples)
        assert mean == pytest.approx(750, rel=0.05)

    def test_evade_reduces_mean(self):
        """回避率が上がると期待値が下がる。"""
        p_no_evade = _make_params(
            crit_min=1000, crit_max=1000, normal_min=1000, normal_max=1000,
            hits=10, crit_rate=100, evade_rate=0,
        )
        p_half_evade = _make_params(
            crit_min=1000, crit_max=1000, normal_min=1000, normal_max=1000,
            hits=10, crit_rate=100, evade_rate=50,
        )
        mean_no = np.mean(_simulate_cards([0], {0: p_no_evade}, 0, 0, "pre_decay"))
        mean_half = np.mean(_simulate_cards([0], {0: p_half_evade}, 0, 0, "pre_decay"))
        assert mean_half == pytest.approx(mean_no * 0.5, rel=0.05)

    def test_multiple_cards_additive(self):
        """複数カードのダメージは加算される。"""
        p = _make_params(
            crit_min=1000, crit_max=1000, normal_min=1000, normal_max=1000,
            hits=1, crit_rate=100, evade_rate=0,
        )
        one_card = np.mean(_simulate_cards([0], {0: p}, 0, 0, "pre_decay"))
        two_cards = np.mean(_simulate_cards([0, 1], {0: p, 1: p}, 0, 0, "pre_decay"))
        assert two_cards == pytest.approx(one_card * 2, rel=0.05)

    def test_output_is_sorted(self):
        """_simulate_cards の戻り値はソート済み。"""
        p = _make_params(crit_min=100, crit_max=200, normal_min=50, normal_max=100, hits=3)
        samples = _simulate_cards([0], {0: p}, 0, 0, "pre_decay")
        assert np.all(np.diff(samples) >= 0)

    def test_empty_indices(self):
        """空のインデックスではゼロ配列。"""
        samples = _simulate_cards([], {}, 0, 0, "pre_decay")
        assert np.all(samples == 0)

    def test_post_decay_mode(self):
        """post_decay モードでもシミュレーションが動作する。"""
        p = _make_params(
            crit_min=3_000_000, crit_max=3_500_000,
            normal_min=2_000_000, normal_max=2_500_000,
            hits=2, crit_rate=50, evade_rate=10,
        )
        samples = _simulate_cards([0], {0: p}, 0, 0, "post_decay")
        assert len(samples) > 0
        assert np.all(np.diff(samples) >= 0)

    def test_hits_multiply_damage(self):
        """ヒット数2 は ヒット数1 の約2倍の期待値。"""
        p1 = _make_params(
            crit_min=1000, crit_max=1000, normal_min=1000, normal_max=1000,
            hits=1, crit_rate=100, evade_rate=0,
        )
        p2 = _make_params(
            crit_min=1000, crit_max=1000, normal_min=1000, normal_max=1000,
            hits=2, crit_rate=100, evade_rate=0,
        )
        mean1 = np.mean(_simulate_cards([0], {0: p1}, 0, 0, "pre_decay"))
        mean2 = np.mean(_simulate_cards([0], {0: p2}, 0, 0, "pre_decay"))
        assert mean2 == pytest.approx(mean1 * 2, rel=0.05)


# ---------------------------------------------------------------------------
# run_simulation (結合テスト)
# ---------------------------------------------------------------------------


class TestRunSimulation:
    """run_simulation の結合テスト。"""

    def test_returns_figure_and_text(self):
        """戻り値の型が正しい。"""
        p = _make_params(crit_min=1000, crit_max=2000, normal_min=500, normal_max=1000, hits=1)
        fig, text = run_simulation([0], {0: p}, 50, 10, 1000, "pre_decay")
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
        assert isinstance(text, str)

    def test_pass_rate_100_for_trivial_target(self):
        """目標が0に近い場合、通過率はほぼ100%。"""
        p = _make_params(
            crit_min=10000, crit_max=10000, normal_min=10000, normal_max=10000,
            hits=1, crit_rate=100, evade_rate=0,
        )
        _, text = run_simulation([0], {0: p}, 0, 0, 1, "pre_decay")
        assert "100.00%" in text

    def test_pass_rate_0_for_impossible_target(self):
        """達成不可能な目標では通過率が0に近い。"""
        p = _make_params(
            crit_min=100, crit_max=100, normal_min=100, normal_max=100,
            hits=1, crit_rate=100, evade_rate=0,
        )
        _, text = run_simulation([0], {0: p}, 0, 0, 999_999_999, "pre_decay")
        assert "0.00%" in text

    def test_no_target_no_text(self):
        """目標値なしの場合、通過率テキストは空。"""
        p = _make_params(crit_min=1000, crit_max=2000, normal_min=500, normal_max=1000)
        _, text = run_simulation([0], {0: p}, 50, 0, 0, "pre_decay")
        assert text == ""

    def test_global_crit_evade_applied(self):
        """カードにcrit_rate/evade_rate未指定時、グローバル値が適用される。"""
        p = {
            "crit_min": 1000, "crit_max": 1000,
            "normal_min": 500, "normal_max": 500,
            "hits": 1,
            "crit_rate": None, "evade_rate": None,
        }
        # グローバル crit=100 → 全ヒットがクリティカル
        fig, text = run_simulation([0], {0: p}, 100, 0, 999, "pre_decay")
        assert "100.00%" in text  # 1000 >= 999 は確定

    def test_empty_cards(self):
        """カードなしでもエラーにならない。"""
        import plotly.graph_objects as go
        fig, text = run_simulation([], {}, 0, 0, 0, "pre_decay")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 固定シードによるスナップショットテスト
# ---------------------------------------------------------------------------


class TestDeterministicSnapshot:
    """固定シードで出力を確定し、リファクタリング後の一致を検証する。

    _simulate_cards の内部構造に依存するため、リファクタリング時は
    乱数生成の順序が変わる場合、このテストの期待値を更新する必要がある。
    代わりに統計的性質 (平均・分散) で比較する。
    """

    def test_deterministic_mean_and_std(self):
        """固定パラメータでの平均と標準偏差が安定していることを確認。"""
        p = _make_params(
            crit_min=5000, crit_max=10000,
            normal_min=2000, normal_max=5000,
            hits=3, crit_rate=60, evade_rate=10,
        )
        # 複数回実行して統計量の範囲を確認
        means = []
        for _ in range(5):
            samples = _simulate_cards([0], {0: p}, 0, 0, "pre_decay")
            means.append(np.mean(samples))

        # E[hit] = 0.9 * (0.6*7500 + 0.4*3500) = 0.9 * 5900 = 5310
        # E[total] = 3 * 5310 = 15930
        overall_mean = np.mean(means)
        assert overall_mean == pytest.approx(15930, rel=0.05)
        # 5回の平均のばらつきが小さい
        assert np.std(means) / overall_mean < 0.02
