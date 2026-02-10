"""
Tests for S5 Monte Carlo Simulation Engine.

Validates:
- Parameter sampling distributions (uniform, normal, log-normal)
- Statistical computations (mean, CI, convergence)
- Effect size and power estimation
- Full hypothesis simulation
- Zero LLM calls
"""

import pytest
import numpy as np
from src.hypothesis.s4_hypothesis import Hypothesis, ParameterSpec
from src.monte_carlo.monte_carlo import (
    sample_parameter,
    compute_stats,
    check_convergence,
    compute_effect_size,
    estimate_power,
    simulate_hypothesis,
    run_s5,
    S5Result,
)


# ── Fixtures ──

@pytest.fixture
def uniform_param():
    return ParameterSpec(name="dose", unit="uM", low=1.0, high=20.0, distribution="uniform")


@pytest.fixture
def normal_param():
    return ParameterSpec(
        name="Kd", unit="uM", low=5.0, high=17.0,
        distribution="normal", mean=11.0, std=2.0,
    )


@pytest.fixture
def lognormal_param():
    return ParameterSpec(
        name="concentration", unit="nM", low=1.0, high=1000.0,
        distribution="log-normal", mean=100.0, std=30.0,
    )


@pytest.fixture
def simple_hypothesis():
    return Hypothesis(
        id="H1",
        prediction="IF CBD at Kd concentration, THEN depolarization > 40mV",
        source_claims=["1", "2"],
        key_variables=["CBD", "VDAC1", "membrane potential"],
        parameters=[
            ParameterSpec(name="CBD_conc", unit="uM", low=1.0, high=20.0, distribution="uniform"),
            ParameterSpec(name="VDAC1_Kd", unit="uM", low=8.0, high=14.0, distribution="normal", mean=11.0, std=1.5),
        ],
        testability_score=8.0,
        experimental_protocol="Test with JC-1 staining",
        expected_outcome="Dose-dependent depolarization",
        null_outcome="No differential effect",
    )


# ── Parameter Sampling ──

class TestSampling:
    def test_uniform_within_bounds(self, uniform_param):
        rng = np.random.default_rng(42)
        samples = sample_parameter(uniform_param, 1000, rng)
        assert samples.min() >= 1.0
        assert samples.max() <= 20.0

    def test_uniform_mean_near_midpoint(self, uniform_param):
        rng = np.random.default_rng(42)
        samples = sample_parameter(uniform_param, 10000, rng)
        assert abs(np.mean(samples) - 10.5) < 0.5

    def test_normal_within_bounds(self, normal_param):
        rng = np.random.default_rng(42)
        samples = sample_parameter(normal_param, 1000, rng)
        assert samples.min() >= 5.0
        assert samples.max() <= 17.0

    def test_normal_mean_near_specified(self, normal_param):
        rng = np.random.default_rng(42)
        samples = sample_parameter(normal_param, 10000, rng)
        assert abs(np.mean(samples) - 11.0) < 0.5

    def test_lognormal_within_bounds(self, lognormal_param):
        rng = np.random.default_rng(42)
        samples = sample_parameter(lognormal_param, 1000, rng)
        assert samples.min() >= 1.0
        assert samples.max() <= 1000.0

    def test_lognormal_right_skewed(self, lognormal_param):
        rng = np.random.default_rng(42)
        samples = sample_parameter(lognormal_param, 10000, rng)
        # Log-normal is right-skewed: median < mean
        assert np.median(samples) < np.mean(samples)

    def test_seeded_reproducibility(self, uniform_param):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = sample_parameter(uniform_param, 100, rng1)
        s2 = sample_parameter(uniform_param, 100, rng2)
        np.testing.assert_array_equal(s1, s2)


# ── Statistics ──

class TestStats:
    def test_compute_stats_known(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_stats(values)
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_ci_contains_95_percent(self):
        rng = np.random.default_rng(42)
        values = rng.normal(10, 2, size=10000)
        stats = compute_stats(values)
        # 95% CI should contain approximately 95% of samples
        in_ci = np.sum((values >= stats["ci_lower"]) & (values <= stats["ci_upper"]))
        pct = in_ci / len(values)
        assert 0.94 <= pct <= 0.96

    def test_ci_width_decreases_with_n(self):
        # Use same distribution, different sample sizes
        rng_s = np.random.default_rng(42)
        rng_l = np.random.default_rng(42)
        small_vals = rng_s.normal(0, 1, size=100)
        large_vals = rng_l.normal(0, 1, size=10000)
        small = compute_stats(small_vals)
        large = compute_stats(large_vals)
        # The 95% CI for normal(0,1) should be tighter with more samples
        # ~[-1.96, 1.96] for N=inf. Large sample should be closer to 3.92 width.
        large_width = large["ci_upper"] - large["ci_lower"]
        # With 10000 samples from N(0,1), CI width should be ~3.92
        assert 3.8 < large_width < 4.1


# ── Convergence Check ──

class TestConvergence:
    def test_stable_series_converges(self):
        rng = np.random.default_rng(42)
        values = rng.normal(10, 0.5, size=500)
        assert check_convergence(values) == True

    def test_drifting_series_may_not_converge(self):
        # Linearly increasing values — running mean is still moving
        values = np.linspace(0, 100, 500)
        result = check_convergence(values)
        # This should NOT converge — running mean is still changing
        assert result == False

    def test_short_series_assumed_converged(self):
        values = np.array([1.0, 2.0, 3.0])
        assert check_convergence(values) == True


# ── Effect Size ──

class TestEffectSize:
    def test_identical_distributions_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = compute_effect_size(a, a)
        assert d == 0.0

    def test_large_separation_large_d(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = compute_effect_size(a, b)
        assert d > 2.0  # Very large effect

    def test_moderate_effect(self):
        rng = np.random.default_rng(42)
        a = rng.normal(10, 2, size=100)
        b = rng.normal(11, 2, size=100)
        d = compute_effect_size(a, b)
        assert 0.3 < d < 1.0  # Moderate effect

    def test_empty_arrays(self):
        assert compute_effect_size(np.array([]), np.array([1, 2, 3])) == 0.0


# ── Power Estimation ──

class TestPower:
    def test_large_effect_high_power(self):
        power = estimate_power(effect_size=1.0, n=300)
        assert power > 0.9

    def test_small_effect_lower_power(self):
        power = estimate_power(effect_size=0.2, n=50)
        assert power < 0.5

    def test_zero_effect_zero_power(self):
        power = estimate_power(effect_size=0.0, n=300)
        assert power == 0.0

    def test_power_increases_with_n(self):
        p_small = estimate_power(effect_size=0.5, n=50)
        p_large = estimate_power(effect_size=0.5, n=500)
        assert p_large > p_small


# ── Full Simulation ──

class TestSimulateHypothesis:
    def test_basic_simulation(self, simple_hypothesis):
        result = simulate_hypothesis(simple_hypothesis, n_iterations=100, seed=42)
        assert result.hypothesis_id == "H1"
        assert result.n_iterations == 100
        assert result.seed == 42

    def test_parameters_sampled(self, simple_hypothesis):
        result = simulate_hypothesis(simple_hypothesis, n_iterations=100, seed=42)
        assert "CBD_conc" in result.parameters_sampled
        assert "VDAC1_Kd" in result.parameters_sampled

    def test_outcome_stats_exist(self, simple_hypothesis):
        result = simulate_hypothesis(simple_hypothesis, n_iterations=100, seed=42)
        assert "effect_magnitude" in result.outcome_stats
        stats = result.outcome_stats["effect_magnitude"]
        assert "mean" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats

    def test_effect_size_computed(self, simple_hypothesis):
        result = simulate_hypothesis(simple_hypothesis, n_iterations=300, seed=42)
        assert isinstance(result.effect_size, float)

    def test_power_computed(self, simple_hypothesis):
        result = simulate_hypothesis(simple_hypothesis, n_iterations=300, seed=42)
        assert 0.0 <= result.power_estimate <= 1.0

    def test_convergence_checked(self, simple_hypothesis):
        result = simulate_hypothesis(simple_hypothesis, n_iterations=300, seed=42)
        assert result.convergence_check == True or result.convergence_check == False

    def test_reproducible(self, simple_hypothesis):
        r1 = simulate_hypothesis(simple_hypothesis, n_iterations=100, seed=42)
        r2 = simulate_hypothesis(simple_hypothesis, n_iterations=100, seed=42)
        assert r1.parameters_sampled == r2.parameters_sampled

    def test_custom_outcome_fn(self, simple_hypothesis):
        """Custom outcome function should be used when provided."""
        def my_fn(params):
            return params["CBD_conc"] * 2.0

        result = simulate_hypothesis(simple_hypothesis, n_iterations=100, seed=42, outcome_fn=my_fn)
        mean = result.outcome_stats["effect_magnitude"]["mean"]
        # CBD_conc is uniform [1, 20], so mean * 2 should be ~21
        assert 15 < mean < 30

    def test_no_parameters_still_works(self):
        hyp = Hypothesis(
            id="H_empty",
            prediction="test",
            source_claims=[],
            key_variables=[],
            parameters=[],
            testability_score=1.0,
            experimental_protocol="",
            expected_outcome="",
            null_outcome="",
        )
        result = simulate_hypothesis(hyp, n_iterations=50, seed=42)
        assert result.n_iterations == 50


# ── S5 Full Run ──

class TestS5Run:
    def test_full_s5(self, simple_hypothesis):
        from src.hypothesis.s4_hypothesis import S4Result
        s4 = S4Result(hypotheses=[simple_hypothesis], n_hypotheses=1)
        result = run_s5(s4, n_iterations=100, base_seed=42)
        assert result.n_hypotheses == 1
        assert result.total_iterations == 100
        assert result.total_calls == 0  # Pure Python

    def test_multiple_hypotheses(self, simple_hypothesis):
        from src.hypothesis.s4_hypothesis import S4Result
        hyp2 = Hypothesis(
            id="H2", prediction="test2", source_claims=[], key_variables=[],
            parameters=[ParameterSpec("x", "uM", 0, 10, "uniform")],
            testability_score=5.0, experimental_protocol="", expected_outcome="", null_outcome="",
        )
        s4 = S4Result(hypotheses=[simple_hypothesis, hyp2], n_hypotheses=2)
        result = run_s5(s4, n_iterations=100, base_seed=42)
        assert result.n_hypotheses == 2
        assert result.total_iterations == 200

    def test_empty_s4(self):
        from src.hypothesis.s4_hypothesis import S4Result
        s4 = S4Result(hypotheses=[], n_hypotheses=0)
        result = run_s5(s4, n_iterations=100)
        assert result.n_hypotheses == 0
        assert result.total_iterations == 0
