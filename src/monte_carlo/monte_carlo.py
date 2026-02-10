"""
S5 — Monte Carlo Simulation Engine.

Pure Python. ZERO LLM calls. This is where hypotheses become numbers.

Takes S4 ParameterSpecs, samples from specified distributions
(uniform, normal, log-normal), evaluates hypothesis-specific outcome
functions, and returns statistical summaries with 95% confidence intervals.

300+ iterations per hypothesis, seeded for reproducibility.

The legacy bioelectric simulator (engines/simulators/) handles
domain-specific V_mem/Ca2+/GJ dynamics. This module is the
general-purpose Monte Carlo wrapper that feeds any hypothesis
through parameter sampling and statistical analysis.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

from src.hypothesis.s4_hypothesis import Hypothesis, ParameterSpec


@dataclass
class SimulationResult:
    """Result of simulating one hypothesis."""
    hypothesis_id: str
    prediction: str
    n_iterations: int
    parameters_sampled: dict       # param_name → {mean, std, min, max, ci_lower, ci_upper}
    outcome_stats: dict            # outcome_name → {mean, std, ci_lower, ci_upper, median}
    effect_size: float             # Cohen's d or equivalent
    power_estimate: float          # Approximate statistical power
    convergence_check: bool        # Did samples converge? (std stabilized)
    seed: int


@dataclass
class S5Result:
    """Complete S5 stage output."""
    simulations: list[SimulationResult]
    n_hypotheses: int = 0
    total_iterations: int = 0
    total_calls: int = 0           # Always 0 — pure Python


def sample_parameter(
    spec: ParameterSpec,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n values from a ParameterSpec's distribution.

    Supports:
    - uniform: U(low, high)
    - normal: N(mean, std) clipped to [low, high]
    - log-normal: exp(N(log(mean), std)) clipped to [low, high]
    """
    if spec.distribution == "normal":
        mean = spec.mean if spec.mean is not None else (spec.low + spec.high) / 2
        std = spec.std if spec.std is not None else (spec.high - spec.low) / 6
        samples = rng.normal(mean, std, size=n)
        # Clip to bounds
        samples = np.clip(samples, spec.low, spec.high)

    elif spec.distribution == "log-normal":
        mean = spec.mean if spec.mean is not None else (spec.low + spec.high) / 2
        if mean <= 0:
            mean = (spec.low + spec.high) / 2
        log_mean = np.log(max(mean, 1e-10))
        log_std = spec.std / mean if (spec.std and mean > 0) else 0.3
        samples = rng.lognormal(log_mean, log_std, size=n)
        samples = np.clip(samples, spec.low, spec.high)

    else:  # uniform (default)
        samples = rng.uniform(spec.low, spec.high, size=n)

    return samples


def compute_stats(values: np.ndarray) -> dict:
    """Compute summary statistics with 95% CI."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def check_convergence(values: np.ndarray, window: int = 50) -> bool:
    """Check if running mean has stabilized.

    Compares std of the last `window` running means to the first `window`.
    Convergence = the running mean isn't still drifting.
    """
    if len(values) < window * 2:
        return True  # Not enough samples to check

    running_means = np.cumsum(values) / np.arange(1, len(values) + 1)
    early_std = np.std(running_means[:window])
    late_std = np.std(running_means[-window:])

    # Converged if late variation is much smaller than early
    if early_std == 0:
        return True
    return late_std / early_std < 0.3


def compute_effect_size(
    treatment: np.ndarray,
    control: np.ndarray,
) -> float:
    """Cohen's d effect size between treatment and control."""
    n1, n2 = len(treatment), len(control)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = np.mean(treatment), np.mean(control)
    var1, var2 = np.var(treatment, ddof=1), np.var(control, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float(abs(mean1 - mean2) / pooled_std)


def estimate_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Approximate statistical power from effect size and sample size.

    Uses the normal approximation for a two-sample t-test.
    """
    if effect_size <= 0 or n <= 1:
        return 0.0

    # z_alpha for two-tailed test
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n / 2)

    # Power = P(reject H0 | H1 is true)
    power = 1 - norm.cdf(z_alpha - ncp)
    return float(min(max(power, 0.0), 1.0))


def simulate_hypothesis(
    hypothesis: Hypothesis,
    n_iterations: int = 300,
    seed: int = 42,
    outcome_fn: Optional[Callable] = None,
) -> SimulationResult:
    """Run Monte Carlo simulation for a single hypothesis.

    Args:
        hypothesis: S4 Hypothesis with ParameterSpecs.
        n_iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        outcome_fn: Optional custom function(params_dict) → float.
            If None, uses a default sensitivity analysis.

    Returns:
        SimulationResult with full statistics.
    """
    rng = np.random.default_rng(seed)

    # Sample all parameters
    param_samples = {}
    for spec in hypothesis.parameters:
        param_samples[spec.name] = sample_parameter(spec, n_iterations, rng)

    # Compute parameter statistics
    param_stats = {}
    for name, samples in param_samples.items():
        param_stats[name] = compute_stats(samples)

    # Run outcome function for each iteration
    if outcome_fn is not None:
        outcomes = np.array([
            outcome_fn({name: samples[i] for name, samples in param_samples.items()})
            for i in range(n_iterations)
        ])
    else:
        # Default: sensitivity analysis
        # Compute a composite "effect magnitude" as the normalized
        # distance from each parameter's midpoint
        outcomes = _default_outcome(hypothesis, param_samples, n_iterations)

    # Compute outcome statistics
    outcome_stats = {"effect_magnitude": compute_stats(outcomes)}

    # Check convergence
    converged = check_convergence(outcomes)

    # Effect size: compare upper vs lower quartile of primary parameter
    effect_size = 0.0
    if hypothesis.parameters:
        primary = param_samples[hypothesis.parameters[0].name]
        median = np.median(primary)
        lower_outcomes = outcomes[primary < median]
        upper_outcomes = outcomes[primary >= median]
        if len(lower_outcomes) > 1 and len(upper_outcomes) > 1:
            effect_size = compute_effect_size(upper_outcomes, lower_outcomes)

    # Estimate power
    power = estimate_power(effect_size, n_iterations)

    return SimulationResult(
        hypothesis_id=hypothesis.id,
        prediction=hypothesis.prediction,
        n_iterations=n_iterations,
        parameters_sampled=param_stats,
        outcome_stats=outcome_stats,
        effect_size=round(effect_size, 4),
        power_estimate=round(power, 4),
        convergence_check=converged,
        seed=seed,
    )


def _default_outcome(
    hypothesis: Hypothesis,
    param_samples: dict[str, np.ndarray],
    n_iterations: int,
) -> np.ndarray:
    """Default outcome function: normalized effect magnitude.

    For each iteration, compute how far each parameter is from
    its midpoint (normalized to [0,1]), then take the weighted sum.
    This gives a proxy for "how strong is the effect in this sample."
    """
    if not hypothesis.parameters:
        return np.ones(n_iterations) * 0.5

    n_params = len(hypothesis.parameters)
    contributions = np.zeros(n_iterations)

    for spec in hypothesis.parameters:
        samples = param_samples[spec.name]
        # Normalize to [0, 1] within the parameter range
        range_width = spec.high - spec.low
        if range_width > 0:
            normalized = (samples - spec.low) / range_width
        else:
            normalized = np.full(n_iterations, 0.5)
        contributions += normalized / n_params

    return contributions


def run_s5(
    s4_result,  # S4Result from hypothesis stage
    n_iterations: int = 300,
    base_seed: int = 42,
    outcome_fns: Optional[dict[str, Callable]] = None,
) -> S5Result:
    """Run S5 Monte Carlo for all hypotheses.

    Args:
        s4_result: Output from run_s4() with hypotheses list.
        n_iterations: Iterations per hypothesis.
        base_seed: Base random seed (each hypothesis gets base_seed + index).
        outcome_fns: Optional dict mapping hypothesis_id → outcome function.

    Returns:
        S5Result with all simulation results.
    """
    hypotheses = s4_result.hypotheses if hasattr(s4_result, 'hypotheses') else []

    simulations = []
    total_iterations = 0

    for i, hyp in enumerate(hypotheses):
        seed = base_seed + i

        # Use custom outcome function if provided
        fn = None
        if outcome_fns and hyp.id in outcome_fns:
            fn = outcome_fns[hyp.id]

        result = simulate_hypothesis(
            hypothesis=hyp,
            n_iterations=n_iterations,
            seed=seed,
            outcome_fn=fn,
        )
        simulations.append(result)
        total_iterations += n_iterations

    return S5Result(
        simulations=simulations,
        n_hypotheses=len(simulations),
        total_iterations=total_iterations,
        total_calls=0,  # Pure Python — zero API calls
    )
