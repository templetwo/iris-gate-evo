"""
IRIS Gate Evo — API Key Preflight Check

Tests all required API keys before a pipeline run starts.
Catches misconfiguration early — before burning budget on partial runs.

Each provider gets a minimal test call (list models or tiny completion)
to verify the key is valid and the endpoint is reachable.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field

import litellm

from src.models import MODELS, VERIFY_MODEL
from src.pulse.pulse import PROVIDER_PREFIX


# Provider → environment variable name
PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}


@dataclass
class KeyCheckResult:
    provider: str
    model_name: str
    env_var: str
    key_present: bool
    key_valid: bool = False
    latency_s: float = 0.0
    error: str = ""


@dataclass
class PreflightResult:
    checks: list[KeyCheckResult] = field(default_factory=list)
    all_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.key_valid)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.key_valid)

    @property
    def failed_providers(self) -> list[str]:
        return [c.provider for c in self.checks if not c.key_valid]


def check_key_present(provider: str) -> tuple[bool, str]:
    """Check if the API key environment variable exists and is non-empty."""
    env_var = PROVIDER_ENV_KEYS.get(provider, "")
    if not env_var:
        return False, f"Unknown provider: {provider}"
    val = os.environ.get(env_var, "").strip()
    return bool(val), env_var


async def check_key(model_name: str, model_config: dict) -> KeyCheckResult:
    """Test a single API key with a minimal completion call.

    Uses max_tokens=1 to minimize cost (~$0.0001 per check).
    """
    provider = model_config["provider"]
    key_present, env_var = check_key_present(provider)

    result = KeyCheckResult(
        provider=provider,
        model_name=model_name,
        env_var=env_var,
        key_present=key_present,
    )

    if not key_present:
        result.error = f"{env_var} not set or empty"
        return result

    # Build a minimal test call
    prefix = PROVIDER_PREFIX.get(provider, "openai/")
    model_id = f"{prefix}{model_config['id']}"

    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    # Providers that use openai/ prefix but their own key + base URL
    base_url = model_config.get("base_url")
    if provider in ("xai", "deepseek") and base_url:
        kwargs["api_base"] = base_url
        kwargs["api_key"] = os.environ.get(env_var, "")

    t_start = time.monotonic()

    try:
        response = await litellm.acompletion(**kwargs)
        result.latency_s = round(time.monotonic() - t_start, 2)
        result.key_valid = True
    except Exception as e:
        result.latency_s = round(time.monotonic() - t_start, 2)
        err_str = str(e)
        # max_tokens truncation is a SUCCESS — the key authenticated and model responded
        if "max_tokens" in err_str.lower() or "output limit" in err_str.lower():
            result.key_valid = True
            return result
        # Rate limit means key IS valid — it authenticated, just got throttled
        if "rate" in err_str.lower() and "limit" in err_str.lower():
            result.key_valid = True
            result.error = "rate-limited (key valid)"
            return result
        # Quota exceeded also means key authenticated — billing issue, not auth
        if "quota" in err_str.lower() or "exceeded" in err_str.lower():
            result.key_valid = True
            result.error = "quota warning (key valid)"
            return result
        # Classify the error
        if "auth" in err_str.lower() or "api key" in err_str.lower() or "401" in err_str:
            result.error = f"Authentication failed: {type(e).__name__}"
        elif "404" in err_str or "not found" in err_str.lower():
            result.error = f"Model not found: {model_config['id']}"
        elif "timeout" in err_str.lower() or "connect" in err_str.lower():
            result.error = f"Connection failed: {type(e).__name__}"
        else:
            result.error = f"{type(e).__name__}: {err_str[:120]}"

    return result


async def run_preflight(
    models: list[str] | None = None,
    include_verify: bool = True,
) -> PreflightResult:
    """Run preflight checks on all required API keys.

    Args:
        models: Which models to check. None = all five mirrors.
        include_verify: Whether to check the Perplexity verify key.

    Returns:
        PreflightResult with per-provider check details.
    """
    t_start = time.monotonic()

    # Build the list of checks to run
    targets = {}
    model_names = models or list(MODELS.keys())
    for name in model_names:
        if name in MODELS:
            targets[name] = MODELS[name]

    if include_verify:
        targets["verify"] = VERIFY_MODEL

    # Run all checks in parallel
    tasks = [check_key(name, config) for name, config in targets.items()]
    checks = await asyncio.gather(*tasks)

    result = PreflightResult(
        checks=list(checks),
        total_time_s=round(time.monotonic() - t_start, 2),
    )

    # Determine pass/fail
    # The 5 core models must pass. Verify is a warning if missing.
    core_checks = [c for c in checks if c.model_name != "verify"]
    verify_checks = [c for c in checks if c.model_name == "verify"]

    all_core_passed = all(c.key_valid for c in core_checks)
    result.all_passed = all_core_passed

    if verify_checks and not verify_checks[0].key_valid:
        result.warnings.append(
            f"Perplexity key not valid — VERIFY stage will be skipped. "
            f"({verify_checks[0].error})"
        )

    return result


def run_preflight_sync(
    models: list[str] | None = None,
    include_verify: bool = True,
) -> PreflightResult:
    """Synchronous wrapper for run_preflight()."""
    return asyncio.run(run_preflight(models, include_verify))


def format_preflight(result: PreflightResult) -> str:
    """Format preflight results for terminal display."""
    lines = []
    lines.append("PRE-FLIGHT — API Key Check")
    lines.append("-" * 40)

    for c in result.checks:
        if c.key_valid:
            status = "OK"
            detail = f"{c.latency_s}s"
        elif not c.key_present:
            status = "MISSING"
            detail = c.error
        else:
            status = "FAIL"
            detail = c.error

        tag = f"  [{status:>7s}] {c.model_name:<10s} ({c.provider})"
        lines.append(f"{tag} — {detail}")

    lines.append("-" * 40)

    if result.all_passed:
        lines.append(f"All {result.n_passed} core models ready. ({result.total_time_s}s)")
    else:
        failed = ", ".join(result.failed_providers)
        lines.append(f"BLOCKED: {result.n_failed} key(s) failed: {failed}")

    for w in result.warnings:
        lines.append(f"  WARNING: {w}")

    return "\n".join(lines)
