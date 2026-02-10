"""
Tests for API Key Preflight Check.

Validates:
- Environment variable detection
- Provider→env_var mapping
- PreflightResult aggregation
- Format output
- Async test_key with mocked LiteLLM
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.preflight import (
    PROVIDER_ENV_KEYS,
    KeyCheckResult,
    PreflightResult,
    check_key_present,
    check_key,
    run_preflight,
    format_preflight,
)
from src.models import MODELS, VERIFY_MODEL


# ── Provider Mapping ──

class TestProviderMapping:
    def test_all_core_providers_mapped(self):
        """Every model provider has an env var mapping."""
        for name, config in MODELS.items():
            provider = config["provider"]
            assert provider in PROVIDER_ENV_KEYS, f"{name}: {provider} missing from PROVIDER_ENV_KEYS"

    def test_verify_provider_mapped(self):
        assert VERIFY_MODEL["provider"] in PROVIDER_ENV_KEYS

    def test_expected_env_vars(self):
        assert PROVIDER_ENV_KEYS["anthropic"] == "ANTHROPIC_API_KEY"
        assert PROVIDER_ENV_KEYS["mistral"] == "MISTRAL_API_KEY"
        assert PROVIDER_ENV_KEYS["xai"] == "XAI_API_KEY"
        assert PROVIDER_ENV_KEYS["google"] == "GOOGLE_API_KEY"
        assert PROVIDER_ENV_KEYS["deepseek"] == "DEEPSEEK_API_KEY"
        assert PROVIDER_ENV_KEYS["perplexity"] == "PERPLEXITY_API_KEY"


# ── Key Presence Check ──

class TestCheckKeyPresent:
    def test_key_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        present, env_var = check_key_present("anthropic")
        assert present is True
        assert env_var == "ANTHROPIC_API_KEY"

    def test_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        present, env_var = check_key_present("openai")
        assert present is False
        assert env_var == "OPENAI_API_KEY"

    def test_key_empty_string(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "   ")
        present, _ = check_key_present("xai")
        assert present is False

    def test_unknown_provider(self):
        present, env_var = check_key_present("unknown_provider")
        assert present is False


# ── KeyCheckResult ──

class TestKeyCheckResult:
    def test_fields(self):
        r = KeyCheckResult(
            provider="anthropic",
            model_name="claude",
            env_var="ANTHROPIC_API_KEY",
            key_present=True,
            key_valid=True,
            latency_s=0.42,
        )
        assert r.key_valid is True
        assert r.latency_s == 0.42

    def test_defaults(self):
        r = KeyCheckResult(
            provider="mistral",
            model_name="mistral",
            env_var="MISTRAL_API_KEY",
            key_present=False,
        )
        assert r.key_valid is False
        assert r.error == ""


# ── PreflightResult Aggregation ──

class TestPreflightResult:
    def test_n_passed(self):
        r = PreflightResult(checks=[
            KeyCheckResult("a", "m1", "K1", True, key_valid=True),
            KeyCheckResult("b", "m2", "K2", True, key_valid=True),
            KeyCheckResult("c", "m3", "K3", True, key_valid=False),
        ])
        assert r.n_passed == 2
        assert r.n_failed == 1

    def test_failed_providers(self):
        r = PreflightResult(checks=[
            KeyCheckResult("anthropic", "claude", "K1", True, key_valid=True),
            KeyCheckResult("mistral", "mistral", "K2", True, key_valid=False),
            KeyCheckResult("xai", "grok", "K3", False, key_valid=False),
        ])
        assert r.failed_providers == ["mistral", "xai"]

    def test_all_passed_false_by_default(self):
        r = PreflightResult()
        assert r.all_passed is False

    def test_empty_checks(self):
        r = PreflightResult(checks=[])
        assert r.n_passed == 0
        assert r.n_failed == 0
        assert r.failed_providers == []


# ── test_key (async, mocked) ──

class TestCheckKey:
    def test_key_missing_returns_early(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = asyncio.run(check_key("claude", MODELS["claude"]))
        assert result.key_present is False
        assert result.key_valid is False
        assert "not set" in result.error

    @patch("src.preflight.litellm.acompletion")
    def test_key_valid(self, mock_acompletion, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "p"
        mock_acompletion.return_value = mock_response

        result = asyncio.run(check_key("claude", MODELS["claude"]))
        assert result.key_present is True
        assert result.key_valid is True
        assert result.error == ""

    @patch("src.preflight.litellm.acompletion")
    def test_auth_error(self, mock_acompletion, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "bad-key")
        mock_acompletion.side_effect = Exception("401 Unauthorized: Invalid API key")

        result = asyncio.run(check_key("mistral", MODELS["mistral"]))
        assert result.key_present is True
        assert result.key_valid is False
        assert "Authentication" in result.error

    @patch("src.preflight.litellm.acompletion")
    def test_model_not_found(self, mock_acompletion, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "ok-key")
        mock_acompletion.side_effect = Exception("404 Not Found")

        result = asyncio.run(check_key("mistral", MODELS["mistral"]))
        assert result.key_valid is False
        assert "not found" in result.error.lower()

    @patch("src.preflight.litellm.acompletion")
    def test_connection_error(self, mock_acompletion, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ok")
        mock_acompletion.side_effect = ConnectionError("Connection refused")

        result = asyncio.run(check_key("deepseek", MODELS["deepseek"]))
        assert result.key_valid is False
        assert "Connection" in result.error

    @patch("src.preflight.litellm.acompletion")
    def test_minimal_tokens(self, mock_acompletion, monkeypatch):
        """Verify the test call uses max_tokens=1 to minimize cost."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "p"
        mock_acompletion.return_value = mock_response

        asyncio.run(check_key("claude", MODELS["claude"]))
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["max_tokens"] == 1

    @patch("src.preflight.litellm.acompletion")
    def test_xai_uses_api_base(self, mock_acompletion, monkeypatch):
        """xAI should pass api_base for its non-standard endpoint."""
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "p"
        mock_acompletion.return_value = mock_response

        asyncio.run(check_key("grok", MODELS["grok"]))
        call_kwargs = mock_acompletion.call_args[1]
        assert "api_base" in call_kwargs
        assert "x.ai" in call_kwargs["api_base"]


# ── run_preflight (async, mocked) ──

class TestRunPreflight:
    @patch("src.preflight.check_key")
    def test_all_pass(self, mock_check_key):
        async def _ok(name, config):
            return KeyCheckResult(
                provider=config["provider"], model_name=name,
                env_var="KEY", key_present=True, key_valid=True, latency_s=0.1,
            )
        mock_check_key.side_effect = _ok

        result = asyncio.run(run_preflight(include_verify=False))
        assert result.all_passed is True
        assert result.n_passed == 5
        assert result.n_failed == 0

    @patch("src.preflight.check_key")
    def test_one_fails(self, mock_check_key):
        async def _mixed(name, config):
            valid = (name != "mistral")
            return KeyCheckResult(
                provider=config["provider"], model_name=name,
                env_var="KEY", key_present=True, key_valid=valid, latency_s=0.1,
                error="" if valid else "auth failed",
            )
        mock_check_key.side_effect = _mixed

        result = asyncio.run(run_preflight(include_verify=False))
        assert result.all_passed is False
        assert result.n_failed == 1
        assert "mistral" in result.failed_providers

    @patch("src.preflight.check_key")
    def test_verify_warning_not_blocking(self, mock_check_key):
        """Perplexity failure should warn but not block."""
        async def _verify_fail(name, config):
            valid = (name != "verify")
            error = "key missing" if name == "verify" else ""
            return KeyCheckResult(
                provider=config["provider"], model_name=name,
                env_var="KEY", key_present=(name != "verify"),
                key_valid=valid, latency_s=0.1, error=error,
            )
        mock_check_key.side_effect = _verify_fail

        result = asyncio.run(run_preflight(include_verify=True))
        assert result.all_passed is True  # Core models all passed
        assert len(result.warnings) == 1
        assert "Perplexity" in result.warnings[0]

    @patch("src.preflight.check_key")
    def test_subset_models(self, mock_check_key):
        async def _ok(name, config):
            return KeyCheckResult(
                provider=config["provider"], model_name=name,
                env_var="KEY", key_present=True, key_valid=True, latency_s=0.1,
            )
        mock_check_key.side_effect = _ok

        result = asyncio.run(run_preflight(models=["claude", "mistral"], include_verify=False))
        assert result.n_passed == 2
        assert len(result.checks) == 2


# ── Format Output ──

class TestFormatPreflight:
    def test_all_ok_format(self):
        result = PreflightResult(
            checks=[
                KeyCheckResult("anthropic", "claude", "ANTHROPIC_API_KEY", True, True, 0.3),
                KeyCheckResult("mistral", "mistral", "MISTRAL_API_KEY", True, True, 0.5),
            ],
            all_passed=True,
            total_time_s=0.5,
        )
        output = format_preflight(result)
        assert "OK" in output
        assert "PRE-FLIGHT" in output
        assert "All 2 core models ready" in output

    def test_failure_format(self):
        result = PreflightResult(
            checks=[
                KeyCheckResult("anthropic", "claude", "ANTHROPIC_API_KEY", True, True, 0.3),
                KeyCheckResult("mistral", "mistral", "MISTRAL_API_KEY", False, False, 0.0, error="MISTRAL_API_KEY not set"),
            ],
            all_passed=False,
            total_time_s=0.3,
        )
        output = format_preflight(result)
        assert "MISSING" in output
        assert "BLOCKED" in output

    def test_warnings_displayed(self):
        result = PreflightResult(
            checks=[],
            all_passed=True,
            warnings=["Perplexity key not valid — VERIFY stage will be skipped."],
        )
        output = format_preflight(result)
        assert "WARNING" in output
        assert "Perplexity" in output
