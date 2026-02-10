"""
Tests for model registry integrity.

Validates that the five mirrors are correctly specified
and token budgets follow the compression rule.
"""

from src.models import MODELS, VERIFY_MODEL, TOKEN_BUDGETS


class TestModelRegistry:
    def test_five_models(self):
        assert len(MODELS) == 5

    def test_expected_names(self):
        expected = {"claude", "gpt", "grok", "gemini", "deepseek"}
        assert set(MODELS.keys()) == expected

    def test_all_have_required_fields(self):
        required = {"id", "provider", "base_url", "temperature", "max_tokens"}
        for name, model in MODELS.items():
            for field in required:
                assert field in model, f"{name} missing {field}"

    def test_model_ids_are_current(self):
        """These are the exact 2026-02-09 strings. Do not change."""
        assert MODELS["claude"]["id"] == "claude-opus-4-6"
        assert MODELS["gpt"]["id"] == "gpt-5.2"
        assert MODELS["grok"]["id"] == "grok-4-1-fast-reasoning"
        assert MODELS["gemini"]["id"] == "gemini-2.5-pro"
        assert MODELS["deepseek"]["id"] == "deepseek-chat"

    def test_temperatures_are_identical(self):
        temps = [m["temperature"] for m in MODELS.values()]
        assert len(set(temps)) == 1, "All models should have identical temperature"
        assert temps[0] == 0.7

    def test_max_tokens_are_identical(self):
        tokens = [m["max_tokens"] for m in MODELS.values()]
        assert len(set(tokens)) == 1, "All models should have identical max_tokens"

    def test_verify_model_exists(self):
        assert VERIFY_MODEL["id"] == "perplexity"


class TestTokenBudgets:
    def test_budgets_decrease(self):
        assert TOKEN_BUDGETS["S1"] >= TOKEN_BUDGETS["S2_start"]
        assert TOKEN_BUDGETS["S2_start"] >= TOKEN_BUDGETS["S2_end"]
        assert TOKEN_BUDGETS["S2_end"] >= TOKEN_BUDGETS["S3"]

    def test_s1_is_800(self):
        assert TOKEN_BUDGETS["S1"] == 800

    def test_s3_is_600(self):
        assert TOKEN_BUDGETS["S3"] == 600
