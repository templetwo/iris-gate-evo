"""
Tests for the Anonymizer.

Validates that:
- Mirror labels are randomized every round
- No model identity leaks into anonymized output
- Audit mapping is preserved but separate
"""

import pytest
from src.parser import ParsedResponse, Claim
from src.stages.anonymizer import anonymize_round, build_debate_prompt


def _make_parsed(model: str, claim_text: str) -> ParsedResponse:
    """Helper: create a parsed response with one claim."""
    return ParsedResponse(
        model=model,
        claims=[Claim(statement=claim_text, type=1, confidence=0.8)],
        raw=f"CLAIM: {claim_text}\nTYPE: 1\nCONFIDENCE: 0.8",
    )


@pytest.fixture
def five_responses():
    return [
        _make_parsed("claude", "CBD binds VDAC1"),
        _make_parsed("gpt", "Cancer cells are depolarized"),
        _make_parsed("grok", "ROS drives apoptosis"),
        _make_parsed("gemini", "TRPV1 contributes to selectivity"),
        _make_parsed("deepseek", "Mitochondrial permeability increases"),
    ]


class TestAnonymization:
    def test_returns_five_anonymized(self, five_responses):
        anon, mapping = anonymize_round(five_responses, round_num=1, seed=42)
        assert len(anon) == 5
        assert len(mapping) == 5

    def test_labels_are_a_through_e(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        labels = {a["label"] for a in anon}
        assert labels == {"A", "B", "C", "D", "E"}

    def test_no_model_names_in_output(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        for a in anon:
            assert "claude" not in str(a).lower()
            assert "gpt" not in str(a).lower()
            assert "grok" not in str(a).lower()
            assert "gemini" not in str(a).lower()
            assert "deepseek" not in str(a).lower()

    def test_mapping_preserves_identity(self, five_responses):
        _, mapping = anonymize_round(five_responses, round_num=1, seed=42)
        model_names = set(mapping.values())
        assert model_names == {"claude", "gpt", "grok", "gemini", "deepseek"}

    def test_randomization_changes_per_round(self, five_responses):
        """Mirror A in round 1 should NOT be the same model as Mirror A in round 2."""
        _, mapping_r1 = anonymize_round(five_responses, round_num=1, seed=42)
        _, mapping_r2 = anonymize_round(five_responses, round_num=2, seed=42)

        # At least one label should map to a different model
        changes = sum(
            1 for label in mapping_r1
            if label in mapping_r2 and mapping_r1[label] != mapping_r2[label]
        )
        assert changes > 0, (
            f"Anonymization did NOT change between rounds!\n"
            f"Round 1: {mapping_r1}\nRound 2: {mapping_r2}"
        )

    def test_different_seeds_give_different_mappings(self, five_responses):
        _, mapping_a = anonymize_round(five_responses, round_num=1, seed=100)
        _, mapping_b = anonymize_round(five_responses, round_num=1, seed=999)
        # Very unlikely to be identical with different seeds
        assert mapping_a != mapping_b

    def test_claims_preserved_in_anonymized(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        all_statements = set()
        for a in anon:
            for c in a["claims"]:
                all_statements.add(c["statement"])
        assert "CBD binds VDAC1" in all_statements
        assert "ROS drives apoptosis" in all_statements


class TestDebatePrompt:
    def test_prompt_contains_mirror_labels(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        prompt = build_debate_prompt("test question?", anon, round_num=1, token_budget=800)
        assert "[Mirror A]" in prompt
        assert "[Mirror B]" in prompt

    def test_prompt_contains_question(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        prompt = build_debate_prompt("test question?", anon, round_num=1, token_budget=800)
        assert "test question?" in prompt

    def test_prompt_has_no_model_names(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        prompt = build_debate_prompt("test?", anon, round_num=1, token_budget=800)
        assert "claude" not in prompt.lower()
        assert "gpt" not in prompt.lower()

    def test_prompt_includes_round_number(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=3, seed=42)
        prompt = build_debate_prompt("test?", anon, round_num=3, token_budget=700)
        assert "ROUND 3" in prompt

    def test_prompt_includes_token_budget(self, five_responses):
        anon, _ = anonymize_round(five_responses, round_num=1, seed=42)
        prompt = build_debate_prompt("test?", anon, round_num=1, token_budget=700)
        assert "700" in prompt
