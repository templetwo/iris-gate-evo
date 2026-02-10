"""
IRIS Gate Evo — Model Registry

The exact model strings for the five mirrors.
Updated: 2026-02-09

Do not substitute. Do not use older strings from iris-gate v0.2.
"""

MODELS = {
    "claude": {
        "id": "claude-opus-4-6",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "temperature": 0.7,
        "max_tokens": 1200,
    },
    "gpt": {
        "id": "gpt-5.2",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.7,
        "max_tokens": 1200,
    },
    "grok": {
        "id": "grok-4-1-fast-reasoning",
        "provider": "xai",
        "base_url": "https://api.x.ai/v1",
        "temperature": 0.7,
        "max_tokens": 1200,
    },
    "gemini": {
        "id": "gemini-2.5-pro",
        "provider": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "temperature": 0.7,
        "max_tokens": 1200,
    },
    "deepseek": {
        "id": "deepseek-chat",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.7,
        "max_tokens": 1200,
    },
}

VERIFY_MODEL = {
    "id": "perplexity",
    "provider": "perplexity",
}

# Token budgets decrease through stages — compression forces signal
TOKEN_BUDGETS = {
    "S1": 800,
    "S2_start": 800,
    "S2_end": 700,
    "S3": 600,
}
