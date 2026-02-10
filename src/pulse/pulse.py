"""
PULSE — Parallel Unified LLM Scientific Engine

Dispatches the compiled prompt to all five models simultaneously
via LiteLLM async. Returns structured responses with timing and
token metadata for SPM tracking.

Five mirrors. One prompt. No model knows it isn't alone.
"""

import asyncio
import time
from typing import Optional

import litellm

from src.models import MODELS


# LiteLLM provider prefixes for routing
PROVIDER_PREFIX = {
    "anthropic": "anthropic/",
    "openai": "openai/",
    "mistral": "mistral/",      # Mistral via LiteLLM native
    "xai": "openai/",           # xAI uses OpenAI-compatible API
    "google": "gemini/",        # LiteLLM routes gemini/ to Google
    "deepseek": "openai/",      # DeepSeek uses OpenAI-compatible API
    "perplexity": "perplexity/", # Perplexity via LiteLLM native
}


async def _call_model(
    name: str,
    model_config: dict,
    prompt: str,
    token_budget: int,
) -> dict:
    """Call a single model and return structured response with metadata.

    Never raises — captures errors so one failure doesn't kill the pulse.
    """
    model = MODELS[name]
    prefix = PROVIDER_PREFIX.get(model["provider"], "openai/")
    model_id = f"{prefix}{model['id']}"

    # Build kwargs for LiteLLM
    base_max_tokens = min(model_config["max_tokens"], token_budget + 400)

    # Gemini 2.5 Pro is a "thinking" model — its internal reasoning tokens
    # count against max_tokens. Without headroom, thinking exhausts the budget
    # and content comes back None. Give it 4x headroom.
    if model["provider"] == "google" and "2.5" in model["id"]:
        base_max_tokens = max(base_max_tokens * 4, 8192)

    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": model_config["temperature"],
        "max_tokens": base_max_tokens,
    }

    # Providers using openai/ prefix need their own key + base URL
    if model["provider"] in ("xai", "deepseek"):
        import os
        from src.preflight import PROVIDER_ENV_KEYS
        kwargs["api_base"] = model["base_url"]
        env_var = PROVIDER_ENV_KEYS.get(model["provider"], "")
        if env_var:
            kwargs["api_key"] = os.environ.get(env_var, "")

    t_start = time.monotonic()

    try:
        response = await litellm.acompletion(**kwargs)

        t_elapsed = time.monotonic() - t_start
        content = response.choices[0].message.content
        usage = response.usage

        return {
            "model": name,
            "model_id": model["id"],
            "status": "ok",
            "content": content,
            "tokens_prompt": getattr(usage, "prompt_tokens", 0),
            "tokens_completion": getattr(usage, "completion_tokens", 0),
            "latency_s": round(t_elapsed, 2),
            "error": None,
        }

    except Exception as e:
        t_elapsed = time.monotonic() - t_start
        return {
            "model": name,
            "model_id": model["id"],
            "status": "error",
            "content": None,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "latency_s": round(t_elapsed, 2),
            "error": f"{type(e).__name__}: {e}",
        }


async def fire(
    compiled: dict,
    models: Optional[list[str]] = None,
) -> dict:
    """Fire the pulse — dispatch compiled prompt to all models in parallel.

    Args:
        compiled: The compiler output dict (from compiler.compile()).
        models: Optional list of model names to dispatch to.
                Defaults to all five mirrors.

    Returns:
        Dict with 'responses' list, 'meta' timing/cost summary,
        and 'session_id' from the compiled input.
    """
    prompt = compiled["prompt"]
    token_budget = compiled["token_budgets"]["S1"]
    model_configs = compiled["models"]

    # Which models to fire
    target_models = models or list(model_configs.keys())

    t_pulse_start = time.monotonic()

    # Fire all models in parallel
    tasks = [
        _call_model(name, model_configs[name], prompt, token_budget)
        for name in target_models
        if name in model_configs
    ]

    responses = await asyncio.gather(*tasks)

    t_pulse_total = time.monotonic() - t_pulse_start

    # Build summary metadata
    ok_responses = [r for r in responses if r["status"] == "ok"]
    total_prompt_tokens = sum(r["tokens_prompt"] for r in responses)
    total_completion_tokens = sum(r["tokens_completion"] for r in responses)

    return {
        "session_id": compiled["session_id"],
        "responses": list(responses),
        "meta": {
            "models_dispatched": len(tasks),
            "models_ok": len(ok_responses),
            "models_failed": len(tasks) - len(ok_responses),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_latency_s": round(t_pulse_total, 2),
            "slowest_model": max(responses, key=lambda r: r["latency_s"])["model"] if responses else None,
            "fastest_model": min(responses, key=lambda r: r["latency_s"])["model"] if responses else None,
        },
    }


def fire_sync(compiled: dict, models: Optional[list[str]] = None) -> dict:
    """Synchronous wrapper for fire(). Use from non-async contexts."""
    return asyncio.run(fire(compiled, models))
