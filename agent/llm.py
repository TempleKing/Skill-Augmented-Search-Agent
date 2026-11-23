from functools import lru_cache
from langchain_openai import ChatOpenAI

from config import settings


def is_retryable_llm_error(error: Exception) -> bool:
    """Return whether an LLM failure is likely transient."""
    name = type(error).__name__.lower()
    if any(term in name for term in (
        "timeout", "connection", "ratelimit", "rate_limit"
    )):
        return True

    status_code = getattr(error, "status_code", None)
    return status_code == 429 or (
        isinstance(status_code, int) and status_code >= 500
    )


def configure_llm_for_round(llm: ChatOpenAI, search_round: int):
    """Apply the configured thinking schedule (baseline: all ten rounds)."""
    is_deepseek = "api.deepseek.com" in settings.llm_base_url.casefold()
    should_disable = (
        is_deepseek
        and settings.llm_enable_thinking
        and search_round >= settings.llm_thinking_rounds
        and hasattr(llm, "model_copy")
    )
    if not should_disable:
        return llm

    return llm.model_copy(update={
        "extra_body": {"thinking": {"type": "disabled"}},
        "reasoning_effort": None,
    })


def configure_llm_for_required_tool(llm: ChatOpenAI):
    """DeepSeek requires thinking to be off for tool_choice='required'."""
    is_deepseek = "api.deepseek.com" in settings.llm_base_url.casefold()
    if not is_deepseek or not hasattr(llm, "model_copy"):
        return llm
    return llm.model_copy(update={
        "extra_body": {"thinking": {"type": "disabled"}},
        "reasoning_effort": None,
    })


@lru_cache
def get_llm():
    provider_url = settings.llm_base_url.casefold()
    api_key = settings.llm_api_key
    if not all([
        api_key,
        settings.llm_base_url,
        settings.llm_model
    ]):
        raise ValueError(
            "Missing LLM_API_KEY, LLM_BASE_URL "
            "or LLM_MODEL in .env"
        )

    options = dict(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        # Retries are handled explicitly by the Agent graph so they are
        # observable through hooks and have a clear terminal state.
        max_retries=0
    )
    if settings.llm_max_tokens > 0:
        options["max_tokens"] = settings.llm_max_tokens

    if "api.deepseek.com" in provider_url:
        options["extra_body"] = {
            "thinking": {
                "type": (
                    "enabled"
                    if settings.llm_enable_thinking
                    else "disabled"
                )
            }
        }
        if settings.llm_enable_thinking:
            options["reasoning_effort"] = settings.llm_reasoning_effort
    return ChatOpenAI(**options)
