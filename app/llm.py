"""One chat client for three providers, via their OpenAI-compatible endpoints.

OpenAI, Google Gemini and Anthropic all accept the OpenAI Chat Completions format
(including tool calling), so a single client with a per-provider base URL covers them.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

import httpx

from app.config import Settings, get_settings

PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,  # the SDK default
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1/",
}
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-5-5",
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # parsed JSON; {} if the model sent invalid JSON
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """The assistant turn, in the format the API expects to see echoed back."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in self.tool_calls
            ]
        return message


class LLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None,
        timeout: float,
        max_retries: int,
        max_tokens: int,
        temperature: float | None = 0.0,
        http_client: httpx.Client | None = None,  # injectable for tests
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature  # 0 for factual QA where allowed
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    raw_arguments=call.function.arguments or "{}",
                )
            )
        return LLMResponse(content=message.content, tool_calls=calls)


def build_llm(settings: Settings) -> LLM | None:
    """Return the configured LLM, or None for the offline extractive baseline."""
    if settings.llm_provider == "extractive":
        return None
    temperature = settings.llm_temperature
    if temperature is None:
        temperature = None if settings.llm_provider == "anthropic" else 0.0
    if settings.llm_api_key is None:
        raise RuntimeError(f"LLM_API_KEY is required for LLM_PROVIDER={settings.llm_provider}")
    return OpenAICompatibleLLM(
        model=settings.llm_model or DEFAULT_MODELS[settings.llm_provider],
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url or PROVIDER_BASE_URLS[settings.llm_provider],
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_output_tokens,
        temperature=temperature,
    )


@lru_cache
def get_llm() -> LLM | None:
    return build_llm(get_settings())
