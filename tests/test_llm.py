"""The real LLM client, exercised against a mocked HTTP transport (no network, no key)."""

import json

import httpx
import pytest

from app.config import Settings
from app.llm import OpenAICompatibleLLM, build_llm


def make_llm(handler, base_url="https://example.test/v1/") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url=base_url,
        timeout=5,
        max_retries=0,
        max_tokens=100,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def completion(message: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        },
    )


def test_sends_openai_format_and_parses_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_documents", "arguments": '{"query": "E3"}'},
                    }
                ],
            }
        )

    tools = [{"type": "function", "function": {"name": "search_documents", "parameters": {}}}]
    response = make_llm(handler).chat(
        [{"role": "user", "content": "hi"}], tools=tools, tool_choice="none"
    )

    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["tools"] == tools and seen["body"]["tool_choice"] == "none"
    assert response.tool_calls[0].name == "search_documents"
    assert response.tool_calls[0].arguments == {"query": "E3"}
    assert response.as_message()["tool_calls"][0]["function"]["arguments"] == '{"query": "E3"}'


def test_invalid_tool_arguments_become_empty_dict():
    def handler(request):
        return completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "search_documents", "arguments": "{not json"},
                    }
                ],
            }
        )

    assert make_llm(handler).chat([{"role": "user", "content": "x"}]).tool_calls[0].arguments == {}


def test_build_llm_provider_selection():
    assert build_llm(Settings(llm_provider="extractive")) is None
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        build_llm(Settings(llm_provider="gemini", llm_api_key=None))
    gemini = build_llm(Settings(llm_provider="gemini", llm_api_key="k"))
    assert "generativelanguage.googleapis.com" in str(gemini._client.base_url)
    assert gemini._temperature == 0.0
    claude = build_llm(Settings(llm_provider="anthropic", llm_api_key="k"))
    assert str(claude._client.base_url).startswith("https://api.anthropic.com/v1")
    assert claude._temperature is None  # recent Claude models reject temperature < 1 here


def test_temperature_omitted_when_none():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return completion({"role": "assistant", "content": "ok"})

    llm = make_llm(handler)
    llm._temperature = None
    llm.chat([{"role": "user", "content": "x"}])
    assert "temperature" not in seen["body"]
