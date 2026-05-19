"""Provider adapter for the teaching agents.

The lesson files use an Anthropic-shaped interface:
`client.messages.create(...)`, `response.stop_reason`, and content blocks with
`.type`, `.text`, `.name`, `.input`, `.id`. This adapter keeps that shape while
allowing either Anthropic Messages or Azure OpenAI Chat Completions underneath.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


def get_model_id() -> str:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "azure_openai":
        return os.environ["AZURE_OPENAI_DEPLOYMENT"]
    return os.environ["MODEL_ID"]


def create_llm_client():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "azure_openai":
        return AzureOpenAICompatClient()
    if provider == "anthropic":
        return AnthropicCompatClient()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _normalize_block(block: Any) -> Any:
    if isinstance(block, (TextBlock, ToolUseBlock)):
        return block
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return TextBlock(text=getattr(block, "text", ""))
    if block_type == "tool_use":
        return ToolUseBlock(
            id=getattr(block, "id"),
            name=getattr(block, "name"),
            input=getattr(block, "input", {}) or {},
        )
    return block


def _normalize_response(response: Any) -> Any:
    return SimpleNamespace(
        stop_reason=getattr(response, "stop_reason", None),
        content=[_normalize_block(block) for block in response.content],
    )


def _anthropic_block(block: Any) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, dict):
        return block
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id"),
            "name": getattr(block, "name"),
            "input": getattr(block, "input", {}) or {},
        }
    raise TypeError(f"Unsupported content block: {block!r}")


def _anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [_anthropic_block(block) for block in content]
        converted.append({**message, "content": content})
    return converted


def _openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in tools
    ]


def _openai_messages(
    system: str | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if system:
        converted.append({"role": "system", "content": system})

    for message in messages:
        role = message["role"]
        content = message.get("content")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if role == "assistant" and isinstance(content, list):
            text_parts = []
            tool_calls = []
            for block in content:
                block = _normalize_block(block)
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })
            converted.append({
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
                "tool_calls": tool_calls or None,
            })
            continue

        if role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    converted.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(block.get("content", "")),
                    })
                elif isinstance(block, dict) and block.get("type") == "text":
                    converted.append({"role": "user", "content": block.get("text", "")})
                else:
                    converted.append({"role": "user", "content": str(block)})
            continue

        converted.append({"role": role, "content": str(content)})

    return converted


class AnthropicCompatClient:
    def __init__(self):
        from anthropic import Anthropic

        if os.getenv("ANTHROPIC_BASE_URL"):
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        self._client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        kwargs["messages"] = _anthropic_messages(kwargs["messages"])
        return _normalize_response(self._client.messages.create(**kwargs))


class AzureOpenAICompatClient:
    def __init__(self):
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        self.messages = self

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        params = {
            "model": model,
            "messages": _openai_messages(system, messages),
            **kwargs,
        }
        openai_tools = _openai_tools(tools)
        if openai_tools:
            params["tools"] = openai_tools
        if max_tokens is not None:
            token_param = os.getenv(
                "AZURE_OPENAI_MAX_TOKENS_PARAM",
                "max_completion_tokens",
            )
            params[token_param] = max_tokens
        response = self._client.chat.completions.create(**params)
        message = response.choices[0].message
        blocks: list[Any] = []
        if message.content:
            blocks.append(TextBlock(text=message.content))
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": call.function.arguments}
            blocks.append(ToolUseBlock(
                id=call.id,
                name=call.function.name,
                input=arguments,
            ))
        return SimpleNamespace(
            stop_reason="tool_use" if message.tool_calls else response.choices[0].finish_reason,
            content=blocks,
        )
