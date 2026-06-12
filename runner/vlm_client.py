"""OpenAI 兼容的 VLM 客户端 —— 独立实现，支持 function-calling 与 token 用量统计。

默认对接 Qwen3-VL-8B(SiliconFlow 等 OpenAI 兼容端点)。真跑需要 base_url + api_key +
model(后面由用户补)。本模块不依赖任何外部 agent 框架。

设计
----
- `send(messages, tools=None) -> AssistantResponse`：第一个位置参数是 messages，方便被
  budgetsee.instrument.instrument_sender 包装(发送前挂视觉历史策略)。
- 返回里带 usage(prompt/completion/cached tokens)，供成本台账。SiliconFlow/OpenAI 的
  usage 字段名可能不同，已做容错抽取。
- 解析 native tool_calls；无 tool_calls 时 content 即最终回复。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON 字符串(原样保留，便于解析失败时排错)

    def parsed_args(self) -> Dict[str, Any]:
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0   # 命中缓存的输入 token(若端点提供)，用于 cache-aware 成本

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class AssistantResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


def parse_usage(usage_json: Dict[str, Any]) -> Usage:
    """从 API usage 字段抽取，兼容不同命名(cached_tokens / prompt_tokens_details.cached_tokens)。"""
    if not usage_json:
        return Usage()
    cached = usage_json.get("cached_tokens", 0)
    details = usage_json.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = cached or details.get("cached_tokens", 0)
    return Usage(
        prompt_tokens=usage_json.get("prompt_tokens", 0),
        completion_tokens=usage_json.get("completion_tokens", 0),
        cached_tokens=cached or 0,
    )


def parse_response(resp_json: Dict[str, Any]) -> AssistantResponse:
    """把一次 chat.completions 响应解析成 AssistantResponse。"""
    choices = resp_json.get("choices", [])
    if not choices:
        return AssistantResponse(raw=resp_json)
    message = choices[0].get("message", {})
    tool_calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        tool_calls.append(ToolCall(
            id=tc.get("id", "call_0"),
            name=fn.get("name", ""),
            arguments=fn.get("arguments", "") or "",
        ))
    return AssistantResponse(
        content=message.get("content"),
        tool_calls=tool_calls,
        usage=parse_usage(resp_json.get("usage", {})),
        raw=resp_json,
    )


class OpenAICompatibleClient:
    """OpenAI 兼容 chat.completions 客户端(Qwen3-VL-8B 等)。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def send(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AssistantResponse:
        """发送一次请求。messages 是第一个位置参数，便于被 instrument_sender 包装。"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return parse_response(r.json())
