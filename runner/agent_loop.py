"""多轮 agent 循环 —— 把 VLM client + 可插拔工具 + BudgetSee 视觉历史策略串起来。

整体流程(一个 rollout / 一道题)
================================
  1. 起一个持久 CodeInterpreter 内核(原图预载)，建 RolloutCtx。
  2. 初始 messages = [system, user(原图 + 问题)]。
  3. 每轮携带完整视觉历史发送(policy=full)；累积的 messages 始终保留完整。
     (注：早期的视觉历史压缩层 budgetsee 已移除；policy/keep_original 参数保留但当前不生效。)
  4. 循环 max_turns：
       发送 → 解析 →
         有 tool_calls：执行工具，把 assistant(带 tool_calls) + tool(结果) +
                        user(产出图) 追加进 messages，continue。
         无 tool_calls：content 即最终答案，break。
     期间累计 token 用量、产出图数；命中 max_images / 用尽轮次 → 记退化标记。
  5. 返回 EpisodeResult(答案 + 完整 messages + 成本台账 + 退化标记 + 视觉历史统计)。

工具无关：循环只依赖 BaseTool 抽象与 OpenAI 消息格式，加新工具不动这里。
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .tools import BaseTool, RolloutCtx, ToolRegistry, ToolResult
from .vlm_client import AssistantResponse, Usage

# code_interpreter 沙箱可选导入(纯逻辑测试时可不依赖)
try:
    from .code_interpreter import CodeInterpreter
except Exception:  # pragma: no cover
    CodeInterpreter = None


DEFAULT_TIR_SYSTEM_PROMPT = (
    "You are a visual reasoning agent. You are given image(s) and a question. "
    "You can call the `code_interpreter` tool to run Python and analyze the image(s) "
    "(preloaded as PIL variables original_image, original_image_1, ...). Produced images "
    "become tool_image_N and are shown to you. Reason step by step, use the tool as needed, "
    "then give your final answer wrapped in <answer>...</answer>."
)


# ---------------- 消息构造 ----------------

def image_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def build_initial_messages(system_prompt: str, question: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_to_data_uri(im)}} for im in images
    ]
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def assistant_message_with_tool_calls(resp: AssistantResponse) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": resp.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in resp.tool_calls
        ],
    }


def tool_result_message(tool_call_id: str, result: ToolResult) -> Dict[str, Any]:
    return {
        "role": "tool",
        "content": json.dumps({"result": result.text}),
        "tool_call_id": tool_call_id,
    }


def tool_images_message(result: ToolResult) -> Optional[Dict[str, Any]]:
    """把工具产出图打成一条独立 user 消息(与 vs-skills 格式一致，供 visual_history 压缩)。"""
    if not result.images:
        return None
    content: List[Dict[str, Any]] = []
    names = []
    for name, img in result.images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(img)}})
        names.append(name)
    content.append({"type": "text", "text": f"[{', '.join(names)}]"})
    return {"role": "user", "content": content}


def extract_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else text.strip()) or None


# ---------------- 结果 ----------------

@dataclass
class EpisodeResult:
    final_answer: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)   # 完整(未压缩)历史
    usage: Usage = field(default_factory=Usage)
    num_turns: int = 0
    num_tool_calls: int = 0
    num_tool_images: int = 0
    degenerate: Dict[str, bool] = field(default_factory=dict)       # turn_exhausted / max_images / no_answer
    instrument_stats: Dict[str, Any] = field(default_factory=dict)  # 视觉历史压缩统计(含 resend_factor)

    @property
    def is_degenerate(self) -> bool:
        return any(self.degenerate.values())


# ---------------- 主循环 ----------------

def run_episode(
    question: str,
    input_images: List[Image.Image],
    send_fn: Callable[..., AssistantResponse],
    registry: ToolRegistry,
    enabled_tools: List[str],
    *,
    policy: str = "full",
    keep_original: bool = True,
    system_prompt: str = DEFAULT_TIR_SYSTEM_PROMPT,
    max_turns: int = 10,
    max_images: int = 15,
    code_interpreter: Optional[Any] = None,
    image_paths: Optional[List[str]] = None,
) -> EpisodeResult:
    """跑一道题。

    send_fn(messages, tools) -> AssistantResponse —— 真跑传 OpenAICompatibleClient.send，
    测试传 mock。code_interpreter 可外部注入(便于 mock)；否则按 image_paths 起一个内核。
    """
    own_ci = False
    if code_interpreter is None and "code_interpreter" in enabled_tools:
        if CodeInterpreter is None:
            raise RuntimeError("CodeInterpreter 不可用，请在 budgetsee 环境运行或外部注入。")
        code_interpreter = CodeInterpreter(image_paths=image_paths or [])
        own_ci = True

    ctx = RolloutCtx(code_interpreter=code_interpreter)
    tools_schema = registry.enabled_schemas(enabled_tools)
    messages = build_initial_messages(system_prompt, question, input_images)

    wrapped_send = send_fn  # 历史压缩(budgetsee)已移除；trace_skill 始终携带完整历史(policy=full)

    result = EpisodeResult(messages=messages)

    try:
        for turn in range(max_turns):
            if result.num_tool_images >= max_images:
                result.degenerate["max_images"] = True
                break

            resp: AssistantResponse = wrapped_send(messages, tools_schema)
            result.usage.add(resp.usage)
            result.num_turns = turn + 1

            if resp.has_tool_calls:
                messages.append(assistant_message_with_tool_calls(resp))
                for tc in resp.tool_calls:
                    result.num_tool_calls += 1
                    try:
                        tool: BaseTool = registry.get(tc.name)
                        tr = tool.run(tc.parsed_args(), ctx)
                    except KeyError as e:
                        tr = ToolResult(text=f"Error: {e}", error=True)
                    messages.append(tool_result_message(tc.id, tr))
                    img_msg = tool_images_message(tr)
                    if img_msg is not None:
                        messages.append(img_msg)
                        result.num_tool_images += len(tr.images)
                continue

            # 无 tool_calls → 最终答案
            messages.append({"role": "assistant", "content": resp.content or ""})
            result.final_answer = extract_answer(resp.content)
            break
        else:
            result.degenerate["turn_exhausted"] = True
    finally:
        if own_ci and code_interpreter is not None:
            code_interpreter.shutdown()

    if result.final_answer is None:
        result.degenerate["no_answer"] = True
    return result
