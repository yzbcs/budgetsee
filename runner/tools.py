"""可插拔工具抽象 —— 加新 benchmark 的工具 = 实现一个 BaseTool 子类 + 注册。

设计要点
========
agent 循环与 BudgetSee 视觉历史层都是**工具无关**的：它们只认 OpenAI 消息格式里的
image_url block，不关心图是哪个工具产的。所以加工具只动工具层，循环/visual_history/
instrument 一行不改。

- BaseTool   : 工具接口。schema() 给 OpenAI function-calling 声明；run() 执行并返回
               文本 + 产出图。
- ToolResult : 一次工具调用的结果(text + images)。
- ToolRegistry: 名字→工具实例。每 benchmark 用一份 enabled 列表选子集(对应原始实现的
               ENABLED_TOOLS)。
- RolloutCtx : 单个 rollout 的运行时资源(如持久 CodeInterpreter 内核)，传给 run()。

TIR 只注册 code_interpreter；VisualToolBench/MMSearch-Plus/AgentVista 后续再加
web_search/visit/image_search/crop 等，方式相同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class ToolResult:
    text: str = ""
    images: List[Tuple[str, Image.Image]] = field(default_factory=list)  # [(tool_image_N, PIL), ...]
    error: bool = False


@dataclass
class RolloutCtx:
    """单个 rollout 的运行时资源容器。不同工具按需取用各自字段。"""
    code_interpreter: Any = None   # runner.code_interpreter.CodeInterpreter 实例
    save_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """工具接口。子类需实现 name/description/parameters_schema/run。"""

    name: str = ""
    description: str = ""

    def parameters_schema(self) -> Dict[str, Any]:
        """返回 JSON Schema 的 parameters 部分(OpenAI function-calling)。"""
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        """完整 OpenAI function 声明。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    def run(self, args: Dict[str, Any], ctx: RolloutCtx) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}，已注册: {list(self._tools)}")
        return self._tools[name]

    def enabled_schemas(self, enabled: List[str]) -> List[Dict[str, Any]]:
        """按 benchmark 的 enabled 列表返回工具声明，喂给模型。"""
        return [self.get(n).schema() for n in enabled]

    def names(self) -> List[str]:
        return list(self._tools)


# ============================================================================
# code_interpreter 工具(TIR 唯一工具)
# ============================================================================

class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = (
        "Execute Python code in a persistent sandbox to process or analyze images. "
        "Preloaded PIL variables: original_image, original_image_1, ... and tool_image_N "
        "(images you produced in previous calls). To produce an image, return the object, "
        "use plt.show(), or save it (e.g., image.save('out.png')). Available: PIL, NumPy, "
        "OpenCV(cv2), scikit-image, matplotlib."
    )

    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        }

    def run(self, args: Dict[str, Any], ctx: RolloutCtx) -> ToolResult:
        if ctx.code_interpreter is None:
            return ToolResult(text="Error: no code_interpreter in context.", error=True)
        code = args.get("code", "")
        res = ctx.code_interpreter.run(code)
        if res.timed_out:
            return ToolResult(text="Error: code execution timed out.", error=True)
        if res.error:
            return ToolResult(text=res.error, images=res.images, error=True)
        return ToolResult(text=res.text, images=res.images)
