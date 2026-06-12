"""TIR 风格的 code_interpreter 沙箱 —— 基于持久 Jupyter 内核，完全独立实现。

为什么是这个形态
================
TIR-Bench 上 agent 唯一的工具就是 code_interpreter（TIR 的 agent 唯一工具）。
视觉历史的累积来自：模型写 Python(PIL/NumPy/OpenCV)处理原图 → 产出图被捕获为
`tool_image_N` → 下一轮作为图片回灌。因此沙箱必须满足：
  1. 跨轮持久：同一个内核，上一轮的变量(含 tool_image_N)下一轮还能用 ⇒ 用 jupyter 内核而非每轮 exec。
  2. 原图预载：`original_image / original_image_N` 作为 PIL 对象直接可用。
  3. 三种产图方式都捕获：返回对象 / `plt.show()`(inline) / `image.save(...)`。

整体执行流程（run 一次代码）
---------------------------
  execute(code) → 轮询 iopub 消息直到 idle：
    stream            → 收 stdout/stderr 文本
    execute_result/display_data:
        image/png     → base64 解码成 PIL，登记为新产出图
        text/plain    → 收文本
    error             → 收 traceback
  再扫 work_dir 下新出现的图片文件（捕获 image.save 产物）
  对每张新图：命名 tool_image_{全局递增}，存盘，并注入内核变量供下一轮引用。

安全
----
这是在执行模型生成的代码。务必跑在隔离环境（容器/受限用户）。本类不做沙箱级隔离，
只做超时与内核重启；隔离边界由部署环境保证。
"""

from __future__ import annotations

import base64
import glob
import io
import os
import queue
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

try:
    from jupyter_client import KernelManager
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 jupyter_client，请在 budgetsee 环境安装：pip install jupyter_client ipykernel") from e


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


@dataclass
class CodeResult:
    """一次代码执行的结果。"""
    text: str = ""                                   # stdout / 文本输出 / 结果 repr
    images: List[Tuple[str, Image.Image]] = field(default_factory=list)  # [(tool_image_N, PIL), ...]
    error: Optional[str] = None                      # traceback 文本，None 表示无错
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


class CodeInterpreter:
    """持久 Jupyter 内核沙箱。一个实例对应一道题(一个 rollout)的整段多轮代码执行。

    用法：
        ci = CodeInterpreter(image_paths=["/path/puzzle.png"])
        r1 = ci.run("crop = original_image.crop((0,0,100,100)); crop")   # 返回对象→产图
        r2 = ci.run("tool_image_1.size")                                  # 上一轮产图可复用
        ci.shutdown()
    """

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        timeout: float = 60.0,
        kernel_name: str = "python3",
    ):
        self.timeout = timeout
        self._owns_work_dir = work_dir is None
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="budgetsee_ci_")
        os.makedirs(self.work_dir, exist_ok=True)
        self._tool_image_counter = 0

        self._km = KernelManager(kernel_name=kernel_name)
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=60)

        self._preload(image_paths or [])

    # ---------------- setup ----------------

    def _preload(self, image_paths: List[str]) -> None:
        """把原图拷进 work_dir，在内核里预载成 original_image / original_image_N。"""
        lines = [
            "import os",
            f"os.chdir({self.work_dir!r})",
            "from PIL import Image",
            "import numpy as np",
            "get_ipython().run_line_magic('matplotlib', 'inline')",
        ]
        for i, src in enumerate(image_paths):
            dst = os.path.join(self.work_dir, f"original_{i}.png")
            Image.open(src).convert("RGB").save(dst)
            var = "original_image" if i == 0 else None
            lines.append(f"original_image_{i + 1} = Image.open('original_{i}.png')")
            if var:
                lines.append(f"original_image = Image.open('original_{i}.png')")
        setup_code = "\n".join(lines)
        res = self.run(setup_code, _capture_images=False)
        if not res.ok:
            raise RuntimeError(f"沙箱预载失败：{res.error or 'timeout'}")

    # ---------------- core ----------------

    def run(self, code: str, _capture_images: bool = True) -> CodeResult:
        """在持久内核里执行一段代码，捕获文本/产出图/错误。"""
        existing_files = set(glob.glob(os.path.join(self.work_dir, "*")))

        result = CodeResult()
        msg_id = self._kc.execute(code)
        inline_images: List[Image.Image] = []

        import time
        deadline = time.time() + self.timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                result.timed_out = True
                self._km.interrupt_kernel()
                # 排空被中断 cell 的残留消息(KeyboardInterrupt + idle)，否则会污染下一次 run
                self._drain_until_idle(msg_id, max_wait=10.0)
                break
            try:
                msg = self._kc.get_iopub_msg(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mtype = msg["header"]["msg_type"]
            content = msg["content"]

            if mtype == "status":
                if content.get("execution_state") == "idle":
                    break
            elif mtype == "stream":
                result.text += content.get("text", "")
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "image/png" in data:
                    inline_images.append(self._decode_png(data["image/png"]))
                elif "text/plain" in data:
                    result.text += data["text/plain"] + "\n"
            elif mtype == "error":
                result.error = self._strip_ansi("\n".join(content.get("traceback", [])))

        if not _capture_images:
            return result

        # 1) inline 产出图（返回对象 / plt.show / display）
        for img in inline_images:
            self._register_image(img, result)
        # 2) 扫描新落盘的图片文件（image.save 产物），按文件名排序保证确定性
        new_files = sorted(set(glob.glob(os.path.join(self.work_dir, "*"))) - existing_files)
        for path in new_files:
            if path.lower().endswith(_IMAGE_EXTS) and not os.path.basename(path).startswith("tool_image_"):
                try:
                    self._register_image(Image.open(path).convert("RGB"), result, source_path=path)
                except Exception:
                    pass
        return result

    def _register_image(self, img: Image.Image, result: CodeResult, source_path: Optional[str] = None) -> None:
        """给产出图命名 tool_image_N、存盘，并注入内核变量供下一轮引用。"""
        self._tool_image_counter += 1
        name = f"tool_image_{self._tool_image_counter}"
        save_path = os.path.join(self.work_dir, f"{name}.png")
        img.save(save_path)
        result.images.append((name, img))
        # 注入内核：下一轮代码可直接用 tool_image_N（兑现 prompt 的承诺）
        self._kc.execute(f"{name} = Image.open({save_path!r})", silent=True)

    def _drain_until_idle(self, msg_id: str, max_wait: float = 10.0) -> None:
        """读掉 iopub 上 msg_id 的残留消息，直到它的 idle 状态出现(或超过 max_wait)。

        超时打断后调用：被中断的 cell 仍会异步发出 error/idle，不排空会被下一次 run 误读。
        """
        import time
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                msg = self._kc.get_iopub_msg(timeout=min(deadline - time.time(), 1.0))
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if msg["header"]["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
                return

    # ---------------- helpers ----------------

    @staticmethod
    def _decode_png(b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    @staticmethod
    def _strip_ansi(text: str) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def shutdown(self) -> None:
        try:
            self._kc.stop_channels()
            self._km.shutdown_kernel(now=True)
        finally:
            if self._owns_work_dir and os.path.isdir(self.work_dir):
                shutil.rmtree(self.work_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
