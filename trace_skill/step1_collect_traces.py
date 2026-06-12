#!/usr/bin/env python3
"""step1: 用 qwen3-vl-8b-thinking 跑 TIR-Bench 易失败子集，逐题录 trace + 花费。

目的
====
  采集 VLM 在 thinking-with-image 任务上的真实失败 trace（每轮 reasoning / 工具代码 / token /
  花费），作为后续诊断(step2)与蒸 skill(step3)的原料。

整体流程(一题一个 rollout)
==========================
  1. 从本地 TIR 数据(val_shuffled.json + 图)按 data_source 过滤到目标子集，每子集取前 N 题。
  2. 对每题：起持久 code_interpreter 沙箱(原图预载) → run_episode(policy=full, max_turns=10)。
     - policy=full：携带完整视觉历史，观察模型"自然状态"下的迭代/打转行为。
     - 用 traced_send 包住发送函数，逐轮录：prompt/completion/cached token、assistant content、
       thinking 的 reasoning_content、tool_calls(模型写的代码)。
  3. 分类每题结局：correct / wrong_answered / turn_exhausted / max_images / no_answer。
  4. 落盘：每题一个 trace json(图片 data URI 脱水成占位符) + 子集级 summary + 控制台摘要。

依赖：复用根目录通用 agent 基建 `runner/`（run_episode/code_interpreter/vlm_client/tir_data），
本目录的 `verifier_lenient`(宽容验证器)。

用法
----
  # 模型配置写入 trace_skill/.env（见 .env.example），或用环境变量：
  #   QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  #   QWEN_MODEL=qwen3-vl-8b-thinking ; QWEN_API_KEY=...
  python trace_skill/step1_collect_traces.py \\
    --subsets tir_bench_jigsaw tir_bench_instrument tir_bench_refcoco tir_bench_maze tir_bench_spot_difference \\
    --n-per-subset 5 --max-turns 10 --concurrency 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

# 把项目根加进 sys.path，复用根目录的 runner / budgetsee
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # trace_skill 本目录

from runner.agent_loop import EpisodeResult, run_episode
from runner.code_interpreter import CodeInterpreter
from runner.gate import Prices, episode_cost_yuan
from runner.tools import CodeInterpreterTool, ToolRegistry
from runner.tir_data import TIRSample, load_tir_samples
from runner.vlm_client import AssistantResponse, OpenAICompatibleClient
from verifier_lenient import LenientTIRVerifier  # 任务感知宽容验证器(本任务专用)

def load_dotenv(path: Optional[str] = None) -> None:
    """若 trace_skill/.env 存在，把里面的 KEY=VALUE 灌进 os.environ(已存在的不覆盖)。

    把 API key 写进 gitignore 的 .env，无需每次 export、也不进 git。
    """
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# 默认 5 个易失败子集(TIR-Bench 论文 arXiv:2511.01833 最低准确率/最深迭代)
DEFAULT_SUBSETS = [
    "tir_bench_jigsaw",
    "tir_bench_instrument",
    "tir_bench_refcoco",
    "tir_bench_maze",
    "tir_bench_spot_difference",
]

# 本地数据默认位置(download_tir.py 下载到此)
DEFAULT_TIR_JSON = os.path.join(_ROOT, "assets/input/tir/val_shuffled.json")
DEFAULT_TIR_IMG = os.path.join(_ROOT, "assets/input/tir")  # 与 json 内 "TIR-Bench/data/..." 前缀拼接


class TopPClient(OpenAICompatibleClient):
    """OpenRouter / 阿里百炼(DashScope) / 自建 vLLM 通用对接：加 top_p / seed + 思维链回传 + 流式聚合。

    平台差异(按 base_url 自动适配)：
    - OpenRouter：思维链在 message.reasoning，用 include_reasoning=True 触发；非流式即可。
    - 百炼 DashScope：思维链在 message.reasoning_content；thinking 模型多强制 stream=True，故自动走流式。
    - 自建 vLLM(`--reasoning-parser deepseek_r1`)：思维链同样在 message.reasoning_content（非流式响应里也有，
      usage 也照常返回），默认非流式即可，需要时 --stream 显式开。tool_calls：若 vLLM 未带
      `--enable-auto-tool-choice --tool-call-parser hermes`，模型会把调用当文本吐 → 已由 _recover_text_tool_call 兜底。
    - 流式时用 stream_options.include_usage 拿 token 账；temp=0.6/top_p=0.95、seed 可复现。
    """

    def __init__(self, *args, top_p: float = 0.95, seed: Optional[int] = None,
                 stream: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_p = top_p
        self.seed = seed
        self._is_openrouter = "openrouter" in self.base_url
        self._is_dashscope = "dashscope" in self.base_url or "aliyuncs" in self.base_url
        # stream=None 自动判定：仅百炼 thinking 强制流式；vLLM/自建默认非流式(也完整支持，
        # reasoning_content 与 usage 在非流式响应里都有)，需要时用 --stream 显式开。
        self.stream = stream if stream is not None else (self._is_dashscope and "thinking" in self.model.lower())

    def _payload(self, messages, tools) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self._is_openrouter:
            payload["include_reasoning"] = True   # 百炼会拒未知参数，故仅 OpenRouter 发
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def send(self, messages, tools=None):
        import requests
        from runner.vlm_client import parse_response
        payload = self._payload(messages, tools)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        if not self.stream:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            resp = parse_response(r.json())
        else:
            # 流式：聚合 delta 成一条完整响应(content / reasoning_content / tool_calls / usage)
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout, stream=True)
            r.raise_for_status()
            resp = self._aggregate_stream(r)
        return _recover_text_tool_call(resp)

    def _aggregate_stream(self, resp):
        """把 SSE 流聚合成 OpenAI 非流式同形 dict，再交给 parse_response 解析。"""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}
        finish_reason = None
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {}) or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                if delta.get("reasoning"):
                    reasoning_parts.append(delta["reasoning"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": "", "type": "function",
                                                        "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
        message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        from runner.vlm_client import parse_response
        return parse_response({"choices": [{"message": message, "finish_reason": finish_reason}],
                               "usage": usage})


def _extract_first_json_object(text: str) -> Optional[str]:
    """从 text 里抠出第一个完整的 {...}(括号配平，跳过字符串内的括号/转义)。失败返回 None。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _recover_text_tool_call(resp: AssistantResponse) -> AssistantResponse:
    """模型把工具调用当文本吐在 content 里(非原生 tool_calls)时，解析出来合成 tool_call。

    Qwen3-VL-Thinking 偶发不走原生 function-calling，而是把调用写成 content 文本——
    可能是 <tool_call>{...}</tool_call>、```json {...}```、或裸 {"name":...,"arguments":...}。
    不修的话 agent loop 会把它当最终答案、1 轮就停(假阳性)。这里兜底（与常见 vs-skills 做法一致）。
    """
    if resp.tool_calls or not resp.content:
        return resp
    text = resp.content.strip()
    cand = None
    mt = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if mt:
        cand = mt.group(1)
    else:
        mf = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if mf:
            cand = mf.group(1)
        elif text.startswith("{") and '"name"' in text and ("arguments" in text or "parameters" in text):
            cand = _extract_first_json_object(text)
    if not cand:
        return resp
    try:
        obj = json.loads(cand)
    except json.JSONDecodeError:
        return resp
    fn = obj.get("function", obj)
    name = fn.get("name")
    args = fn.get("arguments", fn.get("parameters", {}))
    if not name:
        return resp
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    from runner.vlm_client import ToolCall
    resp.tool_calls = [ToolCall(id="call_text_0", name=name, arguments=args)]
    # 把工具调用片段从 content 剥掉(避免污染答案抽取)
    stripped = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    resp.content = "" if stripped == cand else stripped.replace(cand, "").strip()
    return resp


def filter_subsets(samples: List[TIRSample], subsets: List[str], n_per: int) -> List[TIRSample]:
    """按 data_source(存在 raw 里)过滤到目标子集，每子集取前 n_per(保持 json 顺序，可复现)。"""
    by_sub: Dict[str, List[TIRSample]] = defaultdict(list)
    for s in samples:
        ds = s.raw.get("data_source", s.category)
        if ds in subsets:
            by_sub[ds].append(s)
    picked: List[TIRSample] = []
    for ds in subsets:
        picked.extend(by_sub.get(ds, [])[:n_per])
    return picked


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 image_url 的 base64 data URI 换成占位符(<image: N chars>)，避免 trace 文件爆炸。"""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    new_content.append({"type": "image_url",
                                        "image_url": {"url": f"<image:{len(url)} chars>"}})
                else:
                    new_content.append(block)
            out.append({**m, "content": new_content})
        else:
            out.append(m)
    return out


def classify(res: EpisodeResult, correct: bool) -> str:
    """把一题结局分类。turn_exhausted = 用尽 10 轮还没答 = 导师说的'死循环/10轮答不出'。"""
    if correct:
        return "correct"
    if res.degenerate.get("turn_exhausted"):
        return "turn_exhausted"   # 死循环主信号
    if res.degenerate.get("max_images"):
        return "max_images"
    if res.degenerate.get("no_answer"):
        return "no_answer"
    return "wrong_answered"       # 答了但答错


def count_repeated_tool_calls(turn_log: List[Dict[str, Any]]) -> int:
    """统计与之前某轮完全相同的工具调用次数(代码原样重复) —— 死循环的硬指标。"""
    seen = set()
    repeats = 0
    for t in turn_log:
        for tc in t.get("tool_calls", []):
            key = tc.get("arguments", "")
            if key in seen:
                repeats += 1
            else:
                seen.add(key)
    return repeats


def run_one(sample: TIRSample, client, registry, verifier, prices,
            max_turns: int, max_images: int) -> Dict[str, Any]:
    """跑一题，返回完整 trace 记录(含逐轮 token / reasoning / 工具代码 + 结局 + 成本)。"""
    turn_log: List[Dict[str, Any]] = []

    def traced_send(messages, tools=None) -> AssistantResponse:
        resp = client.send(messages, tools)
        msg = (resp.raw.get("choices") or [{}])[0].get("message", {}) if resp.raw else {}
        # OpenRouter 思维链在 message.reasoning；SiliconFlow/Qwen 原生在 reasoning_content
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        turn_log.append({
            "turn": len(turn_log) + 1,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "cached_tokens": resp.usage.cached_tokens,
            "reasoning_content": reasoning,
            "content": resp.content,
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls],
        })
        return resp

    ci = CodeInterpreter(image_paths=sample.image_paths)
    t0 = time.time()
    try:
        res = run_episode(
            question=sample.question,
            input_images=sample.load_images(),
            send_fn=traced_send,
            registry=registry,
            enabled_tools=["code_interpreter"],
            policy="full",                 # 不压历史，看自然状态
            keep_original=True,
            max_turns=max_turns,
            max_images=max_images,
            code_interpreter=ci,
        )
    finally:
        ci.shutdown()
    latency = time.time() - t0

    correct = verifier.verify(res.final_answer, sample.solution,
                              sample.raw.get("data_source", sample.category))
    outcome = classify(res, correct)
    cost = episode_cost_yuan(res.usage, prices)

    return {
        "doc_id": sample.doc_id,
        "subset": sample.raw.get("data_source", sample.category),
        "question": sample.question,
        "solution": sample.solution,
        "final_answer": res.final_answer,
        "correct": correct,
        "outcome": outcome,
        "num_turns": res.num_turns,
        "num_tool_calls": res.num_tool_calls,
        "num_tool_images": res.num_tool_images,
        "repeated_tool_calls": count_repeated_tool_calls(turn_log),
        "degenerate": res.degenerate,
        "latency_s": round(latency, 2),
        "usage": res.usage.as_dict(),
        "cost_yuan": cost,
        "turn_log": turn_log,                        # 逐轮：token / 思维链 / 工具代码
        "messages": sanitize_messages(res.messages), # 完整对话(图片脱水)
    }


def main():
    ap = argparse.ArgumentParser(description="采集 Qwen3-VL-8B-Thinking 在 TIR 5 子集上的 trace + 花费")
    ap.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    ap.add_argument("--n-per-subset", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=8192, help="Thinking 吃 token(reasoning+答案共用)，给大点防截断")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=600.0, help="单次请求超时秒(自建 vLLM 长推理给大点)")
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None,
                    help="显式开/关流式(默认 auto：百炼 thinking 流式、vLLM 非流式更好算账)")
    ap.add_argument("--tir-json", default=os.environ.get("TIR_JSON", DEFAULT_TIR_JSON))
    ap.add_argument("--tir-img", default=os.environ.get("TIR_IMAGE_FOLDER", DEFAULT_TIR_IMG))
    # 百炼 qwen3-vl-8b-thinking 实际计价(CNY/token)：输入(思考) ¥0.5/M、输出(思考) ¥5/M
    ap.add_argument("--in-price", type=float, default=0.5e-6, help="CNY/token 输入(百炼思考 ¥0.5/M)")
    ap.add_argument("--out-price", type=float, default=5.0e-6, help="CNY/token 输出(百炼思考 ¥5/M)")
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/output"))
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 limit 题(调试)")
    ap.add_argument("--concurrency", type=int, default=6, help="并发题数(thinking 慢，靠并发提吞吐)")
    args = ap.parse_args()

    load_dotenv()  # 若 trace_skill/.env 存在则加载 QWEN_* 配置
    base = os.environ.get("QWEN_BASE_URL")
    key = os.environ.get("QWEN_API_KEY")
    model = os.environ.get("QWEN_MODEL")
    if not (base and key and model):
        sys.exit("缺 VLM 配置：请 export QWEN_BASE_URL / QWEN_API_KEY / QWEN_MODEL。")

    client = TopPClient(base_url=base, api_key=key, model=model,
                        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        seed=args.seed, timeout=args.timeout, stream=args.stream)
    prices = Prices(in_miss=args.in_price, in_hit=args.in_price, out=args.out_price)
    registry = ToolRegistry(); registry.register(CodeInterpreterTool())
    verifier = LenientTIRVerifier()

    all_samples = load_tir_samples(args.tir_json, args.tir_img)
    samples = filter_subsets(all_samples, args.subsets, args.n_per_subset)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"[load] {len(all_samples)} 总样本 → 过滤 {args.subsets} 每子集 {args.n_per_subset} → 跑 {len(samples)} 题")
    print(f"[model] {model}  temp={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens} max_turns={args.max_turns}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join(args.output_dir, f"traces_{stamp}")
    os.makedirs(trace_dir, exist_ok=True)

    def work(s: TIRSample) -> Dict[str, Any]:
        try:
            rec = run_one(s, client, registry, verifier, prices, args.max_turns, args.max_images)
        except Exception as e:
            rec = {"doc_id": s.doc_id, "subset": s.raw.get("data_source"), "error": str(e),
                   "outcome": "error", "correct": False}
        with open(os.path.join(trace_dir, f"{rec['doc_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec

    records: List[Dict[str, Any]] = []
    done = 0
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(work, s): s for s in samples}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            err = f" ERROR={rec['error'][:60]}" if rec.get("outcome") == "error" else ""
            print(f"  [{done}/{len(samples)}] {rec['doc_id']:24s} outcome={str(rec.get('outcome')):14s} "
                  f"turns={rec.get('num_turns','-')} rep={rec.get('repeated_tool_calls','-')} "
                  f"imgs={rec.get('num_tool_images','-')} ${rec.get('cost_yuan',0):.5f} "
                  f"ans={str(rec.get('final_answer'))[:20]!r} gt={rec.get('solution')!r}{err}", flush=True)

    # 子集级聚合
    summary: Dict[str, Any] = {"config": vars(args), "model": model, "n": len(records), "by_subset": {}}
    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_sub[r.get("subset", "unknown")].append(r)
    for sub, rs in by_sub.items():
        oc = Counter(r.get("outcome") for r in rs)
        n = len(rs)
        def mean(key):
            xs = [r.get(key, 0) for r in rs if isinstance(r.get(key), (int, float))]
            return sum(xs) / len(xs) if xs else 0.0
        loop_n = oc.get("turn_exhausted", 0) + oc.get("max_images", 0) + oc.get("no_answer", 0)
        summary["by_subset"][sub] = {
            "n": n,
            "accuracy": oc.get("correct", 0) / n if n else 0.0,
            "loop_or_fail_rate": loop_n / n if n else 0.0,   # 死循环/答不出占比
            "outcomes": dict(oc),
            "avg_turns": mean("num_turns"),
            "avg_tool_images": mean("num_tool_images"),
            "avg_repeated_calls": mean("repeated_tool_calls"),
            "avg_cost_yuan": mean("cost_yuan"),
            "avg_prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rs) / n if n else 0,
            "avg_completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rs) / n if n else 0,
        }

    # 死循环/失败题清单(给 step2 蒸 skill 用)
    fail_outcomes = {"turn_exhausted", "max_images", "no_answer", "wrong_answered"}
    summary["fail_cases"] = [
        {"doc_id": r["doc_id"], "subset": r.get("subset"), "outcome": r.get("outcome"),
         "num_turns": r.get("num_turns"), "repeated_tool_calls": r.get("repeated_tool_calls"),
         "cost_yuan": r.get("cost_yuan")}
        for r in records if r.get("outcome") in fail_outcomes
    ]

    sum_path = os.path.join(args.output_dir, f"summary_{stamp}.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("子集级摘要")
    for sub, a in summary["by_subset"].items():
        print(f"  {sub:26s} n={a['n']:2d} acc={a['accuracy']:.2f} "
              f"死循环/答不出={a['loop_or_fail_rate']:.2f} avg轮={a['avg_turns']:.1f} "
              f"avg重复调用={a['avg_repeated_calls']:.1f} $/题={a['avg_cost_yuan']:.5f} "
              f"in_tok={a['avg_prompt_tokens']:.0f}")
    print(f"\ntrace 目录: {trace_dir}")
    print(f"summary:    {sum_path}")
    print(f"失败题数(供 step2 蒸 skill): {len(summary['fail_cases'])}")


if __name__ == "__main__":
    main()
