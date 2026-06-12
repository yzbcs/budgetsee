#!/usr/bin/env python3
"""step2: 读 step1 的 trace，做死循环诊断 → 输出每子集的"失败/打转模式"报告(蒸 skill 的原料)。

为什么需要这步
==============
  导师要"据 trace 蒸 skill 让题做对 + 省 token"。蒸 skill 前必须先看清：模型在每个子集上
  到底怎么死循环 / 答不出的——是反复跑同一段代码？是 reasoning 越想越乱？是面积/坐标系统性
  估错？本脚本把 step1 的逐题 trace 聚成可读的诊断，定位每个子集的"主病灶"，作为 skill 草案的依据。

输出
----
  - 控制台：每子集 outcome 分布、死循环题清单、典型重复代码片段、reasoning 长度分布。
  - assets/output/diagnosis_<stamp>.md：人读诊断报告(交接 + 手工蒸 skill 用)。

用法
----
  python trace_skill/step2_analyze_traces.py --trace-dir trace_skill/assets/output/traces_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List


def load_traces(trace_dir: str) -> List[Dict[str, Any]]:
    recs = []
    for p in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        with open(p, encoding="utf-8") as f:
            recs.append(json.load(f))
    return recs


def code_of(tc: Dict[str, Any]) -> str:
    """从 tool_call.arguments(JSON 串)里抽出 code 字段(失败则原样返回)。"""
    try:
        return json.loads(tc.get("arguments", "{}")).get("code", "")
    except Exception:
        return tc.get("arguments", "")


def diagnose_one(rec: Dict[str, Any]) -> Dict[str, Any]:
    """单题诊断：抽出工具代码序列、重复代码、reasoning 长度、是否死循环。"""
    turns = rec.get("turn_log", [])
    codes = [code_of(tc) for t in turns for tc in t.get("tool_calls", [])]
    code_counts = Counter(codes)
    repeated_snippets = {c: n for c, n in code_counts.items() if n > 1 and c}
    reasoning_lens = [len(t.get("reasoning_content") or "") for t in turns]
    return {
        "doc_id": rec.get("doc_id"),
        "subset": rec.get("subset"),
        "outcome": rec.get("outcome"),
        "num_turns": rec.get("num_turns"),
        "num_tool_calls": rec.get("num_tool_calls"),
        "repeated_tool_calls": rec.get("repeated_tool_calls"),
        "n_distinct_codes": len(code_counts),
        "repeated_snippets": repeated_snippets,
        "reasoning_chars_total": sum(reasoning_lens),
        "reasoning_chars_per_turn": reasoning_lens,
        "final_answer": rec.get("final_answer"),
        "solution": rec.get("solution"),
        "cost_yuan": rec.get("cost_yuan"),
    }


def main():
    ap = argparse.ArgumentParser(description="死循环诊断：读 trace → 每子集失败/打转模式")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/output"))
    ap.add_argument("--top-snippets", type=int, default=3, help="每子集展示几条最典型的重复代码")
    args = ap.parse_args()

    recs = load_traces(args.trace_dir)
    if not recs:
        raise SystemExit(f"trace 目录为空: {args.trace_dir}")
    diags = [diagnose_one(r) for r in recs]

    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in diags:
        by_sub[d["subset"]].append(d)

    lines: List[str] = [f"# TIR 死循环诊断报告\n", f"> trace 来源: `{args.trace_dir}`  样本数: {len(diags)}\n"]
    print("=" * 72)
    for sub, ds in by_sub.items():
        n = len(ds)
        oc = Counter(d["outcome"] for d in ds)
        loops = [d for d in ds if d["outcome"] in ("turn_exhausted", "max_images", "no_answer")]
        wrong = [d for d in ds if d["outcome"] == "wrong_answered"]
        avg_turns = sum(d["num_turns"] or 0 for d in ds) / n
        avg_rep = sum(d["repeated_tool_calls"] or 0 for d in ds) / n
        avg_rlen = sum(d["reasoning_chars_total"] for d in ds) / n

        # 全子集最常见的重复代码片段
        snip = Counter()
        for d in ds:
            for c, k in d["repeated_snippets"].items():
                snip[c[:160]] += k

        print(f"\n## {sub}  n={n}")
        print(f"   outcomes: {dict(oc)}")
        print(f"   avg_turns={avg_turns:.1f} avg_repeated_calls={avg_rep:.1f} avg_reasoning_chars={avg_rlen:.0f}")
        print(f"   死循环/答不出 {len(loops)} 题, 答错 {len(wrong)} 题")

        lines.append(f"\n## {sub} (n={n})\n")
        lines.append(f"- outcome 分布: `{dict(oc)}`\n")
        lines.append(f"- 平均轮数 {avg_turns:.1f} · 平均重复工具调用 {avg_rep:.1f} · 平均 reasoning 字符 {avg_rlen:.0f}\n")
        lines.append(f"- 死循环/答不出 **{len(loops)}** 题, 答错 **{len(wrong)}** 题\n")
        if loops:
            lines.append(f"- 死循环题: {', '.join(d['doc_id'] for d in loops)}\n")
        if snip:
            lines.append(f"- 典型重复代码片段(片段→总重复次数):\n")
            for c, k in snip.most_common(args.top_snippets):
                lines.append(f"  - ×{k}: `{c.strip()[:120]}`\n")

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"diagnosis_{stamp}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\n诊断报告: {out}")


if __name__ == "__main__":
    main()
