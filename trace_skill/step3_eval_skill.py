#!/usr/bin/env python3
"""step3: skill 注入再评测 —— 同批题 baseline(无 skill) vs skill 配对对比 accuracy + token/成本。

任务闭环的最后一环
==================
  导师要的是"据 trace 蒸 skill 让这 5 题做对 + 省 token"。step1 采 trace、step2 诊断、(人/LLM)
  据诊断写出 skill(SKILL.md)；本脚本验证 skill 到底有没有用：

    对同一批样本(同 seed/同温度)跑两个条件——
      baseline : 原 system prompt(无 skill)
      skill    : 原 system prompt + 注入的 skill 文本
    配对比较 accuracy / 轮数 / token / 成本，证明"做对率↑ 且 token↓"。

  skill 注入点 = system prompt 追加(run_episode 的 system_prompt 参数)，最小侵入、可控变量干净。

用法
----
  python trace_skill/step3_eval_skill.py --skill trace_skill/skills/loop_breaker.md \
         --n-per-subset 10
  # 只在 step1 标记的失败题上验证(更聚焦)：
  python trace_skill/step3_eval_skill.py --skill ... --doc-ids-from trace_skill/assets/output/summary_*.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# 复用 step1 的 TopPClient / filter_subsets / run_one(按文件路径载入，不复制代码)
_spec = importlib.util.spec_from_file_location("step1_collect", os.path.join(_HERE, "step1_collect_traces.py"))
step1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step1)

from runner.agent_loop import DEFAULT_TIR_SYSTEM_PROMPT
from runner.gate import Prices
from runner.tools import CodeInterpreterTool, ToolRegistry
from runner.tir_data import load_tir_samples
from verifier_lenient import LenientTIRVerifier


def load_skill(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def doc_ids_from_summary(pattern: str) -> Optional[List[str]]:
    """从 step1 summary 的 fail_cases 取失败题 doc_id(支持 glob，取最新)。"""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        summ = json.load(f)
    return [c["doc_id"] for c in summ.get("fail_cases", [])]


def load_baseline_from_traces(trace_dir: str, verifier) -> List[Dict[str, Any]]:
    """从已采 trace 目录重建 baseline 记录(用宽容验证器重判分，与 skill 侧同口径)。"""
    recs = []
    for p in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        if "usage" not in d:   # 跳过 error 记录
            continue
        sub = d.get("subset")
        correct = verifier.verify(d.get("final_answer"), d.get("solution"), sub)
        recs.append({
            "doc_id": d["doc_id"], "subset": sub, "correct": correct,
            "outcome": d.get("outcome"), "num_turns": d.get("num_turns", 0),
            "num_tool_images": d.get("num_tool_images", 0),
            "final_answer": d.get("final_answer"), "solution": d.get("solution"),
            "usage": d.get("usage", {}), "cost_yuan": d.get("cost_yuan", 0.0),
        })
    return recs


def run_condition(samples, client, registry, verifier, prices, system_prompt,
                  max_turns, max_images, concurrency=6) -> List[Dict[str, Any]]:
    """跑一个条件(baseline 或 skill)，复用 step1.run_one 但换 system_prompt。并发跑。"""
    import concurrent.futures as cf

    def work(s):
        # 直接调底层 run_episode 以便换 prompt(step1.run_one 写死默认 prompt)
        return _run_one_with_prompt(s, client, registry, verifier, prices,
                                    system_prompt, max_turns, max_images)

    recs, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(work, s): s for s in samples}
        for fut in cf.as_completed(futs):
            r = fut.result(); recs.append(r); done += 1
            print(f"  [{done}/{len(samples)}] {r['doc_id']:22s} {str(r['outcome']):14s} "
                  f"turns={r['num_turns']} ${r['cost_yuan']:.4f} ans={str(r['final_answer'])[:18]!r} "
                  f"gt={str(r['solution'])[:18]!r} {'✓' if r['correct'] else '✗'}", flush=True)
    return recs


def _run_one_with_prompt(sample, client, registry, verifier, prices,
                         system_prompt, max_turns, max_images) -> Dict[str, Any]:
    """run_one 的变体：可指定 system_prompt(用于注入 skill)。轻量记录(不存完整 messages)。"""
    from runner.agent_loop import run_episode
    from runner.code_interpreter import CodeInterpreter
    from runner.gate import episode_cost_yuan

    ci = CodeInterpreter(image_paths=sample.image_paths)
    try:
        res = run_episode(
            question=sample.question,
            input_images=sample.load_images(),
            send_fn=client.send,
            registry=registry,
            enabled_tools=["code_interpreter"],
            policy="full", keep_original=True,
            system_prompt=system_prompt,
            max_turns=max_turns, max_images=max_images,
            code_interpreter=ci,
        )
    finally:
        ci.shutdown()
    correct = verifier.verify(res.final_answer, sample.solution,
                              sample.raw.get("data_source", sample.category))
    return {
        "doc_id": sample.doc_id,
        "subset": sample.raw.get("data_source", sample.category),
        "correct": correct,
        "outcome": step1.classify(res, correct),
        "num_turns": res.num_turns,
        "num_tool_images": res.num_tool_images,
        "final_answer": res.final_answer,
        "solution": sample.solution,
        "usage": res.usage.as_dict(),
        "cost_yuan": episode_cost_yuan(res.usage, prices),
    }


def agg(recs: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(recs) or 1
    return {
        "n": len(recs),
        "accuracy": sum(r["correct"] for r in recs) / n,
        "avg_turns": sum(r["num_turns"] for r in recs) / n,
        "avg_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in recs) / n,
        "avg_completion_tokens": sum(r["usage"]["completion_tokens"] for r in recs) / n,
        "avg_cost_yuan": sum(r["cost_yuan"] for r in recs) / n,
    }


def main():
    ap = argparse.ArgumentParser(description="skill 注入再评测：baseline vs skill 配对对比")
    ap.add_argument("--skill", required=True, help="SKILL.md 路径")
    ap.add_argument("--subsets", nargs="+", default=step1.DEFAULT_SUBSETS)
    ap.add_argument("--n-per-subset", type=int, default=10)
    ap.add_argument("--doc-ids-from", default=None, help="只跑这些 doc_id(从 step1 summary fail_cases 取，支持 glob)")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tir-json", default=os.environ.get("TIR_JSON", step1.DEFAULT_TIR_JSON))
    ap.add_argument("--tir-img", default=os.environ.get("TIR_IMAGE_FOLDER", step1.DEFAULT_TIR_IMG))
    ap.add_argument("--in-price", type=float, default=0.5e-6, help="CNY/token 输入(百炼思考 ¥0.5/M)")
    ap.add_argument("--out-price", type=float, default=5.0e-6, help="CNY/token 输出(百炼思考 ¥5/M)")
    ap.add_argument("--output-dir", default=os.path.join(_HERE, "assets/output"))
    ap.add_argument("--reuse-baseline", default=None,
                    help="复用已采 trace 目录当 baseline(只跑 +skill 侧，省钱)；样本取该目录的 doc_id")
    args = ap.parse_args()

    step1.load_dotenv()  # 复用 step1 的 .env 加载
    base = os.environ.get("QWEN_BASE_URL"); key = os.environ.get("QWEN_API_KEY")
    model = os.environ.get("QWEN_MODEL")
    if not (base and key and model):
        sys.exit("缺 VLM 配置：请 export QWEN_BASE_URL / QWEN_API_KEY / QWEN_MODEL。")

    skill_text = load_skill(args.skill)
    skill_prompt = DEFAULT_TIR_SYSTEM_PROMPT + "\n\n# Task-specific skill (follow this)\n" + skill_text

    client = step1.TopPClient(base_url=base, api_key=key, model=model,
                              temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, seed=args.seed)
    prices = Prices(in_miss=args.in_price, in_hit=args.in_price, out=args.out_price)
    registry = ToolRegistry(); registry.register(CodeInterpreterTool())
    verifier = LenientTIRVerifier()

    all_samples = load_tir_samples(args.tir_json, args.tir_img)

    if args.reuse_baseline:
        # 复用已采 trace 当 baseline：样本=该目录的 doc_id，baseline 用宽容验证器重判分
        base_recs = load_baseline_from_traces(args.reuse_baseline, verifier)
        keep_ids = [r["doc_id"] for r in base_recs]
        by_id = {s.doc_id: s for s in all_samples}
        samples = [by_id[d] for d in keep_ids if d in by_id]
        print(f"[eval] skill={args.skill}  复用 baseline {args.reuse_baseline}")
        print(f"[eval] baseline {len(base_recs)} 题(已采，不重跑) → 只跑 +skill；model={model}")
    else:
        samples = step1.filter_subsets(all_samples, args.subsets, args.n_per_subset)
        only = doc_ids_from_summary(args.doc_ids_from) if args.doc_ids_from else None
        if only is not None:
            keep = set(only)
            samples = [s for s in samples if s.doc_id in keep]
        print(f"[eval] skill={args.skill}  样本 {len(samples)} 题  model={model}")
        print("\n=== 条件 A: baseline(无 skill) ===")
        base_recs = run_condition(samples, client, registry, verifier, prices,
                                  DEFAULT_TIR_SYSTEM_PROMPT, args.max_turns, args.max_images)

    print("=== 条件 B: + skill ===")
    skill_recs = run_condition(samples, client, registry, verifier, prices,
                               skill_prompt, args.max_turns, args.max_images)

    # 配对对比(总体 + 子集)
    result = {"skill": args.skill, "model": model, "config": vars(args),
              "overall": {"baseline": agg(base_recs), "skill": agg(skill_recs)},
              "by_subset": {}, "per_sample": []}
    bsub: Dict[str, List] = defaultdict(list); ssub: Dict[str, List] = defaultdict(list)
    for r in base_recs: bsub[r["subset"]].append(r)
    for r in skill_recs: ssub[r["subset"]].append(r)
    for sub in bsub:
        result["by_subset"][sub] = {"baseline": agg(bsub[sub]), "skill": agg(ssub[sub])}
    bmap = {r["doc_id"]: r for r in base_recs}
    for r in skill_recs:
        b = bmap.get(r["doc_id"], {})
        result["per_sample"].append({
            "doc_id": r["doc_id"], "subset": r["subset"],
            "baseline_correct": b.get("correct"), "skill_correct": r["correct"],
            "baseline_turns": b.get("num_turns"), "skill_turns": r["num_turns"],
            "baseline_prompt_tok": b.get("usage", {}).get("prompt_tokens"),
            "skill_prompt_tok": r["usage"]["prompt_tokens"],
            "baseline_cost": b.get("cost_yuan"), "skill_cost": r["cost_yuan"],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"skill_eval_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    o = result["overall"]
    def line(tag, a):
        print(f"  {tag:9s} acc={a['accuracy']:.3f} avg轮={a['avg_turns']:.1f} "
              f"in_tok={a['avg_prompt_tokens']:.0f} out_tok={a['avg_completion_tokens']:.0f} "
              f"$/题={a['avg_cost_yuan']:.5f}")
    print("\n" + "=" * 72)
    print("总体对比(baseline vs +skill)")
    line("baseline", o["baseline"]); line("+skill", o["skill"])
    dacc = o["skill"]["accuracy"] - o["baseline"]["accuracy"]
    dcost = o["skill"]["avg_cost_yuan"] - o["baseline"]["avg_cost_yuan"]
    dtok = o["skill"]["avg_prompt_tokens"] - o["baseline"]["avg_prompt_tokens"]
    print(f"  Δaccuracy={dacc:+.3f}  Δin_tok={dtok:+.0f}  Δ$/题={dcost:+.5f}")
    print(f"  目标达成(做对↑ 且 省token): {'是' if dacc > 0 and dcost < 0 else '否/部分'}")
    print(f"\n结果: {out}")


if __name__ == "__main__":
    main()
