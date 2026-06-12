"""Feasibility gate 聚合与判定 —— 纯函数，可离线测试。

把每个策略(full/last)的逐题结果聚成指标，算两把尺：
  成本 headroom   = ¥(full) / ¥(last)
  accuracy headroom = acc(full) − acc(last)
两者围成的矩形 = BudgetSee 的发挥空间。成本按 cache-aware ¥ 算(见 metrics_spec)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .agent_loop import EpisodeResult
from .vlm_client import Usage


@dataclass
class Prices:
    """¥ / token。SiliconFlow Qwen3-VL-8B 默认：输入 0.5/M、输出 2/M；
    缓存命中价未知，默认与未命中相同(扁平计价，保守=无缓存收益)。"""
    in_miss: float = 0.5e-6
    in_hit: float = 0.5e-6
    out: float = 2.0e-6


def episode_cost_yuan(usage: Usage, prices: Prices) -> float:
    miss = max(usage.prompt_tokens - usage.cached_tokens, 0)
    return miss * prices.in_miss + usage.cached_tokens * prices.in_hit + usage.completion_tokens * prices.out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(records: List[Tuple[EpisodeResult, bool]], prices: Prices) -> Dict[str, Any]:
    """records = [(EpisodeResult, correct), ...] → 一个策略的聚合指标。

    accuracy 只在**非退化**样本上算(退化样本既无有效答案，也污染成本)；同时单列
    退化率与"含退化的总体 accuracy"，两个口径都给，避免藏猫腻。
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    correct_all = [1.0 if ok else 0.0 for _, ok in records]
    degenerate = [r.is_degenerate for r, _ in records]
    non_deg = [(r, ok) for (r, ok) in records if not r.is_degenerate]

    costs = [episode_cost_yuan(r.usage, prices) for r, _ in records]
    return {
        "n": n,
        "accuracy": _mean(correct_all),                                  # 含退化
        "accuracy_nondegenerate": _mean([1.0 if ok else 0.0 for _, ok in non_deg]),
        "degenerate_rate": _mean([1.0 if d else 0.0 for d in degenerate]),
        "yuan_per_sample": _mean(costs),
        "prompt_tokens_per_sample": _mean([r.usage.prompt_tokens for r, _ in records]),
        "completion_tokens_per_sample": _mean([r.usage.completion_tokens for r, _ in records]),
        "cached_tokens_per_sample": _mean([r.usage.cached_tokens for r, _ in records]),
        "tool_images_per_sample": _mean([r.num_tool_images for r, _ in records]),
        "turns_per_sample": _mean([r.num_turns for r, _ in records]),
        "resend_factor": _mean([r.instrument_stats.get("resend_factor", 1.0) for r, _ in records]),
    }


def gate_decision(full_agg: Dict[str, Any], last_agg: Dict[str, Any]) -> Dict[str, Any]:
    """由 full / last 两个聚合算 headroom 与 Go/No-Go(口径对齐 metrics_spec)。"""
    cost_headroom = (full_agg["yuan_per_sample"] / last_agg["yuan_per_sample"]
                     if last_agg.get("yuan_per_sample") else float("inf"))
    token_headroom = (full_agg["prompt_tokens_per_sample"] / last_agg["prompt_tokens_per_sample"]
                      if last_agg.get("prompt_tokens_per_sample") else float("inf"))
    acc_headroom = full_agg["accuracy"] - last_agg["accuracy"]

    # gate：成本矩形够大(>=2x)且 accuracy 有可保护空间(>=3分=0.03)才值得做 BudgetSee
    cost_ok = cost_headroom >= 2.0
    acc_ok = acc_headroom >= 0.03
    if cost_ok and acc_ok:
        verdict = "GO"
    elif cost_ok and not acc_ok:
        verdict = "NO-GO(last 已够用，无 accuracy 可保护)"
    elif not cost_ok and acc_ok:
        verdict = "NO-GO(旧图不占成本，无成本可省)"
    else:
        verdict = "NO-GO(矩形太小)"

    return {
        "cost_headroom": cost_headroom,
        "token_headroom": token_headroom,
        "accuracy_headroom": acc_headroom,
        "cost_ok(>=2x)": cost_ok,
        "accuracy_ok(>=0.03)": acc_ok,
        "verdict": verdict,
    }
