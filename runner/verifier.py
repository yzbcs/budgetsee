"""答案验证器 —— 独立、确定性的归一化 exact-match，可插拔(以后可换 LLM judge)。

为什么 exact-match 够用(对 gate 而言)
====================================
gate 看的是 accuracy headroom = acc(full) − acc(last) 的**差值**。只要两个基线用同一个
验证器，验证器是否完美校准不影响差值的有效性——它只需**一致**。原始实现用的是 LLM judge
(compute_score，需 VERIFIER_API_KEY，且报告里出过 bug)；这里先用确定性 exact-match
避免再引一个 API 依赖与不稳定源。Verifier 是协议，后续要更高保真可换 LLMJudgeVerifier。

归一化规则
----------
小写、去首尾空白、去 <answer> 包裹、压缩空白、去首尾标点/引号；再尝试：
数值等价(float 比较)、单选字母(ground_truth 是单个 A-Z 时，看预测里独立字母)。
"""

from __future__ import annotations

import re
from typing import Optional


_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_PUNCT_EDGE = re.compile(r"^[\s\"'`.,:;!?()\[\]{}]+|[\s\"'`.,:;!?()\[\]{}]+$")


def strip_answer_tags(s: str) -> str:
    m = _ANSWER_TAG.search(s)
    return m.group(1) if m else s


def normalize(s: str) -> str:
    s = strip_answer_tags(str(s))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = _PUNCT_EDGE.sub("", s)
    return s.strip()


def _numeric_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False


def _choice_match(pred: str, gt: str) -> bool:
    """ground_truth 是单个字母(选择题)时，看预测里是否以该字母作答。"""
    if len(gt) != 1 or not gt.isalpha():
        return False
    letters = re.findall(r"\b([a-z])\b", pred)
    return gt in letters


class ExactMatchVerifier:
    """确定性归一化匹配。verify(prediction, ground_truth) -> bool。"""

    def verify(self, prediction: Optional[str], ground_truth: Optional[str]) -> bool:
        if prediction is None or ground_truth is None:
            return False
        p, g = normalize(prediction), normalize(ground_truth)
        if not g:
            return False
        if p == g:
            return True
        if _numeric_equal(p, g):
            return True
        if _choice_match(p, g):
            return True
        return False
