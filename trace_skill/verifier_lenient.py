"""任务感知的宽容验证器 —— 修掉严格 exact-match 把"格式不符"误判为"答错"的问题。

为什么需要
==========
  step1 实测发现：spot_difference 题目要求输出 `{"different_patches":[...]}` JSON，模型也确实
  输出了正确 JSON，但 gt 是逗号串 "9, 13, 14, 16, 17, 20"，runner 的 ExactMatchVerifier 拿
  JSON 比逗号串 → 即使感知对了也判错。这是 vs-skills 报告里"answer_parse_failure 主导"的老问题。
  experiment_design.md 把"输出契约"列为受控变量，要求宽容抽取，accuracy 才测得准、skill 收益才公平。

按任务类型分派
==============
  - 单选(refcoco/部分 instrument)：gt 是单字母 A-Z → 从预测里宽容抽取所选字母。
  - 数值(部分 instrument/math)：gt 是数 → 抽第一个数，按相对容差比。
  - 集合(spot_difference)：gt 是一组补丁下标 → 抽数字按**集合**比(顺序无关)。
  - 序列(jigsaw)：gt 是排列 → 抽数字按**有序序列**比。
  抽数字统一兼容：JSON({"different_patches":[...]} / {"arrangement":[...]})、[..] 方括号、逗号/空格串。

  不修改 runner/，作为本任务(trace_skill)的可插拔验证器；step1/step3 传入 subset 即按类型判。
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_INT = re.compile(r"-?\d+")
_FLOAT = re.compile(r"-?\d+(?:\.\d+)?")

# 哪些子集按集合比 / 按序列比(其余走 choice/numeric/文本)
SET_TASKS = {"tir_bench_spot_difference"}
SEQ_TASKS = {"tir_bench_jigsaw"}


def _strip_tags(s: str) -> str:
    m = _ANSWER_TAG.search(s)
    s = m.group(1) if m else s
    # 去掉残留的孤立 </answer>/<answer>
    return re.sub(r"</?answer>", "", s, flags=re.IGNORECASE).strip()


def _extract_int_list(s: str) -> List[int]:
    """从任意格式抽出整数序列(保持出现顺序)。优先 JSON 列表 → [..] → 全文整数。"""
    s = _strip_tags(s)
    # 1) JSON 对象里第一个 list 值(different_patches / arrangement / 任意键)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return [int(x) for x in v if isinstance(x, (int, float)) or str(x).lstrip("-").isdigit()]
        if isinstance(obj, list):
            return [int(x) for x in obj if isinstance(x, (int, float)) or str(x).lstrip("-").isdigit()]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # 2) 第一个 [...] 方括号
    mb = re.search(r"\[([^\[\]]*)\]", s)
    if mb:
        nums = _INT.findall(mb.group(1))
        if nums:
            return [int(x) for x in nums]
    # 3) 兜底：全文整数(去掉明显的键名数字干扰前，先看是否像纯列表)
    return [int(x) for x in _INT.findall(s)]


def _extract_choice(s: str) -> Optional[str]:
    """从预测里抽单选字母。优先 <answer>X</answer> / 'answer: X' / 末尾独立字母。"""
    s0 = _strip_tags(s).strip()
    # 整串就是一个字母
    if len(s0) == 1 and s0.isalpha():
        return s0.upper()
    m = re.search(r"answer\s*[:：]?\s*\(?([A-Za-z])\)?\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 独立字母(取最后一个，通常是结论)
    letters = re.findall(r"\b([A-Fa-f])\b", s0)
    return letters[-1].upper() if letters else None


def _extract_number(s: str) -> Optional[float]:
    s0 = _strip_tags(s)
    m = _FLOAT.search(s0)
    return float(m.group()) if m else None


class LenientTIRVerifier:
    """任务感知宽容验证器。verify(pred, gt, subset) -> bool。"""

    def verify(self, prediction: Optional[str], ground_truth: Optional[str],
               subset: Optional[str] = None) -> bool:
        if prediction is None or ground_truth is None:
            return False
        pred, gt = str(prediction).strip(), str(ground_truth).strip()
        if not gt:
            return False

        # 集合 / 序列任务
        if subset in SET_TASKS:
            pe, ge = _extract_int_list(pred), _extract_int_list(gt)
            return bool(ge) and set(pe) == set(ge)
        if subset in SEQ_TASKS:
            pe, ge = _extract_int_list(pred), _extract_int_list(gt)
            return bool(ge) and pe == ge

        # gt 是单字母 → 单选
        gt_clean = _strip_tags(gt).strip()
        if len(gt_clean) == 1 and gt_clean.isalpha():
            return _extract_choice(pred) == gt_clean.upper()

        # gt 是纯数 → 数值(相对容差 1e-3，绝对 1e-6)
        g_num = _extract_number(gt)
        if g_num is not None and _FLOAT.fullmatch(gt_clean.replace(" ", "")):
            p_num = _extract_number(pred)
            if p_num is None:
                return False
            tol = max(1e-6, abs(g_num) * 1e-3)
            return abs(p_num - g_num) <= tol

        # gt 含多个数(逗号/方括号列表)但子集未知 → 默认按集合比(更宽容)
        ge = _extract_int_list(gt)
        if len(ge) > 1:
            return set(_extract_int_list(pred)) == set(ge)

        # 兜底：归一化文本相等
        norm = lambda x: re.sub(r"\s+", " ", _strip_tags(x).lower()).strip(" .,:;!?\"'`")
        return norm(pred) == norm(gt)
