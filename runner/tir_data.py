"""TIR-Bench 数据加载 —— 独立读取，按类目分层抽样。

TIR 样本 schema
  doc_id   : 样本 id
  problem / question : 题面，含 <image> 占位
  images   : 图片相对路径列表(相对 image_folder)
  solution : ground_truth 答案
  source/category(可选): 类目；缺失时从 doc_id 前缀推断(如 'refcoco_524' → 'refcoco')

数据获取(用户后补)：
  HF DjangoJungle/TIR-Bench 的 val_shuffled.json + 图片。
  set TIR_JSON 指向 json，TIR_IMAGE_FOLDER 指向图片根目录。
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

_IMAGE_PLACEHOLDER = re.compile(r"<image>", re.IGNORECASE)


@dataclass
class TIRSample:
    doc_id: str
    question: str                 # 已去掉 <image> 占位的题面
    image_paths: List[str]        # 绝对路径
    solution: str                 # ground_truth
    category: str = "unknown"
    raw: Dict = field(default_factory=dict)

    def load_images(self) -> List[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.image_paths]


def _derive_category(sample: Dict) -> str:
    for key in ("category", "source", "task", "subset"):
        v = sample.get(key)
        if v:
            return str(v)
    doc_id = str(sample.get("doc_id", ""))
    # 'refcoco_524' → 'refcoco'；'tir_bench_color_12' → 'tir_bench_color'
    m = re.match(r"^([a-zA-Z_]+?)(?:_\d+)+$", doc_id)
    return m.group(1) if m else (doc_id or "unknown")


def load_tir_samples(json_path: str, image_folder: str) -> List[TIRSample]:
    """读 TIR json，解析成 TIRSample 列表(解析图片绝对路径，不立即载图)。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 list，实际 {type(data)}")

    samples: List[TIRSample] = []
    for item in data:
        question = item.get("problem", item.get("question", ""))
        question = _IMAGE_PLACEHOLDER.sub("", question).strip()
        rels = item.get("images", []) or []
        abs_paths = [os.path.join(image_folder, r) for r in rels]
        samples.append(TIRSample(
            doc_id=str(item.get("doc_id", item.get("question_id", f"idx_{len(samples)}"))),
            question=question,
            image_paths=abs_paths,
            solution=str(item.get("solution", "")),
            category=_derive_category(item),
            raw=item,
        ))
    return samples


def stratified_sample(samples: List[TIRSample], n: int, seed: int = 42) -> List[TIRSample]:
    """按类目分层抽 n 个(尽量每类均匀)，保证 gate 能按类目看 headroom。"""
    import random
    rng = random.Random(seed)
    by_cat: Dict[str, List[TIRSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)
    for v in by_cat.values():
        rng.shuffle(v)

    cats = sorted(by_cat)
    picked: List[TIRSample] = []
    idx = 0
    # 轮转各类目取，直到取满 n 或取尽
    while len(picked) < n and any(by_cat[c] for c in cats):
        c = cats[idx % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop())
        idx += 1
    return picked[:n]


def category_counts(samples: List[TIRSample]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.category] += 1
    return dict(sorted(counts.items()))


def resolve_paths_from_env() -> Optional[tuple]:
    """从环境变量取 (json_path, image_folder)，缺失返回 None。"""
    j = os.environ.get("TIR_JSON")
    f = os.environ.get("TIR_IMAGE_FOLDER")
    if j and f:
        return j, f
    return None
