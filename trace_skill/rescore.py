#!/usr/bin/env python3
"""离线重算分：用宽容验证器(verifier_lenient)重判已采 trace，对比严格 vs 宽容准确率。

零 API 成本——只读 trace 里的 final_answer/solution/subset 重新判分，看"严格验证器把多少
格式不符误判成了答错"。用法：
  python trace_skill/rescore.py --trace-dir trace_skill/assets/output/traces_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runner.verifier import ExactMatchVerifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifier_lenient import LenientTIRVerifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    args = ap.parse_args()

    strict = ExactMatchVerifier()
    lenient = LenientTIRVerifier()

    recs = []
    for p in sorted(glob.glob(os.path.join(args.trace_dir, "*.json"))):
        recs.append(json.load(open(p, encoding="utf-8")))

    by = defaultdict(list)
    flipped = []
    for d in recs:
        ans, sol, sub = d.get("final_answer"), d.get("solution"), d.get("subset")
        s_ok = strict.verify(ans, sol)
        l_ok = lenient.verify(ans, sol, sub)
        by[sub].append((s_ok, l_ok))
        if l_ok and not s_ok:
            flipped.append((d["doc_id"], ans, sol))

    print(f"{'subset':26s} {'严格acc':>8s} {'宽容acc':>8s} {'被救回':>6s}")
    ts = tl = 0
    for sub in sorted(by):
        rs = by[sub]; n = len(rs)
        s = sum(a for a, _ in rs); l = sum(b for _, b in rs)
        ts += s; tl += l
        print(f"{sub:26s} {s/n:8.2f} {l/n:8.2f} {l - s:6d}")
    N = len(recs)
    print(f"{'总计':26s} {ts/N:8.2f} {tl/N:8.2f} {tl - ts:6d}")

    if flipped:
        print(f"\n被宽容验证器救回的题({len(flipped)})：")
        for doc, ans, sol in flipped:
            print(f"  {doc}: ans={str(ans)[:50]!r} gt={str(sol)[:50]!r}")
    else:
        print("\n无题被救回(严格判错的都是真·感知错，非格式问题)。")


if __name__ == "__main__":
    main()
