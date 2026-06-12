#!/usr/bin/env python3
"""下载 TIR-Bench 数据集到 assets/input/tir/（数据不进 git，clone 后跑此脚本复现）。

数据源：HuggingFace `DjangoJungle/TIR-Bench`（公开）。下载 val_shuffled.json + 全部图片
（13 子集 1241 张，约 1.7GB），目录结构与 json 内相对路径 `TIR-Bench/data/<sub>/<id>.png` 对齐。

用法：
  python download_tir.py                 # 直连
  python download_tir.py --proxy http://127.0.0.1:7890   # 走代理（或设环境变量 HTTPS_PROXY）
  python download_tir.py --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time

import requests

REPO = "DjangoJungle/TIR-Bench"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/input/tir")


def get(url, proxies, timeout=60):
    for _ in range(3):
        try:
            r = requests.get(url, proxies=proxies, timeout=timeout)
            if r.status_code == 200:
                return r.content
        except Exception:
            time.sleep(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None

    os.makedirs(OUT_ROOT, exist_ok=True)
    # 1) 标注 json
    json_path = os.path.join(OUT_ROOT, "val_shuffled.json")
    if not os.path.exists(json_path):
        print("下载 val_shuffled.json ...")
        data = get(BASE + "val_shuffled.json", proxies)
        if data is None:
            raise SystemExit("下载 val_shuffled.json 失败（检查网络 / --proxy）")
        open(json_path, "wb").write(data)
    items = json.load(open(json_path))
    rels = sorted({p for it in items for p in it.get("images", [])})
    print(f"共 {len(items)} 样本 / {len(rels)} 张图，下载到 {OUT_ROOT}")

    def dl(rel):  # rel = "TIR-Bench/data/<sub>/<id>.png"
        out = os.path.join(OUT_ROOT, rel)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return "skip"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        content = get(BASE + rel.replace("TIR-Bench/", "", 1), proxies, timeout=40)
        if content is None:
            return "FAIL:" + rel
        open(out, "wb").write(content)
        return "ok"

    t = time.time(); ok = skip = fail = 0; fails = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for st in ex.map(dl, rels):
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail += 1; fails.append(st)
    print(f"完成：ok={ok} skip={skip} fail={fail}（{time.time()-t:.0f}s）")
    if fails:
        print("失败：", fails[:20])


if __name__ == "__main__":
    main()
