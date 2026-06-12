# Skill Generation for Thinking-with-Image Benchmarks

> 从 **skill 生成的基础工作**出发，研究 **TIR 等 thinking-with-image（工具集成视觉推理）benchmark** 的 skill 方法：
> 用 VLM 在这些 benchmark 上的真实失败 trace，蒸出可注入的 skill，让模型把题**做对**、同时**省 token**。

---

## 1. 背景与动机

多模态 Agent 在 **thinking-with-image** 类任务上（解题过程中调用工具产图 / 裁剪 / 分析、多轮视觉推理）失败率高，且弱基座模型常**过度思考**——单题生成上万 token 的推理却仍答错。

我们想验证：**能否用模型自己的失败 trace 蒸出 skill**，在不换基座的前提下同时拿到两个收益——
- **做对**：把准确率从地板抬起来；
- **省 token**：砍掉无效的过度思考与反复试错。

---

## 2. 方法（四步闭环）

```
跑 benchmark 易失败子集
        │
        ▼
① 采集 trace ── 逐题记录每轮 reasoning / 工具代码 / 工具结果 / token / 花费
        │
        ▼
② 诊断根因 ── 从真 trace 归纳失败模式（写码 bug / 退化值 / 幻觉 / 过度思考 / 纯感知错…）
        │
        ▼
③ 蒸 skill ── 据根因写可注入的 skill（写码配方 + 校验闸门 + token 纪律 + 分任务配方）
        │
        ▼
④ 验证 ── baseline vs +skill 配对，证明「做对↑ 且 省 token↓」
```

第一个完整实现与跑通案例在 **[`trace_skill/`](trace_skill/README.md)**。

---

## 3. Benchmark 路线图

| Benchmark | 状态 | 说明 |
|-----------|------|------|
| **TIR-Bench** | ✅ 进行中 | 13 类 thinking-with-image 任务（迷宫 / 拼图 / 占比 / 仪表 / 找不同…）。论文 arXiv:2511.01833。已全量下载到 `assets/input/tir/`，Round 1 已在 5 个最易失败子集上跑通采集→诊断→skill→验证 |
| **VisualToolBench** | ⏳ 计划中 | 多轮视觉工具使用 benchmark，下一个接入对象。按同一套方法（采 trace→诊断→蒸 skill→验证）扩展 |
| 其它（MMSearch-Plus / AgentVista 等） | 🔭 候选 | 视进展再评估是否纳入 |

> 接入新 benchmark 的工作量主要在「数据加载 + 该 benchmark 需要的视觉工具」，`runner/` 的多轮工具循环与 skill 验证流程可直接复用。

---

## 4. 当前进展（TIR-Bench Round 1，2026-06-12）

模型 `qwen3-vl-8b-thinking`（阿里百炼），5 个最易失败子集 25 题：

- **失败形态**：不是 10 轮死循环，而是**内部过度思考**（reasoning 3.4 万~11.7 万字符、输出 token 1.96 万/题），2-4 轮就答错。
- **skill v0 效果**：准确率 **0.04 → 0.16（+12pt，4×）**，但 token **+23%**（省 token 尚未达成）。
- **核心洞察**：skill 的「**可执行配方**」机制（如占比估计）→ 又对又省；「**修码重试 sanity gate**」机制在救不了的硬任务上 → token 爆炸反噬。
- **下一步（skill v1）**：①每任务给具体配方；②硬任务装止损闸防 token 爆炸；③压 reasoning 真省 token。

完整数据、管线、实验推进历程见 **[`trace_skill/README.md`](trace_skill/README.md)**。

---

## 5. 仓库结构

```
exp20_budgetsee/                  （目录名为遗留名，见下方说明）
├── README.md                     本文件：项目总览
├── runner/                       通用 agent 基建（与具体 benchmark/skill 无关，可复用）
│   ├── agent_loop.py             多轮工具循环（VLM client + 可插拔工具 + 完整视觉历史）
│   ├── code_interpreter.py       持久 Jupyter 沙箱（原图预载、跨轮变量、三路产图捕获）
│   ├── vlm_client.py / tools.py / tir_data.py / verifier.py / gate.py
├── trace_skill/                  ⭐ 方法实现（采集 / 诊断 / skill / 验证）
│   └── README.md                 方法权威文档
├── assets/input/tir/             TIR-Bench 数据（13 子集 1241 图 + val_shuffled.json）
└── docs/                         （留待新文档）
```

> **关于目录名**：本项目原名 "BudgetSee"（一个视觉历史压缩的方法层），现已停做、相关代码删除。
> 目录名 `exp20_budgetsee` 与 conda 环境名 `budgetsee` 为遗留名，沿用以避免改动大量绝对路径。

---

## 6. 快速开始

```bash
# 0) 环境（python 3.10）
conda create -n tirskill python=3.10 -y && conda activate tirskill
pip install -r requirements.txt

# 1) 下载 TIR-Bench 数据（约 1.7GB，不在仓库里）
python download_tir.py            # 国内可加 --proxy http://127.0.0.1:7890

# 2) 配置模型——复制 trace_skill/.env.example 为 trace_skill/.env 并填 key（已 gitignore）
#   QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#   QWEN_MODEL=qwen3-vl-8b-thinking
#   QWEN_API_KEY=...

# 3) 采 trace（TIR 5 子集各 5 题）
python trace_skill/step1_collect_traces.py \
  --subsets tir_bench_jigsaw tir_bench_instrument tir_bench_refcoco tir_bench_maze tir_bench_spot_difference \
  --n-per-subset 5 --max-turns 10 --concurrency 6

# 4) 诊断   5) skill 验证（复用 baseline 省钱）
python trace_skill/step2_analyze_traces.py --trace-dir trace_skill/assets/output/traces_<stamp>
python trace_skill/step3_eval_skill.py --skill trace_skill/skills/tir_skill_v0.md \
  --reuse-baseline trace_skill/assets/output/traces_<stamp>
```

详细用法、参数、产物说明见 **[`trace_skill/README.md`](trace_skill/README.md)**。

---

## 7. 相关资源

- **[`trace_skill/README.md`](trace_skill/README.md)** — 方法实现权威文档
- TIR-Bench 论文：arXiv:2511.01833 ｜ 数据：HF `DjangoJungle/TIR-Bench`
- 方法上借鉴已发表的 **XSkill / ACE-Skill**（skill / experience 蒸馏）的失败分析与占比估计思路
