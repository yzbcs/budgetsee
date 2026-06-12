# trace_skill — TIR 失败 trace → skill 任务

> 导师布置的独立任务：用 **qwen3-VL-8B 跑 TIR-Bench 上最易失败的子集**，逐题记录 trace 与花费，
> 据真 trace 蒸一/几个 **skill**，让这些题"**做对 + 省 token**"。
>
> 这是项目主线（原 BudgetSee 视觉历史压缩方法层已停做并删除）。**复用**根目录通用 agent 基建 `runner/`，
> 不改它。产物自包含在 `trace_skill/assets/`。conda 环境名 `budgetsee` 是遗留名。

---

## 1. 结论先行（Round 1，2026-06-12，真跑 25 题 ¥2.5）

- **不是 10 轮死循环**：qwen3-vl-8b-thinking 在 TIR 上平均只跑 2.8 轮、≤4 轮就答，没有一题耗尽 10 轮。
  真正的病是**内部过度思考**——单题 reasoning 3.4 万~11.7 万字符、输出 token 1.96 万/题。"死循环"是思维层面的，不是轮次层面的。
- **skill v0 效果**：accuracy **0.04 → 0.16（+12pt，4×）**，但 token **+22%、¥ +23%**（省 token 未达成）。
- **核心洞察**：skill 的两个机制一个真香一个反噬——
  - ✅ **可执行配方**（refcoco 可见像素占比）→ **双赢**：refcoco 0→60%，救回题更省（¥0.059→0.029）。
  - ❌ **"sanity gate 让你修代码重试"** → 在救不了的硬任务（jigsaw/spot/maze）上**反噬**：token 29k→42k 爆炸、把蒙对的 maze_510 跑坏。
  - 一句话：**有配方处又对又省，只有"再试"无配方处更费更错。**

---

## 2. 5 个目标子集（TIR-Bench 论文定，非经验猜）

`jigsaw / instrument / refcoco / maze / spot_difference`，据 **TIR-Bench 论文 arXiv:2511.01833 Table2**
（工具调用类模型每任务准确率，越低越易深迭代）：jigsaw 7.6%(PyVision 开源)/16.4%(o3-TU)——论文实录
"近 55 次迭代→3.6 万次暴力尝试"；instrument 21.3% 反复裁剪；refcoco 31.7% 分割不收敛；maze 弱模型 15.8%；
spot 36-41% 反复比对。论文任务名↔HF 目录映射：**Proportion=refcoco、Low-Light=contrast、Rotated OCR=ocr**。

数据：完整 TIR-Bench（13 子集 1241 图 + `val_shuffled.json`）在 `../assets/input/tir/`。
`data_source` 字段区分子集；图片相对路径前缀 `TIR-Bench/data/<sub>/<id>.png`，故 `TIR_IMAGE_FOLDER=../assets/input/tir`。

---

## 3. 代码与管线

| 文件 | 职责 |
|------|------|
| `step1_collect_traces.py` | 采 trace + 算账。`TopPClient`(百炼/OpenRouter 通用) + 逐轮录 token/reasoning/工具代码 + 结局分类 + 并发 |
| `step2_analyze_traces.py` | 读 trace 出失败诊断 md（重复代码片段、reasoning 长度、失败题清单） |
| `step3_eval_skill.py` | baseline vs +skill 配对评测，验证"做对↑ 且 省token↓"。`--reuse-baseline` 复用已采 trace 只跑 +skill |
| `verifier_lenient.py` | `LenientTIRVerifier` 任务感知宽容验证器（spot 按集合 / jigsaw 按序列 / 选择数值多格式抽取） |
| `rescore.py` | 离线用宽容验证器重判已采 trace（零 API 成本） |
| `skills/tir_skill_v0.md` | 证据驱动 skill v0（写码规则 + sanity gate + token 纪律 + 分任务配方），step3 作 system prompt 注入 |

### 模型 / 平台（阿里百炼 DashScope，OpenAI 兼容）
- `QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`、`QWEN_MODEL=qwen3-vl-8b-thinking`、key 写 `.env`（gitignore）。
- 计价：输入(思考) **¥0.5/M**、输出(思考) **¥5/M**（输出贵 10×，是成本主项）。
- 两个坑已在 `step1` 处理：①thinking 模型强制流式 → 自动开流式 + SSE 聚合（含 `stream_options.include_usage` 取 token 账）；
  ②模型偶发把 tool_call 当文本 JSON 吐（非原生 tool_calls）→ `_recover_text_tool_call` 兜底解析（裸 JSON / `<tool_call>` 标签）。

### 怎么跑
```bash
# 环境：python 3.10 + pip install -r ../requirements.txt；数据先 `python ../download_tir.py`
# 1) 采 trace（5 子集各 5 题，mt10，并发 6，~¥2）
python trace_skill/step1_collect_traces.py \
  --subsets tir_bench_jigsaw tir_bench_instrument tir_bench_refcoco tir_bench_maze tir_bench_spot_difference \
  --n-per-subset 5 --max-turns 10 --concurrency 6
# 2) 诊断
python trace_skill/step2_analyze_traces.py --trace-dir trace_skill/assets/output/traces_<stamp>
# 3) skill 验证（复用 baseline 只跑 +skill）
python trace_skill/step3_eval_skill.py --skill trace_skill/skills/tir_skill_v0.md \
  --reuse-baseline trace_skill/assets/output/traces_<stamp>
# 离线重算分（换验证器后）
python trace_skill/rescore.py --trace-dir trace_skill/assets/output/traces_<stamp>
```

产物：`assets/output/`（`traces_<stamp>/` 逐题 trace、`summary_*.json`、`diagnosis_*.md`、`skill_eval_*.json`），日志 `assets/log/`。

---

## 4. Round 1 失败画像（全 25 题扫描的根因命中）

| 根因 | 命中/25 | skill 可治? |
|------|---------|------------|
| 工具代码报错(Traceback) | 16 (64%) | ✅ 写码规则 |
| 全程无可用工具输出 | 12 (48%) | ✅ 配方 |
| 脑内模拟工具(幻觉) | 13 (52%) | ✅ token 纪律 |
| 过度思考 >4万字符 | 14 (56%) | ✅ token 纪律 |
| `Image.open(原图)` bug | 8 (32%) | ✅ 一句话治 |
| 工具返回退化值(0.00/[]) | 6 (24%) | ✅ sanity gate |
| 残余：工具正常但纯感知错(instrument 读数/jigsaw 36 块) | ~20-30% | ⚠️ 8B 硬伤，难救 |

验证器修正：原 `ExactMatchVerifier` 把格式不符疑似误判，故建 `LenientTIRVerifier`（11/11 单测过）。
但重算分 **0 题被救回**——24 个答错全是真·感知错，非格式问题。结论：这弱基座上"做对"要真功夫，光规范格式救不回。

---

## 5. 实验推进历程

### Round 1 — 采 trace + 诊断 + skill v0 验证
- **动机**：导师要"跑易失败子集 → 据 trace 蒸 skill 做对 + 省 token"。
- **方案**：论文 Table2 选 5 子集（弃经验广擒，省钱）；百炼 qwen3-vl-8b-thinking 跑 25 题 mt10；step2 诊断 + 全题根因扫描；据真 trace 写证据驱动 skill v0；step3 复用 baseline 跑 +skill。
- **结果**：acc 0.04→0.16（+12pt），但 token/¥ +23%。refcoco 0→60% 是主要收益且更省；jigsaw/spot token 爆炸、maze 退步。
- **核心发现**：skill 的"配方机制"双赢、"修码重试机制"在硬任务反噬（见 §1）。失败是内部过度思考非 10 轮死循环。
- **下一步洞察 → v1**：①每任务都给 refcoco 级具体配方（instrument/maze/spot 到代码级）；②硬任务装止损闸（2 次拿不到干净结果立即出最佳估计、禁 retry，治 token 爆炸）；③明确压 reasoning 真正省 token。

---

## 6. 方法借鉴（已发表工作）
- **XSkill / ACE-Skill** 的失败分析与 skill / experience 蒸馏思路。
- refcoco 占比估计配方借鉴 ACE-Skill 的 "Accurate Visual Proportion Estimation"（可见像素计数 + 几何 footprint 兜底）。
