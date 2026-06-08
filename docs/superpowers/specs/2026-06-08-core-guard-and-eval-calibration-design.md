# Core-Guard + Eval Calibration 设计文档(安全竖梁 v1 + 质量竖梁强化)

> 状态:已批准(2026-06-08,用户口头确认"按方案做")。本文件是该里程碑的单一事实来源。
> **断点续作入口**:新会话从这里读起即可继续,无需依赖被压缩的对话上下文。

## 0. 一句话目标

补齐 anvil「三根竖梁」中最空的一根——**安全**:新增 `packages/core/guard`(圈1 普适:注入检测 + 结构化输出约束),并强化「质量」竖梁(eval judge 校准 + golden 扩容)。知识库(packages/kb)是第一个消费方。

## 1. 背景与定位

anvil 架构模型:三同心圆(圈1普适/圈2 RAG/圈3 Agent)× 六层(L1网关→L6体验)× 三竖梁(可观测/质量/安全)。

当前三梁完成度:
- 可观测 ✅ 强(P0-M3 自研 OTLP→Langfuse)
- 质量 🟡 有但小(P0-M4 RAGAS 四指标 + CI recall 门;golden 16 条、judge 未校准、无在线闭环)
- 安全 🔴 近乎空(仅 kb-api 的可选 Bearer + CORS + 生成侧"拒答" prompt)

**本里程碑专攻"圈1 普适该补的"两项缺口:安全 guardrails + 评测规模/可信度。** 平台层安全(多租户隔离/审计日志/PII/限流)刻意留给 P4「AI员工」,不在本里程碑。

## 2. 架构调整(关键)

安全是横切关注点且圈1普适 → 组件必须落在 `packages/core`,所有产品复用,**不得塞进 kb**:

```
packages/core/
├─ gateway/   (已有,L1/L2)
├─ obs/       (已有,可观测竖梁)
├─ eval/      (已有,质量竖梁)  ← 本里程碑扩展:judge 校准 + golden 扩容
└─ guard/     (★新增,安全竖梁 v1)  ← anvil_guard
   ├─ injection.py   注入检测:关键词/正则快路(确定性硬判)+ 可选 LLM 语义兜底(走 gateway)
   └─ structured.py  结构化输出:json_schema 约束(provider 支持则用)/ 否则 parse+retry 统一封装

packages/kb/  ← 接线:查询前插一道 guard.injection 检查(命中注入 → 拒绝/降级,不进检索)
```

架构图/蓝图(anvil-notes)的"安全竖梁"原本是空框,出图时补 guard 节点。

## 3. 范围(本里程碑做什么)

### G1. `packages/core/guard` 注入检测(injection.py)
- `detect_injection(text) -> InjectionVerdict{is_injection: bool, category: str, matched: list[str], confidence: float}`
- 关键词/正则快路(确定性,零延迟):覆盖常见提示注入模式(忽略以上指令 / ignore previous / 角色扮演越权 / 泄露 system prompt / 代码块伪装指令等),中英双语
- 可选 LLM 语义兜底(`detect_injection_llm`,走 `anvil_gateway.chat` + 结构化输出),默认关闭(快路够用时不调 LLM)
- TDD:手写对抗用例集(正例:各类注入;负例:正常含"忽略"等词的良性提问),锁定召回/误报

### G2. `packages/core/guard` 结构化输出约束(structured.py)
- 统一封装"要模型吐合法 JSON"的两条路径:① provider 支持 `response_format=json_schema` 时走原生约束 ② 否则走 prompt 指令 + parse + 白名单校验 + 一次 retry(把现在散落在 eval/judge 与 gateway 的手写 JSON 逻辑收敛到一处)
- `structured_chat(model, messages, schema) -> dict`(走 gateway,失败 retry,最终抛 StructuredOutputError)
- DeepSeek 对 json_schema strict 的支持先实测确认,以实测为准选路径(实测结论写进注释)

### G3. eval judge 校准(packages/core/eval 扩展)
- 新增 `calibration.py`:给定 (judge 分数, 人工标注分数) 配对,算一致性(Cohen's κ 或 Spearman),输出校准报告
- golden 增加 ~10-20 条带**人工标注分**的样本(标注是虚构语料上的,自己定)用于校准
- CLI/脚本:`anvil-eval calibrate` 跑校准并打印 κ;κ 低于阈值则警告"judge 不可信"

### G4. golden 扩容(packages/kb)
- kb.jsonl 从 16 条扩到 ~50 条:补更多换述/多跳/边界/拒答用例,尤其针对 KB-M2 暴露的 kb-16 类换述难例
- 防腐烂测试同步覆盖新增条目

### G5. kb 接线 + 实验 + 文档 + PR
- kb 查询路径(cli + kb-api)在检索前调 `guard.detect_injection`,命中 → 拒绝(返回安全话术,不进检索/不调生成)
- 实验:① 注入拦截率(对抗集上 guard 的召回/误报)② judge 校准 κ 值 ③ golden 扩容后三模式 eval 复跑对比
- examples/04-kb/README 加「安全竖梁 + 评测强化(Core-Guard)」章节,原样记录数据
- README/CLAUDE.md 状态更新;全量回归;PR

## 4. 纪律(沿用既有流程)

- 每个子任务 TDD + 两段式审查(实现者 + spec审查 + 质量审查),里程碑末 opus 终审,CI 绿后合并
- 公开仓红线:无公司项目名(one-policy/forge/iic 等)、无本机绝对路径、无真实邮箱;commit 用 robben766 noreply
- guard 的注入用例、结构化封装均为**通用安全工程**,不含任何具体业务

## 5. 验收

- guard:对抗用例集注入召回 ≥ 目标 + 良性误报 ≤ 目标(具体数字执行时手算锁定)
- structured:DeepSeek 路径实测能稳定吐合法 JSON
- 校准:输出真实 κ 值(不修饰,低就低)
- golden:50 条,防腐烂测试绿
- 全仓 ruff + pytest(-m "not live")绿;web build 绿(若动 web)

## 6. 不做(本里程碑明确边界)

多租户隔离 / 审计日志 / PII 脱敏 / 限流配额 / 在线反馈闭环 / A/B 灰度 —— 属平台层,留给 P4。OCR 也不在本里程碑(另立)。
