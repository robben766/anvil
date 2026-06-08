# 多模型评测陪审团(Council-Jury)设计文档

> 状态:已批准(2026-06-08,头脑风暴收敛后用户确认"做 (i) 评测陪审团")。本文件是该产品的单一事实来源。
> **断点续作入口**:新会话从这里读起即可继续,无需依赖被压缩的对话上下文。

## 0. 一句话目标

把 anvil 质量竖梁从「单 LLM-judge + Cohen's κ」升级到「**校准过的多模型评测陪审团**」:同一待评对象交给多个不同模型的陪审员**独立**按统一 rubric 打分,聚合出带置信度、**显式标注分歧**的裁决,并用实验回答"**陪审团是否真的打得过最好的单个评委、是否更贴近人工**"。

## 1. 背景:为什么是"评测陪审团",而不是"协商聊天会议"

P2 原设想是"多模型协商会议(辩论/共创)"。头脑风暴中用户指出其亲历的两个失败:
- **坑1**:全靠手写角色 persona,定义不好则收敛差、结果差(persona 脆弱)。
- **坑2**:整套不如直接用 Claude Code / Codex / Gemini 等强 agent CLI——裸 chat 模型空对空辩论,把幻觉平均并不会变对;弱模型集成打不过有工具会迭代的强 agent。

收敛结论:**多模型集成只在"判断/评测"上真能赢单模型(陪审团 > 单评委,研究证实),在"生成/共创"上是强 CLI 赢(那是 P3 的事)。** 因此 P2 收窄为**评测陪审团**——它落在 ensemble 真有效的地方、吃 anvil 自己的 eval 狗粮、不与 CLI 抢饭碗,且把"多 seat 并行→结构化→聚合→标分歧"的**编排原语**练出来供 P3 复用。

正面修两个坑:陪审员**独立打分、互不可见**(评测要独立性,天然避开谄媚锚定/同质化);角色 = **功能性 rubric 分项**(库内固定,不让用户手写 persona);每个陪审员**喂参考/证据接地**,不空对空。

明确砍掉:领域判定简报(原 (ii),未见真实痛点)。

## 2. 架构归属

新包 `packages/core/council`(圈1 普适),依赖 `gateway`(多 provider 陪审员)+ `eval`(rubric/校准)+ `guard`(结构化输出)。编排原语(并行扇出 / 聚合 / 分歧度量)被 P3 编码团队复用;反过来陪审团可评 P3 agent 的产出。

```
packages/core/council/
├─ seats.py        陪审员 = (model) × (rubric);并行扇出,各自 structured_chat 打分
├─ rubric.py       评分维度库(功能角色,非手写人格):如 正确性/证据充分/完整性/反例与风险
├─ aggregate.py    聚合(分项中位数/均值 + 总分)+ 分歧探测(离散度超阈值标分歧点)
├─ agreement.py    多评委一致性:手写 Fleiss' κ(>2 评委)+ 陪审团 vs 人工 + 最佳单评委对比
├─ verdict.py      裁决数据模型:总分/分项/置信度/分歧点/各员留痕
└─ cli.py          anvil-council judge / calibrate
```

## 3. 范围(里程碑,由简入繁)

### CJ-M1:陪审团核心引擎
- `seats.py`:给定 (case, [model 列表], rubric),**并行**调用每个模型经 `guard.structured_chat` 产出 `{per_criterion: {score, reason}, overall, reason}`;陪审员之间**独立、互不可见**。
- `rubric.py`:内置一套通用功能维度(如 正确性/证据充分性/完整性/相关性),可配置;**不暴露手写 persona**。
- `aggregate.py`:分项聚合(中位数为主,均值可选)+ 总分;**分歧探测**——某分项陪审员打分离散度(极差或标准差)超阈值即标该分项为"分歧",附离散陪审员的理由。
- `verdict.py`:`Verdict{overall, per_criterion, confidence, disagreements: list, jurors: list[JurorScore]}`。
- CLI `anvil-council judge --case <jsonl> --models deepseek-chat,qwen-plus`,打印裁决。
- TDD:聚合/分歧探测纯函数手算对照;陪审员调用 respx mock。

### CJ-M2:多评委一致性 + 校准 + 核心实验
- `agreement.py`:手写 **Fleiss' κ**(多评委,手算对照锚点;复用/对照 eval 已有的 pairwise Cohen's κ)+ 陪审团聚合分 vs 人工 κ + **每个单评委 vs 人工 κ**。
- 校准数据集:扩 `packages/core/eval/golden/calibration.jsonl`(Core-Guard 已建 14 条带 human_score)到 ~30 条,覆盖好/部分/错三档。
- CLI `anvil-council calibrate`:跑陪审团 + 各单评委,输出 κ 对比表。
- **核心实验**(真实数字,含负向):陪审团 κ_vs_human 与**最佳单评委** κ_vs_human 对比 + 评委间 Fleiss' κ;诚实记录陪审团是否真有增益(可能没有——强+弱平均反而拉低,照实写,对照 KB Contextual 负结果口径)。激活 task #22:百炼 qwen-plus live 接入(第二个真陪审员)。

### CJ-M3:接入 eval 管线(吃狗粮)
- `anvil-eval` 增 `--jury`(或 council 侧封装):用陪审团替/补单 judge 评 anvil 自己的 KB 回答。
- 实验:单 judge vs 陪审团 在 KB 生成侧评测上的稳定性/分歧对比。
- 文档:`examples/` 加「多模型评测陪审团」章节,原样记录 κ 与实验数据;README/CLAUDE 状态更新。

### CJ-M4(可选,后置)
极简 web 裁决视图(各陪审员打分 + 分歧矩阵)。**首期不做**,留作后续。

## 4. 核心数据流

```
case{question, answer, reference/evidence, rubric}
  → 并行: 每个 model 经 structured_chat 按 rubric 打分(独立,互不可见)
  → aggregate: 分项中位数 + 总分;离散度超阈值 → 标分歧
  → verdict{overall, per_criterion, confidence, disagreements[], jurors[]}
  → (CJ-M2 离线) calibrate: 陪审团/各单评委 各自 vs 人工 κ + 评委间 Fleiss' κ
```

## 5. 纪律

- 每子任务 TDD + 两段式审查(实现者 + spec审查 + 质量审查),里程碑末 opus 终审,CI 绿后合并。
- 公开仓红线:无公司项目名、无本机绝对路径、无真实邮箱;commit 用 robben766 noreply。
- 复用优先:structured_chat / judge_json / Cohen's κ / gateway 多 provider / golden 数据集均现成,council 只加"多 seat 编排 + 聚合 + 分歧 + 多评委校准"。

## 6. 验收

- 陪审团引擎:多模型并行打分 + 聚合 + 分歧探测,聚合/分歧/Fleiss' κ 均有手算对照测试。
- 校准:输出真实 κ(陪审团 / 各单评委 / 评委间),不修饰;诚实回答"陪审团是否打过最佳单评委"。
- 百炼 live 接入打通(qwen-plus 作第二陪审员)。
- 全仓 ruff + pytest(-m "not live")绿。

## 7. 不做(明确边界)

辩论 / 交叉质询(评测要独立性,刻意反着来)· 内容共创(留 CLI / P3)· 领域判定简报(已砍)· 多轮 / 过程 HITL / 自动替人最终裁决 · web 视图(CJ-M4 后置)。Claude/GPT 等更多陪审员:接口预留,首期只 DeepSeek + 百炼。
