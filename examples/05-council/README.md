# 示例 05:多模型评测陪审团(Council-Jury)

把质量竖梁从「单 LLM-judge + Cohen's κ」升级到「校准过的多模型评测陪审团」:多个不同
模型的陪审员**独立**按统一 rubric 打分 → 聚合(分项中位数 + 标分歧)→ 用 Cohen's κ /
Fleiss' κ 校准,并诚实回答「陪审团是否真的打得过最好的单个评委」。

## 设计要点(正面修两个已知坑)

- 角色 = **功能性评分维度**(correctness/evidence/completeness/relevance),库内固定,
  **不让用户手写人格 prompt**(persona 脆弱是上一版失败主因)。
- 陪审员**独立、互不可见**——评测要独立性,天然避开谄媚锚定与同质化。
- 陪审员**喂参考/证据接地**,不空对空。

## 命令

- `anvil-council judge --dataset cases.jsonl --models deepseek-chat,qwen-plus` — 跑陪审团,打印裁决
- `anvil-council calibrate --dataset calibration.jsonl` — 陪审团 vs 人工 vs 最佳单评委 κ 对比

## 核心实验(真实数据,30 条 calibration golden,DeepSeek + 百炼 Qwen 双评委)

| 评委 | vs 人工 Cohen's κ |
|---|---|
| deepseek-chat | 0.526 |
| qwen-plus(百炼) | 0.681 |
| **陪审团(聚合)** | **0.626** |

- **陪审团跑赢最佳单评委:NO**(最佳单评委 = qwen-plus 0.681 > 陪审团 0.626)。
- **评委间一致性 Fleiss' κ = 0.815**(两个评委彼此高度一致)。

**诚实结论(IP 金料,负向结果)**:在这一集上,2 评委陪审团**没有**打过最强的单评委——
陪审团 κ(0.626)夹在两评委之间,把较弱的 deepseek 平均进来反而把强的 qwen 拖下来了。
关键洞察:**评委间 Fleiss' κ 高达 0.815,说明两个评委高度冗余、并不互补**——彼此说的差不多,
平均一下基本还是原样,反而被略弱的那个拉低。**教训:集成的增益来自评委的"互补性"(低相互
一致 + 各自高质量),而非"评委越多越好"或"评委越一致越好";高度一致的评委是冗余,弱评委会
稀释强评委。** 这与 KB 的 Contextual Retrieval 负结果同口径——照实记录,不修饰。

**可复现性观察**:即便 temperature=0,deepseek 的 κ 在多次运行间有 ~0.05 抖动(0.53~0.57),
provider 侧并非完全确定性——多步审议类系统的评测需注意这点。后续:补势均力敌但**视角互补**的
异构评委(Claude/GPT)、或按校准 κ 给评委加权再聚合,可能翻转该结论。

## Dogfood:陪审团评 anvil 自己的 KB 回答

陪审团对 6 条 KB 回答打分,**正确抓出故意写弱的答案**:"重疾赔付比例?"答"赔保额的一半"
(参考是按 100% 给付)被判 总分 0.00 / correctness 0.0;并在偏简略的回答上标出分歧维度
(如"理赔第一步:先报案"标 evidence 分歧)。证明陪审团能把低质量答案与分歧显式暴露出来。

## 复用底座

structured_chat(guard)/ judge 思路(eval)/ Cohen's κ(eval)/ gateway 多 provider /
golden 数据集 —— 几乎全是现成,council 只新增「多 seat 并行 + 聚合 + 分歧 + Fleiss' κ +
陪审团对比」。多 seat 并行/聚合/分歧这套编排原语供 P3 复用。百炼 qwen-plus live 链路在此验通。
