# anvil 🔨

**Build your own AI stack, from scratch, to truly understand it.**

anvil(铁砧)是一个 AI 工程深度学习项目:自研造轮子实现 LLM 应用的核心模块,框架只做参照。底座之上将依次锻造四个产品:

```
packages/core(P0 公共底座)
  ├─ gateway   统一模型网关:多 provider / fallback / 成本与缓存命中记账
  ├─ obs       OTEL 标准可观测:自写采集层 + Langfuse 看图
  └─ eval      评测:手写 RAGAS 四指标 + golden set CI 门禁
apps/(递进式四产品)
  ① 通用知识库 → ② 多模型协商会议 → ③ AI 团队编码 → ④ AI 员工
```

设计文档:[docs/superpowers/specs/](docs/superpowers/specs/)

> Built in public. 每个里程碑配套一篇深度文章与可运行示例(examples/)。

## 进度

- [x] M1 骨架 + CI + Langfuse
- [x] M2 gateway:统一调用 / fallback / 缓存命中记账 → [examples/01-hello-gateway](examples/01-hello-gateway/)
- [ ] M3 obs · M4 eval · M5 proxy
