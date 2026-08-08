# Repro-Agent 源码阅读入口

现在项目只有一条运行路径，不需要先理解历史 `solo/full` 实验代码。

## 一次运行怎么走

```text
run_xxx.py
  -> catalog.make_config(task, attempt)
  -> YAML Manifest + 可选 Hook
  -> OracleConfig
  -> run_oracle()
  -> Router
       -> 低风险：Reproducer -> Execute
       -> 陌生任务：Navigator -> Reproducer -> Execute
       -> 语义风险：Navigator -> Reproducer -> Critic -> Execute -> Reviewer
  -> 第一次失败：补充 Navigator -> Reviewer -> Repair -> Execute
  -> 重复失败：升级 full -> Critic -> Execute -> Reviewer -> Repair
  -> verify_run()：最后使用私有 gold 判卷一次
```

## 建议阅读顺序

1. `evals/tasks/distilbert_sst2.yaml`：最简单的标量预测任务。
2. `evals/tasks/openood_ebo.yaml`：不用 hook 声明复杂资源、双后端和分组指标。
3. `evals/catalog.py`：任务名怎样映射到 manifest 和 hook。
4. `evals/manifest/`：怎样解析任务、准备 workspace 并生成 `OracleConfig`。
5. `agent/pipeline.py`：只看 `_node_*`、`_decide()`、`_build_graph()`。
6. `agent/orchestration/roles.py`：LLM 可以调用哪些工具。
7. `agent/orchestration/repair.py`：patch-first 怎样应用补丁。
8. `verify/check.py`：为什么最终判卷与 Agent 隔离。

## 三类 LLM 工具

```text
search_repo       搜索文件、符号和相关代码片段
runtime_probe     检查 import、函数参数、路径或 CLI
submit_xxx        提交路线、handoff、代码、审查或 patch
```

`run_rag_role()` 把合适的工具绑定给当前角色；`run_agent()` 循环执行 LLM
选择的工具，直到角色提交结果或耗尽预算。

## State 和磁盘

LangGraph State 是普通字典，保存 `short / assisted / full` 协作级别、轮次、
失败类型、是否通过公开 contract，以及报告路径。完整 `eval.py`、执行日志、
Navigator handoff 和 Reviewer 报告保存在 workdir，避免把大文本在每个节点
之间重复复制。

## 第一次先忽略

- token 和成本统计；
- transcript、RAG trace、命令 replay；
- BM25 排序细节；
- 各 ML 仓库的 checkpoint 下载过程；
- `ScriptedLLM` 测试替身。

先能口述“manifest 出题 -> Router 选择起点 -> 真实执行 -> 失败逐级增加角色
-> 私有 verifier 判卷”，再深入每个模块。
