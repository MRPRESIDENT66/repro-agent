# Repro-Agent 源码阅读入口

不要从头读完整仓库。先沿着一次真实运行往下看。

## 一次运行怎么走

```text
run_xxx.py
  -> OracleConfig：准备任务、工作目录、执行函数和隐藏判卷函数
  -> run_oracle()：启动流水线，最后调用独立 verifier
  -> ReproductionPipeline.run()：运行 LangGraph
  -> _node_navigate()：Navigator 查仓库并提交报告
  -> _node_reproduce()：Reproducer 查仓库并生成 eval.py
  -> _node_critique()：Critic 检查并改写 eval.py
  -> _node_execute()：系统真正执行 eval.py
       -> _review()：full 模式下，Reviewer 检查代码和执行结果
  -> _node_repair()：失败时，Repair 修改并再次执行
  -> _decide()：通过、禁止修复或耗尽预算时结束
  -> verify_run()：使用隐藏目标和 gold 独立判卷
```

`solo` 只走 Reproducer 和一次 Execute。

`solo-repair` 在失败后进入 Repair，但没有 Navigator、Critic、Reviewer。

`full` 使用全部角色，并在每次执行后调用 Reviewer。

## 建议阅读顺序

1. `run_distilbert_multi_rag.py`：最短入口。
2. `evals/oracles/distilbert_sst2.py`：一个任务如何生成 `OracleConfig`。
3. `agent/pipeline.py`：只看 `_node_*`、`_decide()` 和 `_build_graph()`。
4. `agent/roles.py`：看 `_RoleTools` 的三个 LLM 工具方法。
5. `agent/loop.py`：看 LLM 如何逐轮选择并调用一个工具。
6. `agent/repair.py`：看 patch-first 如何校验和应用补丁。
7. `verify/check.py`：最后再看独立验证器。

`agent/artifacts.py` 只负责保存结果和日志，第一次学习可以跳过。

## 角色工具只看这三个方法

```text
_RoleTools.search_repository()  搜索仓库并返回相关代码片段
_RoleTools.probe_runtime()       受限检查包、函数参数、路径或 CLI
_RoleTools.submit_artifact()     提交报告、代码、审查或补丁
```

`run_rag_role()` 负责把这三个方法绑定给当前 LLM。

`run_agent()` 只做一个循环：调用 LLM、执行一个工具、把工具结果放回消息，直到提交成功或耗尽步数。

## 需要认识的 Python 语法

- `self.xxx`：当前对象保存的数据或方法。
- `@dataclass`：自动生成初始化函数的数据类。
- `TypedDict`：给字典标注固定字段；LangGraph State 仍然是普通字典。
- `Callable[[str], str]`：接收字符串并返回字符串的函数。
- `str | None`：值可以是字符串，也可以没有值。
- 函数参数中的单独 `*`：后面的参数必须写名字，避免传错位置。
- `@property`：调用时像字段，例如 `policy.full_team`，实际会执行一个方法。
- `make_xxx_validator()`：先根据当前任务生成一个校验函数，后面重复使用。

## 第一次学习先忽略

- token 和成本统计；
- transcript、trace、replay 文件保存；
- BM25 排序细节；
- Oracle 内具体数据集和 checkpoint 下载逻辑；
- `ScriptedLLM` 测试替身。

先能口述“配置任务 -> 角色查仓库和交付 -> 系统执行 -> 失败修复 -> verifier 独立判卷”，再进入这些细节。
