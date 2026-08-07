# Repro-Agent

[English](README.md) | [中文](README.zh-CN.md)

Repro-Agent 是一个**研究原型(research prototype)**:面向代码仓库任务的**盲测多智能体复现基准 + 运行时(blind multi-agent reproduction benchmark + runtime)**。一队角色分工的 LLM 智能体在盲测条件下复现已发布的 ML 结果:用 **native tool calling** 自主读 repo、写评测脚本、执行、**从真实执行失败中自我修复(self-correction)**,最终由一个**独立、fail-closed 的评测器(evaluation harness)**从逐样本 artifact 重算指标——智能体全程看不到目标数字。

**定位要诚实**:这是 6 项开发任务加 1 个冻结后 held-out 仓库上的 N=5 **原型规模证据**,不是经过大规模验证的通用 runtime。详见 [项目边界](#项目边界)。

编排用 **LangGraph**(一个由角色节点组成的 `StateGraph`,带条件式修复循环);检索、失败分类修复、隔离执行、盲测验证器都直接实现在 provider-agnostic 的 OpenAI 兼容 API 上。部分工具能力还通过 **MCP** 独立暴露给外部客户端。

| 核心证据 | 结果 |
|---|---:|
| 6 项任务的 fresh full-pipeline coverage | **30 次运行中 27 次通过独立验证** |
| 4 任务消融：`solo` → `solo-repair` → `full` | **7/20 → 14/20 → 17/20 通过验证** |
| 最难任务 OpenOOD：`solo` → `full` | **0/5 → 4/5 通过验证** |
| 冻结后 held-out 仓库：Sentence-Transformers STS-B | **5/5 通过，5/5 无流程异常** |

![Architecture: blind inputs feed a generic role pipeline that emits per-sample predictions, which an independent verifier recomputes against pinned gold labels.](docs/architecture.svg)

## 本项目展示的能力(技术点对照)

| 能力 | 在本项目里是什么 | 位置 |
|---|---|---|
| **Multi-agent orchestration(LangGraph 编排)** | LangGraph `StateGraph`,角色节点(Navigator→Reproducer→Critic→执行→Reviewer→Repair)+ 条件式修复循环 + per-role 上下文隔离 | [`agent/pipeline.py`](agent/pipeline.py) |
| **Tool use / function calling(工具调用)** | 原生 OpenAI function-calling 的 agent loop、顺序工具派发 | [`agent/loop.py`](agent/loop.py) |
| **Tool interoperability(MCP)** | 仓库检索和受限诊断/命令接口通过 Model Context Protocol 独立暴露 | [`mcp_server.py`](mcp_server.py) |
| **Self-correction(Reflexion 式自我修复)** | 失败分类驱动、execution-grounded 的修复闭环,patch-first 优先于盲目重写 | [`agent/repair.py`](agent/repair.py)、[`agent/failure.py`](agent/failure.py) |
| **RAG / retrieval(检索增强)** | 面向代码仓库的检索:BM25 + 路径/符号信号 + LLM rerank + 动态 query rewriting | [`retrieval/`](retrieval/) |
| **LLM evaluation & guardrails(评测与护栏)** | 盲测、fail-closed 验证器从逐样本 artifact 重算指标,隐藏目标/gold 不进入 Agent 上下文 | [`verify/`](verify/) |
| **Isolated execution(隔离执行)** | subprocess 会话 + 可选的资源受限 Docker 会话和断网 | [`exec/`](exec/) |
| **Observability(可观测)** | per-call token + 成本核算、全链路 transcript、可复算 `commands.sh` | [`agent/llm.py`](agent/llm.py) |
| **Evaluation methodology(评测方法)** | budget-fair 重复运行消融、平均成本、失败模式拆解 | [`evals/`](evals/) |
| **Deterministic agent testing(确定性测试)** | `ScriptedLLM` 零 API/token 驱动整条控制流,快速可复现 | [`tests/`](tests/) |

技术栈:Python、**LangGraph**、**MCP**(Model Context Protocol)、OpenAI 兼容 function calling(provider-agnostic,可跑 DeepSeek/任意 OpenAI 风格端点)、BM25 检索、Docker、`pytest`。

## 为什么不是普通 Agent Demo

很多“AI 复现论文”的 demo 最后只输出一个 aggregate 数字，例如 `accuracy=94.82%`。这个数字本身没有证明力：Agent 可能硬编码、抄 README、只跑部分样本，或者根本没做真实评测。

这个项目重点解决两个问题：

1. **可信性问题**：Agent 不能只汇报分数，必须输出逐样本预测/分数等 artifact；verifier 用隐藏 gold labels 或确定性规则自己重算指标。
2. **泛化问题**：不同仓库的入口、数据加载、checkpoint、preprocessing、metric 都不一样；如果每个任务都手写提示词，本质是在做数据录入，不是在做 Agent 系统。

## 核心设计

### 1. 盲测 + verifier 重算

Agent 从不看到目标值。它必须生成公开协议要求的结果文件，例如 `predictions.json`。Verifier 读取自己的隐藏标签或验收规则，重新计算指标。

因此以下情况都会 fail-closed：

- 只打印 aggregate 数字；
- 样本数不对；
- artifact 格式错误；
- 只硬编码 aggregate 数字或提交不可重算结果；
- 指标可重算但超过容差。

**当前所有任务都走重算路径(`recompute_fn`)**:判定结果是从逐样本输出对隐藏 gold 现算出来的。旧的 provenance 启发式(只判断"代码看起来像不像评测",可被 dead-code block 伪造)只作为未迁移任务的 fallback 保留,**本基准里没有任何任务用它**。

### 2. 通用角色提示词 + 公共任务规格

Agent 的角色 prompt、RAG、执行、修复逻辑是通用的。每个任务只定义：

- 公开任务描述；
- 公开执行命令；
- 公开 artifact contract；
- workspace 如何准备；
- verifier 如何隐藏地判分。

也就是说，任务规格只负责出题和阅卷，不给 Agent 注入解题提示。

## 系统架构

一次 run 是一个多角色 pipeline。每个角色都有独立 LLM 上下文和工具预算，避免一个角色的错误推理污染后续角色。

| 角色 | 职责 |
|---|---|
| **Navigator** | 阅读仓库，产出基于证据的执行计划：入口、数据/模型资产、metric 语义、未决问题。 |
| **Reproducer** | 根据公开任务、Navigator handoff 和检索到的源码，生成完整评测程序。 |
| **Critic** | 执行前审查代码是否符合仓库证据和 artifact contract。 |
| **execute** | 在真实 subprocess / Docker 环境中运行生成的评测脚本。 |
| **Reviewer** | 根据执行日志和 verifier 的公开诊断做 post-execution 审查。 |
| **Repair ×N** | 根据真实错误日志、失败分类和 public diagnostics 做 patch-first 修复并重跑。 |

### RAG repo navigation

这里的 RAG 不是默认依赖向量库，而是面向代码仓库的检索增强：

- BM25 lexical search；
- exact path / symbol signals；
- LLM rerank；
- query-centered source snippets。

Dense embedding 不是默认路径必须项。

### Tool use

系统提供受控工具：

- `search_repo`：在 workspace 源码中检索相关文件和片段；
- `runtime_probe`：受限运行时探针，用于 import smoke、函数签名、路径列表、CLI help；
- shell / Docker session：执行生成的评测脚本；
- verifier：从 artifact 中重算指标。

`runtime_probe` 是软建议，不是强制门槛。Failure classifier 可以建议 probe，但当源码证据足够时，Repair 可以直接提交。

### Self-correction:Failure classifier + patch-first repair

执行失败后，系统会先根据执行日志和 verifier diagnostics 分类：

- `import_error`
- `api_mismatch`
- `missing_path`
- `missing_artifact`
- `malformed_artifact`
- `semantic_mismatch`
- `timeout`
- `workflow_error`

分类结果会进入 Repair 上下文，指导下一步是 search、probe、patch 还是 full-file fallback。Repair 默认先提交精确 old/new patch，避免每轮全文件重写破坏已正确的代码；patch 不可用时再 fallback 到完整文件重写。

## Observability(可观测)

每次 LLM 调用都累计 token 用量和成本(含 cache-hit 计价),一次 run 的成本就是两次快照之差。每次 run 还会产出完整的逐角色 transcript、RAG/probe trace,以及可复算的 `commands.sh`,使任何判定结果事后都可审计、可复现。

## MCP Server

部分能力通过 **Model Context Protocol** 独立暴露在 [`mcp_server.py`](mcp_server.py)：外部 MCP 客户端可以调用仓库检索、受限 probe 和临时目录命令执行。主 Pipeline 仍使用原生 Function Calling 和任务专用的 Session/Docker 后端，并不通过 MCP 调度内部工具。

```bash
python mcp_server.py   # stdio 传输
```

> MCP 命令工具使用临时工作目录和 subprocess，并不是 Pipeline 的可选 Docker 后端；两者都不是抵御恶意代码的强安全边界。

## 实验结果

当前实验摘要放在 [evals/RESULTS.md](evals/RESULTS.md)。fresh evaluation
共报告 75 次 DeepSeek V4 Flash 运行：70 次开发任务实验，外加冻结后
Sentence-Transformers held-out full-pipeline N=5。另有一批5次的holdout pilot
因新Oracle误拒绝合法的CLI输出路径代码而明确排除，不计入正式结果。

## Pipeline Conditions

所有条件使用同一套通用 role prompts 和相同执行预算（1 次初始执行 + 最多 4 次后续执行），区别只在编排深度和是否使用执行反馈。

- `solo`：只有 Reproducer，一次执行。
- `solo-repair`：Reproducer + Repair，根据真实执行错误修复，最多 5 次执行。
- `full`：Navigator + Reproducer + Critic + Reviewer + Repair，最多 5 次执行。

这个版本更适合作为项目展示：对比“一次生成”“执行反馈修复”“完整多角色协作”，避免过多消融条件让代码和讲解变复杂。

4 任务汇总结果为：`solo` 7/20、`solo-repair` 14/20、`full` 17/20
通过独立 verifier。在各自通过 verifier 的样本中，无流程异常的比例分别为
`solo` 7/7、`solo-repair` 14/14、`full` 12/17（70.6%）。这说明角色分工
提高了解题上限，但 Reviewer/handoff 也引入了额外流程失败面。

## 目录结构

- `agent/pipeline.py`：顶层编排状态机和执行/修复循环。
- `agent/contracts.py`：公共任务上下文和通用 code/report/review 校验。
- `agent/types.py`：task/runtime 共享配置类型。
- `agent/repair.py`：patch-first 修复和修复校验。
- `agent/diagnostics.py`：通用 public-contract diagnostics。
- `agent/runtime_probe.py`：受限 import/signature/path/CLI probe。
- `agent/generic_prompts.py`：任务无关的角色提示词。
- `agent/failure.py`：基于执行日志和 verifier diagnostics 的失败分类器。
- `retrieval/`：代码仓库检索和 snippet 提取。
- `exec/`：subprocess / Docker 执行会话。
- `verify/`：确定性验收和指标重算。
- `evals/oracles/`：每个任务的出题/阅卷配置。
- `run_*_multi_rag.py`：任务 runner，`PIPELINE` 选择实验条件。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

配置聊天模型：

```bash
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=deepseek-v4-flash
LLM_THINKING=disabled
```

部分任务需要本地预置模型、数据集缓存或 Docker 镜像，具体见对应的 `evals/oracles/` 文件。

## 运行

```bash
python run_distilbert_multi_rag.py
PIPELINE=solo-repair python run_openood_multi_rag.py
PIPELINE=full python run_robustbench_multi_rag.py
```

准备并运行冻结后的 held-out Sentence-Transformers 任务：

```bash
.venv-oracle/bin/python scripts/prepare_sentence_transformers_stsb.py
.venv/bin/python scripts/run_holdout_n5.py --batch holdout_v2_n5
```

OpenOOD 默认使用断网的 Docker/CPU 后端。Apple Silicon 可以对可信、人工检查过
的仓库启用更快的宿主机 MPS 后端:

```bash
.venv-oracle/bin/pip install --target repos/OpenOOD/.mps-site numpy==1.26.4
OPENOOD_EXECUTION_BACKEND=mps PIPELINE=solo-repair python run_openood_multi_rag.py
```

MPS 模式仍保留盲测工作区、无密钥子进程环境、完整样本契约和独立 verifier，
但不具备 Docker 的容器与断网隔离。新实验会在结果文件和 manifest 中记录后端。

测试——单元测试不依赖 LLM/Docker/网络,约 1 秒跑完:

```bash
pytest                 # 快速单元测试(integration 测试默认跳过)
pytest -m integration  # 依赖 Docker daemon 的测试
```

## 项目边界

把话说在前面,免得 claim 超过证据:

- **原型规模的评测。** 主消融覆盖 4 任务 × 3 条件 × N=5，full-pipeline coverage 覆盖 6 任务 × N=5；另有 1 个冻结后 held-out 仓库 N=5，但单个仓库还不能代表完整 held-out 集合，也没有置信区间。这些数字是**原型证据**,不是基准定论。
- **通用性在智能体层,不是端到端。** 一套任务无关的 agent 跨 6 个 ML 代码生态,但**每个新任务都要手写一个评测适配器**(任务描述 + 执行命令 + 样本契约 + 隐藏 gold + 工作区准备)。这不是"任意 repo 零配置复现"。
- **失败分类器是规则式的。** 它是对 stdout/stderr/diagnostics 的 **execution-grounded 正则/规则**分类器,负责给 LLM 拼修复上下文——**不是"智能"自动诊断**。推理在 Repair 智能体里。
- **检索未做规模化优化。** 每次检索都现扫仓库(`load_corpus` 遍历目录树),**无缓存、无增量索引**;超大仓库要先做工程优化才能上生产。
- **是隔离,不是安全沙箱。** 这是面向协作式 agent 的实验完整性运行时。Verifier 拒绝不可验证输出,目标和 gold 按设计不进入 Agent 上下文；执行在隔离工作目录中运行,可选 Docker + 断网。宿主机 MPS 更快但隔离更弱,两种模式都**不保证抵御恶意代码**。

更准确地说,本项目的 claim 不是"任意 repo 零配置自动复现",而是:

> 给定公开任务、运行命令、artifact contract 和隐藏 verifier 资产后,同一套 generic multi-agent runtime 能自动读 repo、写评测脚本、执行、修复,并由 verifier 做 fail-closed 验收。
