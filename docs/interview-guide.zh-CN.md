# Repro-Agent 面试讲解

## 60 秒介绍

Repro-Agent 是一个面向机器学习代码仓库的盲测自动复现框架。任务作者通过
Manifest 声明仓库、资产、公开输出协议和私有评测配置；Agent 只能看到公开任务和
代码，不能看到 gold labels、目标结果或最终判卷逻辑。

系统先由 Router 根据仓库规模和指标语义选择 short、assisted 或 full 路径。
Reproducer 检索仓库并生成评测脚本，脚本在隔离运行环境中真实执行；失败后 Reviewer
结合日志和公开 diagnostics 定位问题，Repair 使用 patch-first 方式修改当前代码。
流程结束后，Private Verifier 根据逐样本输出和隐藏标签独立重算指标。

当前五个开发任务各运行五次，Verifier 通过 18/25；两个 held-out 仓库均达到 5/5
Verifier 通过。结果同时暴露了 OpenOOD 语义修复和结构化提交仍不稳定的问题。

## 为什么需要 Router

固定 full 流程会让所有任务都调用 Navigator、Critic、Reviewer 和 Repair。简单任务
不需要这些角色，额外调用只会增加成本、延迟和结构化提交失败的机会。

Router 只做一次风险规划，不写代码。它根据公开任务和低成本仓库特征决定初始路径：

- `short`：任务明确，Reproducer 直接生成并执行。
- `assisted`：仓库陌生，先由 Navigator 调研。
- `full`：存在分数方向、聚合、攻击配置等高语义风险，加入 Critic 和 Reviewer。

真实失败仍可让短流程逐步升级，因此 Router 只决定起点，不会永久限制后续能力。
DistilBERT 的正式 N=5 全部走 short，平均一次评测、成本约 ¥0.012，证明选择性编排
确实减少了简单任务的开销。

## 为什么 Verifier 必须独立

Agent 生成的代码既可能写错，也可能直接打印一个看似正确的 aggregate metric，所以
系统不相信 `eval.py` 打印的 accuracy。Agent 必须提交完整逐样本预测，Private
Verifier 再使用隐藏标签重算指标。

这里不是把模型推理运行两次：

1. `eval.py` 运行模型一次，生成 `predictions.json`。
2. Verifier 只读取预测和 gold labels，执行轻量的 accuracy、Spearman 或 AUROC 计算。

公开 Contract Check 只检查文件路径、结构、数量和不依赖 gold 的语义约束；Private
Verifier 才能访问 gold、expected 和 tolerance。任何文件缺失、数量不完整、字段异常
或指标超出容差都会 fail closed。

## 为什么 Repair 使用 patch-first

一次执行失败通常只说明局部调用、路径、参数或预处理有问题。每次重新生成完整文件
会丢掉已经验证过的工作代码，并引入新的错误。

Repair 因此优先提交精确的 `old -> new` 局部替换：

- `old` 必须来自当前文件并且只出现一次。
- 修改后重新执行代码和公开 Contract Check。
- 补丁过大、没有真实变化或没有解决确定性问题时会被拒绝。
- 只有补丁提交反复失败时，才使用完整文件作为 fallback。

循环在公开协议通过、达到最多五次真实评测，或流程发生不可恢复异常时停止。Private
Verifier 只在流程结束后运行，避免 Repair 根据隐藏目标反向调参。

## OpenOOD 为什么失败

OpenOOD 不是普通分类准确率任务。它需要三个 checkpoint、ID 与两个 OOD 数据集，
并要求 OOD 分数高于 ID；Verifier 先分别计算 AUROC，再按数据集和 checkpoint 分层
平均。代码能够运行和输出 50,379 个分数，不代表方向与聚合语义正确。

当前 N=5 只有 1/5 通过：

- 两次输出完整，但分别得到 87.0916 和 87.5113，超出 87.58 ± 0.05 的容差。
- 一次正确得到 87.5823。
- 一次 Repair 没有提交合法结构化结果，最终没有可重算 artifact。
- 一次得到 12.4177，几乎等于 `100 - 87.5823`，明确说明分数方向被反转。

这说明 Router 能识别 score-direction 风险，但 Reviewer/Repair 还不能稳定地把风险证明
转化为正确补丁。它是当前系统的真实边界：执行修复能力较强，复杂指标的语义修复仍
不稳定。面试时不要把它包装成高成功率系统，应强调独立 Verifier 正是为了让这种
“能运行但语义错误”的结果无法蒙混通过。

## 常见追问

### 为什么多 Agent 不用一个大 Prompt

不同角色拥有独立上下文和明确交付物，Navigator 负责仓库证据，Reproducer 负责代码，
Reviewer 负责执行后诊断，Repair 负责最小修改。分工提高困难任务的能力上限，但也会
增加成本和结构化交付失败，所以系统没有让所有任务固定走 full。

### LangGraph State 为什么不保存完整代码和日志

State 只保存决定下一条边的轻量元数据，例如执行次数、协作级别和报告路径。完整代码、
日志和报告保存在 workdir，避免 State 过大，同时保留可以审计和回放的磁盘证据。

### Manifest 和 Hook 的边界

Manifest 声明仓库、资产、输出协议、执行方式和验证参数。通用框架自动生成
`OracleConfig`。只有动态脱敏、特殊 provisioning 或自定义 verifier 无法声明时才写
轻量 Hook；新增普通任务不修改 Agent 编排代码。

### 项目最大的不足

它主要解决已有仓库、数据和 checkpoint 条件下的评测复现，不是从论文开始训练模型的
通用科研 Agent。开发任务成功率为 72%，N=5 规模也不足以作为统计显著的研究结论。
项目价值主要在盲测协议、执行修复、独立验证和可审计工程，而不是新的模型算法。
