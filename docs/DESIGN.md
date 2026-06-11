# 📄 Repro Agent — ML 研究产物复现 Agent(秋招 · 最终版)

> **一句话:** 给定**一个官方代码仓库 + 一个目标指标 + 论文声称值**,Agent 自主**在大仓库里检索导航、在隔离环境里配好依赖、运行公开评测、确定性验证结果,并产出可重放的审计证据**。
>
> **项目重心(明确):** **RAG(大仓库检索导航)+ 上下文管理(长 debug 轨迹压缩 + 隔离)= 主角**;**多 Agent = 配角**(后期加,价值是隔离,诚实度量);**持久隔离执行 = 精瘦但必须稳的地基**(最先验证,但不镀金)。
>
> **正式定位:** ML Research Artifact Reproduction Agent——自动运行并验证 ML 论文官方仓库中**公开、轻量的评测结果**。**不声称复现整篇论文。**

> **范围纪律:** 只写承诺做、能跑出数字的内容。复现是公认难题——**成功率不会高,卖点是"诚实量化 Agent 能复现到哪一阶段、卡在哪、为什么"。** 哪怕最终只稳定复现 40%,只要评测可信,就是好项目。

---

## 一、核心叙事(面试 30 秒版)

**核心问题:Agent 能否在受控环境中,把一个陌生研究仓库的公开评测结果跑出来,并留下可审计证据?**

- **会找(RAG · 主角):** 官方 repo 上千文件塞不进上下文 → 检索导航定位 eval 入口 / checkpoint 加载;配完整检索基线阶梯证明它值多少。
- **会管(上下文 · 主角):** 一次复现几十轮"写码→跑→报错→修",轨迹会撑爆窗口 → 压缩/预算/隔离。
- **会修(自修复):** 依赖冲突、缺包、弃用 API——环境地狱是自修复最真实的舞台。
- **会拆(多 Agent · 配角):** Lead 拆多指标、Reproducer 各自隔离上下文、Verifier 独立核对;**收益是隔离而非速度/成功率,用 ablation 说话**。
- **会测:** 7 阶段复现率 + 失败 taxonomy + 按论文聚类显著性 + 时间切分 held-out 泛化。
- **会跑(地基):** shell/文件级 Agent + 持久隔离环境会话,按信任等级选后端。

差异化钩子:**别人的 Agent demo 在干净 toy 任务上;我的在"代码会腐烂、环境会爆炸"的真实仓库上,RAG 真用于大仓库导航、上下文真会爆,而且诚实量化它能复现到哪一步。**

---

## 二、它是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 跑通仓库里**已有的、公开的、轻量 eval** | 从零实现论文方法 |
| 主走**加载公开 checkpoint 跑 eval**(不训练) | 完整训练复现 SOTA |
| 单仓库、**单目标指标**、固定 commit | 一次复现整篇论文所有结果 |
| 确定性验证 + 可重放证据 | LLM 拍脑袋说"复现成功" |
| 度量"走到哪一阶段" | 只看最终成/败 |

---

## 三、架构总览

```
                    ┌─────────────── 主角 ───────────────┐
[manifest:repo+commit+claim]                              │
        │                                                 │
        ▼                                                 │
   仓库检查 → ★RAG 检索导航(定位 eval 入口/配置/权重加载)   │
        │                                                 │
        ▼                                                 │
   执行计划 → 隔离环境会话(持久)                            │
        │                                                 │
        ▼              ★上下文管理(长轨迹压缩 + 预算 + 隔离) │
   ReAct over shell/file 动作 + 分级自修复 ────────────────┘
        │
        ▼
   日志抽取指标 → 确定性验证(actual vs expected)→ 可重放审计报告

  ── 配角(M5 加):Lead 拆多指标 → 并行/串行 Reproducer(各自隔离上下文)→ Verifier ──
  ── 地基:持久隔离执行会话 + 按信任等级的可插拔后端(精瘦,最先验证)──
```

---

## 四、主角之一:RAG 仓库检索导航

官方 repo 上千文件塞不进上下文 → 检索导航找"eval 入口 / checkpoint 加载 / 配置"。**这是 RAG 在本项目里真核心、不 contrived 的体现**(私有 + 大)。

**消融基线必须是阶梯(不拿"全仓库塞进去"当稻草人):**

```
rg/find/tree 关键词 → README 优先导航 → BM25 → BM25+embedding → +reranker
```

报告每级的"入口定位成功率",量化检索到底值多少 pp。

> 边界:**一篇论文塞得进上下文,所以不对论文做 RAG**——RAG 只用于大仓库,懂边界。

---

## 五、主角之二:上下文管理

复现的上下文难点不是"文档太大",是**agentic 轨迹会爆**:

| 机制 | 说明 | 度量 |
|---|---|---|
| 长轨迹压缩 | 旧步骤摘要、丢过时输出、留关键状态(已装什么/在哪一步) | 关压缩掉 X pp / 省 token Y% |
| 上下文预算 | 给定预算,在"贴更多仓库证据 vs 留给推理"间分配 | token 会计 |
| 多 Agent 隔离 | 每个 Reproducer 只扛自己那摊脏 debug 轨迹,不互相污染 | 隔离 vs 不隔离消融 |
| 检索注入 | 只把"够用的"仓库片段进上下文,全量留库内引用 | — |

---

## 六、配角:多 Agent(M5 加,诚实度量)

**不作为早期核心。** 顺序:单 Reproducer 闭环跑稳 → 建可靠 benchmark → 再加 Lead(拆多指标)+ Verifier(独立核对)→ **最后用 ablation 验证它是否真有收益**。

诚实预期:单机上同时跑多个 eval 会内存打架、更慢 → 现实里多半**串行**。所以**多 Agent 的收益主要是上下文/职责隔离,不是速度或成功率;证不出就如实说**。

---

## 七、地基:执行层(精瘦但必须稳)

> **不本末倒置:执行层够用就行、最先验证、不镀金;深度全砸在 RAG/上下文上。** 但它必须稳——它一塌,后面全废。

### 7.1 按信任等级选后端(绝不裸跑)

| 仓库来源 | 后端 | 隔离 |
|---|---|---|
| 用户上传的任意 repo | **Docker / VM / 远程隔离** | 强(断网/资源限/只读宿主) |
| 人工审核过的 benchmark repo | 子进程 + MPS(快速迭代) | 中——但 `pip install` 仍拉未审计依赖,**默认仍走 Docker 更稳** |

### 7.2 持久会话(关键架构)

复现是 `clone → 装 → 跑 → 改 → 再跑`,**全程在同一个不断演化的工作目录/容器里**,Agent 往里发 `exec`——**不是每步起新空容器**。

### 7.3 网络分两阶段(改正"全程断网"的矛盾)

```
Provision 阶段:受限网络 → clone、装依赖、下 checkpoint/数据 → 校验 SHA256
  └─ 注意:pip install 会带网执行 setup.py 的任意代码,真正外传风险在这一步
     → provision 也在隔离容器内跑(碰不到宿主密钥/文件);权重尽量宿主下好再只读挂载
Execution 阶段:网络关闭(Docker `none`)→ 跑仓库代码和 eval
```

### 7.4 Docker 与 MPS 是两条路(benchmark 必须标后端)

Mac 上 Docker 容器**用不了 MPS**(Docker Desktop GPU 仅 Windows/WSL2+NVIDIA)。所以:`DockerBackend`(Mac)= CPU;`SubprocessBackend` = MPS 但只跑审核过的;真 GPU 隔离 = 远程 Linux+NVIDIA。**每个 benchmark 任务标注用哪个后端**,否则 15 分钟约束没法保证。

### 7.5 动作空间(地基的硬骨头)

`shell`(git/pip/python/ls/rg/cat)· `read_file` / `edit_file`(改弃用 API/路径)· `read_logs`(抓指标)· 持久状态(venv/依赖/数据跨步骤保留)。**这是和现有项目差最多的部分。**

---

## 八、自修复(环境地狱)

| 连续失败 | 升级 |
|---|---|
| 1 | 回喂 traceback,改了重试 |
| 2 | 检索 repo 的 README/requirements/issues 线索,据此修 |
| 3 | 换思路(换 torch 版本/换入口/跳过可选依赖) |

记录**失败类型 taxonomy**(依赖冲突/缺包/弃用 API/shape/OOM/数据缺失/无 eval 流程……)——项目最有价值的产出之一。

---

## 九、验证:确定性,不是又一个 LLM

LLM 只负责**抽取**(给证据),**最终比对由代码做**:

```yaml
claim:  { metric: top1_accuracy, dataset: CIFAR-10 test, expected: 95.50, tolerance: 0.30 }
result: { actual: 95.42, evidence: { command: "python eval.py --ckpt w.pth", log_line: "Acc@1 95.42" } }
verdict: { match: true, abs_diff: 0.08 }   # ← 确定性代码算出来的
```

---

## 十、评估体系(脊柱)

### 10.1 7 阶段复现率

```
1 repo_inspected → 2 entrypoint_located → 3 environment_ready →
4 evaluation_started → 5 evaluation_completed → 6 metric_extracted → 7 claim_matched
```

报告**每阶段通过率**。**样例/自带小数据最多到 `evaluation_completed`;只有 claim 对应的完整数据集才能进 `claim_matched`**(否则复现率失真)。

### 10.2 eligibility 与 outcome 分离(防止 Agent 失败后偷偷移出分母)

```yaml
eligibility: { supported: true|false, reason: ... }   # 跑前决定
outcome: success | agent_failure | artifact_blocked | unsupported
```
`unsupported`:跑前就不在支持范围,**不进分母** · `artifact_blocked`:跑后发现外部资源失效,单独报 · `agent_failure`:oracle 能跑但 Agent 没跑通 · **失败后不得临时移出分母**。

### 10.3 每次复现记录

改过哪些文件 · 执行了哪些命令 · 失败类型 · token/时间/成本 · **是否需要人工介入**。

### 10.4 统计纪律

- **按论文聚类 bootstrap**(同一论文多 claim 相关,当独立样本会虚高 n)。
- 小样本 → 主打**效应量 + 分阶段通过率 + 失败 taxonomy**,显著性点到为止、CI 跨 0 如实报告。

### 10.5 held-out = 时间切分(个人项目唯一可执行的版本)

```
1. 用 6 dev repo 开发 Agent
2. 冻结代码/Prompt/参数
3. 再选并人工验证 3 个新 repo(此时 Agent 已冻结,验证不污染)
4. 跑 Agent
5. 无论结果如何,不再改 Agent
```

---

## 十一、Benchmark 规格(6 dev + 3 held-out)

每个仓库**固定并人工核验**:

```yaml
- paper: <标题/链接>
  repo: <url>; commit_sha: <固定>
  backend: docker_cpu | subprocess_mps | remote_gpu      # 必标
  checkpoint: { url: <...>, sha256: <...> }
  dataset: { name: CIFAR-10 test, prep: <如何获取完整数据> }   # 验证 claim 必须用完整集
  target: { metric: top1_accuracy, expected: 95.5, tolerance: 0.3 }
  canonical_path: <人工验证过、能复现的标准命令序列>          # oracle
  eligibility: { supported: true, reason: null }
```

**选仓库倾向**:多选**带自定义 eval 流程的研究仓库**(Agent 价值才显出来),少量"HF 一行就能跑"的当冒烟。

---

## 十二、数据域:小图像分类 / 小 NLP 分类

| 域 | 数据/指标(小、可精确匹配) |
|---|---|
| **图像分类** | CIFAR-10/100 test(10k 张)、Fashion-MNIST;top-1 acc |
| **小 NLP 分类** | SST-2 / AG News / IMDB dev;accuracy / F1 |

避开:ImageNet 全量 val(太大/慢)、需训练才出数的、gated 数据。

---

## 十三、M1(去风险的第一钉,**完全离线**)

**先只验证最大风险:持久状态化执行闭环能不能跑通。** 不做下载/RAG/自动选环境。

只做:
- 一个人工验证过的真实 PyTorch repo,**已克隆、固定 commit**
- checkpoint + **完整数据**已缓存、校验哈希
- 一个**持久 Docker CPU 容器**(完全离线)
- 动作:`shell` / `read_file` / `edit_file`
- **≤12 个 Agent 步骤**;每命令超时 + 整任务 15 分钟超时
- 保存所有命令、输出、文件 diff
- 结构化抽取指标 + 确定性比对
- **挑一个"≤12 步够得着"的低摩擦 repo**(入口清晰、近乎能跑)

**验收:** 删容器重来,Agent **仅凭任务 manifest** 自主复现该指标,并输出**可重放的命令序列 + 证据**(无任何针对该 repo 的硬编码)。

> 这个闭环跑稳,再加自动下载 / 网络分阶段 / 第二个异构 repo。**M1 完全离线 → 顺带把外传风险归零。**

---

## 十四、技术栈

| 层 | 技术 |
|---|---|
| 语言 | Python 3.12 |
| 编排 | LangGraph(条件路由 + checkpointer + 子图) |
| LLM | LangChain + OpenAI 兼容(Qwen / DeepSeek) |
| 执行 | 持久会话:Docker(任意/CPU)/ 子进程+MPS(审核过)/ 远程(可选) |
| 检索 | ripgrep + BM25 + Qdrant/embedding + reranker(阶梯) |
| ML | PyTorch(MPS / CPU) |
| 评估 | 自建 harness(阶段状态 + 结构化证据)+ scipy/numpy(clustered bootstrap) |
| 前端 | Gradio / CLI(流式复现过程 + 审计报告) |
| 基建 | pytest · ruff · GitHub Actions(lint + 单测 + 1 repo 冒烟) |

---

## 十五、项目结构(新建,**不围绕现有 Analyst 改**)

```
repro-agent/
├── agent/
│   ├── actions.py        # shell / read_file / edit_file / read_logs
│   ├── session.py        # ★ 持久隔离环境会话(exec 进同一容器)
│   ├── loop.py           # ReAct + 分级自修复
│   └── plan.py           # 仓库检查 → 定位入口 → 执行计划
├── retrieval/            # ★主角:repo 检索导航(rg→BM25→embedding→reranker)
├── context/              # ★主角:长轨迹压缩 + 预算 + 隔离
├── exec/backends.py      # Docker / Subprocess / Remote(按信任等级,两阶段网络)
├── verify/check.py       # 结构化 claim/result + 确定性比对
├── agents/               # 配角(M5):lead / reproducer / verifier
├── evals/
│   ├── harness.py        # 7 阶段 + 结构化证据 + 失败 taxonomy
│   ├── significance.py   # clustered bootstrap(按论文)
│   └── benchmark/        # 6 dev + 3 held-out 的 yaml 规格
├── app.py
└── tests/
```

---

## 十六、对现有项目的复用(诚实:约 20%,其余重写)

**直接复用:** `core/llm.py` · `evals/significance.py`(作基座,需加 clustered 变体)· `ScriptedLLM` 零成本测试 · Docker sandbox **部分设计** · tiered-repair **思想**。
**需大幅重写:** `analyst.py`(→ shell/文件/持久状态)· `graph.py`(→ 真条件路由 + checkpoint)· `harness.py`(→ 阶段 + 结构化证据)· `memory/`+`kb/`(→ 仓库索引 + 代码检索)· 执行层(→ **持久工作目录**)。

> **进场预期:基本是个新项目(约 80% 重写)**,带走的是 LLM 接口、显著性思路、tiered-repair 思路和评估纪律。**别低估排期。**

---

## 十七、里程碑(倒排)

| 阶段 | 目标 | 验收 |
|---|---|---|
| **M1** | **执行闭环 de-risk(地基,精瘦)** | 完全离线、单 repo、shell/file 动作、持久容器、≤12 步、确定性验证、**可重放序列**。删容器重来只凭 manifest 复现。**可行性生死关。** |
| M2 | **自修复 + 网络分阶段 + benchmark 起步** | 分级自修复 + 自动下载校验 + 两阶段网络;3→ benchmark;失败 taxonomy |
| **M3** | **★RAG 仓库导航(主角)** | 检索阶梯(rg→README→BM25→+embedding→+reranker);入口定位消融带 CI;dev 扩到 6 |
| **M4** | **★上下文管理(主角)** | 长轨迹压缩 + 预算;消融出"省 token / 不掉精度" |
| M5 | **多 Agent(配角)+ 评估定稿** | Lead/Reproducer/Verifier + 隔离,**诚实测隔离收益**;10 dev + **时间切分 held-out** + clustered bootstrap |
| M6 | **demo + 收尾** | Gradio(喂 held-out repo)+ README(只写已实现+数字)+ 冻结 |

> **永不砍:** 持久隔离执行 + 确定性验证 + 分阶段评估 + M1 闭环 + RAG 导航 + 上下文管理。**多 Agent / 高级压缩由真实失败驱动着加。**

---

## 十八、简历写法(数字以实测填入)

> 自研 ML 研究产物复现 Agent:以 **LangGraph** 编排"检索导航→隔离执行→自修复→确定性验证"闭环。**RAG 仓库导航**(ripgrep+BM25+embedding+reranker 阶梯)定位陌生大仓库的评测入口,较 BM25 基线 +X pp(按论文聚类 bootstrap);**长 debug 轨迹的上下文压缩/预算**省 token Y%、不掉精度;**shell/文件级 Agent + 持久隔离环境会话**(按信任等级在 Docker 隔离与 MPS 性能间权衡)自主配环境跑 eval,分级自修复应对依赖/API 错误;**确定性验证**(LLM 抽取 + 代码比对 + 可重放证据)替代 LLM 自评;自建 **7 阶段复现率 + 失败 taxonomy** 评测,在 N 个异构仓库上分阶段通过率为…;多 Agent 上下文隔离层经 ablation 度量;held-out 仓库(时间切分)验证泛化。

---

## 十九、面试问答弹药

| 高频题 | 答案出处 |
|---|---|
| RAG 为什么是核心、又为什么不对论文用? | repo 大→必要;论文塞得下→不用;懂边界,配了基线阶梯 |
| context 爆了怎么办? | 长轨迹压缩 + 预算 + 多 Agent 隔离,有消融数字 |
| 跑任意 repo 安全吗? | 按信任分级;任意 repo 强制 Docker/VM;`pip install` 风险用两阶段网络兜 |
| 多 Agent 真有用吗? | 收益在隔离不在速度;ablation 决定,没用就老实说 |
| Verifier 会幻觉吗? | 抽取归 LLM、比对归代码、留可重放证据 |
| 固定几个 repo 还算 Agent 吗? | 固定的是评估集不是 Agent;零特判;时间切分 held-out 现场跑没见过的 |
| 复现率才 40% 是失败吗? | 不是——度量的就是可复现性;诚实量化比假装万能更可信 |
| 怎么证明好? | 7 阶段通过率 + 失败 taxonomy + 聚类 bootstrap + 泛化 |

---

## 二十、Roadmap(未实现,README 标注,简历不写)

训练级复现(GPU 服务器)· 跨仓库"复现踩坑"记忆库 · AST/调用图增强检索 · 自动写复现 PR 回上游 · 封装 MCP server · 接受用户任意 repo 的产品化(强隔离前提)

---

## 一句话总结

> **主角是 RAG 仓库导航 + 长轨迹上下文管理;多 Agent 是诚实度量的配角;shell/文件级 Agent 在持久隔离环境里跑通真实仓库的公开 eval,自修复扛环境地狱,确定性代码做验证,7 阶段+taxonomy+聚类 bootstrap+时间切分 held-out 做诚实度量。** 安全按信任分级、绝不裸跑;执行精瘦最先验证、深度全砸在 RAG 与上下文上。范围砍到"单 repo 单指标 ≤15 分钟",M1 先把离线闭环钉死——每个词背后都有一个跑出来的数字。
