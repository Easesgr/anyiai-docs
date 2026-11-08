---
title: 如何把复杂工程任务封装成 Agent Skill？辅助脚本、状态管理与人机确认机制
icon: mdi:package-variant-closed
tag:
- Agent Skills
- 任务封装
- 工作流
- 稳定性
description: 讲解如何将多步骤工程任务封装为Agent Skill，通过确定性脚本辅助模型决策，规范绝对路径与自解释报错，并由宿主实现人机确认与双层验收。
---

在日常开发与运维中，如果直接让大模型仅凭自然语言提示词去完成一项复杂任务——例如“拉取生产节点的日志压缩包、解压清洗、聚合错误率并生成报表，最后清理临时目录”，往往会遇到一系列典型问题：大模型写出的 Shell 单行管道命令容易因正则转义失败；在批量读取大文件时容易造成上下文溢出；甚至可能在切换当前目录后误删重要文件。

解决这些问题的有效思路，是把复杂任务规范化封装为一个包含**使用说明（SKILL.md）**、**本地辅助脚本（scripts/）**与**参考文档（references/）**的 Skill。模型负责理解动态需求与决策分支，本地脚本负责确定性的文件处理与数值计算，宿主程序负责安全拦截与权限隔离。

在上一篇[《一个可复用的 Agent Skill 应该包含哪些内容？》](/frontier/agent-skills/reusable-skill.md)中，我们梳理了基础 Skill 的元数据与边界格式。本文以**多服务日志审计与数据归档**为贯穿案例，详解如何将多步骤复杂工程任务封装为健壮的 Skill，协调模型动态决策与本地确定性脚本，建立路径防漂移、自解释报错与人机协同确认机制。

![复杂工程任务中 Agent、SKILL.md、辅助脚本与运行环境协同的工作台示意图](/images/content-rebuild/b6112f0e597bb5c8/cover.png)

---

## 任务拆解：模型动态决策与确定性辅助脚本的分工

封装复杂任务的第一步，是厘清**模型推理**与**本地程序代码**各自适合处理的工作。根据 Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 中的建议，高效的智能体系统往往把预定义代码路径（Workflows）与模型自主决策（Agents）结合起来，而不是让大模型从头推导所有计算与数据遍历。

![复杂任务中大模型动态决策与本地确定性脚本的职责划分拓扑图](/images/content-rebuild/b6112f0e597bb5c8/fig-1.png)

*图 1 解释：左侧代表模型的主观决策域，负责语义理解与参数组装；右侧代表宿主机的确定性计算域，负责 I/O、正则与数值计算。两者通过结构化标准接口双向交互。*

在日志审计与归档的任务中，两者的分工界限非常清晰：

1. **下沉到辅助脚本的确定性工作（Deterministic Steps）**：
   - **批量与大文件 I/O**：遍历成百上千个日志文件、解压 `.tar.gz` 压缩包、按行扫描文本。如果将原始日志全部作为上下文喂给模型，不仅消耗大量 Token，还容易因上下文窗口截断而丢失关键信息。
   - **精确数值与统计计算**：如统计各服务的 QPS、计算错误率分布、纳秒时间戳对齐以及计算 SHA-256 校验和。这类操作交给 Python 或通用脚本，结果确定且执行速度远快于模型推导。
   - **格式规范校验与回滚**：校验数据是否符合预设字段模式（Schema），执行事务性文件重命名或临时目录清理。

2. **交由大模型承担的动态决策工作（Agentic Steps）**：
   - **自然语言意图理解与策略选择**：分析用户非标准的诉求（例如“帮我看看昨晚支付模块为什么告警激增”），判断需要审计哪几个服务的日志区间。
   - **参数提取与脚本调用组装**：从用户指令或上一轮执行结果中，提取出日期、服务名、阈值等参数，拼装成脚本所需的调用命令。
   - **异常归因与纠错分支**：当辅助脚本返回非零状态码或格式警告时，模型根据脚本输出的诊断信息判断原因，决定是调整参数重试，还是查阅参考文档。

这种分工让模型聚焦于它擅长的语义理解与动态编排，而把稳定性要求极高的读写与数值计算留给可重复运行的代码。

<ReadMoreLock />

---

## 目录组织与渐进披露：管理 scripts/ 与 references/ 资源

按照 [Agent Skills 规范（agentskills.io/specification）](https://agentskills.io/specification)，一个标准的工程级 Skill 采用目录形式交付，并通过**渐进披露（Progressive Disclosure）**机制降低初始上下文占用。

渐进披露的核心在于分层：模型在刚启动时不必加载所有说明文档与辅助代码，而是根据需要分步拉取。

![Agent Skill 目录组织规范与渐进披露加载阶段流向图](/images/content-rebuild/b6112f0e597bb5c8/fig-2.png)

*图 2 解释：展示了冷启动（仅加载元数据）、Skill 激活（加载主指引）、任务推进（按需加载参考文档或执行本地脚本）的三层递进机制。*

以我们的日志审计任务为例，典型的 Skill 目录结构如下：

```text
log-audit-expert/
├── SKILL.md                  # 主说明文件：包含元数据、步骤指引与报错恢复规则
├── scripts/                  # 辅助执行脚本：由模型通过命令行或工具调用运行
│   ├── parse_logs.py         # 高性能日志解析与指标计算
│   └── generate_report.py    # 报表生成与哈希校验
├── references/               # 按需查阅的静态文档
│   ├── error_codes.md        # 常见业务错误码与排查指引
│   └── compliance_rules.md   # 数据审计合规要求与指标口径
└── assets/                   # 模板与静态资产
    └── report_template.html  # 审计报告 HTML 模板
```

在 `SKILL.md` 中，通过标准的 YAML Frontmatter 声明元数据：

```yaml
---
name: log-audit-expert
description: 分析并审计系统运行日志，排查错误码并生成合规审计报告。当用户需要处理原始日志文件、定位故障或输出归档报表时使用。
compatibility: Requires python 3.10+, pip packages: pandas, jinja2
allowed-tools: Bash(python:*) Read
metadata:
  author: platform-team
  version: "1.2.0"
---
```

### 渐进披露的三层加载过程

1. **系统初始化（元数据阶段）**：宿主环境仅读取各 Skill 的 `name` 与 `description`（通常在 100 Token 左右），注入系统提示词。大模型借此获知当前有哪些技能可用，而不会消耗多余上下文。
2. **技能激活（主指引阶段）**：当用户提出日志审计需求，模型识别到该 Skill 的描述最契合，触发激活。此时宿主将 `SKILL.md` 的正文注入当前上下文。规范建议将 `SKILL.md` 正文控制在 500 行以内，集中说明核心执行步骤与容错规则。
3. **按需查阅与运行（资源阶段）**：诸如数千行的业务错误码对照表（`references/error_codes.md`），不在激活时全量加载。而是在 `SKILL.md` 中说明：“若遇到未定义的业务错误码，读取 `references/error_codes.md` 进行映射”。模型在实际遇到该情况时，再调用文件读取工具加载该文档；运行数据时，则调用 `scripts/parse_logs.py`。

需要说明的是，渐进披露能够有效减少初始上下文消耗，但并不能保证模型永远不会超出上下文限制，也不能保证模型在每种情况下都能 100% 选对参考文件。它依赖于模型遵循指令的能力以及提示词中相对路径引导的清晰度。

关于 Skill 机制与 MCP 协议的边界分工，可以进一步查阅[《MCP 和 Skills 有什么区别？分别适用于什么场景？》](/interview/agent-interview/mcp-vs-skills.md)。

---

## 状态与环境对齐：工作区绝对路径与运行依赖隔离

在多步工程任务中，Agent 调用外部脚本最常见的隐蔽故障是**当前工作目录漂移（CWD Drift）**。

![相对路径产生目录漂移与绝对路径对齐工作区上下文的对比示意图](/images/content-rebuild/b6112f0e597bb5c8/fig-3.png)

*图 3 解释：左侧演示模型在前置命令中切换了子目录，导致后续相对路径失效；右侧通过统一挂载点解析为标准绝对路径，确保无歧义寻址。*

### 为什么辅助脚本必须强制规范化绝对路径？

当 Agent 拥有类似 `Bash` 的终端执行权限时，若在步骤 1 执行了 `cd /workspace/project/src`，终端会话的当前工作目录就已经改变。如果在随后的步骤 2 中，模型按照默认习惯执行相对路径调用：

```bash
python scripts/parse_logs.py --input ./raw_data.log
```

此时系统会在 `/workspace/project/src/scripts/` 下寻找脚本，并在 `/workspace/project/src/` 下寻找日志文件，随即触发 `FileNotFoundError`。
更危险的是写操作：如果模型使用相对路径写入 `./output.json`，生成的文件将被写在子目录下，导致后续验证逻辑找不到产物，或者意外覆写子目录中的同名文件。

因此，在 `SKILL.md` 中必须写入明确的**路径规范要求**：

```markdown
### 路径使用规范
1. 所有提供给脚本的文件路径，必须通过环境变量 `$WORKSPACE_ROOT` 拼接为绝对路径。
2. 辅助脚本本身的调用必须使用绝对路径，例如：
   python "${WORKSPACE_ROOT}/skills/log-audit-expert/scripts/parse_logs.py"
3. 严禁假定终端当前工作目录处于工作区根目录。
```

### 运行时依赖声明与环境隔离

大模型无法感知宿主机是否安装了特定版本的 Python 库。如果脚本依赖第三方包（如 `pandas`），在设计时应采取以下措施：
- 在 `compatibility` 字段显式声明运行环境要求（例如 `Requires python 3.10+, pandas`）。
- 辅助脚本内部编写前置依赖自检逻辑，在缺少模块时输出明确的退出提示。
- 在宿主环境中，推荐通过轻量虚拟环境隔离运行工具（例如使用 `uv run` 执行脚本），避免全局依赖冲突。

---

## 失败处理与自纠错：结构化自解释报错与重试控制

在真实环境中，脚本执行可能遇到网络抖动、文件权限缺失、格式不匹配或磁盘满载等问题。一个实用的 Skill 不应直接把未经处理的 Python 原始栈轨迹（Traceback）甩给模型，因为长串的系统调用栈不仅消耗上下文，还容易干扰模型的语义判断。

![辅助脚本自解释错误捕获与模型自纠错重试的闭环流程图](/images/content-rebuild/b6112f0e597bb5c8/fig-4.png)

*图 4 解释：展示脚本抛出结构化自解释错误后，模型捕获字段、归因问题、修改入参并重新执行，最终达到退出码 0 或熔断重试的闭环。*

### 编写面向模型的自解释脚本

辅助脚本应当在内部捕获可预期的异常，并将错误以**结构化 JSON** 形式输出到标准错误流（stderr）。这包含三个核心字段：
- `error_code`：明确的错误枚举码。
- `message`：对发生情况的简明描述。
- `suggestion`：给调用者（模型）的针对性修改建议。

下面是一个具体的日志解析脚本范例：

```python
# scripts/parse_logs.py
import sys
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--max-entries", type=int, default=1000)
    args = parser.parse_args()

    target_path = Path(args.input_file).resolve()

    # 1. 检查文件是否存在
    if not target_path.exists():
        err_payload = {
            "status": "FAILED",
            "error_code": "ERR_FILE_NOT_FOUND",
            "message": f"Target log file not found at: {target_path}",
            "suggestion": "Please check the absolute file path, or search the directory for alternate .log files."
        }
        print(json.dumps(err_payload, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    # 2. 检查编码格式并读取
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        err_payload = {
            "status": "FAILED",
            "error_code": "ERR_ENCODING_NOT_UTF8",
            "message": "File could not be decoded with UTF-8.",
            "suggestion": "Convert file encoding to UTF-8 or run with --encoding=gbk parameter."
        }
        print(json.dumps(err_payload, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)

    # 3. 处理完成，标准输出返回结构化结果
    result = {
        "status": "SUCCESS",
        "processed_lines": min(len(lines), args.max_entries),
        "sample_head": lines[0].strip() if lines else ""
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### 在 SKILL.md 中配置容错协议与重试上限

在 `SKILL.md` 中，我们为模型提供与上述错误码一一对应的修复动作，并设定重试次数上限：

```markdown
### 错误排查与重试规则
当脚本以非 0 状态码退出时，读取 stderr 的 JSON 输出，对照以下规则处理：
1. 若 `error_code` 为 `ERR_FILE_NOT_FOUND`：
   - 检查传入的绝对路径是否存在拼写错误。
   - 若路径无误，执行一次目录列表命令寻找可能的备份文件，更新路径后重新调用。
2. 若 `error_code` 为 `ERR_ENCODING_NOT_UTF8`：
   - 增加 `--encoding gbk` 参数重新调用脚本。
3. 重试次数上限：针对同一种错误，最多重试 2 次。若重试 2 次后仍未成功，立即停止执行并向用户报告故障原因与 stderr 详情，严禁捏造虚假执行数据。
```

通过对错误码进行枚举指导，并引入明确的重试上限，可以显著减少模型在面对报错时的无序试错与 Token 浪费。

---

## 宿主协作与安全验收：高危操作拦截与双层核验

在很多复杂工程任务中，Agent 拥有修改文件、调用远程接口或执行删除命令的权限。仅靠在提示词里告诉模型“请小心不要误删文件”是不足以提供确定性安全保障的。在生产工程落地中（如 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 框架所强调的持久化状态与人机介入），必须依靠**宿主环境的硬性安全门禁**与**程序级验收**。

![高危工具操作拦截与宿主介入的人机协同确认时序示意图](/images/content-rebuild/b6112f0e597bb5c8/fig-5.png)

*图 5 解释：时序图展示模型发起关键破坏性动作时，宿主安全网关阻断执行并持久化挂起状态，经工程师审查同意后才恢复执行并提交物理状态。*

### 操作分级与宿主拦截机制

我们将任务中的动作划分为三种风险等级：

| 等级 | 动作特征 | 典型操作 | 宿主处理策略 |
| :--- | :--- | :--- | :--- |
| **L1: 只读操作** | 不产生任何持久化副作用 | 文本解析、日志扫描、计算指标 | 宿主放行，自动执行 |
| **L2: 幂等写入** | 仅影响临时工作目录或新文件 | 生成报表 HTML、写入临时缓存 | 校验写入路径在安全白名单内后放行 |
| **L3: 破坏性变更** | 覆盖已有文件、删除原始数据、外发网络调用 | 清理原始日志目录、覆盖配置文件 | **宿主拦截，挂起工作流并等待人工审批** |

需要明确的是：**高危操作的拦截必须发生在宿主工具执行层**。当模型生成了带有清理命令的工具调用时，宿主系统的拦截中间件（如 LangGraph 的 `interrupt`）拦截该调用，将当前任务状态保存，并在界面向用户呈现具体的命令与待删除文件列表。只有当用户点击“确认”后，宿主才真正将命令下发给终端执行。

`SKILL.md` 的职责，是在提示词层面要求模型在产生此类调用前输出清晰的变更摘要，以便用户审批：

```markdown
### 高危操作前置说明要求
在发起包含删除（如 rm/unlink）或覆盖现有文件的工具调用前：
1. 必须在回答中先行输出待影响文件的清单与变更原因。
2. 明确提示用户确认，等待系统拦截器完成授权确认后再继续。
```

### 双层验收机制

当任务全部步骤完成后，如何确认输出结果符合预期？应当建立物理层与语义层的双层核验：

![工程任务交付时的脚本物理检验与模型语义验收双重防线示意图](/images/content-rebuild/b6112f0e597bb5c8/fig-6.png)

*图 6 解释：展示底层物理校验（状态码与文件完整性）与上层语义验收（业务合规与格式契合）双重检验通过后才最终向用户交付结果。*

1. **底层物理检验（程序判定）**：
   - 辅助脚本的退出码（Exit Code）严格为 `0`。
   - 目标产物文件（如 `report.html`）在指定绝对路径存在，且文件大小大于预期阈值（避免空文件）。
   - 校验输出文件的哈希值，确保文件未在传输或写入中截断。

2. **上层语义核验（模型判定）**：
   - 模型对照 `references/compliance_rules.md` 中的合规清单，核对报表内是否包含了必要的指标字段（如错误数、Top3 告警模块）。
   - 检查报表文本中是否残留未渲染的模板占位符（如 `{{error_summary}}`）。

只有当两层验收均通过后，任务才标记为成功交付。

---

## 综合实战演练：一个数据归档 Skill 的构建

现在，我们把上述各模块组装成一个完整的微型 Skill：`archive-data-master`。它的功能是：扫描指定工作区目录下的多个 CSV 数据，利用本地脚本进行清洗合并与 SHA-256 校验，当需要删除原始 CSV 时必须触发人工确认。

### 步骤 1：建立标准目录

```text
archive-data-master/
├── SKILL.md
├── scripts/
│   └── aggregate_data.py
└── references/
    └── schema_spec.md
```

### 步骤 2：编写 `SKILL.md`

```markdown
---
name: archive-data-master
description: 批量清洗并归档多源 CSV 数据文件。当用户需要合并离散数据表、计算哈希并生成归档包时调用。
compatibility: Python 3.10+, pandas installed
allowed-tools: Bash(python:*) Read
---

# 数据归档任务执行指引

## 1. 路径锚定
所有操作基于环境变量 `$WORKSPACE_ROOT` 展开：
- 待处理数据目录：`${WORKSPACE_ROOT}/data/incoming`
- 归档产物目录：`${WORKSPACE_ROOT}/data/archive`

## 2. 执行确定性合并脚本
调用本地脚本进行批量合并与哈希计算：
```bash
python "${WORKSPACE_ROOT}/skills/archive-data-master/scripts/aggregate_data.py" \
  --input-dir "${WORKSPACE_ROOT}/data/incoming" \
  --output-dir "${WORKSPACE_ROOT}/data/archive"
```

## 3. 错误恢复协议
- 若脚本返回状态码 1 且包含 `NO_VALID_FILES`：
  检查是否存在备用目录 `${WORKSPACE_ROOT}/data/backup`，若存在则更新 `--input-dir` 参数重试 1 次。
- 若报错提示列名不匹配：
  读取 `references/schema_spec.md` 确认各列命名标准，指导后续处理。
- 重试上限为 2 次，超限后停止并上报错误。

## 4. 高危清理安全门禁
合并成功后，若用户要求删除 `${WORKSPACE_ROOT}/data/incoming` 下的原始文件：
- 严禁静默执行 `rm -rf`。
- 必须先列出即将删除的文件清单，并向宿主触发审批确认请求。
```

### 步骤 3：编写配套辅助脚本 `scripts/aggregate_data.py`

```python
import sys
import json
import hashlib
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    # 校验目录有效性
    if not in_dir.exists() or not in_dir.is_dir():
        err = {"status": "ERROR", "code": "NO_VALID_FILES", "detail": f"Directory {in_dir} does not exist."}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)

    csv_files = list(in_dir.glob("*.csv"))
    if not csv_files:
        err = {"status": "ERROR", "code": "NO_VALID_FILES", "detail": "No CSV files found in input directory."}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_file = out_dir / "merged_archive.csv"

    # 执行文件合并（代码保证确定性）
    total_rows = 0
    with open(merged_file, "w", encoding="utf-8") as outfile:
        for idx, fpath in enumerate(csv_files):
            with open(fpath, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile):
                    # 第一份文件保留表头，其余文件跳过表头
                    if idx > 0 and line_num == 0:
                        continue
                    outfile.write(line)
                    total_rows += 1

    # 计算产物 SHA-256 作为物理验收证据
    with open(merged_file, "rb") as f:
        sha256_val = hashlib.sha256(f.read()).hexdigest()

    # 成功输出
    output_info = {
        "status": "SUCCESS",
        "merged_path": str(merged_file),
        "total_rows": total_rows,
        "sha256": sha256_val
    }
    print(json.dumps(output_info, ensure_ascii=False))
    sys.exit(0)

if __name__ == "__main__":
    main()
```

在这个案例中，所有的批量合并、表头跳过、行数统计和 SHA-256 运算完全由本地 Python 脚本稳定执行；大模型则负责解析用户的路径要求、在遇到错误时根据结构化 JSON 识别备选目录，并在执行删除时提供清晰清单供宿主发起人工审批。这种协同机制，正是让 Agent 能够稳定处理多步骤复杂工作的基石。

---

## 思考题

1. 在什么场景下，把一段计算逻辑写成独立的 Python 脚本比直接让模型生成 Python 代码就地执行更稳妥？
2. 为什么单纯在 `SKILL.md` 里写“删除文件前必须问用户”不能替代宿主系统层面的人机拦截机制？

更多相关内容可参考同专题的其他文章：
- [Agent Skills 是什么？它和普通 Prompt、工具调用有什么区别？](/frontier/agent-skills/what-is-agent-skills.md)
- [一个可复用的 Agent Skill 应该包含哪些内容？说明、输入、步骤和边界怎么写？](/frontier/agent-skills/reusable-skill.md)
- [返回 Agent Skills 专题目录](/frontier/agent-skills/README.md)
