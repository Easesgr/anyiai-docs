---
title: MCP 和 Agent Skills 有什么区别？分别适用于什么场景？
icon: mdi:compare
tag:
- MCP
- Skills
- 对比
description: 对比 MCP 通信协议与 Agent Skills 规范的架构定位、上下文开销、执行主体与安全边界，给出协同落地模式与面试口述思路。
---

在开发基于大模型的 Agent 时，很多人常常分不清两个概念：一个是 Anthropic 主导的 **MCP（Model Context Protocol）**，另一个是以 `SKILL.md` 为规范的 **Agent Skills**。

有的开发者把 MCP 当成单纯的“工具库”，把 Skill 当成简单的“提示词模板”；还有人认为两者功能重叠，非要二选一。实际上，**MCP 是一套连接外部世界的应用层通信协议**，规定了宿主系统如何与外部服务交换数据和调用工具；而 **Agent Skills 是一套静态的能力规约包**，用来沉淀特定领域的业务操作规范、前置条件与执行脚本。它们分别处于通信层与知识规范层，经常协同工作。

![MCP 协议插头与 Skill 操作手册架构示意图](/images/content-rebuild/c53bfb0a081a8384/cover.png)

---

## 面试标准口述：如何回答两者区别

### 基础版口述（60 秒清晰回答）

> “MCP 和 Agent Skills 处于不同的架构层次：
>
> 1. **定位不同**：MCP 是**应用层通信协议**，解决的是‘如何统一接入外部服务’的问题。它规范了宿主（Host）、客户端（Client）与服务端（Server）之间的连接生命周期、工具列表发现和 JSON-RPC 过程调用。
> 2. **形态不同**：Agent Skills 是一套**基于文件系统的能力与知识规约**（核心是 `SKILL.md`），解决的是‘如何引导模型按业务规范完成复杂任务’的问题。它为模型提供任务指引、前置要求、领域参考资料以及可选的本地执行脚本。
> 3. **执行主体不同**：MCP 工具的真正逻辑运行在独立的 Server 进程或远程网络服务中；而 Skill 依赖宿主环境解析，正文内容会被注入到模型的上下文中，指导模型分步思考，附带的脚本也是由宿主沙箱或本地环境直接执行。
>
> 总结一句话：**MCP 提供的是标准化的‘外部工具插头’，Agent Skills 提供的是业务领域的‘操作指导手册’。**”

### 贯穿案例：合同合规审查任务

为了直观理解两者的分工，我们以一个具体的工程任务为例：**“从企业内网拉取采购合同，核算违约金条款是否合规，并在工单系统中记录异常。”**

- **只用 MCP 会遇到什么？** 宿主给模型挂载了两个 MCP Server：一个读数据库（`read_contract_db`），一个写工单（`create_jira_issue`）。模型能调用这些接口，但它不知道这笔采购遵循哪一版合同法、违约金比例超过 20% 时该如何标记黄色预警、需要提取哪些特定字段。如果全靠模型自由发挥，很容易漏审或格式不合规。
- **只用 Skill 会遇到什么？** 开发者写了一份详尽的 `contract-audit` 技能包，里面有完整的法律审核标准、正则提取规则。但如果需要访问带动态权限认证的企业内网数据库，把连接凭证和鉴权代码全写在本地脚本或提示词里，既难维护，又容易引入安全风险。
- **两者协同如何工作？** 宿主为 Agent 挂载只读数据库与工单系统的 MCP Server，负责安全可靠的外部数据读写；同时为 Agent 配置 `contract-audit` 的 Agent Skill，用标准作业规范引导模型按照“拉取合同 -> 依据规范校验 -> 触发预警并创建工单”的步骤稳定执行。

<ReadMoreLock />

---

## Agent Skills 的形态：不只是提示词

根据 [Agent Skills 规范标准](https://agentskills.io/specification)，一个标准的 Skill 是由文件系统规范组织的能力目录，而不是随手粘贴的一句 System Prompt：

```text
contract-audit/
├── SKILL.md              # 必需：元数据与操作指引（核心入口）
├── scripts/              # 可选：由宿主或 Agent 执行的脚本（如 extract_terms.py）
├── references/           # 可选：详细的技术参考文档、合规规章（按需读取）
└── assets/               # 可选：静态表格模板、示例配置
```

![Agent Skills 目录体系与渐进式揭示流程图](/images/content-rebuild/c53bfb0a081a8384/fig-1.png)

### SKILL.md 结构与渐进式揭示（Progressive Disclosure）

一个合规的 `SKILL.md` 由 YAML Frontmatter 元数据和 Markdown 说明正文组成：

```markdown
---
name: contract-audit
description: 提取采购合同条款，核验履约时间节点并校验违约金上限。在审查合同或排查法律风险时使用。
compatibility: Requires python >= 3.10, PyPDF2
metadata:
  author: legal-audit-team
  version: "1.1.0"
allowed-tools: Bash(pdf:*) Read
---

# 合同审查操作规范

## 执行步骤
1. 运行 `scripts/extract_terms.py` 提取合同结构化文本。
2. 参照 `references/statutory_limits.md` 逐项比对双方违约责任。
3. 若约定的违约金超过法定上限，在审查报告中给出风险提示。
```

标准 Agent Skills 依赖宿主实现**渐进式揭示（Progressive Disclosure）**，用来缓解上下文占用：

1. **元数据发现阶段**：宿主启动时，只读取所有 Skill 的 `name` 与 `description`（通常几十到上百个 Token），作为模型可见的技能清单目录常驻在上下文。
2. **指令激活阶段**：当用户的任务匹配到该技能时，宿主将该技能的 `SKILL.md` 正文注入当前会话，指导模型进入特定角色的执行路径。
3. **资源按需拉取阶段**：正文中提到的 `references/` 详细参考文件，只有在模型需要查阅具体条目时，才通过文件读取工具加载；`scripts/` 中的代码也只在需要运算时才被宿主执行。

需要说明的是，渐进式揭示依赖宿主程序的具体调度策略。Skill 本身并不是一个独立的进程，它不具备主动监听网络或自动管理连接的能力，本质是一份结构化的指示资产。

---

## MCP 的通信架构与执行机制

与基于文件系统的 Skill 不同，[Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/learn/architecture) 解决的是**跨进程与跨网络的上下文交互问题**。

![MCP 与 Skill 在 Agent 运行时中的拓扑与执行主体对比](/images/content-rebuild/c53bfb0a081a8384/fig-2.png)

MCP 采用客户端-服务端架构：

- **MCP Host**：Agent 运行环境本体（如 Claude Desktop 或各种自主开发的 Agent 框架）。
- **MCP Client**：运行在 Host 内部的协议客户端，负责与具体的 MCP Server 建立会话连接。
- **MCP Server**：独立的后端程序，可以运行在本地，也可以部署在远程服务器上。它向 Host 暴露三类上下文能力：**Prompts**（提示模板）、**Resources**（结构化数据源）和 **Tools**（可调用的功能函数）。
- **通信传输层**：MCP 官方标准定义了基于标准输入输出的 `stdio` 传输（常用于本地子进程）和基于网络传输的 `Streamable HTTP`（利用 HTTP POST 与流式响应传输，早期版本也包含 HTTP+SSE）。通信内容采用 JSON-RPC 2.0 格式。

![任务执行时序：MCP 协议调用 vs Skill 指令驱动](/images/content-rebuild/c53bfb0a081a8384/fig-3.png)

### 运行时调用链路的差异

当模型决定“使用某项能力”时，两者的执行时序完全不同：

- **MCP 的工具调用链路**：
  1. Host 启动时通过 `tools/list` 获取工具的完整 JSON Schema；
  2. 模型根据 Schema 生成结构化的函数调用参数；
  3. Client 向 MCP Server 发送 `tools/call` 请求；
  4. **MCP Server 进程执行具体逻辑**（如连接只读数据库执行 SQL），将结果打包为 `CallToolResult` 回传给 Client 并写回上下文。
- **Agent Skill 的执行链路**：
  1. 模型匹配到技能元数据，宿主将 `SKILL.md` 正文载入会话；
  2. 模型阅读其中的指引，自行规划接下来的思考与动作；
  3. 若说明中建议运行 `scripts/extract_terms.py`，**宿主通过本地沙箱或命令行工具去执行该脚本**，并将终端标准输出带回上下文。

---

## 架构与特性对比

以下表格梳理了两者在工程实现上的核心差异：

| 维度 | MCP 协议架构 | Agent Skills 规约包 |
| :--- | :--- | :--- |
| **本质属性** | 跨进程 / 跨网络的应用层通信协议 | 基于文件系统的静态任务规范与指示包 |
| **规范依据** | [Model Context Protocol 架构规范](https://modelcontextprotocol.io/docs/learn/architecture) | [Agent Skills 规范标准](https://agentskills.io/specification) |
| **通信媒介** | JSON-RPC 2.0（支持 `stdio` 与 `Streamable HTTP`） | 本地文件读取、上下文注入、本地命令行或沙箱执行 |
| **存在形式** | 可独立部署运行的服务端程序（如 Node/Python 进程、Docker） | 结构化文件目录（包含 `SKILL.md` 及可选的脚本与资料） |
| **执行主体** | 独立的 MCP Server 进程，与宿主相互隔离 | 宿主模型推理上下文与宿主本地执行环境 |
| **适用内容** | 动态接口调用、实时外部数据拉取、跨网络资源集成 | 业务操作流程（SOP）、代码规范、核算逻辑与领域指引 |
| **凭证管理** | 外部系统核心凭据收敛在 Server 端，模型无感知；但远程 Server 仍需 Host 维护连接鉴权 | 本地脚本若需联网，凭据通常需由宿主注入环境变量或配置文件，模型有接触或泄露风险 |

---

## 上下文开销对比：不能简单说“谁更省 Token”

在面试中，常有候选人脱离场景背诵“Skill 比 MCP 更省 Token”。事实上，Token 的消耗必须放在**相同工作量、相同功能覆盖度**的前提下，分阶段具体分析。

![两种架构在初始化、激活与执行三个阶段的 Token 开销对比](/images/content-rebuild/c53bfb0a081a8384/fig-4.png)

### 算例假设与基准条件

- **基准场景**：系统接入 10 项独立的业务能力（例如：合同解析、汇率查询、工单提交等）。当前用户的具体请求只涉及其中 1 项（合同解析）。
- **MCP 方案**：由 MCP Server 提供这 10 个工具的 JSON Schema 定义，每个工具定义（含参数说明、字段校验）平均假设为 120 Tokens。
- **Skill 方案**：遵循标准渐进式揭示规范，配置 10 个 Skill 目录。每个 Skill 的 `name` 与 `description` 平均假设为 40 Tokens；正文 `SKILL.md` 包含详细操作指南，平均假设为 1200 Tokens。

### 分阶段 Token 消耗推演

1. **初始化阶段（发现可用能力）**
   - **MCP**：宿主需要让模型感知所有可用工具，通常在系统提示区一次性加载全部 10 个工具的完整 Schema。初始开销为：
     $$10 \times 120 = 1200\text{ Tokens}$$
   - **Skill**：根据规范，宿主只需加载 10 个 Skill 的元数据说明，初始开销为：
     $$10 \times 40 = 400\text{ Tokens}$$
   - *阶段结论*：在初始化时，Skill 的轻量目录占用的上下文明显小于全量展开的 MCP 工具列表。

2. **激活与决策阶段（确定使用某项能力）**
   - **MCP**：模型由于已具备工具 Schema，直接生成 `tools/call` 请求，**没有额外的指令注入开销**（增加开销为 0 Token）。
   - **Skill**：模型判断任务命中 `contract-audit`，发出读取请求后，宿主将约 1200 Tokens 的 `SKILL.md` 全文注入对话上下文。
   - *阶段结论*：在激活单项具体能力时，Skill 因载入详细业务指引，其会话上下文瞬间增量高于 MCP。

3. **执行与结果返回阶段**
   - **MCP**：Server 执行完毕后返回 JSON 格式结果。如果外部接口返回的数据结构冗余，会直接占满上下文，需要在 Server 端做好字段过滤或切片。
   - **Skill**：若 Skill 通过脚本处理数据，可以在本地执行后只将格式化后的提炼结果输出给模型；当面对复杂规章时，还能让模型按需读取 `references/` 中的特定段落。

**总结**：MCP 是一次性平铺所有接口定义，单次决策链路直接；Agent Skills 采用元数据与详细正文分离的方式，降低了初始常驻开销，但在技能展开后会引入成百上千 Token 的业务指导说明。因此，**脱离“当前启用了多少项能力”和“指令展开深度”，孤立地下结论说谁比谁省 Token 是不成立的**。

---

## 协同落地：Skill 做指引，MCP 做执行

在真实的 Agent 系统中，**MCP 与 Agent Skills 并不是互斥的替代关系，而是经常上下配合的组合关系**。

> **Agent Skill 是指导步骤的操作手册（SOP），MCP 是执行动作的标准化工具箱（Tools）。**

如果只给 Agent 一堆 MCP 工具，模型面对复杂多步骤分支时，容易出现调用顺序混乱、死循环或者漏掉前置判断的情况；如果把所有数据访问逻辑全用 Python 脚本写在 Skill 里面，又会导致代码重复、失去集中鉴权与远程部署的能力。

![Skill 业务 SOP 指导 LLM 调度底层 MCP 工具协同架构图](/images/content-rebuild/c53bfb0a081a8384/fig-5.png)

### 落地分工示例

依然回到前述的合同审查场景：

1. **底层设施层（MCP Servers）**：
   - `Database-MCP-Server`：暴露 `query_contract_by_id(id)`，内部管理企业内网只读数据连接池；
   - `Ticketing-MCP-Server`：暴露 `create_audit_ticket(title, risk_level, detail)`，通过企业统一网关打通工单系统。
2. **业务规范层（Agent Skill：`contract-audit`）**：
   - 在 `SKILL.md` 中约束严格的操作顺序：
     - 步骤一：通过 `Database-MCP` 读取合同原始内容；
     - 步骤二：利用 Skill 附带的 `scripts/parse_pdf.py` 本地提取文本表格，比对 `references/rules.md` 中的违约责任上限；
     - 步骤三：若违约条款超出标准，整理风险摘要，最后调用 `Ticketing-MCP` 创建黄色风险工单并终止流程。

通过这种搭配，MCP 提供了标准化的原子接口并收敛凭证，Skill 则在上下文层面提供业务流程 SOP 指引，降低多步骤推理的无序性，两者各展所长。需要注意，Skill 本质仍是基于自然语言的上下文指导，不能提供 100% 确定性的执行保证；若业务流程要求完全确定、不可偏离，仍应在宿主侧结合确定性状态图或代码分支进行刚性控制。

---

## 深入追问与常见误区

### 追问 1：为什么不能把所有操作规范都写在 MCP 工具的 Description 里面？

**面试考核点**：考察对 Tool Schema 设计边界与上下文成本的理解。

**原因解释**：
- MCP 工具的 `description` 设计初衷是为 [Tool Calling 的参数匹配](/interview/agent-interview/tool-calling.md) 提供语义指导，让模型明白“这个参数是什么意思、什么情况下该调它”。
- 如果把长达上千字的业务背景、法规条文、判断流程和异常处理样例全部硬塞进工具描述，意味着**每一次只要加载该工具，这上千字就会常驻在上下文**。当系统挂载了十几个工具时，上下文很容易被撑爆。
- 此外，跨越多个工具的执行逻辑（如“先调 A、若状态为 X 则调 B、若为 Y 则调 C”）属于工作流和业务规约，硬塞给单项工具在逻辑上也不合理。合理的做法是将原子接口留在 MCP，将多步骤协同指引剥离到 Skill。

### 追问 2：涉及企业内网 OAuth2 鉴权与敏感数据时，为什么优先选 MCP？

**面试考核点**：考察 Agent 系统与外部企业系统集成时的安全隔离设计能力。

**原因解释**：
- 如果只依赖静态 Skill 和本地脚本，脚本为了连接内网 API，往往需要读取本地配置文件、环境变量甚至直接在提示词中传入 Token。一旦 Agent 面临恶意内容引发的 [Prompt 注入风险](/interview/agent-interview/agent-security.md)，攻击者有可能诱导模型泄露本地环境变量或参数。
- MCP Server 采用独立进程或后端服务部署。所有与企业系统的复杂 OAuth 握手、刷新令牌、字段脱敏、RBAC 权限校验，全部封装在 MCP Server 内部。Host 客户端只需要维护与 MCP Server 之间的连接凭证，模型本身接触不到内网系统的核心密钥，显著收敛了攻击面。
- **需要明确的是，MCP 协议本身并不保证安全**：协议规范只定义上下文交换消息格式，不内置传输加密或访问控制机制。跨网络部署时必须依赖底层的 TLS/HTTPS 与宿主层应用鉴权；同时 MCP Server 必须对输入参数做严格校验与最小权限控制，防范模型遭遇间接提示词注入后被诱导滥用合法接口。

![选型决策矩阵与安全鉴权隔离分析图](/images/content-rebuild/c53bfb0a081a8384/fig-6.png)

### 追问 3：单个 Skill 的激活等价于一次函数调用吗？

**面试考核点**：考察候选人对模型推理与工具调用机制的精确认知。

**原因解释**：
- **完全不等价**。
- 函数调用（Tool Calling）是模型输出符合特定 JSON 格式的结构化文本，由宿主拦截后执行代码并回传结果，属于一个单次的、明确的动作交互。
- Skill 的激活则是宿主将对应的指令注入到当前上下文中。激活后，模型并没有执行特定的外部函数，而是切换了当前的“角色背景”和“思考指引”。在后续的多轮推理中，模型可能会依据 Skill 的说明先后发起多次不同的工具调用，或者只是在本地运行脚本进行数据分析。

---

## 选型决策速查指南

在做架构设计或面试总结时，可以通过以下逻辑快速做出技术选型：

1. **看接口性质**：如果需要接入的是独立的后端服务、第三方网络 API、长生命周期的数据库连接池，优先将其开发为 **MCP Server**。
2. **看沉淀内容**：如果沉淀的是团队的代码重构模式、文档排版格式规范、复杂的特定行业操作流程（SOP），优先编写为 **Agent Skill（SKILL.md）**。
3. **看安全与权限**：凡是涉及复杂动态凭据、租户鉴权隔离与后台审计追踪的场景，优先用 **MCP Server** 在后端收敛鉴权。
4. **看最终落地**：追求稳定与可扩展的 Agent 系统通常是**“以 MCP 为原子基座，以 Skill 为业务编排指引”**，两者分层协同。
