---
title: 代码 Agent 的基本执行流程是什么？从规划到验证的完整步骤
icon: mdi:arrow-decision-outline
tag:
- AI编程面试
- 代码Agent
- 执行流程
description: 介绍代码 Agent 的基本执行流程：任务分解、ReAct 循环、双重门禁验证与 Git 快照回溯自愈，附带伪代码与面试口述模板。
---

面试官提问“代码 Agent 的基本执行流程是什么”时，重点在于考察候选人是否清楚代码 Agent 不是单次模型调用，而是由外部宿主程序编排的“状态管理-工具调用-结果验证-异常回退”循环流程。

学习本篇前，建议先阅读 [AI 编程中如何设计工具调用？](/interview/ai-coding-interview/tool-design.md)，了解模型如何通过结构化协议与文件系统和终端进行交互。

![代码 Agent 状态机执行控制流全景](/images/content-rebuild/37e6ed41142baddc/cover.png)

---

## 1. 核心回答：30 秒面试口述模板

面试时可以用清晰的四步框架进行口述：

> “代码 Agent 是由大模型决策、宿主环境提供确定性执行与真实反馈的状态控制系统。其基本执行流程包含四个阶段：
>
> 1. **任务分解与动态规划**：将用户需求分解为结构化的任务清单，并在工具探索获得新信息后动态调整步骤；
> 2. **ReAct 执行循环**：模型按照‘思考（Thought）- 工具调用（Action）- 真实环境观察（Observation）’不断循环，执行搜索、代码阅读与文件编辑；
> 3. **分层双重验证**：代码修改后先经过低耗时、本地运行的静态检查（语法分析器、Linter 或类型检查器），拦截排版和语法错误；通过后再运行针对性的动态测试（如单元测试），验证业务功能；
> 4. **回溯与自愈**：验证失败时将报错信息注入上下文让模型修复；若单步尝试超过重试上限，宿主程序会利用 Git 快照撤销当前脏修改，换用其他思路或降级向人工求助。
>
> 系统通过步数配额、Token 预算、测试通过状态与无进展检测共同构成明确的退出条件。”

按照 [Anthropic 关于构建有效 Agent 的分类](https://www.anthropic.com/engineering/building-effective-agents)，系统架构通常区分为两类：
* **工作流（Workflows）**：通过预定义的代码路径编排模型与工具，流程更具可预测性与一致性；
* **智能体（Agents）**：由大模型根据环境反馈动态决定下一步动作和工具调用，更具灵活性。

生产环境中的代码 Agent（例如 Cursor、Claude Code）通常是二者的组合：**外层是由宿主程序按预设规则编排的工作流（负责状态流转、Git 快照回滚与门禁拦截等手写代码步骤），内层由模型驱动 ReAct 循环进行局部探索与编辑**。这种架构与 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 提出的“在同一张状态图里混合手写代码步骤与模型决策步骤”的设计完全一致。

---

## 2. 贯穿示例：修复一个登录 Token 泄露 Bug

为了直观展示整个流程，我们以一个具体的工程任务贯穿全文：
> **任务需求**：用户退出登录后，其携带的 JWT Token 仍能在 5 分钟内访问接口。需要修改认证逻辑，在登出时将 Token 写入 Redis 黑名单，并在鉴权中间件增加失效拦截。

接下来各个阶段将展示代码 Agent 如何逐步推进并落地这个修改。

---

## 3. 阶段一：任务分解与动态规划

代码库往往包含成百上千个文件，如果让模型直接一次性修改整个项目，极易产生范围漂移或漏改关联逻辑。因此执行流程的第一步是生成可跟踪的任务清单。

![需求拆解与动态待办列表状态更新图](/images/content-rebuild/37e6ed41142baddc/fig-1.png)

### 静态计划与动态调整的区别
早期简单的实现通常采用静态清单：先生成 1、2、3 步，然后机械执行。但在代码场景下，模型初期的规划往往基于不完整的仓库认知。

例如在我们的贯穿示例中，模型一开始根据通用经验生成的初始计划如下：

```json
{
  "task_id": "fix_token_revocation",
  "status": "IN_PROGRESS",
  "plan": [
    {
      "step_id": 1,
      "description": "在 auth_middleware.py 中定位 token 校验函数",
      "status": "COMPLETED",
      "evidence": "定位到 check_auth() 函数"
    },
    {
      "step_id": 2,
      "description": "在 auth_middleware.py 中直接连接 Redis 检查黑名单",
      "status": "PENDING",
      "evidence": null
    }
  ]
}
```

当执行第 1 步时，工具返回的代码显示该项目并未直接在中间件使用裸 Redis，而是封装了一个 `TokenCacheManager` 单例服务。此时宿主程序要求模型执行**计划刷新（Plan Refinement）**：
1. 将第 2 步修正为“在 `token_cache.py` 中增加黑名单读写接口”；
2. 动态插入第 3 步“在 `auth_middleware.py` 中依赖注入 `TokenCacheManager` 并增加拦截”。

动态任务清单机制确保了模型不会基于早期错误的假设把无用代码写到底。

<ReadMoreLock />

---

## 4. 阶段二：ReAct 执行循环

确定当前子任务后，系统进入核心执行循环。ReAct（Reasoning + Acting）模式由四个连续步骤组成：

![ReAct 循环与环境状态注入时序图](/images/content-rebuild/37e6ed41142baddc/fig-2.png)

1. **思考（Thought）**：模型阅读当前任务说明、历史对话和前一次工具执行的真实输出，推导下一步动作。例如：“需要确认 `TokenCacheManager` 当前的写入方法签名，先调用代码检索工具。”
2. **行动（Action）**：模型输出结构化工具调用（Tool Call），包含工具名称与符合 JSON Schema 的参数，如 `grep_search(query="class TokenCacheManager", path="services/")`。
3. **宿主执行（Execution）**：宿主控制流接管执行权。在确认命令不越界后，在本地沙箱运行命令，获取输出流。
4. **环境观察（Observation）**：宿主将真实命令输出（如标准输出、文件内容片段或退出码）包装为 `role: tool` 消息填入上下文，驱动模型进入下一次思考。

在代码 Agent 中，**Observation 必须来自宿主环境的真实执行反馈**，绝不能由模型自行假设结果。工具调用的具体设计可参考 [AI 编程中如何设计工具调用？](/interview/ai-coding-interview/tool-design.md)。

---

## 5. 阶段三：分层双重验证门禁

当模型完成文件修改后，不能仅凭模型输出的“修改已完成”就直接交付，必须由运行环境进行严格验证。典型的工程设计是**分层双重门禁**：

![静态门禁与动态测试的双重验证过滤漏斗](/images/content-rebuild/37e6ed41142baddc/fig-3.png)

### 第一道门禁：静态检查（Linter / Compiler / 类型检查）
在运行测试之前，先调用本地工具对改动文件执行静态检查（如 TypeScript 项目运行 `tsc --noEmit`，Python 项目运行 `ruff check` 或 `mypy`）。

**为什么优先执行静态检查？**
* **低耗时与零 Token 成本**：静态检查在本地进程以毫秒级完成，无需调用大模型 API；
* **避免污染模型上下文**：若代码存在括号缺失或变量拼写错误，直接运行测试套件往往会输出数十行混乱的 Traceback，极易误导大模型；静态分析器则只返回精准的“第 42 行变量未定义”，错误提示高度聚焦；
* **确定性规则校验**：编译器或语法解析器提供了完全确定的规则校验，不需要把简单语法问题交给模型去概率猜测。

### 第二道门禁：动态验证（单元测试与集成测试）
通过静态检查后，证明代码在语法和类型上没有明显破损。此时宿主程序触发针对性的测试套件：
* **Bug 修复场景**：运行先前编写的失败复现用例，验证该用例是否转为通过；
* **回归防护**：运行与修改模块相关的上下文测试，确保修改没有破坏已有的其他功能。

当静态检查与动态测试均返回 Exit Code 0 时，子任务才被标记为 `VERIFIED`，推动清单前进。

---

## 6. 阶段四：回溯与容错自愈策略

首次修改未通过测试属于正常现象。但若模型连续修改失败，错误代码会不断累加。宿主程序必须具备确定性的回滚机制。

![修改失败时的 Git 快照回滚与重试状态机分支](/images/content-rebuild/37e6ed41142baddc/fig-4.png)

### 常见失败模式
1. **振荡修改**：模型修复了测试 A 导致测试 B 失败，改回测试 B 又导致测试 A 失败，来回摇摆；
2. **脏状态扩散**：为了解决一个报错，模型连续在 6 个不相关的文件里添加临时修补，导致工作区代码结构严重失真；
3. **思路重复**：模型连续多次生成只有变量名微调、但底层逻辑完全一样的错误补丁。

### 宿主程序的控制机制

| 控制机制 | 触发条件 | 宿主具体动作 | 解决的问题 |
| :--- | :--- | :--- | :--- |
| **Git 快照原子回退** | 局部重试达到阈值（如连续 2 次失败） | 运行 `git reset --hard` 或恢复任务开始前的 stash 锚点 | 清理累积的脏改动，让工作区回到干净状态 |
| **报错签名比对** | 连续两次工具调用产生完全一致的报错特征 | 拦截重复的工具调用，在 Prompt 中显式提示“该方案已产生完全相同的报错” | 避免模型在死循环中重复调用 |
| **重试预算熔断** | 单个子任务失败次数达到上限（如 3 次） | 强制终止当前子任务，标记为 `BLOCKED` 并退回规划阶段 | 避免 Token 费用与执行轮次无休止浪费 |
| **负向提示注入** | 上一次方案被验证无效并回滚后 | 在上下文中追加：“已证明方案 X 会引起 Y 错误，已被回滚，请换用其他方案” | 促使模型改变搜索方向 |

---

## 7. 端到端执行控制器代码实现

以下 Python 示例展示了一个精简的代码 Agent 宿主控制循环，体现了状态流转、双重门禁和 Git 回退逻辑：

```python
import subprocess
from typing import Dict, Any, List

class CodingAgentController:
    def __init__(self, workspace_path: str, max_retries: int = 3):
        self.workspace = workspace_path
        self.max_retries = max_retries
        self.history: List[Dict[str, Any]] = []

    def run_cmd(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """在工作区目录执行命令"""
        return subprocess.run(cmd, cwd=self.workspace, capture_output=True, text=True)

    def get_git_head(self) -> str:
        """记录当前干净提交点的哈希值"""
        res = self.run_cmd(["git", "rev-parse", "HEAD"])
        return res.stdout.strip()

    def rollback(self):
        """还原所有未提交的脏改动"""
        self.run_cmd(["git", "reset", "--hard", "HEAD"])
        self.run_cmd(["git", "clean", "-fd"])

    def execute_subtask(self, task_description: str) -> bool:
        initial_head = self.get_git_head()
        retry_count = 0
        feedback = ""

        while retry_count < self.max_retries:
            # 1. 结合历史上下文与上次执行反馈，调用模型获取动作决策
            decision = self.call_llm_step(task_description, feedback, self.history)
            
            # 2. 宿主程序执行文件修改
            self.apply_file_edits(decision.file_edits)

            # 3. 第一道门禁：静态语法与代码风格检查
            linter_res = self.run_cmd(["ruff", "check", "."])
            if linter_res.returncode != 0:
                retry_count += 1
                feedback = f"静态检查报错，请优先修复语法问题:\n{linter_res.stdout}"
                continue # 带着精确报错进入下一轮微调

            # 4. 第二道门禁：针对性单元测试
            test_res = self.run_cmd(["pytest", "tests/test_token_cache.py"])
            if test_res.returncode == 0:
                return True # 两道门禁全部通过，该步骤成功交付
            else:
                retry_count += 1
                feedback = f"单元测试失败:\n{test_res.stdout}"
                
                # 若已连续尝试失败 2 次，清空脏修改，强迫换思路
                if retry_count >= 2:
                    self.rollback()
                    feedback += "\n前序修改已被彻底回滚，请换用不同的实现思路，严禁重复刚才的代码。"

        # 达到重试预算，回滚并退出
        self.rollback()
        return False

    def call_llm_step(self, task: str, feedback: str, history: List[Dict]) -> Any:
        # 伪代码：实际场景中封装 API 调用与 JSON 解析
        pass

    def apply_file_edits(self, edits: List[Any]):
        # 伪代码：实际场景中将补丁写入磁盘文件
        pass
```

![Python 模拟的 Agent 控制流状态转移执行轨迹](/images/content-rebuild/37e6ed41142baddc/fig-5.png)

---

## 8. 面试高频追问与设计边界

![成本预算、探索步数与任务质量的终止决策矩阵](/images/content-rebuild/37e6ed41142baddc/fig-6.png)

### 追问 1：如何定义代码 Agent 的任务终止条件？
终止逻辑必须由宿主控制流强制执行，通常由四类条件共同判定：

1. **正常达成终止**：
   * 规划清单中的所有子项全部处于完成状态；
   * 静态分析和目标测试全部以 Exit Code 0 退出；
   * Git 变更范围符合限定目录（未误改核心配置或锁定文件）。
2. **步数与成本熔断**：
   * 循环轮数达到阈值（例如单任务最多 20-30 轮交互）；
   * 累计消耗的 Token 数量达到预设成本上限。
3. **无进展停滞熔断**：
   * 连续 3 轮工具调用的 `git diff` 结果均为空，或陷入相同的命令死循环。
4. **安全越界拦截**：
   * 检测到高危行为（如修改 `.git` 目录、执行网络外联传输凭据、试图强制推送到保护分支），由宿主沙箱直接挂起并转交人工审核（Human-in-the-Loop）。

### 追问 2：为什么不能在每次修改后直接运行全量测试套件？
部分初学者认为“验证越全面越好”，但在每次模型调用后跑全量测试会带来严重的工程副作用：
* **执行时延急剧上升**：中大型项目的全量测试往往需要数分钟甚至更久。Agent 循环十几次，总体耗时将完全无法接受；
* **偶发失败（Flaky Tests）误导**：大型测试库中常存在由于网络抖动或并发时序偶尔失败的用例，全量运行会引入与当前修改无关的报错，严重干扰模型的推理方向；
* **错误定位困难**：如果几百个测试一同抛出异常，错误信息会挤满上下文窗口，模型无法识别到底是本次修改引发的，还是环境历史原因导致的。

**合理做法**：宿主控制流应配合“测试选择策略（Test Selection）”，只运行被改动模块直接依赖的单元测试，待所有子任务全部完成时，再在最终交付前执行一次全量回归。

### 常见认知误区排查
* ❌ **误区**：“只要大模型支持 100 万 Token 的上下文，测试报错全往里塞就行，模型总能自己搞定。”
  * **事实**：上下文塞入的错误信息越多，噪声和干扰信息就越庞大，模型遵循原始提示指令的能力反而会明显下降。适时的上下文截断、错误归纳与 Git 回滚是维持推理质量的必要工程手段。
* ❌ **误区**：“静态检查没有必要，单元测试覆盖率够高就能代替静态检查。”
  * **事实**：静态检查耗时通常在百毫秒内且不消耗大模型 Token，能以极高确定性过滤掉括号缺失、未声明变量等语法问题，避免把昂贵的时间和算力消耗在无意义的基础报错上。

---

## 关联学习路径

掌握了代码 Agent 的整体控制流后，可以按以下顺序继续深入学习关联知识点：

1. [AI 编程助手的核心原理是什么？](/interview/ai-coding-interview/core-principles.md) —— 理解代码补全与 Agent 控制流在系统定位上的差异；
2. [Claude Code、Cursor、Copilot 如何理解项目上下文？](/interview/ai-coding-interview/project-context.md) —— 了解 Agent 在规划前如何完成项目拓扑与上下文装配；
3. [AI 编程助手如何读取、检索和修改大型代码仓库？](/interview/ai-coding-interview/code-repo-access.md) —— 深入了解文件检索与 AST 精准编辑的工具实现；
4. [AI 编程中如何设计工具调用？](/interview/ai-coding-interview/tool-design.md) —— 学习工具 JSON Schema 设计与沙箱边界隔离；
5. 进阶探索持久化状态与技能沉淀：[Agent 记忆系统](/frontier/agent-memory-system/README.md) 与 [如何把常用任务封装成 Skills](/frontier/agent-skills/skill-encapsulation.md)。
