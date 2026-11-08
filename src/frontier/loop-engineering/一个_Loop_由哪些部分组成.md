---
title: 一个 Loop 由哪些部分组成？生成、验证、反馈与状态管理
icon: mdi:refresh
date: 2026-07-16
tag:
- Loop Engineering
- Agent
description: 拆解循环系统的核心构成：生成器、验证器、反馈构造器、终止策略与状态管理，分析各组件职责、交互流程与边界控制。
---

如果只调用一次大语言模型，并直接把结果交给下游系统，一旦输出中包含语法错误、字段缺失或逻辑漏洞，整个程序就只能中断或抛出异常。大模型输出带有概率性，单次生成很难在复杂约束下做到完全正确。

解决这一问题的常见方法是构建循环系统（Loop）。循环系统的思路是：不把模型的单次输出当成最终结论，而是把“生成候选结果”、“检查是否符合预期”、“根据错误信息修改”组织成一个受控的多轮流程，直到满足测试条件或达到退出限制。

以一个具体任务为例：我们要让模型编写一个 Python 函数 `parse_metadata(text: str) -> dict`，用于提取文章开头的 YAML 属性，并且通过 3 个预设的单元测试。在一个完整的循环中，系统并不是重复发送同一句“请重写”，而是依次协调几个不同职责的组件来推进状态。

![循环系统核心拓扑概览](/images/content-rebuild/22f0b293b6505e9a/cover.png)

## 循环系统的核心组成与流转过程

一个完整的循环控制流程通常由五个部分组成：

1. **状态对象（State Context）**：记录当前任务目标、最新生成的代码或文本、已执行的轮次、历史测试结果等信息。
2. **生成器（Generator）**：负责调用模型，根据当前目标和上一轮的错误反馈，产出新的候选内容。
3. **验证器（Verifier）**：负责运行确定性的规则检查或测试用例，判断候选内容是否达标，并输出结构化的判定结果。
4. **反馈构造器（Feedback Engine）**：把验证器产生的冗长日志或报错信息进行过滤和改写，提炼成适合模型理解的修改建议。
5. **终止控制器（Termination Policy）**：根据测试通过情况、已消耗步数和重复状态检测，裁定流程是“成功退出”、“失败终止”还是“继续下一轮”。

![闭环控制系统的核心拓扑与生命周期流转图](/images/content-rebuild/22f0b293b6505e9a/fig-1.png)

在每一轮迭代中，数据流转遵循明确的顺序：
- 状态输入到**生成器**，生成当前版本的代码；
- 代码交给**验证器**，在沙箱或测试套件中运行并生成测试报告；
- **终止控制器**先检查测试报告和运行状态。如果测试全部通过，流程以成功状态结束；如果达到最大轮数或出现重复错误，流程直接熔断；
- 若判定需要继续修改，测试报告交由**反馈构造器**提炼为修复指引，更新回状态对象，进入下一轮循环。

这种拆分保证了每个组件的职责单一：生成器只负责生成，不负责判定；验证器只负责客观检查，不负责猜测如何修复；终止控制器掌握流程退出的最终决定权。

## 1. 生成器（Generator）：初始产出与针对性修改

生成器的任务是根据给定的要求产出候选结果。在循环系统中，生成器在不同轮次面对的输入有明显差异。

### 初始生成与增量修复
在第 0 轮（初始阶段），生成器接收的是任务说明、输入输出规格以及格式约束，生成第一版候选代码。

进入后续轮次后，生成器的工作模式应切换为“增量修复”。此时如果依然给模型发送最初的任务要求，模型往往会重新写出一份完全不同实现方式的代码，导致之前已经通过的测试用例被再次破坏。因此，增量阶段给生成器的输入应当聚焦在三项内容：
1. 原始任务目标与输入输出要求；
2. 上一轮生成的现有代码；
3. 经过反馈构造器整理的错误说明与定位。

![生成器处理增量反馈与上下文修剪机制](/images/content-rebuild/22f0b293b6505e9a/fig-2.png)

### 控制上下文，避免全量对话堆叠
实现循环时常见的做法是把每一轮的问答历史全量追加到消息列表中。这种做法在轮次增多时会带来两个问题：
- **Token 消耗增长**：前几轮被废弃的错误代码和长篇报错日志重复传入，大幅增加 Prefill 阶段的计算开销与 API 成本。
- **干扰模型注意力**：上下文堆积了大量早期犯过的错误和中间调试思路，模型容易在错误的模式上继续打补丁，甚至陷入越改越乱的循环。

实用的做法是修剪历史上下文：每次调用生成器时，只提供【任务要求】+【上一版本代码】+【本轮错误反馈】，丢弃中间轮次中已经被替换掉的代码快照与冗余对话。

<ReadMoreLock />

## 2. 验证器（Verifier）：提供客观的判定信号

验证器是循环系统能够收敛的前提。关于验证机制的设计，我们在 [Loop 的灵魂不是执行，而是验证](/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md) 中做过专项讨论。如果验证器无法给出明确、准确的对错信号，多轮修改就会失去方向。

![验证器的分层判定与结构化报告产出](/images/content-rebuild/22f0b293b6505e9a/fig-3.png)

验证器不建议单纯依赖“让同一个大模型看一遍代码并自我打分”。大模型容易出现确认偏误，往往会认为自己写出的逻辑完全合理。在代码和数据处理任务中，验证器应尽量引入确定性检查手段：

| 验证层级 | 常用检查工具 | 检查内容 | 结果类型 |
| :--- | :--- | :--- | :--- |
| **语法与规则层** | AST 解析、mypy、flake8、JSON Schema 校验 | 检查是否存在语法错误、类型不匹配或输出格式不合规 | 确定性（通过 / 报错行号） |
| **运行与逻辑层** | pytest、单元测试用例、沙箱执行环境 | 运行代码，检查是否出现异常（Exception）、返回值是否与断言一致 | 确定性（退出码 / 失败用例追踪） |
| **语义与质量层** | 规则打分脚本、独立评审模型（带明确评分准则） | 检查文案风格、解释是否清晰、是否符合业务规范 | 概率性（评分 / 审查意见） |

验证器的输出应统一封装为结构化对象，便于后续程序处理，例如包含以下信息：
- `passed: bool`：当前轮次是否整体通过；
- `failed_cases: list`：未通过的具体用例名、期望输出与实际输出；
- `stderr: str`：测试框架输出的异常堆栈。

## 3. 反馈构造器（Feedback Engine）：提炼可操作的修复信息

直接将测试失败时的几十行底层堆栈日志完整发送给模型，往往修复效果不佳。因为原始堆栈中包含大量测试执行框架内部的调用链路，模型可能抓不住核心错误原因。

反馈构造器的职责是把验证器输出的测试结果提炼成清晰的差分反馈，通常包括三个处理动作：

![反馈构造器将原始错误日志转化为聚焦修复指令](/images/content-rebuild/22f0b293b6505e9a/fig-4.png)

1. **日志剪枝**：过滤掉测试框架运行时的内部文件路径，只保留被测代码抛出异常的行号和关键报错类型（例如 `IndexError: list index out of range`）。
2. **对比断言结果**：把断言失败整理成结构化说明。例如将 `assert parse_metadata("") == {}` 失败明确提炼为：“当输入为空字符串时，函数抛出了未捕获的 KeyError，应返回空字典 `{}`”。
3. **限定修改范围**：在反馈提示中明确告知模型只修复目标函数内部逻辑，不得修改测试用例定义，也不要重新设计无关模块。

经过提炼后，模型接收到的是一份定位明确的工单，而不是混乱的系统日志。

## 4. 终止控制器（Termination Policy）：防范死循环与无序消耗

循环系统必须有明确的安全退出逻辑。如果没有严格的终止控制，遇到模型难以解决的边界条件时，系统会持续重试，造成计算资源浪费。

终止控制器在流程关键节点（如生成后防振荡、测试完成后验收）介入仲裁，根据多项指标综合裁定是否退出：

![终止控制器的多维熔断与退出仲裁判定](/images/content-rebuild/22f0b293b6505e9a/fig-5.png)

- **成功退出（Success）**：验证器返回测试全部通过（`passed == True`）。此时输出最新版本代码，正常结束流程。
- **轮次截断（Max Iterations）**：达到预设的最大迭代次数（例如 4 轮或 5 轮）。如果此时仍未全部通过，标记为“超时失败”，停止重试。
- **状态振荡检测（Oscillation Detection）**：在多轮修改中，模型有时会出现循环往复的现象：第 2 轮为了修 Bug A 改坏了功能 B；第 3 轮修了功能 B，代码又恢复到了第 1 轮带有 Bug A 的状态。控制器可以对每一轮产出的代码计算哈希值（Hash），一旦发现当前代码与历史某一轮完全相同，说明已陷入死循环，应立即熔断退出。
- **退化阻断（Regression Check）**：如果上一轮只错 1 个用例，本轮修改后报错激增到 3 个用例，说明当前的修改策略产生了负面作用。控制器可以记录历史最优版本，回滚到上一版代码再给出新的提示，或者直接熔断交由人工处理。

## 5. 状态对象与环境隔离（State & Environment）

多轮迭代需要维护连续的状态。状态对象通常包含全局任务、当前代码、历史哈希指纹、轮次计数和上一轮的反馈信息。

![状态字典在轮次间的流转与清理隔离机制](/images/content-rebuild/22f0b293b6505e9a/fig-6.png)

除了内存中的变量，如果验证过程涉及真实代码运行或文件生成，还需要注意运行环境的隔离：
- **工作目录隔离**：每次任务应在独立的临时目录中运行，避免多个循环实例并发读写同一个文件。
- **缓存与临时文件清理**：在 Python 运行测试时，生成 `.pyc` 或临时缓存文件可能会影响下一轮的读取，每一轮执行前应确保工作空间状态纯净。

## 最小循环执行框架代码

以下 Python 示例展示了包含状态定义、生成器、验证器、反馈整理与终止裁定的基本循环结构：

```python
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TestReport:
    passed: bool
    failed_reason: str = ""

@dataclass
class LoopState:
    task_goal: str
    iteration: int = 0
    max_iterations: int = 3
    current_code: str = ""
    feedback: Optional[str] = None
    seen_hashes: List[str] = field(default_factory=list)
    status: str = "RUNNING"  # RUNNING | SUCCESS | MAX_ITER_REACHED | OSCILLATION

class SimpleLoopController:
    def __init__(self, generator, verifier, feedback_engine, max_iter=3):
        self.generator = generator
        self.verifier = verifier
        self.feedback_engine = feedback_engine
        self.max_iter = max_iter

    def run(self, task_goal: str) -> LoopState:
        state = LoopState(task_goal=task_goal, max_iterations=self.max_iter)

        while state.iteration < state.max_iterations:
            state.iteration += 1

            # 1. 生成或修改代码
            state.current_code = self.generator.generate(
                goal=state.task_goal,
                current_code=state.current_code,
                feedback=state.feedback
            )

            # 2. 状态振荡检查（防死循环）
            code_hash = hashlib.md5(state.current_code.strip().encode()).hexdigest()
            if code_hash in state.seen_hashes:
                state.status = "OSCILLATION"
                break
            state.seen_hashes.append(code_hash)

            # 3. 运行验证
            report: TestReport = self.verifier.verify(state.current_code)

            # 4. 终止策略判断
            if report.passed:
                state.status = "SUCCESS"
                break

            # 5. 提炼下一轮的反馈信息
            state.feedback = self.feedback_engine.build_feedback(report)

        if state.status == "RUNNING":
            state.status = "MAX_ITER_REACHED"

        return state
```

## 适用边界与局限说明

循环系统能够显著改善自动化任务的通过率，但它并不是万能方案，在实际落地时需要注意其适用条件：
- **严重依赖验证器的完备度**：如果测试用例本身覆盖不全，模型即使完全通过了验证，产出的代码在实际运行中依然可能存在未测到的缺陷。
- **调用延迟与成本成倍增加**：多轮迭代意味着多次调用大模型与测试环境。如果一个任务单次调用需要 3 秒，迭代 3 轮加上测试执行可能耗时 10 秒以上，因此循环架构更适合离线任务、批处理任务或后台代码辅助，而不适合毫秒级响应的实时在线接口。
- **复杂逻辑的收敛上限**：如果模型本身的基础能力不足以理解任务的核心逻辑，给再多的反馈也可能会陷入原地打转。此时单纯增加循环上限并不能解决问题，需要优化初始任务拆解，或者结合人工介入。

掌握了循环的基本结构后，关于如何编写自动化代码修复循环的具体案例，可阅读 [最简单的 Loop：写代码、跑测试、自动修复](/frontier/loop-engineering/最简单的_Loop_写代码_跑测试_自动修复.md)；关于在生产环境中控制 Token 消耗与排查死循环的方案，可参考 [Loop Engineering 的坑：无限循环、烧 Token、越改越差](/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md)。
