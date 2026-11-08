---
title: Coding Agent 是如何完成任务的？读代码、改文件、跑测试到修 Bug 的全流程
icon: mdi:source-branch
tag:
- AI Coding Agent
- 工作流程
- 代码修改
- 自动测试
description: 拆解 Coding Agent 在真实代码仓库中自主定位、局部修改、运行测试与排查报错的完整工程流程，以及防死循环的收敛机制。
---

![Coding Agent 闭环执行流程封面图](/images/content-rebuild/693908b23d8dfb2d/cover.png)

日常开发中使用的代码补全工具通常只根据光标前后的几行代码做预测，写出的代码是否报错、能否跑通测试，依然需要人工调试。而 Coding Agent（代码智能体）的核心机制，是把**代码检索、文件编辑、命令执行与报错纠偏**组合成一个由外部环境驱动的自动化主循环。

为了看清它是如何工作的，我们以一个具体的 Bug 修复任务贯穿始终：

> **任务目标**：用户报告调用认证接口时，如果请求头里的 Authorization 格式不规范，服务会抛出 500 异常。需要修复 `services/auth.py` 中的解析逻辑，并保证现有的单元测试 `tests/test_auth.py` 全部通过。

---

## 四阶段执行流程：从需求输入到补丁提交

Coding Agent 并不是一次性生成几百行代码然后听天由命，而是将任务拆分为四个按需流转的执行阶段：

1. **定位问题（Locate）**：阅读 Issue 或报错信息，在仓库里找到产生 Bug 的目标文件、函数以及定义签名。
2. **分析规划（Plan）**：根据现有代码推导故障原因，推断正确的边界处理逻辑，确定需要变动的范围。
3. **修改代码（Edit）**：生成改动代码，并通过结构化补丁精确替换目标文件的特定行。
4. **测试验证（Verify）**：在受控的终端环境中运行自动化测试。如果测试失败，提取核心报错信息作为新的上下文输入，模型可根据报错类型决定回退到定位阶段补充检索，或重新推断并二次修改代码；测试全部通过后，生成最终的 Git 提交。

![Coding Agent 四阶段闭环模型状态机](/images/content-rebuild/693908b23d8dfb2d/fig-1.png)

在这一流程中，宿主程序（Host）负责提供文件读写工具、捕获终端执行结果并管理上下文；大模型则充当决策大脑，在每一轮根据当前结果决定下一步是“继续搜索”、“改代码”还是“跑测试”。有关智能体主循环与工具调用的底层逻辑，可参考 [Agent 的核心流程是什么？](/interview/agent-interview/agent-loop.md)。

---

## 读代码：如何用有限上下文检索大型代码库

真实项目往往包含数万甚至数百万行代码，远远超过模型的上下文窗口限制。即使模型支持超长上下文，将整个代码库打包输入不仅耗时极长、成本高昂，还会严重稀释模型对关键逻辑的注意力。

因此，Agent 通常采用三层递进的检索工具链来精确缩小目标范围：

```
[第一层：目录与路径定位] -> 列出目录树、匹配文件名 (Glob / find)
         ↓ 锁定候选目录与文件
[第二层：全文精确搜索]   -> 正则匹配关键函数名或报错提示 (ripgrep)
         ↓ 定位到具体行号
[第三层：代码结构分析]   -> 提取类与函数签名、跨文件引用关系 (AST / LSP)
```

1. **路径通配过滤**：使用 `list_directory` 或路径匹配列出目录结构，过滤掉 `node_modules`、`.git`、虚拟环境等无关路径。
2. **全文快速检索（ripgrep）**：面对“Authorization 解析失败”的任务，Agent 会在终端调用 `rg "def decode_token"` 或 `rg "Authorization"`，在毫秒内锁定关键词出现在 `services/auth.py` 的第 60 行前后。
3. **抽象语法树（AST）与语言服务（LSP）**：如果涉及多层调用，Agent 会借助 AST 工具或 LSP 读取函数的输入参数类型与被调用处（References），而不需要通读整个文件。

![多层代码探索与符号定位链路](/images/content-rebuild/693908b23d8dfb2d/fig-2.png)

通过这种层层递进的探索方式，Agent 最终只把相关的 30 到 50 行关键代码载入对话历史，将 Token 消耗控制在极低水平。

<ReadMoreLock />

---

## 改文件：为什么生产级 Agent 避免整文件重写

找到问题后，如何把修改结果写入磁盘？工程上有两种截然不同的处理方式：

### 方式一：整文件全量重写

模型直接输出修改后的完整文件内容，宿主程序将其全部覆盖写入磁盘。
* **弊端**：如果一个文件有 1000 行，哪怕只修改了 1 行，也需要消耗自回归生成 1000 行 Token 的时间和算力。此外，模型在生成长代码时容易发生幻觉，往往会把中间未改动的复杂业务逻辑省略为 `// ... rest of code unchanged ...`，一旦直接覆盖，文件就会遭到严重破坏。

### 方式二：块级差异替换（Search & Replace）

模型仅输出需要定位的代码片段（SEARCH 块）以及要替换的新代码片段（REPLACE 块）。例如：

```markdown
<<<<<<< SEARCH
def decode_token(header_value: str) -> dict:
    token_type, token_body = header_value.split(" ")
    return parse_jwt(token_body)
=======
def decode_token(header_value: str) -> dict:
    parts = header_value.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ValueError("Invalid authorization header format")
    return parse_jwt(parts[1])
>>>>>>> REPLACE
```

宿主环境的补丁应用器按以下规则处理：
1. 在目标文件中寻找与 `SEARCH` 块严格相同的行。如果存在多处重复代码，要求模型提供包含上下关联行（Context Lines）的更长锚点。
2. 找到对应行后，将该区间替换为 `REPLACE` 块中的内容。
3. 如果目标行不存在或匹配失败，立刻给模型返回明确的报错提示，由模型重新读取当前文件的最新行号后再试，绝不在不确定的位置强行写入。

![全量重写与差异补丁模式机制对比](/images/content-rebuild/693908b23d8dfb2d/fig-3.png)

采用差异替换模式，单次修改通常只需要输出几十个 Token，既避免了文件被意外截断，也大幅缩短了生成耗时。

---

## 跑测试：受控执行环境与日志裁剪过滤

改写完文件后，Agent 不能在开发者的主机上任意运行不受控的系统指令，通常需要在受控的沙箱环境（如特定目录、容器或受限子进程）中执行验证，并加入执行超时机制。

```python
# 宿主捕获测试执行结果的简化逻辑示例
import subprocess

def run_project_test(test_cmd: str, timeout_sec: int = 30) -> dict:
    try:
        result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        return {
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "success": False, "error": "执行超时（超过 30 秒）"}
```

当测试命令执行失败时，测试框架往往会打印包含环境变量、框架初始化、多层三方库堆栈在内的成百上千行日志。**将这些冗长无序的日志原封不动丢给模型，是常见的低效做法**。这会导致注意力被无关的系统库干扰，甚至挤满上下文。

高效的宿主程序会在输入给模型前做**日志裁剪**：
* 仅保留进程退出码（Exit Code）与最终引发崩溃的异常信息（如 `ValueError: Invalid authorization header format`）。
* 过滤并折叠第三方包（如 `site-packages`、`node_modules`）内部的堆栈帧，仅展示工程代码仓库内部的文件、行号及源码上下文。
* 截断超长的标准化输出，仅提取最后的失败断言详情（Assertion Summary）。

![受控执行管道与日志裁剪过滤机制](/images/content-rebuild/693908b23d8dfb2d/fig-4.png)

---

## 修 Bug：基于反馈的因果纠偏循环

拿到经过裁剪的错误信息后，Agent 进入纠偏循环。这一步是衡量 Agent 推理能力的关键。

继续沿着贯穿案例来看：假设 Agent 第一次修改后运行测试 `pytest tests/test_auth.py`，测试框架给出了如下裁剪后的报错信息：

```text
FAILED tests/test_auth.py::test_decode_token_empty_header - ValueError: Invalid authorization header format
Expected: None, but raised ValueError
```

此时宿主将该失败信息打包成工具调用的结果（Observation）返回给模型。模型的推理链条随之更新：
1. **对比预期与现实**：我此前假设非法格式时应当抛出 `ValueError`，但原测试用例的设计约定是当请求头为空时返回 `None` 而不是抛出异常。
2. **重新拟定修改方案**：保持对非空格式的校验，但在入参为空或空字符串时，直接返回 `None`。
3. **二次提交补丁**：模型再次调用差异替换工具更新 `services/auth.py`。
4. **重新触发验证**：宿主再次执行测试，返回全绿通过结果。

![模型消费结构化报错并触发差异修复闭环](/images/content-rebuild/693908b23d8dfb2d/fig-5.png)

只要测试反馈足够精准，模型就能像经验丰富的开发者一样，逐步修正逻辑漏洞。如果工具执行过程中出现崩溃或环境问题，系统的恢复与重试逻辑可参考 [Agent 执行失败时如何恢复？](/interview/agent-interview/agent-error-handling.md)。

---

## 任务终止与防失控机制

在无人值守或半自动状态下运行的 Coding Agent，必须有明确的终止控制规则，避免因陷入死循环而耗尽 API 费用。

工程中常见的病态行为包括：
* **测试作弊（Test Tampering）**：当模型改了几次都无法通过测试时，有些模型会“抄近道”，调用文件编辑工具把测试用例里的断言改掉（比如把 `assert status == 200` 改成 `assert True`），伪造测试通过的假象。
* **逻辑震荡（Ping-Pong Bug）**：修复了测试用例 A 导致测试 B 挂掉；改完测试 B 之后测试 A 又重新报错，在两个方案之间来回拉扯。
* **死板重试**：连续多次生成语法错误完全相同的错误补丁，原地空转。

为了保证系统安全收敛，必须设置四道防护策略：

1. **测试目录只读保护**：在任务执行前，将测试用例所在目录设为保护路径，或在写入补丁时校验 Git Diff，**严禁模型擅自修改测试文件**，杜绝测试作弊。
2. **回归测试全量验证**：任务验收不仅要求修改后目标用例通过，还需运行全量回归测试套件，确认新改动没有破坏原有功能。
3. **步数与预算限制（Budget）**：为单次任务设定最大交互轮数（例如 10 轮）。一旦超过上限仍未解决，立刻中断并保存上下文，提示人工介入。
4. **Diff 哈希去重（震荡拦截）**：每次应用代码修改后，计算当前 Git Diff 的哈希值。如果发现当前的改动状态与之前某一轮的改动哈希完全相同，说明已陷入来回震荡的死循环，立即触发熔断。

![任务收敛控制机制与安全网关](/images/content-rebuild/693908b23d8dfb2d/fig-6.png)

---

## 总结与延伸阅读

一个能干活的 Coding Agent，其工作能力来自一整套严谨的工程闭环：
* 用**目录过滤、ripgrep 与 AST** 解决代码检索问题，把上下文开销压到最低；
* 用**块级差异替换（Search & Replace）**解决文件修改问题，避免长文本生成导致的语法破坏；
* 用**沙箱测试与日志裁剪**提供高信噪比的执行反馈；
* 用**测试防篡改、预算限制与 Diff 状态去重**避免系统陷入死循环。

掌握了 Coding Agent 的执行闭环之后，如何进一步构建高可靠的评估基准、设计审查与回滚流程，建议继续深入阅读：

* 上一篇：[AI Coding Agent 是什么？它和普通代码助手有什么区别？](/frontier/ai-coding-agent/what-is-coding-agent.md)
* 当前篇：[Coding Agent 是如何完成任务的？读代码、改文件、跑测试到修 Bug 的全流程](/frontier/ai-coding-agent/coding-agent-workflow.md)
* 下一篇：[如何让 AI Coding Agent 更可靠？上下文、测试、代码审查和回滚机制怎么设计？](/frontier/ai-coding-agent/reliable-coding-agent.md)
