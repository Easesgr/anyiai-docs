---
title: AI Coding Agent 怎么工作？从单行补全到自主修复缺陷的工程路径
icon: mdi:code-tags
description: 解析 AI Coding Agent 的核心工作机制与控制权演进，梳理读改测修工程闭环与门禁设计，提供最小原型实验方案与系统学习路径。
---

在日常开发中，代码补全工具通常只根据当前打开的文件和光标附近的字符预测下一行代码。一旦遇到需要跨文件修改的复杂缺陷，或者修改后引发了测试报错，补全工具就无法继续处理。AI Coding Agent（软件编码智能体）正是为了解决这类问题而设计的系统：它让大模型在受控的软件工程环境中，自主调用工具来探索代码、定位错误、修改文件并运行测试，直到代码达到预期状态。

![AI Coding Agent 模块导读与核心研究框架](/images/content-rebuild/da6586a99b4227d5/cover.png)

---

## 一、从贯穿示例看 Coding Agent 的工作方式

为了直观理解它的工作机制，我们来看一个具体的调试场景：

假设代码仓库中包含计算模块 `calculator.py` 和测试套件 `test_calculator.py`。运行 `pytest` 时，出现如下失败报错：

```text
FAILED test_calculator.py::test_divide_by_zero - ZeroDivisionError: division by zero
```

### 1. 单点补全工具的处理方式
开发者必须手动打开 `calculator.py`，把光标定位到 `divide(a, b)` 函数内部。补全工具会读取光标前后的上下文，给出一行防御性判断提示（如 `if b == 0: raise ValueError(...)`）。至于测试用例到底期望抛出哪种异常、其他文件是否有类似调用、修改后测试能否跑通，工具本身无法获知，完全依赖开发者的人工核对与调试。

### 2. Coding Agent 的处理方式
Agent 接收到的任务是“修复 `pytest` 报告的失败项”。它的执行流程不再局限于光标所在行，而是自主推进一个交互循环：
1. **获取错误**：Agent 调用终端执行工具运行 `pytest`，拿到 `ZeroDivisionError` 的完整调用栈与失败用例名称。
2. **查找源码**：根据调用栈中的文件路径和行号，Agent 调用文件读取工具查看 `calculator.py` 中 `divide` 函数的实现，同时读取 `test_calculator.py` 中对该函数的断言逻辑，确认用例期望抛出的是 `ValueError("divisor cannot be zero")`。
3. **应用修改**：Agent 生成补丁并写入 `calculator.py`。
4. **测试验证**：Agent 再次在终端执行 `pytest`。如果退出码为 0，说明修复成功，任务终止；如果出现新的报错，Agent 则把新报错加入上下文，进入下一轮修复。

根据 Anthropic 在构建 Agent 系统时的定义，这类系统在架构上可以划分为两种形态：
* **工作流（Workflows）**：大模型与工具的调用流程由预先编写的代码路径编排，适合步骤相对明确、包含固定规则的任务（需要注意，代码编排预设路径不等于整个系统百分之百确定，模型生成和外部环境状态仍具有不确定性）；
* **自主代理（Agents）**：模型根据环境的每一次反馈（例如测试输出或检索结果），动态决定下一步该读取哪个文件、调用哪个工具或何时结束，由模型自主掌控执行流程，适合探索性更强的调试场景。

软件工程兼具“确定性规则”（编译器报错、测试断言、语法树结构）与“语义发散性”（需求意图、命名风格、代码重构）。将确定性的检查脚本与大模型的动态探索结合起来，是构建可靠 Coding Agent 的核心思路。

<ReadMoreLock />

---

## 二、从概念到落地的三层进阶内容

本专题通过三篇文章，循序渐进地解析 Coding Agent 的系统设计：

![AI Coding Agent 专题三层知识进阶架构](/images/content-rebuild/da6586a99b4227d5/fig-1.png)

### 1. 概念辨析：从单点补全到自主执行
* **对应文章**：[/frontier/ai-coding-agent/what-is-coding-agent.md](/frontier/ai-coding-agent/what-is-coding-agent.md)
* **解决问题**：厘清 Coding Agent 与传统 Copilot 代码补全工具在系统控制权上的差异。
* **机制重点**：对比不同交互范式下的控制权归属。分析智能体在感知项目环境（读取目录树、解析终端输出）、规划修改步骤与执行具体编辑操作时的状态流转。

### 2. 执行闭环：读、改、测、修的完整链路
* **对应文章**：[/frontier/ai-coding-agent/coding-agent-workflow.md](/frontier/ai-coding-agent/coding-agent-workflow.md)
* **解决问题**：Agent 如何在多个源码文件之间独立排查并修复具体缺陷？
* **机制重点**：拆解“环境探索 $\to$ 缺陷复现 $\to$ 代码定位 $\to$ 补丁生成 $\to$ 回归测试”的标准链路。探讨全文检索、语义索引与精确语法符号（如 AST、ctags）在长代码库中的协同使用方式。

### 3. 可靠性工程：让 Agent 在工程约束下稳定交付
* **对应文章**：[/frontier/ai-coding-agent/reliable-coding-agent.md](/frontier/ai-coding-agent/reliable-coding-agent.md)
* **解决问题**：如何防止 Agent 陷入死循环、破坏不相关的代码或直接篡改测试断言？
* **机制重点**：上下文治理策略（输出截断与分块读取）、确定性门禁（静态语法解析检查、代码风格校验）、容器沙箱隔离，以及基于 Git 提交点的快照保存与回滚机制。

---

## 三、学习路线与前后依赖

学习 Coding Agent 建议遵循由浅入深的工程实践路径。在进入本专题前，建议先掌握模型交互与评测的基础概念：

![Coding Agent 学习路径与前后置模块依赖关系](/images/content-rebuild/da6586a99b4227d5/fig-2.png)

* **前置基础**：阅读 [/fundamentals/llm-basics/README.md](/fundamentals/llm-basics/README.md) 了解大模型的自回归生成与上下文窗口机制；查阅 [/fundamentals/evaluation-metrics/README.md](/fundamentals/evaluation-metrics/README.md) 了解基准评测的度量方式。
* **后续拓展**：完成本专题后，如果要在实际生产中搭建具备状态持久化、支持人工审批介入（Human-in-the-loop）的复杂智能体，可以继续参考 [/architecture/README.md](/architecture/README.md) 中的状态图与系统架构实现。

---

## 四、动手验证：构建最小修复原型

检验是否理解 Coding Agent 机制的最有效方法，是编写一个脱离复杂第三方框架的最小闭环脚本。以下代码展示了一个基于状态循环的单文件 Agent 原型：

### 状态机与执行逻辑伪代码

```python
# 最小 Coding Agent 原型逻辑（伪代码示意）
import ast
import subprocess

def run_tests() -> tuple[bool, str]:
    result = subprocess.run(["pytest", "test_calculator.py"], capture_output=True, text=True)
    return (result.returncode == 0, result.stdout + result.stderr)

def is_syntax_valid(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

initial_source = open("calculator.py").read()
state = {
    "source": initial_source,
    "iteration": 0,
    "max_retries": 3,
    "passed": False
}

state["passed"], test_output = run_tests()

while not state["passed"] and state["iteration"] < state["max_retries"]:
    state["iteration"] += 1
    
    # 1. 组装提示词：包含当前源码与具体的测试报错信息
    prompt = f"Source:\n{state['source']}\n\nTest Error:\n{test_output}\nFix the bug. Return ONLY valid python code."
    new_code = llm.generate(prompt)
    
    # 2. 门禁检查：写入前进行 AST 静态解析，防止语法损坏
    if not is_syntax_valid(new_code):
        print(f"Iteration {state['iteration']}: 生成代码存在语法错误，跳过写入")
        continue
        
    # 3. 写入文件并在测试沙箱中执行验证
    with open("calculator.py", "w") as f:
        f.write(new_code)
    
    state["passed"], test_output = run_tests()
    state["source"] = new_code
    
    # 4. 若测试通过则跳出循环；若耗尽重试仍未通过，触发原子回滚
    if state["passed"]:
        print("测试通过，修改成功完成。")
        break
else:
    if not state["passed"]:
        with open("calculator.py", "w") as f:
            f.write(initial_source)
        print("重试次数耗尽，测试仍未通过，已回滚到初始代码。")
```

![最小 Coding Agent 原型执行与验收回路](/images/content-rebuild/da6586a99b4227d5/fig-3.png)

### 验收标准与边界条件
在运行此类实验时，需要关注三个核心设计约束：
1. **语法门禁拦截**：模型输出的代码在覆盖原文件前，必须经过静态解析校验。若语法不合法，应直接拦截并提示重新生成，避免破坏原有代码库。
2. **以测试结果为反馈**：模型不能仅凭自己的“推测”判断修改是否正确，每一次迭代的输入都必须基于外部测试工具的真实标准输出和退出码。
3. **受限的修改范围**：在原型中必须对可写路径进行限制（例如禁止模型编辑测试文件 `test_calculator.py` 本身），防止模型通过删除测试断言来制造“测试通过”的假象。
