---
title: 代码自修复循环：如何用测试驱动代码自动生成与修复？
icon: mdi:refresh
date: 2026-07-16
tag:
- Loop Engineering
- Agent
description: 通过一个具体的算法修复案例，讲解如何利用子进程沙箱、pytest 退出码与错误提取器构建代码自修复循环。
---

在上一讲[/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md](/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md)中，我们讨论了循环的核心在于是否存在客观且确定性的验证机制。在代码生成场景下，单元测试提供了这种确定性：一段代码要么通过测试，要么抛出异常并给出明确的断言差异。

本篇通过一个具体的 Python 算法案例，拆解一个不依赖任何重型框架的代码自修复循环（Code Self-Repair Loop）。系统通过直接调用子进程、运行测试套件、提取错误信息并构造差分提示词，驱动大语言模型完成代码修复。

![代码自修复闭环全景](/images/content-rebuild/9a9b22178326bc39/cover.png)

---

## 场景定义：以固定测试集驱动代码收敛

代码自修复的目标不是无限制的开放探索，而是让模型生成的代码逐步满足预设的规范。

我们以一个常见的算法需求为例：实现合并重叠区间的函数 `merge_intervals`。在进入循环前，需要明确定义系统的三项要素：

1. **不可变的输入**：包括任务需求描述，以及一份预先写好的单元测试文件（`test_solution.py`）。在整个修复过程中，测试文件处于只读状态，严禁模型修改测试用例。
2. **轮次中的状态**：当前轮次生成的源代码（`solution.py`）、测试执行的退出码与终端输出，以及历史失败信息。
3. **验收判定**：测试运行器的退出码为 `0`（全部测试用例通过）；若达到最大重试轮次（如 3 次）仍未全部通过，则终止并报告失败。

![任务规格与测试驱动开发场景](/images/content-rebuild/9a9b22178326bc39/fig-1.png)

这种结构与测试驱动开发（TDD）的思想一致：由固定的测试用例定义预期行为。如果允许模型同时修改业务代码和测试代码，模型往往会通过删除测试或修改断言来消除报错，造成功能已经正确的假象。

---

## 运行隔离：使用 subprocess 限制超时与资源

大模型生成的代码属于未经验证的外部输入，不能在主程序中直接调用 `exec()` 或 `eval()` 执行。如果生成的代码包含死循环或缺少终止条件的递归，主程序会被直接卡死。

最基础的保护措施是将测试执行放入独立的子进程中，并设置超时上限：

```python
import subprocess
import sys
from pathlib import Path
from typing import Tuple

def execute_pytest_in_sandbox(
    workspace_dir: Path, 
    timeout_seconds: float = 5.0
) -> Tuple[int, str, str]:
    """
    在指定目录以子进程启动 pytest，限制最长运行时间。
    返回: (退出码, 标准输出, 标准错误)
    """
    cmd = [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=short"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        # 超时截断：防止死循环持续占用 CPU
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"Execution timed out after {timeout_seconds} seconds. Suspected infinite loop."
        return -1, stdout, stderr
    except Exception as exc:
        return -2, "", f"Subprocess execution failed: {str(exc)}"
```

<ReadMoreLock />

在这段代码中，`timeout_seconds=5.0` 起到了看门狗的作用。如果模型写出的循环缺少步进语句，子进程会在 5 秒超时后被强制终止并抛出异常，执行函数捕获后约定返回 `-1` 退出码，主进程据此可以生成“运行超时”的错误信号，而不会被动挂起。

需要注意的是，`subprocess` 只能限制运行时间和捕获输出，无法阻止恶意的系统调用或文件篡改。若在生产环境中运行不可信代码，需要使用容器或只读沙箱环境进一步隔离。

![子进程沙箱执行与超时熔断机制](/images/content-rebuild/9a9b22178326bc39/fig-2.png)

---

## 错误信息提取：从 pytest 输出中抓取关键上下文

Pytest 失败时的输出通常很长，夹杂着环境版本、测试耗时汇总和 ANSI 终端控制字符。如果将原始输出未经处理直接放进下一次的模型提示词中，会带来两个问题：

1. **占用上下文窗口**：冗长的环境信息和统计行占用了有效 Token。
2. **分散注意力**：模型可能将注意力放在末尾的统计摘要上，忽视具体的异常类型和变量差分。

一次有效的错误反馈，应当聚焦在三个核心信息上：
- **失败的测试函数名**；
- **异常类型与错误信息**（如 `IndexError: list index out of range` 或 `AssertionError`）；
- **报错的具体代码行上下文**。

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestFailureSignal:
    is_timeout: bool
    failed_test_name: Optional[str]
    error_type: str
    error_message: str
    code_context: str

def extract_failure_signal(returncode: int, stdout: str, stderr: str) -> Optional[TestFailureSignal]:
    """从测试运行结果中提取关键错误信息。若全部通过则返回 None。"""
    if returncode == 0:
        return None
    
    # 超时情况特殊处理
    if returncode == -1:
        return TestFailureSignal(
            is_timeout=True,
            failed_test_name=None,
            error_type="TimeoutExpired",
            error_message="代码执行超时，请检查算法中是否存在死循环或缺少终止条件的递归。",
            code_context=""
        )
    
    combined_output = stdout + "\n" + stderr
    
    # 提取失败的测试函数名（例如 FAILURES: test_empty_intervals）
    test_name_match = re.search(r"_{3,}\s+(test_[a-zA-Z0-9_]+)\s+_{3,}", combined_output)
    failed_test = test_name_match.group(1) if test_name_match else "UnknownTest"
    
    # 提取以 E 开头的错误行和代码上下文行
    error_lines = []
    context_lines = []
    for line in combined_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("E ") or stripped.startswith("E:"):
            error_lines.append(stripped[2:].strip())
        elif "> " in line or "solution.py:" in line:
            context_lines.append(stripped)
            
    return TestFailureSignal(
        is_timeout=False,
        failed_test_name=failed_test,
        error_type=error_lines[0].split(":")[0] if error_lines else "RuntimeError",
        error_message="\n".join(error_lines) if error_lines else combined_output[-300:],
        code_context="\n".join(context_lines[:5])
    )
```

![Pytest输出过滤与结构化提取流程](/images/content-rebuild/9a9b22178326bc39/fig-3.png)

过滤后，回传给模型的不是整屏的终端日志，而是明确的失败位置与报错原因，模型可以直接根据报错回溯当前代码的缺陷。

---

## 构造修复提示词：提供当前代码与具体报错

在自动修复过程中，常见的一个误区是“重开式生成”：一旦测试失败，就把原始需求发给模型，让它“换个思路重新实现一遍”。

从头重写容易丢弃前几轮已经写对的部分，导致原本通过的测试再次失败。更稳定的做法是提供当前代码及具体的测试报错，引导模型在现有基础上做针对性修改：

1. **带行号的当前代码**：作为修改的定位基准；
2. **本次失败的具体用例与异常**：指明具体是在哪种输入下出现的问题；
3. **修改要求**：要求保持原函数签名不变，并输出修复后的完整代码。

![盲目重试与精准差分提示词对比](/images/content-rebuild/9a9b22178326bc39/fig-4.png)

```python
def build_repair_prompt(current_code: str, signal: TestFailureSignal, task_description: str) -> str:
    """构建针对当前错误的修复提示词。"""
    code_with_lines = "\n".join(
        f"{idx + 1:3d} | {line}" for idx, line in enumerate(current_code.splitlines())
    )
    
    prompt = f"""你编写的代码未通过单元测试，请基于以下错误信息进行定向修复。\n\n### 原始需求\n{task_description}\n\n### 当前实现（带行号）\n```python\n{code_with_lines}\n```\n\n### 测试失败信息\n- 失败用例: {signal.failed_test_name or '全局运行时'}\n- 异常类型: {signal.error_type}\n- 核心报错:\n{signal.error_message}\n"""
    if signal.code_context:
        prompt += f"- 报错上下文:\n{signal.code_context}\n\n"
        
    prompt += """### 修复约束\n1. 分析报错信息和出错行，定位逻辑缺陷。\n2. 保持原有的函数名、参数与返回值类型不变。\n3. 仅返回修复后的完整 Python 代码，置于 ```python ... ``` 块中，不要输出其他解释。"""
    return prompt
```

这样，模型可以在具体的代码行和异常之间建立对应关系，把修改范围约束在产生问题的局部逻辑上。

---

## 状态循环实现：迭代控制与终止条件

我们可以把代码自修复建模为一个状态机：

- **INIT / GENERATE**：根据需求生成第一版初始代码。
- **EXECUTE**：在子进程中运行测试，获取退出码。
- **PARSE**：分析退出码。如果为 `0`，转移到 `SUCCESS` 并退出；如果不为 `0`，提取报错信号。
- **REPAIR**：组装包含代码和报错的提示词，调用模型生成新代码。如果轮次达到设定的上限，转移到 `FAILED` 并退出。

![状态机控制流与终止判定分支](/images/content-rebuild/9a9b22178326bc39/fig-5.png)

明确的终止判定避免了在模型无法解决问题时发生无休止的请求和消耗。

---

## 最小可运行实现

下面给出一个完整的实现。为了便于直接运行验证，这里使用一个预设的 `MockLLM` 模拟大模型的两次返回：第一轮输出存在边界缺陷的代码（未处理空列表），第二轮根据测试报错输出修复后的代码。

```python
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ==================== 1. 沙箱执行、信号解析与提示词组装 ====================
@dataclass
class TestFailureSignal:
    is_timeout: bool
    failed_test_name: Optional[str]
    error_type: str
    error_message: str
    code_context: str

def execute_pytest_in_sandbox(
    workspace_dir: Path, 
    timeout_seconds: float = 5.0
) -> Tuple[int, str, str]:
    cmd = [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=short"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"Execution timed out after {timeout_seconds} seconds. Suspected infinite loop."
        return -1, stdout, stderr
    except Exception as exc:
        return -2, "", f"Subprocess execution failed: {str(exc)}"

def extract_failure_signal(returncode: int, stdout: str, stderr: str) -> Optional[TestFailureSignal]:
    if returncode == 0:
        return None
    if returncode == -1:
        return TestFailureSignal(
            is_timeout=True,
            failed_test_name=None,
            error_type="TimeoutExpired",
            error_message="代码执行超时，请检查算法中是否存在死循环或缺少终止条件的递归。",
            code_context=""
        )
    combined_output = stdout + "\n" + stderr
    test_name_match = re.search(r"_{3,}\s+(test_[a-zA-Z0-9_]+)\s+_{3,}", combined_output)
    failed_test = test_name_match.group(1) if test_name_match else "UnknownTest"
    error_lines = []
    context_lines = []
    for line in combined_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("E ") or stripped.startswith("E:"):
            error_lines.append(stripped[2:].strip())
        elif "> " in line or "solution.py:" in line:
            context_lines.append(stripped)
    return TestFailureSignal(
        is_timeout=False,
        failed_test_name=failed_test,
        error_type=error_lines[0].split(":")[0] if error_lines else "RuntimeError",
        error_message="\n".join(error_lines) if error_lines else combined_output[-300:],
        code_context="\n".join(context_lines[:5])
    )

def build_repair_prompt(current_code: str, signal: TestFailureSignal, task_description: str) -> str:
    code_with_lines = "\n".join(
        f"{idx + 1:3d} | {line}" for idx, line in enumerate(current_code.splitlines())
    )
    prompt = f"""你编写的代码未通过单元测试，请基于以下错误信息进行定向修复。\n\n### 原始需求\n{task_description}\n\n### 当前实现（带行号）\n```python\n{code_with_lines}\n```\n\n### 测试失败信息\n- 失败用例: {signal.failed_test_name or '全局运行时'}\n- 异常类型: {signal.error_type}\n- 核心报错:\n{signal.error_message}\n"""
    if signal.code_context:
        prompt += f"- 报错上下文:\n{signal.code_context}\n\n"
    prompt += """### 修复约束\n1. 分析报错信息和出错行，定位逻辑缺陷。\n2. 保持原有的函数名、参数与返回值类型不变。\n3. 仅返回修复后的完整 Python 代码，置于 ```python ... ``` 块中，不要输出其他解释。"""
    return prompt

# ==================== 2. 模拟大模型行为 ====================
class MockLLM:
    """模拟模型：第 1 轮返回缺少边界判断的代码，第 2 轮返回补充了判断的代码。"""
    def __init__(self):
        self.call_count = 0
        
    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self.call_count == 1:
            # 缺陷实现：当 intervals 为空列表时，intervals[0] 会报 IndexError
            return """```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged
```"""
        else:
            # 修复实现：补充空列表保护判断
            return """```python
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged
```"""

def extract_code_block(llm_output: str) -> str:
    """从输出文本中提取 python 代码块。"""
    match = re.search(r"```python\s*(.*?)\s*```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_output.strip()

# ==================== 3. 控制器实现 ====================
class CodeRepairLoop:
    def __init__(self, task_spec: str, test_code: str, max_iterations: int = 3):
        self.task_spec = task_spec
        self.test_code = test_code
        self.max_iterations = max_iterations
        self.llm = MockLLM()  # 实际使用时替换为真实 API 客户端

    def run(self) -> Tuple[bool, str, int]:
        """
        运行修复循环。
        返回: (是否通过, 最终代码, 消耗轮次)
        """
        with tempfile.TemporaryDirectory() as temp_dir_str:
            work_dir = Path(temp_dir_str)
            # 写入只读测试文件
            (work_dir / "test_solution.py").write_text(self.test_code, encoding="utf-8")
            
            current_code = ""
            last_signal = None
            
            for iteration in range(1, self.max_iterations + 1):
                print(f"\n>>> [轮次 {iteration}/{self.max_iterations}] 开始")
                
                # 1. 构造提示词并生成代码
                if iteration == 1:
                    prompt = f"请根据需求实现函数：\n{self.task_spec}\n请仅返回 ```python 代码块。"
                else:
                    assert last_signal is not None
                    prompt = build_repair_prompt(current_code, last_signal, self.task_spec)
                
                raw_output = self.llm.generate(prompt)
                current_code = extract_code_block(raw_output)
                
                # 写入被测文件
                (work_dir / "solution.py").write_text(current_code, encoding="utf-8")
                
                # 2. 运行测试
                ret_code, stdout, stderr = execute_pytest_in_sandbox(work_dir, timeout_seconds=3.0)
                
                # 3. 解析结果
                signal = extract_failure_signal(ret_code, stdout, stderr)
                if signal is None:
                    print(f"[✓] 验证通过！在第 {iteration} 轮通过全部测试。")
                    return True, current_code, iteration
                
                print(f"[!] 测试未通过: {signal.error_type} - {signal.error_message.splitlines()[0]}")
                last_signal = signal
                
            print(f"[✗] 达到最大迭代次数 ({self.max_iterations})，修复未完成。")
            return False, current_code, self.max_iterations

# ==================== 4. 运行测试 ====================
if __name__ == "__main__":
    TASK_SPEC = "编写函数 merge_intervals(intervals)，合并所有重叠的区间，并返回排序后的区间数组。"
    
    # 测试套件：包含常规用例与空列表边界
    TEST_SUITE = """
from solution import merge_intervals

def test_normal_intervals():
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]

def test_empty_intervals():
    assert merge_intervals([]) == []
"""
    
    loop = CodeRepairLoop(TASK_SPEC, TEST_SUITE, max_iterations=3)
    success, final_code, rounds = loop.run()
    print(f"\n最终结果: {'成功' if success else '失败'} (运行轮次: {rounds})")
```

### 实际运行中的边界情况

在将上述逻辑扩展到更复杂的系统时，有三个常见问题需要处理：

1. **语法错误导致测试无法收集**：如果模型返回的代码包含无法解析的语法错误（如括号未闭合），pytest 会在收集测试阶段直接崩溃，此时无法捕获具体的断言错误。提取函数需要能识别 `SyntaxError` 并将语法错误行回传给模型。
2. **修复抖动（Fix Flapping）**：模型在第 2 轮修复了用例 A，但在第 3 轮修改时又破坏了原本正常的用例 B。因此，每一轮迭代都必须运行整个测试套件，不能只运行上一轮失败的用例；只有全部用例通过才算成功。
3. **环境与文件隔离**：生成的代码可能尝试访问文件系统其他目录或发起网络请求。如果代码来源不可控，子进程需要运行在网络受限的轻量容器或 Linux 命名空间内。

![修复抖动与边界防御分析](/images/content-rebuild/9a9b22178326bc39/fig-6.png)

---

## 延伸阅读

本篇讨论了在有明确退出码的环境下如何组织自动修复循环。相关主题与进阶内容可以参考：

- 上一步核心原则回顾：[/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md](/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md)
- 循环架构与 Agent 概念的对应关系：[/frontier/loop-engineering/Loop_Engineering_和_Agent_有什么关系.md](/frontier/loop-engineering/Loop_Engineering_和_Agent_有什么关系.md)
- 避免无效重试与振荡的工程实践：[/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md](/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md)
- 面试中关于执行循环的常见讨论：[/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md)
