---
title: 构建代码生成与调试助手：仓库上下文组装、补丁生成与测试验证
icon: mdi:laptop
description: 介绍终端代码助手的设计与实现，涵盖文件树与 AST 骨架提取、Unified Diff 容错应用、测试沙箱隔离及多轮自动纠错循环。
---

让大语言模型编写独立的算法函数相对容易，但在已有的工程仓库中维护代码，需要处理跨文件关联、精确修改局部代码以及通过测试验证修改结果等问题。

如果直接将所有文件塞入提示词，会快速消耗上下文窗口，并稀释注意力；若要求模型每次都输出修改后的完整源码文件，生成速度慢、消耗 Token 多，还容易在重写过程中误删原有的注释或辅助逻辑。

实用的代码助手通常采用“定位代码 -> 组装分级上下文 -> 生成局部补丁 -> 隔离沙箱测试 -> 依赖报错循环修复”的流程。本文通过一个具体的计算器异常处理案例，拆解这一流程的实现细节、错误处理与验收方法。

![代码助手的闭环工作流：从问题输入到补丁验证](/images/content-rebuild/0a308926bd0f8d00/cover.png)

---

### 1. 代码助手的四个执行阶段

当开发者向助手报告一个 Bug 或提出修改需求时，系统按顺序执行以下四个阶段：

1. **问题定位（Issue Localization）**：从用户输入的自然语言描述或测试用例的 Traceback 中，提取目标文件路径和函数名。
2. **上下文组装（Context Assembly）**：读取仓库目录树与相关模块的符号定义，只提取与目标代码相关的模块骨架与函数切片，拼接成紧凑的提示词。
3. **补丁生成（Patch Generation）**：指示模型输出标准的 Unified Diff 文本差异，仅描述增删改的几行代码及前后上下文，而不重写整个文件。
4. **测试验证与自动纠错（Verification & Debug Loop）**：在受限沙箱中对源码应用补丁，运行自动化测试命令（如 `pytest`），捕获进程退出码（Exit Code）与标准错误。如果测试未通过，将错误信息修剪后再次提交给模型，直到测试通过或达到最大重试次数。

关于此类任务流在更复杂 Agent 系统中的编排方式，可以参考 [AI Coding Agent 的执行流程](/frontier/ai-coding-agent/coding-agent-workflow.md)。

---

### 2. 仓库目录树与轻量符号索引

真实项目往往包含成千上万个文件，其中夹杂着依赖包（如 `node_modules/`、`.venv/`）和编译缓存。若不对文件进行过滤，模型可能会引用错误的文件路径或混淆内部模块。

![轻量目录树生成与依赖符号索引抽取机制](/images/content-rebuild/0a308926bd0f8d00/fig-1.png)

#### 过滤规则与目录树构建
构建上下文的第一步，是遍历项目根目录并生成去除噪音的 ASCII 目录树。

* **排除项**：严格遵循 `.gitignore` 规范，并显式过滤 `.git/`、`__pycache__/`、`dist/` 以及二进制文件。
* **深度截断**：设置目录遍历的最大深度（例如 3 到 4 层），避免过深的文件树挤占上下文。

```python
from pathlib import Path
from typing import Set

DEFAULT_IGNORE: Set[str] = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}

def generate_repo_tree(root_dir: str, max_depth: int = 3, current_depth: int = 0) -> str:
    """生成仓库的过滤后 ASCII 树结构，限制遍历深度"""
    if current_depth > max_depth:
        return ""
    
    root_path = Path(root_dir)
    if not root_path.exists():
        return ""

    lines = []
    try:
        # 目录排在前面，文件排在后面，按字母排序保证输出确定性
        entries = sorted(list(root_path.iterdir()), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return ""

    for entry in entries:
        if entry.name in DEFAULT_IGNORE or entry.name.startswith("."):
            continue
        indent = "  " * current_depth
        if entry.is_dir():
            lines.append(f"{indent}📁 {entry.name}/")
            sub_tree = generate_repo_tree(str(entry), max_depth, current_depth + 1)
            if sub_tree:
                lines.append(sub_tree)
        else:
            lines.append(f"{indent}📄 {entry.name}")
            
    return "\n".join(lines)
```

#### 符号索引表
仅有文件名不足以确定函数归属。助手在初始化时利用轻量解析工具（如 Python 内置的 `ast` 库）提取符号：
* **符号定义表（Symbol Index）**：记录全仓所有类和顶层函数的声明位置，格式为 `{符号名: (文件路径, 起始行, 结束行)}`。
* **导入依赖图（Dependency Graph）**：解析每个文件的 `import` 语句，记录模块间的调用链。当某个函数报错时，系统可沿着依赖链寻找上游类型定义，无需读取全仓所有源码。

---

### 3. 分级上下文组装与 AST 骨架提取

即使模型的上下文容量足以容纳数万行代码，盲目填入全量源码也会降低对关键逻辑的注意力，并增加推理成本。

![AST 骨架提取与局部切片注入机制对比](/images/content-rebuild/0a308926bd0f8d00/fig-2.png)

工程中通常采用**分级上下文组装（Tiered Context Injection）**策略：
* **骨架层（Skeleton）**：将被调用模块的代码压缩为 AST 骨架，保留类名、方法名、参数类型注解和文档说明，函数体替换为省略号 `...`。
* **切片层（Target Slice）**：只将报错堆栈涉及的核心函数、待修改代码段完整传入。
* **断言层（Failure Clue）**：传入触发失败的单元测试用例代码及入参。

下面是利用 Python 3.9+ 的 `ast.unparse` 提取代码骨架的实现：

```python
import ast

class SkeletonTransformer(ast.NodeTransformer):
    """将函数和异步函数的实现体替换为省略号 Constant(...)，保留文档说明"""
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        docstring = ast.get_docstring(node)
        new_body = []
        if docstring:
            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
        new_body.append(ast.Expr(value=ast.Constant(value=...)))
        node.body = new_body
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        docstring = ast.get_docstring(node)
        new_body = []
        if docstring:
            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
        new_body.append(ast.Expr(value=ast.Constant(value=...)))
        node.body = new_body
        return node

def extract_code_skeleton(source_code: str) -> str:
    """解析源码并返回仅包含结构定义的骨架文本"""
    try:
        parsed = ast.parse(source_code)
        transformer = SkeletonTransformer()
        stripped = transformer.visit(parsed)
        ast.fix_missing_locations(stripped)
        return ast.unparse(stripped)
    except SyntaxError:
        # 语法解析失败时降级返回空提示，避免阻断流水线
        return "# [AST 解析失败：源码包含语法错误]"
```

#### 压缩效果算例
假设一个被依赖的数据访问模块 `storage.py` 共有 300 行（约 1500 个 Token），包含 8 个数据库操作方法。通过 AST 提取骨架后，函数实现体均被折叠为 `...`，仅保留方法声明与类型签名，文本减少至约 40 行（约 200 个 Token）。在提供相同接口约定的前提下，提示词中该依赖项的 Token 占用减少了约 80% 以上。

<ReadMoreLock />

---

### 4. Unified Diff 补丁生成与容错应用

让模型输出代码变更时，主要有“全文件重写”与“Unified Diff”两种形式。

#### 为什么优先选用 Unified Diff
1. **降低生成延迟与成本**：自回归解码（Decode）的耗时通常与生成 Token 数量近似线性相关。若一个 800 行的文件仅需改动 3 行，全文件重写需要生成完整 800 行，而 Unified Diff 通常只需要生成不到 20 行，显著缩短等待时间。
2. **保护未改动代码**：模型重写长文件时容易遗漏无关函数、改变既有缩进风格或无意识修改注释。Unified Diff 明确指定了修改范围，便于人工审查。

![Unified Diff 的块结构解析与行号对齐冲突](/images/content-rebuild/0a308926bd0f8d00/fig-3.png)

#### Unified Diff 的格式构成
一个合法的补丁由头部与一个或多个 Hunk 块组成：

```diff
--- a/calculator.py
+++ b/calculator.py
@@ -10,4 +10,6 @@
 class Calculator:
     def divide(self, a: float, b: float) -> float:
-        return a / b
+        if b == 0:
+            raise ValueError("除数不能为0")
+        return a / b
     def multiply(self, a: float, b: float) -> float:
```
* `--- a/calculator.py` 与 `+++ b/calculator.py` 指定原始与目标路径。
* `@@ -10,4 +10,6 @@` 为 Hunk 头部：`-10,4` 表示原文件从第 10 行开始的 4 行；`+10,6` 表示新文件从第 10 行开始的 6 行。
* 减号 `-` 代表需删除的行，加号 `+` 代表新增的行，无前缀为空格的行代表匹配用的上下文行。

#### 模型的常见错误与容错对齐策略
大模型在生成 Unified Diff 时，经常出现两种格式问题：
1. **行号估算偏差（Hunk Header Hallucination）**：模型给出的起始行号往往与真实文件存在偏差（例如写了 `@ -10,4`，而目标代码实际在第 15 行）。如果直接调用系统自带的 `patch` 命令，会因为行号对不上而报错终止。
2. **上下文空白微扰（Context Drift）**：模型在提供前后 3 行上下文时，由于 Token 切分原因可能丢失原文件的行尾空格或使用了不同的换行符（CRLF 与 LF）。

针对这些问题，助手需实现**容错补丁应用器（Tolerant Patch Engine）**：
* **滑动窗口内容匹配**：忽略 Hunk 头部中的绝对行号，提取 Hunk 中不带 `+` 的所有行（即原文件的上下文行和待删除行），在目标文件中按内容从头向后进行滑动窗口匹配。一旦某段连续行的文本与模式一致，即锁定该位置为真实插入点。
* **空白规范化（Whitespace Normalization）**：在内容比对时，先对上下文行和文件内容去除行尾空格（`rstrip()`）并统一换行符，避免因格式差异导致匹配失败。

---

### 5. 沙箱测试执行与 Traceback 浓缩

模型生成的补丁不能保证绝对正确，可能会引入新的语法错误、死循环或破坏既有功能。因此，测试验证是闭环的关键环节。

![测试执行器在沙箱中捕获退出码与堆栈](/images/content-rebuild/0a308926bd0f8d00/fig-4.png)

#### 测试执行器的状态与输入输出
测试执行器（Test Runner）接收补丁修改后的工作区路径，运行测试命令，并返回状态三元组：`(exit_code, stdout, stderr)`。

```python
import subprocess
import shlex
from typing import Tuple

def run_unit_tests(repo_path: str, test_cmd: str = "pytest", timeout_sec: int = 30) -> Tuple[int, str, str]:
    """
    在目标仓库目录中执行测试命令，捕获退出码与输出。
    设置超时时间以防止死循环。
    """
    try:
        # 使用 shlex 切分命令，避免 shell=True 带来的命令注入风险
        cmd_args = shlex.split(test_cmd)
        process = subprocess.run(
            cmd_args,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )
        return process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"测试执行超时（超过 {timeout_sec} 秒），可能存在死循环。"
    except Exception as e:
        return -1, "", f"执行器底层错误: {str(e)}"
```

#### Traceback 日志浓缩
自动化测试框架（如 `pytest`）在遇到错误时，会打印大量环境配置、插件加载日志和冗长的模块载入栈。如果将整段几百行的日志作为下一轮提示词，会挤占有效上下文。

浓缩策略包括：
1. **状态行与堆栈切片**：仅抽取以 `FAILED`、`ERROR` 开头的测试用例断言信息，以及包含目标源码文件路径的局部调用栈。
2. **长度截断**：将浓缩后的报错日志控制在固定字符上限（如 1500 字符以内），优先保留报错类型（如 `ZeroDivisionError`）和失败发生所在的源码行。

---

### 6. Debug 循环状态机与回滚设计

自动修复流程可以建模为一个有限状态机。状态集合包含：当前迭代轮次 $k$、已累积补丁列表、最新测试结果和当前代码快照。

![Debug Loop 状态转移与终止条件设计](/images/content-rebuild/0a308926bd0f8d00/fig-5.png)

```python
from typing import Optional

class DebugLoopEngine:
    def __init__(self, repo_path: str, max_iterations: int = 3):
        self.repo_path = repo_path
        self.max_iterations = max_iterations

    def run_repair_loop(self, issue_description: str) -> bool:
        for iteration in range(1, self.max_iterations + 1):
            print(f"--- 启动调试轮次 [{iteration}/{self.max_iterations}] ---")
            
            # 1. 运行测试，检查当前状态
            exit_code, stdout, stderr = run_unit_tests(self.repo_path)
            if exit_code == 0:
                print("测试全部通过，修复完成。")
                return True
            
            # 2. 提取报错信息
            error_output = stderr if stderr.strip() else stdout
            trimmed_traceback = self._trim_traceback(error_output)
            
            # 3. 构造纠错提示词并请求模型生成 Diff
            prompt = self._build_prompt(issue_description, trimmed_traceback)
            diff_patch = self._call_model_for_diff(prompt)
            if not diff_patch:
                print("模型未返回有效 Diff 补丁，终止循环。")
                return False
            
            # 4. 应用补丁
            applied = self._apply_diff(diff_patch)
            if not applied:
                print("补丁匹配应用失败，准备进入下一轮重试。")
                continue
                
        # 退出循环后执行最终验证，检查最后一轮补丁是否使测试通过
        exit_code, _, _ = run_unit_tests(self.repo_path)
        if exit_code == 0:
            print("测试全部通过，修复完成。")
            return True

        print("已达最大重试次数，未能通过全部测试，开始执行回滚。")
        self._rollback_workspace()
        return False

    def _trim_traceback(self, raw_log: str) -> str:
        """简单截取最后 1500 字符的错误核心信息"""
        return raw_log[-1500:] if len(raw_log) > 1500 else raw_log

    def _build_prompt(self, issue: str, error_log: str) -> str:
        return f"任务需求: {issue}\n测试失败输出:\n{error_log}\n请输出修复该问题的 Unified Diff。"

    def _call_model_for_diff(self, prompt: str) -> Optional[str]:
        # 伪代码：实际调用大模型推理接口，并从回复中解析 ```diff 代码块
        return """--- a/calculator.py
+++ b/calculator.py
@@ -2,2 +2,4 @@
 def divide(a: int, b: int) -> int:
+    if b == 0:
+        raise ValueError("除数不能为0")
     return a // b
"""

    def _apply_diff(self, diff: str) -> bool:
        # 伪代码：调用容错补丁应用器，修改对应文件
        return True

    def _rollback_workspace(self) -> None:
        # 恢复初始状态，防止残留错误代码污染仓库
        subprocess.run(["git", "checkout", "."], cwd=self.repo_path)
```

#### 退出与终止条件
为避免模型反复修改引发无限循环或代码逐步破坏，系统设置三个明确的终止边界：
1. **测试通过**：退出码为 0，且无跳过或删除测试用例的行为，正常结束。
2. **超限熔断**：达到最大尝试轮次（如 3 轮），终止调试，执行版本管理回滚（`git checkout .`），避免脏代码残留。
3. **退化检测**：若第 $k$ 轮测试失败用例数量多于第 $k-1$ 轮，表明当前修改引入了新的破坏，可选择撤销该轮补丁，调整提示词重新尝试。相关防御机制可参考 [如何让 AI Coding Agent 更可靠？](/frontier/ai-coding-agent/reliable-coding-agent.md)。

---

### 7. 安全边界与沙箱权限隔离

自动化运行模型生成的代码和补丁存在安全隐患。如果测试用例中夹杂恶意指令，或者模型生成了破坏性系统调用（如意外删除目录、访问私有密钥），将对开发环境造成直接损害。

![开发机与沙箱容器的权限分界与安全防护](/images/content-rebuild/0a308926bd0f8d00/fig-6.png)

系统需要在宿主机与测试进程之间设立安全隔离防线：

#### 1. 文件覆写路径白名单
在将补丁写入磁盘前，静态分析补丁头部的文件路径：
* **路径越界检查**：解析目标路径为绝对路径，确认其必须在当前仓库根目录之内，拦截包含 `../` 的路径穿越请求。
* **敏感文件保护**：禁止修改 `.git/`、`.env`、配置文件以及系统目录中的文件。

#### 2. 测试执行环境隔离
严禁在宿主机上以特权用户直接运行模型修改后的测试脚本：
* **容器化隔离**：使用一次性 Docker 容器或轻量虚拟沙箱执行测试命令，以普通非 root 用户权限运行。
* **禁用外网通信**：启动测试容器时指定 `--network none`，切断一切对外网络连接，杜绝数据外泄风险。
* **计算资源配额**：通过 cgroups 限制测试容器的最大内存使用量（例如 1GB）与 CPU 占用，并配置超时强杀机制（例如 30 秒），防止死循环耗尽服务器资源。

#### 3. 补丁合并审查机制
即使测试全部通过，代码助手生成的补丁也不应在无人介入的情况下直接推送到生产分支。系统应向终端输出最终的 Diff 对比，由人类开发者进行审查与确认。有关自动化集成与发布的流程规范，可参考 [项目部署上线规范](/projects/project-deployment.md) 以及 [AI 编程面试专题](/interview/ai-coding-interview/README.md)。

---

### 8. 最小闭环工程验证

可以通过以下步骤在本地验证整个流程的执行状态：

#### 步骤 1：准备测试项目
新建一个测试目录 `test_project/`，并在其中创建两个文件：

`calculator.py`：
```python
def divide_integers(a: int, b: int) -> int:
    return a // b
```

`test_calculator.py`：
```python
import pytest
from calculator import divide_integers

def test_divide_normal():
    assert divide_integers(6, 2) == 3

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Division by zero is not allowed"):
        divide_integers(5, 0)
```

#### 步骤 2：执行初始测试捕获失败
在目录中运行测试：
```bash
pytest test_calculator.py
```
终端返回退出码 `1`，标准输出中捕获到失败堆栈：
```text
FAILED test_calculator.py::test_divide_by_zero - ZeroDivisionError: integer division or modulo by zero
```

#### 步骤 3：助手组装上下文并生成补丁
助手读取 `test_calculator.py` 报错内容及 `calculator.py` 源码，构造修复请求。模型生成如下 Unified Diff：

```diff
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,4 @@
 def divide_integers(a: int, b: int) -> int:
+    if b == 0:
+        raise ValueError("Division by zero is not allowed")
     return a // b
```

#### 步骤 4：应用补丁并再次验收
助手将该补丁应用至 `calculator.py`，重新执行测试命令：
```bash
pytest test_calculator.py
```
**验收标准**：进程返回退出码 `0`，终端输出 `2 passed`。调试循环在第一轮成功退出，向开发者打印最终生成的 Diff 补丁，完成验证闭环。
