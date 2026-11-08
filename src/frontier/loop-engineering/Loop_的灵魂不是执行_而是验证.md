---
title: Agent 循环怎么做验证？确定性检查与大模型评审的边界
icon: mdi:refresh
date: 2026-07-16
tag:
- Loop Engineering
- Agent
description: 深入分析 Agent 自动化循环中验证器的设计机制，对比 AST、单元测试等确定性检查与大模型评审的适用边界，给出反馈粒度控制与分层级联方案。
---

在构建能够自我修复或迭代的 Agent 系统时，一个常见的误区是把精力全花在“让模型多试几次”上——在外面套一层 `while` 循环，只要未通过就重新调用模型。然而在实际工程中，单纯增加重试次数往往不会让结果变得更好，模型甚至会在几轮修改后把原本写对的代码改得支离破碎。

自动化循环能否最终停在正确结果上，核心取决于**验证器（Verifier）能否提供准确、定位清晰的反馈**。

在上一篇[/frontier/loop-engineering/一个_Loop_由哪些部分组成.md](/frontier/loop-engineering/一个_Loop_由哪些部分组成.md)中，我们梳理了循环的基础组件。本文将以一个具体的代码生成与过滤任务为线索，分析确定性验证与模型评审的技术边界、误判对循环的破坏，以及工程中如何设计分层级联验证。

![验证在循环工程中的核心地位](/images/content-rebuild/6322273f2fe3445e/cover.png)

---

## 贯穿示例：订单过滤函数的自动生成与修复

为了直观理解验证器的工作机制，我们来看一个贯穿全文的具体任务：让模型编写一个订单过滤函数 `filter_orders(orders, min_amount)`。

业务要求如下：
1. 输入 `orders` 为字典列表，每个字典包含 `id`（整数）、`amount`（浮点数）和 `status`（字符串）。
2. 过滤出 `status == "completed"` 且 `amount >= min_amount` 的订单列表。
3. 如果输入的订单缺少必要字段，跳过该条记录，不抛出异常。

模型在第一次尝试时，往往会出现这类代码：

```python
def filter_orders(orders, min_amount):
    result = []
    for item in orders
        if item["amount"] > min_amount and item["status"] == "completed":
            result.append(item)
    return result
```

这段初次生成的代码存在三个层级的问题：
1. **语法错误**：`for item in orders` 末尾漏掉了冒号 `:`，无法被解释器解析；
2. **业务边界错误**：金额条件写成了 `>` 而非 `>=`；
3. **健壮性缺陷**：如果某条订单缺少 `"amount"` 或 `"status"` 键，直接通过下标访问会抛出 `KeyError`。

要让模型把这段代码自动修好，验证器必须依次捕获这些问题，并在不同的阶段给出合适的诊断信息。

---

## 为什么只加重试不收敛：反馈的信噪比与残差

大模型在循环中修改产物，依赖上下文中的上一轮结果和报错提示。如果验证器给出的反馈是模糊的，模型的修改就会失去方向。

我们可以将验证反馈分为两部分：
- **有效信号**：精准指出哪一行、哪种输入下、实际结果与预期结果相差了什么（即残差信息）；
- **噪声**：缺少定位的全局报错（如只说“测试未通过”）、验证逻辑本身的误判，或者前后矛盾的提示。

当有效信号极少、噪声极大时，模型无法得知具体哪里出了错，倾向于重构整个函数的变量命名甚至算法逻辑。原本写对的 `status == "completed"` 可能在下一轮被无故删掉，导致迭代不仅无法收敛，反而越改越偏。

因此，验证器的首要任务不是简单给出一个布尔值，而是**提取高信噪比的残差**。

---

## 确定性验证机制：语法树、数据校验与单元测试

确定性验证指的是**在固定输入和确定环境下，判定规则具备严格的可重复性，不引入随机采样和主观推断**。

这类检查是工程系统中最稳定的一环，适合检查语法合法性、数据格式规范和明确的输入输出映射。

![确定性验证的硬性检查机制](/images/content-rebuild/6322273f2fe3445e/fig-1.png)

### 1. 结构与语法层：AST 与 Schema
- **抽象语法树（AST）检查**：利用编程语言原生的 Parser 验证代码是否合法。例如在贯穿示例中，模型漏掉了 `for` 循环末尾的冒号，Python 原生 `ast.parse()` 会在毫秒内抛出 `SyntaxError: expected ':'` 并精确标出第 3 行。此时根本不需要启动任何测试或调用大模型，直接阻断即可。
- **结构与数据规范（如 JSON Schema）**：在工具调用与接口通信场景中，用来检查字段类型、必选键和取值范围。校验器能明确报出 `amount is missing` 或 `min_amount must be a number`。

### 2. 静态分析与动态运行层：类型检查与单元测试
- **静态类型与代码规约检查**：通过工具（如 `mypy`、`ruff`、TypeScript 编译检查）捕获未导入模块、类型不匹配或可能触发未捕获异常的访问路径。
- **单元测试断言（Unit Tests）**：构造固定的测试输入与预期输出。针对贯穿示例，可以编写针对边界值与异常输入的断言：
  ```python
  # 边界测试：金额刚好等于阈值应被保留
  assert filter_orders([{"id": 1, "amount": 100.0, "status": "completed"}], 100.0) == [{"id": 1, "amount": 100.0, "status": "completed"}]
  # 健壮性测试：缺失字段不应崩溃
  assert filter_orders([{"id": 2}], 100.0) == []
  ```

### 最小可运行实验：确定性校验器实现

下面是一个轻量级的确定性校验器，演示如何在循环前置流程中串联语法检查与数据结构检查，提取明确的诊断上下文：

```python
import ast
from typing import Dict, Any, Tuple
import jsonschema

def verify_code_and_schema(code_str: str, config_dict: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    # 1. AST 语法确定性校验
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return False, f"[AST SyntaxError] Line {e.lineno}, Col {e.offset}: {e.msg}"

    # 2. JSON Schema 数据约束校验
    try:
        jsonschema.validate(instance=config_dict, schema=schema)
    except jsonschema.ValidationError as e:
        field_path = '.'.join([str(p) for p in e.path]) or 'root'
        return False, f"[Schema Violation] Field '{field_path}': {e.message}"

    return True, "Verification Passed"

# 实验验证：输入存在语法错误的代码与不合规配置
schema_def = {
    "type": "object",
    "properties": {"min_amount": {"type": "number", "minimum": 0}},
    "required": ["min_amount"]
}

broken_code = "def filter_orders(orders, min_amount)\n    return []"  # 缺少冒号
invalid_config = {"min_amount": -5}

passed, diagnostic = verify_code_and_schema(broken_code, invalid_config, schema_def)
print(f"Passed: {passed}")
print(f"Diagnostic: {diagnostic}")
# 输出：
# Passed: False
# Diagnostic: [AST SyntaxError] Line 1, Col 38: expected ':'
```

确定性验证的核心优势是**误报率极低**。语法解析器说代码有语法错误，代码就必定无法运行，大模型可以明确以该报错为依据进行修正。

<ReadMoreLock />

---

## 弱确定性验证机制（LLM-as-a-judge）：适用场景与方差控制

确定性验证虽然可靠，但无法覆盖所有场景：比如长文本的逻辑自洽性、文档文风是否符合规范、代码是否遵循了某种没有形式化规则的架构风格。在这些没有现成测试断言的场景下，往往需要引入大模型本身作为裁判（LLM-as-a-judge）。

大模型裁判存在随机性、位置偏见（倾向于选择靠前或靠后的选项）和打分尺度漂移问题。为了让判定结果稳定，必须通过工程约束将其限制为“弱确定性验证”：

![弱确定性验证与模型量表评测](/images/content-rebuild/6322273f2fe3445e/fig-2.png)

### 1. 结构化布尔量表（Rubric-based Evaluation）
避免让裁判模型打抽象的百分制或 1 到 10 分，数值打分的方差极大且难以映射为具体修改指令。应将评估标准拆解为独立的布尔检查项：
- 检查项 1：是否使用了安全字典访问（如 `.get()`）以防字段缺失时崩溃？（是/否）
- 检查项 2：注释说明是否完整且未捏造参数？（是/否）
- 检查项 3：是否引入了未被允许调用的外部网络库？（是/否）

每个检查项要求输出判断依据与所在行号，直接将评价转为结构化问题列表。

### 2. 成对对比评测（Pairwise Comparison）与顺序反转
让模型在两个候选方案之间挑出更好的一个，通常比直接评估单一样本更稳定。在对比当前轮次生成物与基线版本时，应颠倒候选文本在提示词中的前后顺序运行两次（Swap Evaluation），消除位置偏见：

$$\text{WinRate}(A) = \frac{1}{2} \left( \mathbb{I}_{\text{Judge}(A, B) = A} + \mathbb{I}_{\text{Judge}(B, A) = A} \right)$$

其中 $\mathbb{I}_{\cdot}$ 表示指示函数，当条件满足时取值为 1，否则为 0。只有在两次调换顺序中均胜出，才判定版本 $A$ 优于版本 $B$。

### 3. 参数设定与采样控制
- **单次评估**：裁判模型的采样参数应设为 `Temperature = 0`（贪婪解码），尽可能消除单次生成的随机性与抖动；
- **多数表决（Majority Voting）**：若关键判定节点在边界情形存在歧义，需将采样温度微调至略大于 0（或使用不同的提示词模板/裁判模型阵容）进行多次独立采样，再通过多数票仲裁过滤偶然偏差。

---

## 反馈信息的有效度：布尔标记对比结构化残差

验证失败后，系统传回给生成模型的上下文信息，直接决定了下一轮修改的成功率。提示词如何引导模型关注上下文细节，可参考 [/llm-100/prompt-engineering/Prompt_为什么能影响大模型输出.md](/llm-100/prompt-engineering/Prompt_为什么能影响大模型输出.md)。

![反馈粒度对状态搜索空间的影响](/images/content-rebuild/6322273f2fe3445e/fig-3.png)

我们对比两种不同的反馈设计对模型修改贯穿示例时的影响：

### 策略 A：布尔反馈（低信息量）
```text
[System]: 验证失败，代码未通过单元测试，请修复。
```
- **执行后果**：模型无法获知究竟是哪个测试用例挂了。它可能猜测是变量名不规范，或者误以为逻辑框架不对，于是重写了全部实现，结果把原本正确的字段过滤逻辑改乱了。

### 策略 B：结构化残差上下文（高信息量）
```json
{
  "status": "failed",
  "stage": "unit_test",
  "failed_test": "test_amount_boundary_inclusion",
  "input": {
    "orders": [{"id": 101, "amount": 50.0, "status": "completed"}],
    "min_amount": 50.0
  },
  "expected": [{"id": 101, "amount": 50.0, "status": "completed"}],
  "actual": [],
  "diagnostic": "Order with amount == min_amount was excluded. Check comparison operator.",
  "assertion_line": 38
}
```
- **执行后果**：模型清晰看到了边界输入 `50.0` 和空列表的输出对比，注意力集中在比较运算符上，只需将 `>` 改为 `>=` 即可收敛。

反馈信息本质上是下一次生成的**先验约束**。反馈越结构化、残差越明确，模型探索无用分支的代价就越低。

---

## 假阳性（误报）与假阴性（漏报）对循环稳定性的破坏

在缺陷检测语境下，我们区分以下概念：
- **阳性（Positive）**：判定存在缺陷，即验证**失败（Fail）**；
- **阴性（Negative）**：判定没有缺陷，即验证**通过（Pass）**。

由此带来的两类判定错误，对自动化循环的破坏机制完全不同：

![假阳性与假阴性对循环轨迹的破坏](/images/content-rebuild/6322273f2fe3445e/fig-4.png)

### 1. 假阳性（误报，把对的判为错）
- **产生原因**：静态规则过于机械、测试用例本身写错断言，或者大模型裁判产生幻觉并挑剔不存在的问题。
- **系统表现**：产物明明已经满足业务需求，却被判定为不合格。模型被迫在没有错误的地方“强行修改”，容易改出真实的语法或逻辑 bug，导致重试次数耗尽退出。

### 2. 假阴性（漏报，把错的判为对）
- **产生原因**：测试用例没有覆盖边界条件（例如未测试字段缺失的情况），或者裁判模型草率判定“回答得很好”。
- **系统表现**：**虚假收敛**。循环认为任务已顺利完成并退出，带有隐患的代码直接进入下游执行，引发线上异常。关于循环中失控与虚假终止的更多分析，可参考 [/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md](/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md)。

| 验证机制类型 | 假阳性风险（误报） | 假阴性风险（漏报） | 工程应对机制 |
| :--- | :--- | :--- | :--- |
| **AST 语法解析** | 极低（解析器规则明确） | 极低（仅管语法，不管语义） | 作为首道短路门禁拦截 |
| **单元测试用例** | 低（除非测试脚本逻辑写错） | 中到高（取决于用例完备度） | 补充边界值、空值与异常类型测试 |
| **LLM-as-a-judge** | 中到高（模型挑刺或误解要求） | 中到高（模型忽略深层细节） | 离散化量表 + 成对对比 + 低温采样 |

---

## 构建分层验证策略：由快到慢、由硬到软的级联校验链

如果每轮修改后都去调用昂贵的大模型进行全面评审，或者运行耗时数分钟的完整集成测试，循环的延迟和成本将无法接受。生产中通常采用**级联校验链（Cascading Verification Chain）**：

![分层级联校验链架构设计](/images/content-rebuild/6322273f2fe3445e/fig-5.png)

### 1. 级联顺序设计
1. **Layer 1：结构与语法门禁（< 10ms，零模型开销）**
   - 检查代码 AST 是否合法、JSON/YAML 配置是否满足 Schema 约束。
   - **短路规则**：语法不合法直接拦截，返回解析器错误，后续层级不执行。
2. **Layer 2：静态分析与规则校验（< 500ms，零模型开销）**
   - 执行类型推断工具、静态 Lint，检查未声明变量、危险函数调用。
   - **短路规则**：静态分析失败直接拦截，返回定位到行号的规则报错。
3. **Layer 3：沙箱单元测试（1s ~ 10s，计算资源开销）**
   - 在隔离环境中运行预先编写的测试套件，收集失败断言与输入输出差异。
   - **短路规则**：测试断言失败直接拦截，把残差上下文作为反馈返回。
4. **Layer 4：弱确定性语义审查（1s ~ 5s，高 Token 开销）**
   - 仅在前面的硬性关卡全部通过后，才调用裁判模型检查命名风格、注释完整度或业务逻辑深度。

### 2. 状态阻断与退出仲裁实现

以下伪代码展示了分层校验流水线在循环编排中的控制逻辑：

```python
def execute_verification_pipeline(candidate_artifact: dict) -> dict:
    # Layer 1: 语法门禁（硬验证，成本极低）
    l1_result = run_ast_and_format_check(candidate_artifact["code"])
    if not l1_result["passed"]:
        return {"status": "FAIL", "stage": "L1_SYNTAX", "feedback": l1_result["error_context"]}

    # Layer 2: 静态类型与规则门禁（硬验证，成本低）
    l2_result = run_static_type_check(candidate_artifact["code"])
    if not l2_result["passed"]:
        return {"status": "FAIL", "stage": "L2_STATIC", "feedback": l2_result["diagnostic_log"]}

    # Layer 3: 沙箱测试用例（硬验证，需要执行环境）
    l3_result = run_sandbox_unit_tests(candidate_artifact["code"])
    if not l3_result["passed"]:
        return {"status": "FAIL", "stage": "L3_SANDBOX", "feedback": l3_result["assertion_diff"]}

    # Layer 4: LLM 语义量表审查（弱确定性，消耗 Token）
    l4_result = run_llm_rubric_judge(candidate_artifact["code"], candidate_artifact["requirements"])
    if not l4_result["passed"]:
        return {"status": "FAIL", "stage": "L4_SEMANTIC", "feedback": l4_result["rubric_violations"]}

    # 全链条通过，准许退出循环
    return {"status": "SUCCESS", "stage": "ALL_PASSED", "feedback": "Artifact verified successfully."}
```

通过级联设计，语法和类型错误能在前两个毫秒级阶段被直接打回，大模型裁判不会被浪费在充当基础语法检查器上，从而兼顾了执行效率与验证严谨度。

![验证工程落地与全景闭环复盘](/images/content-rebuild/6322273f2fe3445e/fig-6.png)

---

## 总结与延伸阅读

验证器是 Agent 循环中的方向盘和刹车系统。合理区分确定性验证与模型评审的职责边界，能避免两类常见陷阱：既不指望用死板的正则去穷举复杂的语义需求，也不依赖大模型去肉眼找语法错误。

在接下来的实战文章中，我们将沿着分层验证的思路，构建一个端到端的代码生成与测试修复闭环：
- 实践演练：[/frontier/loop-engineering/最简单的_Loop_写代码_跑测试_自动修复.md](/frontier/loop-engineering/最简单的_Loop_写代码_跑测试_自动修复.md)
- 架构扩展：[/llm-100/agent-engineering/README.md](/llm-100/agent-engineering/README.md)
- 面试答辩：[/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md)
