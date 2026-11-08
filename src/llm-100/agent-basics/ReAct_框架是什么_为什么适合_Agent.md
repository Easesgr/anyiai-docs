---
title: ReAct 模式是什么？大模型如何结合推理与工具调用？
icon: mdi:cpu-64-bit
date: 2026-07-16
tag:
- 大模型100讲
- Agent
description: 拆解 ReAct 模式的思考、行动与观察循环，提供最小可运行 Python 状态机、时序分析、成本算例与可检验动手练习。
---

掌握了 [Tool Calling 是什么？大模型如何调用工具？](Tool_Calling_是什么_大模型如何调用工具.md) 的底层协议之后，工程师需要解决的下一个问题是：**如何让大模型自主编排多步工具调用来完成复杂任务？**

2022 年由 Shunyu Yao 等人提出的 **ReAct（Reasoning + Acting）** 模式，通过将大模型的“显式推理”（Reasoning）与“外部行动”（Acting）交替串联，给出了清晰的解法。本文将通过一个具体的查询场景，解析 ReAct 的运行机制、状态机流转、工程收益与性能边界。

![ReAct 模式原理封面图](/images/content-rebuild/af64b80432611b85/cover.png)

> **图示导读**：讲解员正在交互看板前展示 ReAct 架构的核心理念——模型并非盲目执行代码，而是在“思考（Thought）”、“行动（Action）”与“观察（Observation）”的三步循环中推进状态，直到达成最终目标。

---

### 1. 核心结论与贯穿场景

**核心结论**：ReAct 是一种由宿主程序驱动的运行时控制流。模型负责生成“思考”（Thought，拆解目标）与“行动”（Action，指明工具与入参），宿主程序负责执行工具并回填“观察”（Observation，外部真实结果）。三者循环迭代，直到模型得出最终答案（Final Answer）或触发终止条件。

#### 贯穿场景：物流进度查询
用户提问：“帮我查一下张三最新订单的物流进度。”

系统具备两个外部工具：
1. `get_latest_order(user_name: str)`：根据用户名查询最新订单号。
2. `get_shipping_status(order_id: str)`：根据订单号查询实时物流状态。

在这个任务中，大模型必须经历两轮工具调用与结果分析，无法单步直接解决。

---

### 2. 为什么需要 ReAct：纯推理与纯行动的短板

在 ReAct 提出之前，大模型处理任务主要依赖两种方式：

1. **纯推理（如思维链 Chain-of-Thought, CoT）**：
   模型仅在内部进行逐步思考。但大模型的权重是静态的，缺乏外部数据库访问能力。面对“张三最新订单”这类私有动态数据，模型只能根据上下文猜测甚至虚构订单号与物流，产生事实性幻觉。
2. **纯行动（Act-only）**：
   模型直接输出工具调用指令（例如直接生成 `get_shipping_status(order_id=???)`）。由于没有中间思考步骤来解析“先查人、再查单”的依赖关系，模型容易在参数缺失时猜测入参；一旦外部接口返回错误（如网络超时或订单不存在），缺乏上下文记录的纯行动模式很难分析原因并尝试修正。

ReAct 的工作方式是**推理指导行动，行动验证推理**：模型先通过思考拆解当前子目标，再发起行动；拿到外部返回的事实后再继续思考下一步，从而兼顾推理连贯性与数据真实性。

![纯推理、纯行动与 ReAct 模式对比](/images/content-rebuild/af64b80432611b85/fig-1.png)

> **图示导读**：注意图中左侧 CoT 缺少外部输入导致的幻觉漂移，中间 Act 缺乏思考导致的执行盲目，右侧 ReAct 则通过环境反馈与思考交替形成动态闭环。

---

### 3. 三要素职责边界：Thought、Action 与 Observation

在工程实现中，ReAct 并不改变大模型内部的网络架构，而是依赖宿主程序（Host Runtime）与模型约定的文本格式进行协作：

* **Thought（思考）**：
  * **生成者**：大语言模型（LLM）。
  * **内容**：自然语言文本。模型在此阶段说明当前意图、评估前序工具返回是否有效，并确定接下来的行动。
* **Action（行动）**：
  * **生成者**：大语言模型（LLM）。
  * **内容**：结构化指令（约定的文本格式或原生 Tool Call 协议）。指明工具名称与参数，如 `get_latest_order[张三]`。
* **Observation（观察）**：
  * **生成者**：宿主程序（外部系统）。
  * **内容**：在架构设计上，Observation 必须由宿主程序执行真实工具后回填，大模型不应自行生成。但在纯文本自回归生成时，模型本身缺乏物理屏障来彻底杜绝其伪造观察值，工程上需依赖停止字符串（Stop Sequence）辅助截断，或通过原生 Tool Calling 的结构化消息协议在接口层将环境反馈与模型输出物理隔离。

![Thought-Action-Observation 三要素交互机制](/images/content-rebuild/af64b80432611b85/fig-2.png)

> **图示导读**：观察上下分层的边界：Thought 与 Action 位于模型生成侧，而 Observation 位于宿主程序侧，真实环境的返回值通过宿主桥接回填进对话上下文。

---

### 4. 状态机流转与执行时序

从架构上看，ReAct 是一个有穷状态机：

$$\text{State}_{t+1} = \text{HostUpdate}(\text{State}_t, \text{LLM}(\text{State}_t))$$

具体执行时序包含六个阶段：

1. **上下文组装**：宿主将系统提示词（包含工具定义与格式约定）、历史记录与用户输入合并为 Prompt。
2. **模型生成**：模型开始生成输出。在纯文本协议中通常设置停止字符串（Stop Sequence，例如 `Observation:`）帮助分段，或利用原生 Tool Calling 协议在输出工具调用指令后截断。需注意并非所有模型 API 均支持任意停止字符串。
3. **动作拦截**：当模型生成遇到停止字符串或完成工具调用指令时，本次模型请求结束。宿主程序解析输出中的 Action，提取工具名与入参。
4. **环境执行**：宿主在外部沙箱或真实服务中调用对应函数，捕获真实输出或异常报错。
5. **状态回填**：宿主将执行结果格式化为 `Observation`，追加至对话上下文中。
6. **终止判断**：若模型输出了 `Final Answer:` 或放弃调用工具，循环正常结束；若达到最大步数（Max Steps），触发熔断退出。

![ReAct 状态机流转与执行时序](/images/content-rebuild/af64b80432611b85/fig-3.png)

> **图示导读**：该时序图展示了客户端、宿主运行时与大模型之间的交替调用关系，注意宿主程序在解析 Action 后打断模型并挂起等待外部工具返回的关键节点。

---

### 5. 工程优势：白盒轨迹与动态错误应对

在生产系统中，单次端到端黑盒调用如果失败，开发者往往只能看到一个笼统的错误状态。ReAct 架构带来了两个显著优势：

#### 5.1 过程完全透明（Observability）
每一轮交互留下了完整的追踪记录（Trace）：
* 若模型理解偏离业务，可通过 **Thought** 检查意图拆解；
* 若入参类型错误，可通过 **Action** 校验工具 Schema；
* 若外部服务异常，可通过 **Observation** 定位具体接口错误码与返回值。

#### 5.2 具备动态错误适应能力
外部 API 可能会出现临时抖动或参数无效。当接口返回错误时，宿主程序将错误信息作为 Observation 回传（例如：`Observation: Error 404: Order not found`）。模型在下一轮的 Thought 中读取到该报错，可以尝试更换入参或调用备用工具，而不是让整个服务直接崩溃。

![ReAct 模式下的白盒轨迹与故障定位](/images/content-rebuild/af64b80432611b85/fig-4.png)

> **图示导读**：对比左侧无法定位内部成因的端到端调用，右侧 ReAct 轨迹清晰地暴露出工具异常与模型随后的思考修正过程，大幅降低故障排查难度。

---

### 6. 最小可验证实现：Python 状态循环

以下是一个无外部依赖、可直接运行的 ReAct 状态机实现。我们通过模拟函数展示模型与宿主环境的交互：

```python
import re
from typing import Dict, Callable

# 1. 模拟外部工具集（真实工程中为真实 API 或数据库）
def get_latest_order(user_name: str) -> str:
    db = {"张三": "OD-9527", "李四": "OD-1024"}
    return db.get(user_name, "ERROR: 用户不存在")

def get_shipping_status(order_id: str) -> str:
    logistics = {
        "OD-9527": "包裹正在北京转运中心分拣，预计明日送达",
        "OD-1024": "已签收，签收人：本人"
    }
    return logistics.get(order_id, "ERROR: 无效订单号")

TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "get_latest_order": get_latest_order,
    "get_shipping_status": get_shipping_status
}

# 2. 模拟大模型推理决策（工程中替换为实际大模型 API）
def mock_llm_response(prompt: str) -> str:
    if "Observation:" not in prompt:
        # 第一次推理：拆解目标，调用第一个工具
        return (
            "Thought: 用户要查张三的物流，我需要先查出张三的最新订单号。\n"
            "Action: get_latest_order[张三]"
        )
    elif "OD-9527" in prompt and "北京转运中心" not in prompt:
        # 第二次推理：获取到订单号，继续查物流详情
        return (
            "Thought: 已经拿到订单号 OD-9527，接下来查询具体的物流状态。\n"
            "Action: get_shipping_status[OD-9527]"
        )
    elif "北京转运中心" in prompt:
        # 第三次推理：信息完备，生成最终回答
        return (
            "Thought: 已经获取到完整的物流信息，可以回答用户了。\n"
            "Final Answer: 张三的最新订单号为 OD-9527，当前状态为：包裹正在北京转运中心分拣，预计明日送达。"
        )
    return "Final Answer: 处理失败，无法完成任务。"

# 3. 宿主驱动的 ReAct 循环引擎
def run_react_agent(query: str, max_steps: int = 5) -> str:
    prompt = f"Question: {query}\n"
    action_pattern = re.compile(r"Action:\s*(\w+)\[(.*)\]")
    
    for step in range(max_steps):
        print(f"\n=== 第 {step + 1} 轮循环 ===")
        llm_output = mock_llm_response(prompt)
        print(f"[模型输出]:\n{llm_output}")
        prompt += llm_output + "\n"
        
        # 检查是否达成最终目标
        if "Final Answer:" in llm_output:
            return llm_output.split("Final Answer:")[-1].strip()
            
        # 解析 Action
        match = action_pattern.search(llm_output)
        if not match:
            raise ValueError("模型未按协议输出有效 Action")
            
        tool_name, tool_arg = match.group(1), match.group(2)
        
        # 宿主在外部环境中执行工具并回填 Observation
        tool_func = TOOL_REGISTRY.get(tool_name)
        obs = tool_func(tool_arg) if tool_func else f"ERROR: 工具 {tool_name} 不存在"
        
        observation_entry = f"Observation: {obs}"
        print(f"[宿主注入]: {observation_entry}")
        prompt += observation_entry + "\n"
        
    return "错误：已达到最大迭代步数熔断上限"

if __name__ == "__main__":
    result = run_react_agent("帮我查一下张三最新订单的物流进度")
    print(f"\n>>> 执行完毕，返回结果: {result}")
```

![最小 ReAct 状态机架构与数据流](/images/content-rebuild/af64b80432611b85/fig-5.png)

> **图示导读**：图中清晰展示了上述代码中的循环流向：`prompt` 文本在宿主和模型之间传递，每次迭代不断追加 Thought、Action 与外部返回的 Observation。

---

### 7. 工程瓶颈与物理边界

在实际生产中，ReAct 架构并非万能。引入自主决策的同时，系统也必须承担以下性能与资源代价：

#### 7.1 上下文膨胀与 KV 缓存开销
每一轮追加的 Thought、Action 和 Observation 都会扩大后续调用的 Prompt 长度。

**算例对比（以 3 轮循环为例，假设系统提示词 500 Token，单轮生成 100 Token，单次工具返回 300 Token）**：
* 第 1 轮输入上下文：$500$ Token
* 第 2 轮输入上下文：$500 + 400 = 900$ Token
* 第 3 轮输入上下文：$900 + 400 = 1300$ Token

* **未启用 Prompt Cache 时**：三轮调用的全量 Prefill 累计处理 Token 为 $500 + 900 + 1300 = 2700$ Token。
* **启用 Prompt Cache 时**：前序命中缓存，Prefill 仅需处理新增增量，但无论是否开启缓存，随着上下文长度增长，Decode 阶段维持的 **KV 缓存显存占用** 仍呈阶梯上升，且可能触碰模型的最大上下文窗口限制。

#### 7.2 严格串行的端到端延迟
ReAct 的决策链高度依赖上一步工具的返回值，难以进行前置并行：

$$T_{\text{total}} = \sum_{i=1}^{N} (T_{\text{LLM\_Prefill}, i} + T_{\text{LLM\_Decode}, i} + T_{\text{Tool\_Execution}, i} + T_{\text{Network\_RTT}, i})$$

如果单次大模型调用耗时 1 秒，外部网络工具耗时 1.5 秒，一次 4 轮的 ReAct 交互总延迟将直接超过 10 秒。在对时延敏感的高并发场景（如实时在线客服）中，必须通过限制轮次、裁剪 Observation 长度或采用预定义静态工作流来降低耗时。

![ReAct 循环中的延迟堆叠与 Token 累积](/images/content-rebuild/af64b80432611b85/fig-6.png)

> **图示导读**：注意图表中横轴代表的时间延迟串行堆积，以及纵轴展示的上下文 Token 在多轮交互下的阶梯式膨胀，体现出 ReAct 在高并发高时效场景下的工程取舍。

---

### 8. 可检验动手练习

**学习目标**：通过修改上述最小运行代码，验证当外部工具返回错误时，ReAct 状态机如何根据 Observation 进行动态容错。

#### 练习要求
1. 修改 `get_latest_order` 工具函数，增加一个运行时模拟故障：当入参为 `"王五"` 时，前两次调用返回 `"ERROR: 503 数据库连接超时"`，第三次调用才返回订单号 `"OD-2048"`。
2. 修改 `mock_llm_response` 逻辑：当检测到 `ERROR: 503` 时，模型输出 Thought 说明“接口报错超时，等待重试”，并重新发起一次相同的 Action。
3. 运行 `run_react_agent("帮我查一下王五最新订单的物流进度")`，观察控制台输出。

#### 验收标准
* 循环没有中途崩溃退出；
* 控制台清晰打印出至少两次对同一工具的重复调用及对应的 Observation 错误信息；
* 最终模型在重试成功后正常获取数据并输出 `Final Answer`。
