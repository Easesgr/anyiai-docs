---
title: 什么是工具调用（Tool Calling）？如何用 Spring AI 实现？
icon: mdi:tools
tag:
- Tool Calling
- Spring AI
- 工具
description: 口述两阶段工具调用原理，剖析 JSON Schema 协议与消息时序，结合 Spring AI 演示声明、异常处理与安全校验。
---

![工具调用架构全局透视图](/images/content-rebuild/2ada76d4b15fe2b3/cover.png)

在以大语言模型为核心的应用开发中，如果模型仅局限于输出文本，它的能力就会受限于静态权重与当前对话上下文。在面试与工程实践中，“如何让大模型与外部系统交互”是一个高频考点。

我们在 [/interview/prompt-interview/structured-output.md](/interview/prompt-interview/structured-output.md) 中讨论过结构化数据输出。**工具调用（Tool Calling / Function Calling）**在结构化输出的基础上更进一步：模型根据用户需求输出结构化的调用指令，宿主程序负责实际执行，并将执行结果写回对话上下文，供模型生成最终答复。

本文先提供一段可直接用于面试口述的标准回答，再拆解背后的协议格式、消息时序，以及在 Java 中使用 **Spring AI** 的实现方法与防御边界。

---

## 面试口述：工具调用是如何工作的？

如果面试官要求在两分钟内说明“工具调用的工作过程与角色分工”，可以直接这样回答：

> **口述核心回答**：
> 
> “工具调用本质上是**模型生成调用指令，宿主程序负责参数验证与外部执行**的分工协作。大模型本身没有直接发起 HTTP 请求、查询数据库或执行代码的权限与物理环境。
> 
> 它的完整过程可以分为四个步骤：
> 1. **注入工具协议**：宿主程序（如基于 Spring AI 开发的后端服务）在发起请求时，把工具的名称、业务描述以及基于 JSON Schema 的参数定义随请求体发给模型；
> 2. **生成调用指令**：模型根据用户提问，判断是否需要调用外部工具。若需要，模型生成包含函数名、参数 JSON 字符串以及唯一调用 ID（指令中对应 `id`，后续回传对应 `tool_call_id`）的结构化指令，并将响应结束标志标记为 `tool_calls`；
> 3. **宿主执行并包装结果**：宿主程序拦截该响应，反序列化参数并完成校验，在安全运行环境中执行真实的网络请求或数据库查询，然后把执行结果包装为角色为 `tool` 的 `ToolMessage`；
> 4. **生成最终答复**：宿主程序将原始问题、模型的调用指令和工具的执行结果一起拼接到对话历史中，再次请求模型，模型结合工具数据输出最终给用户的自然语言答复。”

![纯结构化输出与工具调用两阶段交互对比](/images/content-rebuild/2ada76d4b15fe2b3/fig-1.png)

---

## 协议结构：模型如何理解工具签名？

模型能否准确决定是否调用工具，取决于宿主程序向模型发送的**工具定义（Tool Definition）**。在底层，各类服务商的接口都遵循统一的元数据规范。

以常见的兼容规范为例，发送给模型推理接口的请求体中会包含一个 `tools` 列表：

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "帮我查询杭州明天的天气"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询指定城市未来的天气预报，支持获取温度、湿度和天气状况",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "待查询的城市名称，例如：北京、杭州"
            },
            "date": {
              "type": "string",
              "enum": ["today", "tomorrow"],
              "description": "查询日期相对值"
            }
          },
          "required": ["city", "date"]
        }
      }
    }
  ]
}
```

在这段结构中，三个关键字段直接影响模型的行为：

1. **函数描述（`description`）**：这是给模型看的**语义路由说明**。模型通过对比用户输入与各个函数描述的语义相关性，决定调用哪个工具或不调用工具。若描述模糊（例如写成“处理天气数据”），模型容易发生漏选或选错。
2. **参数类型与枚举（JSON Schema）**：对工具入参做类型约束。模型会把输入中的自然语言实体映射为对应字段（例如将“杭州”提取为 `city: "杭州"`，将“明天”映射为 `date: "tomorrow"`）。
3. **字段描述（Property `description`）**：用于消除格式歧义。例如当参数为时间戳时，应在字段描述中注明是“秒级时间戳”还是“ISO-8601 格式”，降低模型生成错误格式的概率。

<ReadMoreLock />

---

## 消息流转时序：从 tool_calls 到 ToolMessage

工具调用不是一次 HTTP 请求就能完成的，它需要在宿主程序和模型服务之间交替进行两次或多次交互。

![多阶段状态机时序图](/images/content-rebuild/2ada76d4b15fe2b3/fig-2.png)

### 交互步骤拆解

1. **发起初次请求**：宿主提交用户提问以及工具声明列表 `tools`。
2. **模型产出调用指令**：模型没有直接生成文字回答，而是返回一条特殊的 `AssistantMessage`：
   ```json
   {
     "role": "assistant",
     "content": null,
     "tool_calls": [
       {
         "id": "call_abc123",
         "type": "function",
         "function": {
           "name": "get_weather",
           "arguments": "{\"city\":\"杭州\",\"date\":\"tomorrow\"}"
         }
       }
     ]
   }
   ```
   此时响应的 `finish_reason` 是 `tool_calls`。注意，如果用户同时问了两个城市，现代模型支持并行工具调用（Parallel Tool Calling），会在 `tool_calls` 列表中同时返回多个独立 ID 的调用项。
3. **宿主执行真实逻辑**：宿主解析出函数名 `get_weather` 与 JSON 参数，在本地反序列化为对象，调用对应的 Spring Bean、下游微服务或查询数据库。在这个过程中，第三方系统的访问凭证（如数据库密码、API Key）留在宿主内存中，不暴露给模型。
4. **回传执行结果**：宿主获取到结果（例如 `{"temp": 24, "condition": "晴朗"}`）后，构造一条角色为 `tool` 的消息，并附带对应的 `tool_call_id`：
   ```json
   {
     "role": "tool",
     "tool_call_id": "call_abc123",
     "content": "{\"temp\": 24, \"condition\": \"晴朗\", \"air_quality\": \"优\"}"
   }
   ```
5. **生成最终答复**：宿主把完整的历史消息 `[UserMessage, AssistantMessage(tool_calls), ToolMessage]` 发送给模型。模型根据工具返回的数据，组织出面向用户的最终语句：“杭州明天天气晴朗，气温约 24 度，空气质量为优。”

更多关于工具协议与外部通信演进的讨论，可参阅 [/fundamentals/llm-basics/function-calling.md](/fundamentals/llm-basics/function-calling.md) 和 [/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md](/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md)。

---

## 使用 Spring AI 实现工具调用

在 Java 生态中，**Spring AI** 对底层的 JSON Schema 生成和多轮调用循环进行了封装。

### 1. 使用 `@Tool` 声明工具

通过 `@Tool` 注解，可以将 Spring 容器中托管的 Bean 方法标记为模型可调用的工具：

```java
package com.example.ai.tools;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

@Component
public class WeatherTools {

    public record WeatherRequest(
        @ToolParam(description = "待查询的城市名称，如：北京、杭州", required = true)
        String city,
        @ToolParam(description = "查询日期相对值，支持 today 或 tomorrow", required = true)
        String date
    ) {}

    public record WeatherResponse(String city, int temperature, String condition) {}

    @Tool(description = "根据城市和日期查询实时与预测天气信息")
    public WeatherResponse fetchWeather(WeatherRequest request) {
        // 模拟调用第三方气象系统接口
        if ("杭州".equals(request.city())) {
            return new WeatherResponse("杭州", 24, "晴朗");
        }
        return new WeatherResponse(request.city(), 20, "多云");
    }
}
```

### 2. 通过 ChatClient 挂载并自动执行

Spring AI 的 `ChatClient` 支持自动执行工具调用。当检测到响应中包含 `tool_calls` 时，框架自动通过反射调用对应方法，回传执行结果并请求最终文本：

```java
package com.example.ai.service;

import com.example.ai.tools.WeatherTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class AssistantService {

    private final ChatClient chatClient;

    public AssistantService(ChatClient.Builder chatClientBuilder, WeatherTools weatherTools) {
        this.chatClient = chatClientBuilder
                .defaultTools(weatherTools) // 注册为全局默认工具
                .build();
    }

    public String ask(String userQuery) {
        return this.chatClient.prompt()
                .user(userQuery)
                .call()
                .content(); // 框架自动处理 ToolCall 拦截、Java 方法执行与 ToolMessage 回传
    }
}
```

![Spring AI 内部处理机制](/images/content-rebuild/2ada76d4b15fe2b3/fig-3.png)

### 3. 跳过模型二次润色直接获取工具执行结果

在某些场景下，我们不需要让模型对工具执行结果进行二次自然语言润色（例如大批量数据报表导出、对首字延迟要求极高或需要保持原始格式的数据通道）：

- **两阶段默认闭环**：宿主执行工具 -> 结果装入 `ToolMessage` 发回模型 -> 模型结合结果生成最终自然语言答复。适用于需要对话润色与多源信息综合的场景；
- **直接获取工具结果**：注意 Spring AI 的 `@Tool` 注解仅包含 `name` 与 `description` 元数据，并不提供类似某些框架的 `returnDirect = true` 注解属性。若业务仅需模型提取结构化参数并执行底层服务，应当在宿主层拦截工具调用指令，由本地执行业务逻辑后直接将结果返回给调用方，主动中断第二轮模型推理，避免长文本报表带来的二次 Token 消耗与延迟。

---

## 异常处理与工程安全边界

工具调用将不可控的模型生成内容连接到了受控的业务系统，工程上必须有明确的容错和防御手段。

![异常拦截与错误自愈时序](/images/content-rebuild/2ada76d4b15fe2b3/fig-4.png)

### 1. 外部异常的包装反馈

当工具执行出现非预期情况（如参数不合法、依赖系统超时）时，宿主不要直接把未处理的系统异常抛给前端造成接口崩溃。在很多场景下，将错误包装成结构化文本写入 `ToolMessage`，能为模型提供修正上下文的机会：

```java
@Tool(description = "根据订单号查询订单物流状态")
public String queryOrderStatus(@ToolParam(description = "10位纯数字订单号") String orderId) {
    try {
        if (!orderId.matches("\\d{10}")) {
            // 参数格式不合规，返回提示信息而不是抛出 500
            return "{\"success\": false, \"errorCode\": \"INVALID_FORMAT\", \"message\": \"订单号必须为10位数字，请提示用户重新输入\"}";
        }
        return orderService.findTrack(orderId);
    } catch (ServiceUnavailableException e) {
        return "{\"success\": false, \"errorCode\": \"SYSTEM_BUSY\", \"message\": \"物流系统暂时无响应，请建议用户稍后重试\"}";
    }
}
```

需要说明的是，模型收到错误信息后尝试修正参数属于**概率性行为**，并不是确定性保证。为了防止模型在错误信息中反复死循环调用，宿主程序必须设置最大调用次数上限（如最多循环 3 次）和熔断策略。

### 2. 参数二次校验与防御注入

模型生成的 `arguments` 本质上是未经信任的外部输入。即便其符合 JSON Schema，也可能携带越界数值、恶意 SQL 片段或提示词注入内容：

- **防线设置**：宿主在执行 Java 代码前，需使用标准的业务验证框架（如 `jakarta.validation`）做边界与格式强校验；涉及数据库交互必须使用参数化查询，不得直接拼接模型生成的字符串。

![生产级安全防护边界](/images/content-rebuild/2ada76d4b15fe2b3/fig-5.png)

### 3. 敏感写操作引入人工确认（Human-in-the-Loop）

根据工具对系统状态的影响，通常划分为两类：
- **只读查询型**：如查天气、查知识库、读系统状态，可以交由系统自动调用；
- **状态变更型**：如转账扣款、重置密码、删除集群节点。这类操作通常不应由自动循环直接落盘执行，而是由宿主在生成操作指令后挂起流程，生成待确认事件交由人工在前端显式点击确认后，再触发真实执行。

---

## 面试追问与常见认知误区

### 追问 1：Spring AI 自动调用很方便，什么情况下必须手动接管？

- **自动流转**：适用于只读、简单且无须外部确认的单次查询场景。
- **需要手动接管的场景**：
  1. **人工确认（Human-in-the-Loop）**：写操作需要在用户界面确认后才触发；
  2. **多租户权限注入**：需要根据当前 HTTP 请求的 Token 动态为工具注入租户 ID 和权限范围，防止水平越权；
  3. **严格的循环计数与熔断**：需要在每一轮循环检查全局耗时、Token 消耗或步数，超过限制时强制中断。

### 追问 2：如果向模型注册了 50 个工具，会有什么问题？如何优化？

- **主要问题**：
  1. **Token 开销增加**：每个工具的 JSON Schema 都占用提示词空间，50 个工具会导致每次请求的基础 Token 成本显著增加，同时增加首字延迟（TTFT）；
  2. **路由准确率下降**：工具过多且描述有交集时，模型容易挑错工具、漏传参数或产生幻觉；
- **优化方法**：
  1. **动态检索（Tool Retrieval）**：先用轻量文本检索或向量相似度，从工具库中根据用户提问检索出最相关的 3~5 个候选工具，再动态放进提示词中；
  2. **分层任务分发**：由主调度模型识别用户所属的业务域，将任务路由给拥有较小工具集合的子系统处理。

![常见面试认知误区对比](/images/content-rebuild/2ada76d4b15fe2b3/fig-6.png)

### 常见错误认知辨析

| 常见错误认知 | 真实机制与事实依据 |
| :--- | :--- |
| **“模型内部可以直接发起 HTTP 请求访问互联网”** | **错误**。模型运行在推理环境中，只负责根据输入计算概率并输出文本。网络请求均由宿主程序（如 Java 进程）代为发起。 |
| **“既然在 JSON Schema 里定义了 required，后端就无需再做入参校验”** | **错误**。模型输出具有随机性，可能漏传字段、填入不合法值或注入恶意字符串。宿主程序必须做常规的参数合法性防御。 |
| **“一次问答只能调用一个工具”** | **错误**。现代模型支持并行工具调用（Parallel Tool Calling），能在一次响应中生成包含多个不同 `tool_call_id` 的调用列表。 |
