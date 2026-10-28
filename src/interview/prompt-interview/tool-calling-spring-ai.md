---
title: 什么是工具调用 Tool Calling？如何利用 Spring AI 实现工具调用？
icon: mdi:tools
tag:
  - Tool Calling
  - Spring AI
  - 工具
---

# 什么是工具调用 Tool Calling？如何利用 Spring AI 实现工具调用？

这道题考的是 AI 模型"动手做事"的能力——能不能从"能说"升级到"能做"。面试官想看你对 LLM 集成外部能力的理解深度。

我从四个方面来讲：工具调用的本质、完整工作流程、@Tool 注解用法、注册方式选择。

---
## AI不止会聊天，还能"动手做事"

先说清楚什么是工具调用。

Tool Calling 就是让大模型能调用 Java 后端函数的能力。模型不再只是生成文字，而是能真正执行操作——查天气、发邮件、调接口、查数据库。

**关键点：模型只生成调用指令，实际执行由 Java 应用完成。**

这就好比给 AI 配了一个"执行者"。军师出主意（LLM 生成 JSON 指令），执行者负责干活（Java 应用调用真实接口）。模型永远拿不到任何 API 密钥，也永远无法直接访问外部系统。

这种安全设计是工具调用的核心价值。模型像个参谋，只出主意不碰枪。

---
<ReadMoreLock />

## 工具调用是怎样工作的？四步走

整个流程分四步：

**第一步：定义工具**。用 Java 写一个带 @Tool 注解的方法，告诉 AI 这个方法能干什么。

**第二步：注册工具**。把定义好的工具注册到 ChatClient 或 ChatModel 里，让 AI 模型能"看见"这些工具。

**第三步：模型请求**。用户提问时，模型会判断要不要调用工具，如果需要就生成一个 JSON 格式的调用请求。

**第四步：执行反馈**。Java 应用收到请求，调用真实方法，把结果返回给模型，模型再组织最终回答。

类比一下，就像点外卖：你下单（提问）→ 商家接单（注册）→ 厨房备餐（模型判断调哪个工具）→ 骑手配送（执行并返回结果）。

工具分两类：**信息检索型**（查天气、查库存）和**操作执行型**（发邮件、下订单）。前者只返回数据，后者可能改变系统状态。

---

## @Tool注解——Spring AI 2.0 的新方式

Spring AI 2.0 主推 @Tool 注解，代码量最少。

java
@Tool(name = "get_weather", description = "查询指定城市的天气情况")
String getWeather(@ToolParam(description = "城市名称", required = true) String city) {
    return "北京今天晴，25度";
}


注解属性说明：
- **name**：工具的唯一标识，模型靠它决定调哪个
- **description**：功能描述，非常重要，模型靠它判断该不该调用
- **returnDirect**：设为 true 时，结果直接返回给用户，不经过模型处理

参数用 @ToolParam 标记，description 帮助模型理解参数含义，required 标注是否必填。

Spring AI 会自动把这个方法转成 JSON Schema，模型就能"看懂"这个工具怎么用。

就像给方法贴了张功能说明书，AI 自己会阅读并判断何时使用。

---

## 工具注册有门道，版本选择要记牢

注册位置分四种：
- ChatClient 的 runtime tools（请求级别）
- ChatClient 的 default tools（全局默认）
- ChatModel 的 runtime tools（模型级别）
- ChatModel 的 default tools（模型全局默认）

**Spring AI 2.0 推荐用 @Tool 注解**，代码最少，Spring 自动扫描注册。

如果需要细粒度控制，用 MethodToolCallback 自己构建回调。

**别用 FunctionCallbackWrapper**，那是旧 API，已经 deprecated 了。

选工具箱的原则：新手用现成的（@Tool），专业玩家自己组装（MethodToolCallback），别碰老古董（FunctionCallbackWrapper）。

还有个注意点：@Tool 注解的方法必须是 Spring Bean，或者在 AOT 编译时额外配置反射，否则扫描不到。

---

## 面试怎么答

**基础版**：

工具调用是让大模型能调用 Java 函数的能力，使 AI 从"能说"升级到"能做"。用 @Tool 注解标记 Java 方法，Spring AI 自动把它转成 JSON Schema 给模型看。完整流程是：定义工具→注册工具→模型判断是否调用→Java 执行并返回结果。工具分信息检索型和操作执行型两类。Spring AI 2.0 推荐直接用 @Tool 注解，代码量最少。

**加分版**：

工具调用的核心是安全架构——模型只生成 JSON 调用指令，100% 由 Java 应用执行，模型永远无法直接访问任何 API。@Tool 的 returnDirect 属性控制结果返回路径，适合简单查询场景。注册位置要理解作用域差异：ChatClient 是请求级别，ChatModel 是模型级别，runtime 是动态的，default 是全局的。还要注意 @Tool 方法必须是 Spring Bean，AOT 编译时需要额外处理反射扫描。工具调用是 Agent 模式的雏形，理解这个对后续学习 Agent 很关键。

---

一句话总结：**工具调用让 AI 从"能说"到"能做"，核心是 @Tool 注解声明 + Java 执行层安全隔离。**
