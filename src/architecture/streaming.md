---
title: 大模型流式响应怎么做？从协议选型、代理缓冲到级联取消
icon: mdi:arrow-split-horizontal
description: 解析大模型流式响应的端到端实现：对比 SSE 与 WebSocket、排查 Nginx 缓冲延迟、实现前后端级联取消以避免算力浪费，以及解析工具调用的混合数据流。
---

在构建基于大语言模型（LLM）的交互应用时，如果采用传统的“等待全部生成完毕再一次性返回”模式，用户往往需要面对数秒甚至数十秒的白屏等待。大模型生成文本是一个自回归解码过程，即逐个预测并输出 Token。流式传输（Streaming）将模型每生成的一个或几个 Token 即时推送到前端展示，使用户感知的等待时间从“整段完成时间”转变为“首 Token 延迟”（Time to First Token, TTFT）。

我们将以一个典型的生产架构贯穿全文：前端通过 Web 界面向应用发起对话请求，流量经过 Nginx 反向代理到达 FastAPI 业务后端，业务后端再调用上游推理引擎（如 vLLM 或云端模型 API）。

![流式输出架构与实时传输协议总览](/images/content-rebuild/b271d5f10e97a7de/cover.png)

要让整条链路稳定顺畅，除了在代码中写一个生成器之外，还需要处理传输协议选型、代理缓冲、主动中断取消以及混合工具调用等一系列具体工程问题。

## 流式协议选型：SSE 与 WebSocket 的工程权衡

在浏览器与后端之间传输增量文本，最常见的两种技术是 **Server-Sent Events (SSE)** 和 **WebSocket**。

| 评估维度 | Server-Sent Events (SSE) | WebSocket |
| :--- | :--- | :--- |
| **底层协议** | 标准 HTTP（HTTP/1.1 或 HTTP/2/HTTP/3） | 独立 TCP 协议（通过 HTTP Upgrade 握手） |
| **通信模式** | 单向推送（服务端 $\rightarrow$ 客户端） | 全双工双向传输（客户端 $\leftrightarrow$ 服务端） |
| **数据编码** | UTF-8 文本（遵循 `data: ...\n\n` 换行规范） | 文本帧或二进制帧 |
| **网关与网络穿透** | 与常规 HTTP 兼容，天然穿透 CDN、WAF 与常规反向代理 | 需要代理显式支持连接升级，长连接易受中间件超时掐断 |
| **连接复用** | 在 HTTP/2 下可多路复用，避免浏览器同域名连接数限制 | 每个连接占用独立 TCP 通道或专用连接池 |
| **身份鉴权** | 支持标准 HTTP Headers（如 Authorization、Cookie） | 浏览器原生 API 不支持自定义 Header，通常需依赖 Query 参数或握手后首帧鉴权 |

对于大部分 LLM 对话场景，客户端单次发送包含历史记录的 Prompt，随后只需被动接收持续推送的文本片段。这种“一问多发”的非对称交互与 SSE 非常契合。WebSocket 虽然支持全双工，但维护心跳、状态机重连和通过安全网关的成本更高，通常只在端到端实时语音、实时白板等需要高频双向低延迟互动的场景中使用。

![SSE 与 WebSocket 协议机制对比](/images/content-rebuild/b271d5f10e97a7de/fig-1.png)

### 浏览器原生 EventSource 的局限与替代方案

需要注意，浏览器原生的 `EventSource` API 仅支持发起 `GET` 请求，且无法在握手时自定义请求标头（如添加 JWT `Authorization`）。由于 LLM 请求通常需要通过 `POST` 发送体积较大的上下文消息，现代前端通常改用 `fetch()` 配合 `ReadableStream`，或使用社区库（如 `@microsoft/fetch-event-source`）手动消费遵循 SSE 文本格式的响应流。

## 后端流式管道：代理缓冲与事件格式规范

如果流式请求经过反向代理时被暂存，前端依然会感知到严重的卡顿。理解缓冲的触发原因并正确配置是保证流式体验的前提。

<ReadMoreLock />

### 代理缓冲穿透机制

以反向代理 Nginx 为例，其默认行为是尽可能高效地吞吐数据：开启 `proxy_buffering on` 时，Nginx 会将后端服务返回的数据暂存在内部缓冲区（通常为 4KB 或 8KB），等到缓冲区被填满或后端主动关闭连接后，才一次性刷写给客户端。这会导致模型输出的前几十个甚至上百个 Token 积压在代理中，破坏了流式交互的效果。

要保证响应块即时透传，需要在反向代理和后端应用协同进行配置：

1. **反向代理配置**：针对流式路由关闭代理缓冲：
   ```nginx
   location /v1/chat/stream {
       proxy_pass http://backend_upstream;
       proxy_buffering off;
       proxy_cache off;
       # 关闭 gzip 实时压缩缓冲，防止压缩器暂存数据块
       gzip off;
   }
   ```
2. **应用响应头**：后端服务返回显式的 HTTP 标头指示网关透传：
   * `Content-Type: text/event-stream; charset=utf-8`
   * `Cache-Control: no-cache, no-transform`
   * `Connection: keep-alive`
   * `X-Accel-Buffering: no`（专门用于指示 Nginx 关闭当前响应的代理缓冲）

![Nginx 代理缓冲与 Chunk 透传机制](/images/content-rebuild/b271d5f10e97a7de/fig-2.png)

### 遵循 W3C 规范的事件流分块

标准的 SSE 响应是由纯文本组成的字符流。每条事件消息由一个或多个字段构成，字段以单换行符 `\n` 结尾，单条事件消息之间必须以双换行符 `\n\n` 分隔：

```text
event: delta
data: {"content": "你"}

event: delta
data: {"content": "好"}

event: citation
data: {"source_id": "doc_12", "snippet": "参考资料内容..."}

event: done
data: [DONE]

: keep-alive

```

其中以冒号 `:` 开头的行为注释行，常用于在模型 Prefill 阶段较长或排队时定时发送心跳包，防止客户端或中间网关因达到闲置超时时间（Idle Timeout）而异常掐断连接。

## 客户端交互生命周期与首 Token 延迟（TTFT）

用户等待体验的核心指标是 **首 Token 延迟（TTFT, Time to First Token）**。从物理链路拆解，它由以下几个环节串联而成：

$$\text{TTFT} = t_{\text{DNS+TCP+TLS}} + t_{\text{网关转发}} + t_{\text{Prompt Prefill}} + t_{\text{首Token生成}} + t_{\text{网络传输与代理透传}} + t_{\text{客户端渲染}}$$

1. **建连与网络传输**：涉及 DNS 解析、TCP 握手和 TLS 握手。通过 HTTP/2 连接复用可省去后续请求的握手耗时。
2. **模型 Prefill 计算**：上游推理引擎加载并计算输入 Prompt 的注意力矩阵，构建初始 KV Cache。Prompt 越长，该耗时通常越长。
3. **首 Token 生成**：Prefill 完成后，模型采样产出第一个 Token 并完成网络分包。
4. **流式透传与渲染**：第一个数据块通过反向代理直接到达前端，前端完成文本反序列化并更新 DOM。

![首 Token 延迟（TTFT）与交互生命周期分解](/images/content-rebuild/b271d5f10e97a7de/fig-3.png)

在流式生成中，一旦 TTFT 完成，用户就已经开始阅读内容。后续各个 Token 之间的时间间隔称为 **块间延迟（Inter-Chunk Latency）**，只要该延迟保持在人类正常阅读速度范围内（例如每秒输出 15~30 个 Token），用户就不会感到明显的等待。

## 主动中断与全链路级联取消

在日常使用中，用户看到前几句输出后发现不符合预期，经常会点击“停止生成”。如果前端仅仅停止在页面上渲染，而后端还在继续驱动模型逐 Token 推理，不仅会浪费上游 GPU 的算力，还会产生不必要的 API 计费开销。

### 级联取消的传播链路

完整的资源回收要求断连信号从客户端一直传递到模型推理引擎：

$$\text{前端 AbortController} \longrightarrow \text{关闭 TCP 连接} \longrightarrow \text{Nginx 感知断开} \longrightarrow \text{后端捕获取消异常} \longrightarrow \text{关闭上游 API 流} \longrightarrow \text{释放槽位 / 停止后续计费}$$

1. **前端触发中断**：前端调用 `AbortController.abort()`，浏览器主动关闭当前的 HTTP 连接（发送 TCP FIN 或 RST 包）。
2. **代理层传递**：Nginx 需配置 `proxy_ignore_client_abort off;`（默认即为 off）。当感知到下游客户端断开连接时，Nginx 紧接着切断与后端业务服务的连接。
3. **业务后端感知**：在基于异步架构（如 Python AsyncIO / Starlette / FastAPI）的后端中，底层 Socket 中断会导致运行中的响应生成任务接收到 `asyncio.CancelledError`。
4. **上游流显式关闭**：业务后端在捕获到取消异常或检测到连接中断后，必须在 `finally` 块中显式关闭与上游模型服务的流式会话（如调用 `await client.aclose()` 或 `await response.aclose()`）。
5. **终止上游生成与资源释放**：
   * **自建推理引擎（如 vLLM）**：检测到底层连接切断后，调度器会中止当前的自回归解码循环，并释放该序列在显存中占用的 KV Cache 槽位，避免 GPU 算力浪费。
   * **外部商用 API**：关闭 HTTP 会话能阻止后续未生成 Token 的解码与计费增长；但调用产生的基础 Prefill 输入 Token 与断开前已生成的输出 Token 仍会被供应商计费，无法全额免除已实际消耗的额度。

![主动中断与全链路资源回收流程](/images/content-rebuild/b271d5f10e97a7de/fig-4.png)

### 级联取消代码实现示例

在基于 FastAPI / Starlette 的异步架构中，客户端断开连接会由 ASGI 容器（如 Uvicorn）检测到底层 Socket 断开，并向当前响应处理协程抛出 `asyncio.CancelledError`。切忌在生成器循环内主动调用 `await request.is_disconnected()`，因为在客户端未发送上行消息时该调用会持续挂起阻塞整个生成流，且会与框架自身的断开监听机制争抢 ASGI receive 通道。

正确的实现应依赖取消异常并在 `finally` 块中执行清理，主动关闭上游连接：

```python
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
UPSTREAM_API = "https://api.upstream-model.local/v1/chat/completions"

async def stream_chat_generator():
    # 建立与上游模型服务的 HTTP 异步客户端
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    try:
        # 向上游发起流式请求
        async with client.stream(
            "POST",
            UPSTREAM_API,
            headers={"Authorization": "Bearer mock-token"},
            json={"model": "llama-3", "stream": True, "messages": [{"role": "user", "content": "你好"}]}
        ) as upstream_resp:
            async for raw_chunk in upstream_resp.aiter_bytes():
                yield raw_chunk
    except asyncio.CancelledError:
        # 当客户端强制切断连接，ASGI 框架向协程抛出 CancelledError
        pass
    finally:
        # 确保在退出生成器或被取消时彻底关闭上游 HTTP 连接，避免模型持续空转
        await client.aclose()

@app.post("/v1/chat/stream")
async def chat_stream_endpoint():
    return StreamingResponse(
        stream_chat_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
    )
```

## 混合流式传输：文本与结构化工具调用的协同

在 Agent 智能体架构中，大模型的输出不只有纯自然语言文本，还会交织着结构化函数调用（Tool Calls）。大模型在调用工具时，同样是逐 Token 流式输出 JSON 字符串的片段。

![文本与结构化工具调用混合流解析](/images/content-rebuild/b271d5f10e97a7de/fig-5.png)

### 增量 JSON 的解析挑战

标准的 `JSON.parse` 无法直接解析不完整的字符串（例如 `{"city": "Bei`）。处理混合流通常有两种策略：

1. **透传参数片段，前端拼装**：
   * 当上游返回包含 `delta.tool_calls` 的 Chunk 时，后端以 `event: tool_call` 将增量参数原样推送给前端。
   * 这种模式后端几乎零延迟开销，但前端需要维护参数缓冲累加器，等到整段 JSON 闭合或接收到 `finish_reason: tool_calls` 标志后，再将拼接完整的 JSON 反序列化。
2. **后端状态机累积并执行**：
   * 如果该工具调用属于内部操作（例如模型自主查询知识库，查询结果需要再喂回给模型），后端会在内存中暂存参数字符串片段。
   * 状态机等待流式结束标志到达，将完整 JSON 反序列化并通过 Pydantic/JSON Schema 校验后执行本地函数，将执行结果构建为新消息再次发起模型的下一轮解码推理。

## 生产可观测性与验收验证

流式系统无法仅靠最终 HTTP 状态码 200 来评估运行质量。长连接下的性能与资源健康度需要专用指标来度量：

![流式可观测性大屏与核心指标验收](/images/content-rebuild/b271d5f10e97a7de/fig-6.png)

### 关键度量指标

1. **首 Token 延迟分位数（TTFT p50 / p95 / p99）**：
   * 反映系统的初始响应敏捷度。若 TTFT p95 显著偏高，通常需要排查 Prefill 阶段的排队情况或网络建连握手耗时。
2. **块间延迟（Inter-Chunk Latency）**：
   * 衡量连续两个 Token 之间推送的间隔时间。如果均值正常但方差很大，说明推理服务可能出现了 GPU 显存换入换出卡顿、负载过高或链路中存在隐式网络丢包。
3. **中断无用 Token 率（Wasted Token Ratio）**：
   * 衡量用户中断后，上游推理引擎仍然无效生成并被丢弃的 Token 比例：

$$\text{Wasted Token Ratio} = \frac{\sum \text{客户端中断后上游继续生成的 Token 数}}{\sum \text{总生成 Token 数}}$$

如果系统上线后该比率明显偏高，通常意味着下游主动断连信号未能在 Nginx 或后端异步协程中被正确感知并传递至上游推理服务。通过端到端的级联取消排查，可以使资源浪费率回落至健康区间。
