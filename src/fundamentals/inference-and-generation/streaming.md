---
title: 流式输出怎么实现？从模型逐 Token 生成到网络分块传输
icon: mdi:message-arrow-left-outline
tag:
- 推理与生成
- 流式输出
- SSE
description: 拆解大模型流式输出全链路：自回归逐步推断、Python异步队列解耦、HTTP分块与SSE协议、UTF-8截断避坑及连接断开资源释放。
---

在调用普通的 Web 接口时，客户端发起 HTTP 请求，服务端在内存中把数据完全拼装完毕，通过响应头的 `Content-Length` 标明大小，随后一次性返回给前端。但在大语言模型的文本生成场景中，如果让服务端等待整段回答全部生成完毕再返回，用户往往需要面对数秒甚至十几秒的空白等待。

大模型应用常见的“打字机式逐字弹出”效果，并不是前端用 JavaScript 刻意制造的视觉特效，而是服务端在模型每算出一个 Token（词元）时，就通过长连接协议立即推送到浏览器或客户端的结果。本文将以用户向助手提问“请用几句话解释量子纠缠”为例，梳理从模型自回归解码、服务端队列解耦、网络层分块传输，到字符集解码与连接中断控制的完整实现机制。

![流式输出端到端交付管道全景图](/images/content-rebuild/8bf013dee926e0ac/cover.png)

### 为什么流式传输能改善体验？首字延迟与字间延迟

大模型的生成过程是**自回归（Autoregressive）**的：模型必须依赖前面已经生成的所有上下文，通过一次神经网络前向计算预测出下一个 Token，随后将新 Token 拼入上下文，再进入下一次前向计算。这个生成周期在物理上分为两个截然不同的阶段：

1. **首字阶段（Prefill 阶段）**：模型并行处理输入的 Prompt，计算所有提示词的键值对并写入显存（相关机制详见 [/fundamentals/inference-and-generation/inference-pipeline.md](/fundamentals/inference-and-generation/inference-pipeline.md) 与 [/fundamentals/inference-and-generation/kv-cache.md](/fundamentals/inference-and-generation/kv-cache.md)），最终产出第 1 个输出 Token。从用户点击发送到接收到第 1 个 Token 的耗时，称为**首字延迟（Time to First Token, TTFT）**。
2. **解码步进阶段（Decode 阶段）**：在首字之后，模型受显存带宽等硬件限制，逐个串行解码后续的 Token。平均每生成一个 Token 所消耗的时间，称为**字间延迟（Time per Output Token, TPOT）**。

假设一次回答需要输出 $N$ 个 Token。在各步计算耗时均匀且无网络剧烈抖动的简化假设下，端到端总生成耗时 $T_{\text{total}}$ 可表示为：

$$T_{\text{total}} = \text{TTFT} + (N - 1) \times \text{TPOT}$$

![一次性响应与流式响应的时序与延迟指标对比](/images/content-rebuild/8bf013dee926e0ac/fig-1.png)

假设生成一段 300 个 Token 的回答，测得 TTFT 约为 300 毫秒，Decode 阶段的 TPOT 约为 30 毫秒：
- **一次性响应模式**：用户必须等待全部生成结束，总等待时间约为 $300 + (300 - 1) \times 30 = 9270\text{ 毫秒}$（即 9.27 秒）。在这 9 秒多内，前端没有任何反馈。
- **流式响应模式**：用户在 300 毫秒后就能看到第 1 个字并开始阅读。随后，系统以约 33 Token/秒的速率持续推送文本。由于普通读者的阅读速度一般在每秒 5 到 10 个汉字左右，流式交付将用户感知到的等待时间从 9 秒多压缩到了 0.3 秒，阅读过程与计算过程完全重叠。

<ReadMoreLock />

### 服务端如何释放逐步计算？生成器与异步队列解耦

普通 Python 函数使用 `return` 返回数据，一旦执行 `return`，函数的执行上下文就会销毁。如果推理逻辑写在普通函数内，外层 Web 服务就只能等待自回归循环遇到结束符 `[EOS]` 或达到长度上限后，才能拿到完整的字符串。

为了将每步产生的 Token 实时交出，现代推理服务普遍使用**生成器（Generator）**或**异步迭代器（Async Iterator）**。

![推理引擎步进计算与异步迭代器的让渡解耦流程](/images/content-rebuild/8bf013dee926e0ac/fig-2.png)

通过 `yield` 关键字，推理代码在执行完单步 Decode 并采样出 Token 后，可以暂停当前函数的执行，把计算出的数据切片交还给调用者，待调用者处理完（如推入网络缓冲区）后再恢复执行。

在实际生产中，运行在 GPU 上的模型推理和运行在 CPU 上的 Web 异步事件循环（Event Loop）往往运行在不同的线程或进程中。两者速率和调度方式不同，必须通过**异步队列**进行解耦，避免网络发送慢反向卡死 GPU 推理，或者 GPU 计算阻塞 Web 服务器处理其他网络请求：

```python
import asyncio
from typing import AsyncGenerator, Optional

class AsyncTokenStreamer:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        # asyncio.Queue 本身非线程安全，跨线程（计算线程 -> 网络协程）通信需绑定事件循环
        self.loop = loop or asyncio.get_running_loop()
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def put(self, token_text: str) -> None:
        """由后台推理计算线程调用，通过 call_soon_threadsafe 线程安全地放入新 Token"""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, token_text)

    def end(self) -> None:
        """推理完成时调用，线程安全地放入 None 作为结束哨兵"""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        """由 Web 传输层消费，逐个拉取 Token 发往客户端"""
        while True:
            token = await self.queue.get()
            if token is None:  # 遇到结束哨兵，安全退出
                self.queue.task_done()
                break
            yield token
            self.queue.task_done()
```

这种生产者-消费者模型使推理计算专注向队列写数据，Web 层按网络发送窗口从队列读数据，实现了计算与 I/O 的平滑连接。

### 网络层协议实现：HTTP 分块传输与 SSE 规范

常规 HTTP/1.1 响应通过 `Content-Length: 1024` 告诉对端响应体一共有多长，浏览器读满指定字节数即认为传输完毕。但在流式生成刚开始时，服务端无法预知最终回答会包含多少个 Token，因此不能设置固定长度的 `Content-Length`。

网络层通过分块传输与文本事件流解决这一问题：

![HTTP 分块传输编码与 SSE 事件流线协议结构](/images/content-rebuild/8bf013dee926e0ac/fig-3.png)

1. **传输编码层**：
   - 在 **HTTP/1.1** 中，服务端响应头设置 `Transfer-Encoding: chunked`。此时数据体被切分为一个个独立的块，每个块由“十六进制表示的块长度 + 回车换行 + 数据内容 + 回车换行”组成，最后以长度为 0 的块标记响应结束。
   - 在 **HTTP/2 与 HTTP/3** 中，传输层原生支持二进制分帧（Framing）。流式数据被直接切分进多个 `DATA` 帧传输，当最后一个 `DATA` 帧带有 `END_STREAM` 标志时表示传输收束，因此在 HTTP/2/3 下不再需要显式声明 `Transfer-Encoding: chunked`。
2. **应用数据层（Server-Sent Events, SSE）**：
   大模型 API 广泛采用 W3C 标准的 SSE（`text/event-stream`）。相比于定义复杂的二进制私有协议，SSE 是基于标准 HTTP 的纯文本线协议（Wire Format），能天然穿透各级代理网关。标准数据格式如下：

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
Transfer-Encoding: chunked

data: {"text": "量子"}

data: {"text": "纠缠"}

data: [DONE]

```

SSE 的核心格式规则十分严格：
- 每一行以特定的字段名开头（如 `data:`、`event:`、`id:`，以冒号 `:` 开头的行则作为保活注释行被客户端忽略）。
- 字段名与内容之间通常用一个空格分隔。
- 每一个完整事件消息必须以**连续两个换行符 `\n\n`** 结尾。客户端解析器以此判定一条消息接收完毕，并触发前端回调。
- 需要特别说明的是，最后一行出现的 `data: [DONE]

` 并不是 W3C SSE 规范内置的语法，而是 OpenAI Chat Completions API 及行业大模型普遍约定的应用层结束标志，客户端据此主动停止监听并关闭请求。

### 协议选型权衡：为什么选 SSE 而不是 WebSocket？

许多开发者会产生疑问：既然要实现实时推流，为什么大模型生成普遍采用 SSE，而不是全双工的 WebSocket？

![SSE 与 WebSocket 在基础设施与双向通信维度的对比](/images/content-rebuild/8bf013dee926e0ac/fig-4.png)

这主要源于大模型文本生成的通信模式与系统运维成本：

1. **通信单向性与业务匹配**：在标准的提示词交互中，客户端提交完 Prompt 后，数据流完全是服务端向客户端的单向输出；在生成过程中，客户端除了可能发出中断信号外，不需要频繁向上行链路注入数据。单向的 SSE 刚好覆盖该业务场景。
2. **现有基础设施兼容性**：SSE 是标准的 HTTP 响应。这意味着团队现有的 API 网关（如 Nginx、Envoy、Kong）、WAF 防火墙、身份鉴权中间件（Bearer Token、Cookie）、链路追踪机制无需进行任何适配，即可开箱即用。而 WebSocket 在握手阶段需要执行协议升级（`HTTP 101 Switching Protocols`），在经过某些反向代理或企业防火墙时，常面临连接被杀、超时重连难、长连接无法被标准无状态网关有效负载均衡等维护问题。
3. **连接复用效率**：基于 HTTP/2 或 HTTP/3 的 SSE 可以与同一个域名下的普通 HTTP 请求完全复用底层 TCP/QUIC 连接，减少频繁握手的开销。

**适用边界**：如果系统属于实时双向语音通话、客户端持续上传音频流且模型持续下发音频流的全双工多模态场景，WebSocket 或 WebRTC 才是更合理的技术选型。

### 避坑一：UTF-8 多字节字符跨 Token 截断与乱码

在实现流式输出时，最容易出现且极具隐蔽性的工程 Bug 是**中文字符偶发乱码**。

在现代主流分词器（如基于 Byte-Pair Encoding 的 tiktoken、Llama 分词器）中，分词的基本单元常常是字节序列（Byte Level），而非 Unicode 字符。在 UTF-8 编码下，一个标准中文汉字通常占用 3 个字节。例如汉字“你”的 UTF-8 编码为三个十六进制字节：`0xE4 0xBD 0xA0`。

如果分词词表由于训练统计规则，把前 2 个字节 `[0xE4, 0xBD]` 划分为 Token A，把第 3 个字节 `[0xA0]` 划分为接下来的 Token B：
- 如果服务端在生成 Token A 后，立即调用 `tokenizer.decode([Token A])`，由于收到的字节序列不完整，单 Token 解码器无法将其转换为合法字符，就会报错或替换为乱码占位符 ``（Unicode 替换字符 `\ufffd`）推向网络。
- 紧接着 Token B 到达，单独解码 `[0xA0]` 同样是非法字节，再次输出一个 ``。前端屏幕上就会留下永久或闪烁的乱码字符。

![UTF-8 多字节拆分在单 Token 解码与增量解码缓冲区下的行为对比](/images/content-rebuild/8bf013dee926e0ac/fig-5.png)

解决这一问题的标准手段是引入**增量解码器（Incremental Decoder）**。增量解码器在内部维护一个未决字节缓冲区：当喂入的字节不足以凑齐合法字符时，暂留在缓冲区；直到后续 Token 带来的字节补全了整个字符，才一次性输出完整文字：

```python
import codecs

def create_streaming_utf8_decoder():
    # 使用 Python 内置的 UTF-8 增量解码器
    # errors='replace' 作为兜底，遇到真正非法的字节才替换
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    
    def decode_chunk(token_bytes: bytes) -> str:
        # final=False 明确告知解码器后续还有流数据，不要把未完结的末尾字节强制报错
        return decoder.decode(token_bytes, final=False)
        
    return decode_chunk

# 验证：将汉字“你”（b'\xe4\xbd\xa0'）人为拆分为两个 Token 字节串
token_a_bytes = b'\xe4\xbd'
token_b_bytes = b'\xa0'

decode = create_streaming_utf8_decoder()

chunk_1 = decode(token_a_bytes)
print(f"接收 Token A 后输出: repr('{chunk_1}')")
# 实际输出: repr('') -> 字节暂存内部缓冲，不吐出错误乱码

chunk_2 = decode(token_b_bytes)
print(f"接收 Token B 后输出: repr('{chunk_2}')")
# 实际输出: repr('你') -> 凑齐 3 字节，正确输出汉字
```

在主流的开源推理框架（如 Hugging Face 的 `TextIteratorStreamer` 或 vLLM 的 Detokenizer 模块）中，内部均集成了增量解码机制，确保发送到网络套接字的每个数据块都由合法的 UTF-8 文本组成。

### 避坑二：客户端主动取消与算力悬挂泄漏

在实际业务中，用户经常在看到前两句话后发现不是想要的内容，点击“停止生成”，或者直接关闭了浏览器标签页。

在网络层面，前端的 `fetch` API 会通过 `AbortController.abort()` 触发取消，向操作系统套接字发出 `TCP RST` 包，或在 HTTP/2 中发出 `RST_STREAM` 帧通知服务端中断连接。

如果服务端对网络连接的中断事件缺乏感知，就会引发严重的**算力悬挂泄漏**：Web 框架虽然无法再把数据推给对端，但后端的推理引擎仍然在浑然不觉地逐步计算接下来的数百个 Token。这不仅空耗 GPU 算力，还会让该请求长时间霸占显存中的 KV Cache 槽位，直接导致集群并发吞吐断崖式下跌。

![客户端 Abort 信号沿网络栈触发推理引擎 abort 回调的级联断开链路](/images/content-rebuild/8bf013dee926e0ac/fig-6.png)

在生产级 Web 框架（以 FastAPI / Starlette 为例）中，必须通过异步状态监听将断开事件传递给推理后端：

```python
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

async def mock_inference_generator(request_id: str):
    """模拟推理引擎，每步 Decode 耗时 40ms"""
    try:
        for i in range(100):
            await asyncio.sleep(0.04)
            yield f"data: Token_{i}\n\n"
    except asyncio.CancelledError:
        # 当上层协程被取消时，触发底层显存释放
        print(f"[推理引擎] 请求 {request_id} 已取消，立即回收显存 KV Cache 槽位")
        raise

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    request_id = "req_9527"

    async def event_publisher():
        try:
            async for chunk in mock_inference_generator(request_id):
                # 主动轮询底层连接是否已经挂断
                if await request.is_disconnected():
                    print(f"[Web层] 客户端断开连接: {request_id}")
                    break
                yield chunk
        finally:
            print(f"[Web层] 退出流式发布循环: {request_id}")

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关键：通知 Nginx 等反向代理关闭响应缓冲，避免数据被网关积攒滞留
            "X-Accel-Buffering": "no"
        }
    )
```

这里的两个工程细节非常关键：
1. **`X-Accel-Buffering: no` 响应头**：如果服务前部署了 Nginx 等反向代理，Nginx 默认会开启代理缓冲（Proxy Buffering），试图积攒满 4KB 或 8KB 数据才一次性发送给浏览器。添加该响应头（或在 Nginx 配置中针对该路由设置 `proxy_buffering off;`），才能确保每个 Token 数据帧一到达网关就被立即刷向用户，防止流式输出退化为大块延迟卡顿。
2. **级联释放调用**：当检测到连接断开（`request.is_disconnected()` 为真或捕获到 `asyncio.CancelledError`）时，服务层需调用推理引擎的取消接口（例如 vLLM 中的 `engine.abort(request_id)`）。引擎随即回收对应请求持有的 PagedAttention 显存物理块，并将其移出调度批次。关于流式高并发网关的更多架构模式可参考 [/architecture/streaming.md](/architecture/streaming.md)。

---

通过自回归逐步推断、异步队列解耦、HTTP 分块协议以及增量解码缓冲，系统即可稳定实现平滑的逐字输出。然而，如果业务要求模型输出的是严格的结构化数据（例如符合特定 Schema 的 JSON）或者函数调用（Function Calling），流式传输就会遇到新的问题：数据在传输过程中处于括号未闭合、键名不完整的半成品状态，传统解析器会直接报错。我们在下一讲 [/fundamentals/inference-and-generation/structured-output.md](/fundamentals/inference-and-generation/structured-output.md) 中，继续讨论流式环境下的语法约束解码与增量解析机制。
