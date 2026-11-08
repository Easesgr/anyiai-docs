---
title: 流式输出选 SSE 还是 WebSocket？大模型通信机制与工程实现
icon: mdi:book-open-page-variant
description: 对比大模型流式输出场景下 SSE 与 WebSocket 的协议机制、网关缓冲陷阱与连接并发瓶颈，给出选型依据与调试代码。
---

大语言模型生成回复采用自回归解码（Autoregressive Decoding），即模型内部根据前文逐个推导后续 Token。在 [/llm-100/app-system/model-api-request-lifecycle.md](/llm-100/app-system/model-api-request-lifecycle.md) 中，我们分析了请求的生命周期：服务端分为处理提示词的 Prefill 阶段与逐字生成的 Decode 阶段。

假设用户请求生成一段 500 字的分析报告，服务端 Prefill 耗时 1.2 秒，后续以每秒 40 个 Token 的速度生成 500 个 Token（Decode 耗时 12.5 秒）。如果采用传统的“单次 HTTP 请求-响应”模式，客户端必须等待整整 13.7 秒直到所有文本生成完毕才能一次性拿到数据；在这十多秒内，用户只能面对空白加载动画。而如果采用流式传输，客户端在 1.2 秒左右接收到首个 Token，随后界面随着生成节奏平滑打字显示，用户的首字等待感知被压缩到只有 Prefill 和首包传输时间。

实现这种增量数据推送，Web 工程中主要有两个方案：**Server-Sent Events（SSE）** 与 **WebSocket**。两者工作层次不同，适用场景也存在明确分界。

![流式输出技术选型封面](/images/content-rebuild/cd6574219e99cb39/cover.png)

---

## 传输时序：全量等待与流式推送的差异

为了直观理解流式输出带来的体验差异，我们先对比两种通信模式在时间轴上的事件流向：

![TTFT 与逐字流式传输时序对比](/images/content-rebuild/cd6574219e99cb39/fig-1.png)

- **传统请求-响应模式**：客户端发起一个 HTTP POST 请求后，服务端在模型推理期间一直挂起连接。直到全部 Token 生成结束，服务端将完整 JSON 字符串组装并返回。用户感知到的首字时间（Time To First Token, TTFT）等于“Prefill 耗时 + 全量 Decode 耗时 + 传输耗时”。
- **流式输出模式**：客户端发起请求后，服务端建立长响应通道。当 Prefill 完成并计算出第 1 个 Token 时，网络层立即将该片段封装并下发；后续每解码出一段增量数据，就立即向客户端推送一个分块（Chunk）。用户感知到的 TTFT 仅包含“Prefill 耗时 + 首个 Token 解码耗时 + 首包传输耗时”。

在标准对话场景中，通信拓扑具有明显的**不对称性**：用户在单轮对话中只发送一次输入参数（提示词、会话历史、采样温度等），之后主要是服务端单向、持续、分块地将文本下发给浏览器。

---

## SSE 工作原理：基于标准 HTTP 的事件流

Server-Sent Events（SSE）是 W3C 规范中定义的基于标准 HTTP 协议的单向事件流技术。它的运行方式是在一次标准 HTTP 握手后，服务端保持响应通道开启，按照规范的纯文本格式连续向客户端推送事件。

![SSE 报文结构与解析流程](/images/content-rebuild/cd6574219e99cb39/fig-2.png)

### 1. 响应头声明
在 HTTP/1.1 中，服务端通过以下响应头告知客户端和沿途代理服务器建立事件流：

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
Transfer-Encoding: chunked
```

- `Content-Type: text/event-stream`：声明响应内容为连续事件流，告知浏览器和代理不要作为普通静态文本整体缓存。
- `Cache-Control: no-cache`：防止浏览器本地以及中间代理缓存分块响应。
- `Connection: keep-alive`：保持底层 TCP 连接复用（注：在 HTTP/2 中该字段已被规范弃用，由数据流 Stream 原生支持）。

### 2. 报文格式规则
SSE 载荷必须是 UTF-8 编码的纯文本。数据由若干字段组成，字段名与内容用冒号分隔，每行以 `\n` 结尾。**每个独立事件必须以连续的两个换行符 `\n\n` 作为结束标志**。

常见规范字段包括：
- `data: <内容>`：事件的数据载荷。多行数据可以连续写多个 `data:` 行。
- `event: <名称>`：自定义事件类型。客户端可以根据事件名监听特定消息（如区分 `message`、`delta` 或 `done`）。
- `id: <标识符>`：事件序号。连接异常断开后，浏览器重连会自动在请求头 `Last-Event-ID` 中带上该值，用于服务端确定重传起点。
- `retry: <毫秒数>`：指定客户端断线后的重连等待间隔。
- `: <注释内容>`：以冒号开头的行为注释行，常用于发送网络心跳保活包。

一个符合 OpenAI 规范的典型 SSE 数据流片段如下：

```text
event: message
id: 1
data: {"choices":[{"delta":{"content":"你好"}}]}

event: message
id: 2
data: {"choices":[{"delta":{"content":"！有什么"}}]}

event: message
id: 3
data: {"choices":[{"delta":{"content":"我可以帮您的？"}}]}

data: [DONE]

```

### 3. 前端读取实现：原生 EventSource 与 Fetch API
浏览器虽然内置了 `EventSource` 接口，但在实际大模型应用开发中较少直接使用，原因在于：
1. `EventSource` 仅支持发送 `GET` 请求，无法通过请求体（Body）直接携带长上下文、历史消息列表和模型超参数；若拼在 URL 查询参数中容易超长或泄露。
2. `EventSource` 无法自定义设置请求头（如 `Authorization: Bearer <Token>`），通常只能依赖 Cookie。

因此，现代前端主流方案是使用原生的 `fetch` 结合 `ReadableStream` 读取响应体（或借助 `@microsoft/fetch-event-source` 库）：

```javascript
async function fetchLLMStream(prompt) {
  const response = await fetch('/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: prompt }],
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // 保留尚未闭合的不完整分块

    for (const block of lines) {
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          const parsed = JSON.parse(data);
          console.log(parsed.choices[0].delta.content);
        }
      }
    }
  }
}
```

---

## WebSocket 工作原理：全双工 TCP 帧通道

WebSocket（RFC 6455）是一种建立在 TCP 基础上的独立全双工应用层协议。它通过 HTTP 请求完成协议升级协商后，直接在底层 TCP 信道上双向独立收发数据帧。

![WebSocket 握手与双向帧传输](/images/content-rebuild/cd6574219e99cb39/fig-3.png)

### 1. 协议升级过程
WebSocket 连接必须先通过一次特殊的 HTTP GET 请求发起升级协商：

```http
GET /chat HTTP/1.1
Host: api.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
```

若服务端支持该协议，则返回状态码 `101 Switching Protocols`：

```http
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

握手完成后，HTTP 语义终止，连接转由 WebSocket 协议接管。数据被拆分为二进制帧（Frame）传输，每个帧头包含操作码（Opcode，区分文本帧、二进制帧、Ping/Pong 帧、关闭帧）与载荷长度。

### 2. 全双工特性的适用边界
WebSocket 的核心优势在于**客户端与服务端在同一信道上互不阻塞地双向通信**：
- **双向流式音频（如实时语音大模型）**：客户端需要持续向服务端发送麦克风采集的 PCM 语音分块，同时服务端持续向下推送合成的音频分块。这种对偶的双向流式传输天然契合 WebSocket。
- **频繁的中间双向干预**：在长文本生成或复杂 Agent 工具链执行期间，客户端需要高频发送暂停、打断、修改中间参数等双向控制信号。

但如果仅仅是“用户问一句，模型答一段”的文本生成场景，WebSocket 的全双工能力往往利用不充分，反而会引入有状态长连接的管理负担。

---

## SSE 与 WebSocket 核心维度对比

在架构选型时，需要权衡协议适配度、运维成本和网络组件兼容性。

![SSE 与 WebSocket 技术维度对比](/images/content-rebuild/cd6574219e99cb39/fig-4.png)

| 对比维度 | Server-Sent Events (SSE) | WebSocket |
| :--- | :--- | :--- |
| **传输层协议** | 标准 HTTP (HTTP/1.1, HTTP/2, HTTP/3) | 专有协议（通过 HTTP 101 升级后的 ws/wss） |
| **通信拓扑** | 单向（服务端下发为主，客户端单次请求触发） | 全双工（客户端与服务端可随时并发读写） |
| **数据编码** | UTF-8 纯文本 | 支持文本（UTF-8）与纯二进制（Blob、ArrayBuffer） |
| **网关与网络设施兼容** | 完全复用现有 HTTP 路由、API 网关、WAF 规则 | 需网关（如 Nginx、Traefik）显式配置连接升级与代理映射 |
| **认证与链路追踪** | 直接传递标准 HTTP Header（Authorization、Trace-ID） | 握手期仅限有限 Header，连接建立后需应用层自定义握手或消息鉴权 |
| **断线与保活机制** | EventSource 规范原生支持自动重连与 `Last-Event-ID`（大模型若改用 Fetch 流式读取则需前端自行实现重连与状态恢复） | 协议层内置 Ping/Pong 控制帧（浏览器 API 未暴露主动发送），断线重连需在应用层自行实现 |
| **浏览器连接数限制** | HTTP/1.1 下单域名通常限 6 个；**HTTP/2 多路复用下无此限制** | 浏览器端无单域名 6 连接限制，受服务端句柄与内存约束 |
| **后端扩展与伸缩** | 易于实现无状态扩容，与普通 REST API 部署体系一致 | 有状态连接，后端集群通常需要维护连接路由表或广播中间件 |

### 选型原则
1. **优先选 SSE**：大模型文本对话、知识库检索（RAG）、代码补全、分析报告流式生成。这些场景是典型的“单次输入 + 单向下行”，SSE 无需额外改造既有 HTTP 鉴权、限流和安全拦截体系。
2. **按需选 WebSocket**：端到端双向实时语音会话、多模态音视频混流交互、客户端高频双向控制（如白板多人协同与高频实时打断）。

---

## 生产环境避坑指南与网关配置

在生产环境中部署 SSE 时，常见的问题往往不是出在模型本身，而是沿途网关代理的缓冲机制把流式变成了整块输出。

### 1. 反向代理的响应缓冲陷阱

![Nginx 反向代理缓冲穿透机制对比](/images/content-rebuild/cd6574219e99cb39/fig-5.png)

绝大多数反向代理（如 Nginx、Envoy、Traefik）默认开启了**响应缓冲（Response Buffering）**。网关为了节省网络 I/O 资源，会尝试攒满 4KB 或 8KB 的数据包后再统一发送给客户端。

大模型吐字时，每个 Token 包装成的 JSON 分块通常只有几十字节。如果网关开启缓冲，这些增量分块就会被积压在反向代理的缓冲区内，直到缓冲区填满或响应结束才一次性推送，导致客户端的首字延迟退化为全量生成时间。

#### Nginx 配置修复
在反向代理的流式输出路由中，必须显式关闭缓冲与代理缓存，并延长读取超时：

```nginx
location /api/v1/chat/completions {
    proxy_pass http://llm_backend_cluster;

    # 1. 关闭响应缓冲，让数据块即时下发给客户端
    proxy_buffering off;

    # 2. 关闭代理层缓存
    proxy_cache off;

    # 3. HTTP/1.1 下保持后端长连接
    proxy_http_version 1.1;
    proxy_set_header Connection "";

    # 4. 延长读写超时，防止大模型长上下文 Prefill 或思考时间过长导致断连
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;

    # 传递客户端真实 IP
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

#### 应用层响应头规避：`X-Accel-Buffering`
如果后端无法直接调整 Nginx 全局配置，可在返回 SSE 响应时，由应用服务主动注入响应头：
```http
X-Accel-Buffering: no
```
Nginx 接收到该响应头后，会自动临时对当前连接关闭代理缓冲。

### 2. HTTP/1.1 浏览器连接数瓶颈与 HTTP/2

在 HTTP/1.1 规范下，浏览器对同一域名同时发起的并发 TCP 连接数通常存在硬性限制（主流浏览器一般为 6 个）。如果用户在多个标签页中同时发起流式会话，或者界面上同时开启了多个长轮询监控，未关闭的 SSE 会一直占用连接名额，导致页面后续的静态资源和普通 API 陷入 Pending 阻塞。

![HTTP/2 多路复用配合流式输出拓扑](/images/content-rebuild/cd6574219e99cb39/fig-6.png)

**应对方案**：线上环境建议全面开启 HTTP/2 或 HTTP/3。在 HTTP/2 中，浏览器与网关之间复用同一条 TCP 物理连接，所有请求与 SSE 流以独立的 Stream 帧并发传输。这从根本上解除了 6 个并发连接数的上限。*(注：HTTP/2 在传输层仍共享单条 TCP 连接，若发生网络丢包，在 TCP 重传完成前仍可能出现传输层等待；若需要彻底消除网络抖动下的队头等待，可进一步演进至基于 UDP/QUIC 的 HTTP/3。)*

### 3. NAT 超时与心跳保活
移动蜂窝网络（4G/5G）或某些内网路由器的 NAT 网关通常设有较短的空闲超时时间（例如 30 至 60 秒无网络包交互即剔除对应映射关系）。如果模型在 Prefill 复杂任务（如读取长达数万字的文档）或执行外部工具时静默超过超时阈值，连接就会被 NAT 路由器单方面阻断。

**解决方法**：服务端应在长时间未生成数据的间隔内，定时（例如每 15 秒）向下游注入一条 SSE 注释行作为心跳包：
```text
: ping

```
浏览器端根据 SSE 规范会自动忽略以冒号开头的注释行，既维持了 TCP 连接的数据活跃度，又不会干扰前端正常的 JSON 数据解析。

---

## 验证实验：本地验证流式输出与缓冲机制

你可以通过一段轻量 Node.js 原生代码，直观检验未缓冲流式推送的工作机制。

### 1. 编写测试服务
新建 `server.mjs`：

```javascript
import http from 'node:http';

const server = http.createServer((req, res) => {
  if (req.url === '/stream' && req.method === 'GET') {
    // 设置 SSE 必需响应头
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      'Access-Control-Allow-Origin': '*'
    });

    const sampleTokens = ['这是', '第一段', '流式', '推送', '的', '大模型', '模拟', '回复', '。'];
    let index = 0;

    // 模拟大模型自回归解码吐字节奏（每 200ms 输出一个词）
    const timer = setInterval(() => {
      if (index < sampleTokens.length) {
        const payload = JSON.stringify({ token: sampleTokens[index] });
        res.write(`event: delta\ndata: ${payload}\n\n`);
        index++;
      } else {
        res.write('event: done\ndata: [DONE]\n\n');
        clearInterval(timer);
        res.end();
      }
    }, 200);

    // 监听客户端主动中断连接（例如用户点击了“停止生成”按钮）
    req.on('close', () => {
      clearInterval(timer);
      console.log('客户端断开连接，及时终止模型推理定时器');
    });
  } else {
    res.writeHead(404).end();
  }
});

server.listen(3000, () => {
  console.log('流式测试服务已启动: http://localhost:3000/stream');
});
```

### 2. 运行与验证
在终端中启动服务：
```bash
node server.mjs
```

新开一个终端窗口，使用 cURL 发起请求并观察输出表现：
```bash
curl -N -H "Accept: text/event-stream" http://localhost:3000/stream
```

**检验要点**：
1. 参数 `-N`（`--no-buffer`）表示强制 cURL 禁用终端本地输出缓冲区。加上该参数后，终端会每隔 200 毫秒逐行打印出内容；若去掉 `-N`，在部分系统环境下 cURL 会积攒到一定字节数后一次性刷屏，这直观复现了网络缓冲对流式体验的影响。
2. 在输出完成前按下 `Ctrl + C` 中断请求，观察服务端终端是否打印 `客户端断开连接，及时终止模型推理定时器`。在大模型工程中，监听连接关闭事件来及时取消底层的模型推理解码，是防止算力浪费的关键做法。

---

## 总结与下一步

大模型应用的流式交互优先选择与现有 Web 体系深度兼容的协议：
- **单向文本交互选 SSE**：依靠轻量、纯文本、原生物理穿透与标准的 HTTP 生态，能够以较低的系统改造代价获得平滑的打字机流式体验。
- **双向高频交互选 WebSocket**：在实时语音会话、多方协同等需要持续双向通信的场景下发挥其全双工优势。

在打通了前后端的流式通信通道后，大模型系统工程下一步需要解决的是：如何结构化地组装、注入并管理提交给模型的提示词？在下一讲 [/llm-100/app-system/prompt-template-system.md](/llm-100/app-system/prompt-template-system.md) 中，我们将详细探讨 Prompt 模板系统的组件划分与参数动态渲染。
