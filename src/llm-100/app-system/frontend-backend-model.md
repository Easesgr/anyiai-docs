---
title: 前端、后端与大模型服务如何协同？
icon: mdi:book-open-page-variant
description: 梳理前端、业务后端与大模型服务的三方职责，深入流式传输、中断级联、合规过滤与错误契约的工程实现。
---

![前端后端模型协同架构封面图](/images/content-rebuild/9982ddec76aa1da4/cover.png)

在传统 Web 应用中，客户端发送一个 HTTP 请求，后端查询数据库或处理业务逻辑，通常在几十到几百毫秒内返回一个完整的 JSON 响应。但在接入大语言模型（LLM）后，这个经典链路遇到了三个实际变化：

1. **生成延迟长**：一次完整的文本回答通常需要数秒甚至数十秒，同步等待容易导致连接超时，前端界面也会陷入长时间无响应。
2. **以 Token 为单位增量输出**：模型采用自回归方式依次生成 Token。服务端每生成一部分内容，就可以通过流式通道推给客户端展示。
3. **算力成本随生成过程持续发生**：模型推理按生成的 Token 数量消耗 GPU 算力并计费。如果用户中途关闭了页面或点击了“停止生成”，若系统没有建立连带的中断机制，后方的模型服务就会继续计算，浪费算力和费用。

为了处理这三个变化，前端界面、业务后端和大模型服务之间需要建立一套明确的分工与协作约定。

---

### 一个贯穿示例：从提问到中断

以一个典型的企业知识库问答场景为例：

1. 用户在前端输入“请帮我总结今年第三季度的销售数据”，点击发送。
2. 前端不会直接访问 OpenAI 或自建模型网关，而是将请求发送给业务后端。
3. 后端验证用户登录状态与剩余额度，在私有数据库和向量库中检索第三季度的销售报表，拼装成包含系统设定、历史对话和检索上下文的完整提示词，再发给模型服务。
4. 模型服务开始逐字推理，并通过流式接口向后端吐出数据块（Chunk）。
5. 后端边接收边做敏感词过滤，将安全的数据块包装为标准事件推向前端。
6. 前端逐字渲染 Markdown 文本。若用户在生成到一半时发现提问有误，点击“停止生成”，前端必须向后端发出中断信号，后端随之中断对模型服务的调用，阻止 GPU 继续空转。

围绕这个流程，我们依次剖析各环节的实现机制与工程细节。

---

### 三方职责划分与非对称数据流

前端、业务后端和模型服务处于不同的网络与安全环境中，分工明确：

![端后端模型三方职责与非对称数据流](/images/content-rebuild/9982ddec76aa1da4/fig-1.png)

*   **前端（客户端）**：负责交互状态管理、流式文本渲染（Markdown 解析、代码高亮、光标跟随）、收集用户操作（输入、重发、停止）并做基础的字符长度校验。
*   **业务后端**：作为业务边界与中枢，负责用户身份认证、配额检查、上下文装配（读取历史会话与 RAG 检索结果）、流式中继、敏感内容过滤、Token 用量记录与数据持久化。
*   **模型服务（如 vLLM、TGI 或第三方 API）**：专注 GPU 资源调度、Continuous Batching 批处理调度和模型推理，持续输出原始 Token 片段。

这种分工导致了**非对称数据流**的特征：
*   **前端到后端**：上行数据量小，通常只包含用户当前输入的单条文本、会话 ID 和少量选项。
*   **后端到模型服务**：后端从数据库中读取历史多轮对话（参见 [/llm-100/app-system/conversation-history.md](/llm-100/app-system/conversation-history.md)），并注入检索到的背景文档，请求体膨胀至数千甚至上万 Token。
*   **模型服务到后端**：下行是高频、小体积的原始流式数据块，夹带模型特有的控制字段（如结束原因 `finish_reason`、工具调用分片等）。
*   **后端到前端**：后端清洗并格式化数据后，通过 Server-Sent Events（SSE）将事件稳定推送给前端渲染。

---

### 为什么前端不能直连模型服务？

大模型服务商通常提供标准的 HTTP 接口，让前端直接调用模型似乎可以减少一次后端转发。但在生产系统中，这种方案存在严重缺陷：

![客户端鉴权与凭证安全隔离机制](/images/content-rebuild/9982ddec76aa1da4/fig-2.png)

1. **API Key 泄露风险**：客户端处于开放环境，保存在前端代码、本地存储或请求头中的主 API Key 可以通过浏览器控制台、反编译或抓包轻易提取。一旦泄露，攻击者可以调用该 Key 耗尽额度。
2. **业务逻辑与审计绕脱**：前端直连第三方模型意味着用户的提问内容和回答完全绕过了自建服务，业务系统无法在数据库中留存聊天记录，无法统计真实 Token 消耗，也无法实施合规审计。
3. **私有知识库无法安全注入**：在 RAG 场景中，提示词需要结合私有文档。若前端直连模型，前端必须先从业务后端拉取企业内部文档，再拼装后传给模型。这不仅增加了额外的网络往返，还将未脱敏的原始文档暴露到了前端环境中。
4. **用户配额与风控失效**：模型服务商仅做针对 Key 级别的访问频率限制（Rate Limit），无法识别你的业务用户身份（例如普通用户每天限额 30 次、付费用户享有更高并发）。

> **注**：在部分端侧语音对讲等对网络往返延迟要求严苛的场景下，某些厂商支持通过后端申请短时有效（如 60 秒）且权限受限的临时 Token（Ephemeral Token）供前端直连使用。但常规的文本交互与企业知识库系统，统一采用“前端持用户登录凭证访问业务后端，业务后端持 Master Key 访问模型”的标准隔离方案。关于网关的具体设计可参考 [/architecture/model-gateway.md](/architecture/model-gateway.md)。

---

### 状态管理与主动中断链路

在流式交互中，前端不能使用传统的简单加载布尔值，需要维护一个状态流转逻辑：

```text
[IDLE 空闲]
   │  用户提交输入 (客户端做空值判断与防抖)
   ▼
[PENDING 等待响应]
   │  收到第一个 SSE 数据包 (记录首字时间 TTFT)
   ▼
[STREAMING 流式渲染] ◄──┐ (增量追加文本、渲染 Markdown)
   │                    │
   ├── 用户点击"停止" ───┼──> [ABORTED 主动终止]
   ├── 传输或业务报错 ───> [ERROR 异常阻断]
   └── 收到结束标记 [DONE] ──> [FINISHED 传输完成]
```

在用户点击发送时，界面执行乐观更新：立即将用户输入上屏，清空输入框，并将“发送”按钮切换为“停止生成”，进入 `PENDING` 状态。

#### 停止生成信号的三级级联传递

用户在前端点击“停止”时，如果仅仅停止前端界面的文本渲染，后端的流式循环和模型 GPU 推理仍会继续进行，直到整段文本生成完毕，这会导致不必要的算力扣费。

正确的中断流程需要建立三级信号传递：

![主动中断信号的三级传递与断链对比](/images/content-rebuild/9982ddec76aa1da4/fig-3.png)

1. **端侧发出中断**：前端调用浏览器的 `AbortController.abort()` 方法。浏览器底层网络层随之中断该 HTTP 请求，向服务端发送 TCP 终止信号。
2. **后端捕获连接断开**：业务后端的 Web 框架（如 Node.js 的 `req.on('close')`，Go 的 `r.Context().Done()`，Python FastAPI 的 `request.is_disconnected()`）捕获到客户端已断开事件。
3. **后端中断上游模型请求**：后端调用模型服务时，必须将自身的取消上下文（Cancel Token / AbortSignal）绑定到发往模型的 HTTP 请求上。一旦检测到客户端离线，后端立即中止与模型服务之间的请求连接。
4. **模型引擎停止计算**：以 vLLM、TGI 等现代推理服务为例，当底层 HTTP 连接断开后，调度器会将对应的请求从连续批处理队列中移除，终止后续解码计算，从而停止显存占用和算力消耗。

如果在开发时忽略了第 3 步，后端的异步逻辑就会继续把模型推过来的 Token 读取完毕，即使前端界面早就不再展示，计费系统仍然会记录完整的 Token 消耗。

---

### 后端中继的三个处理任务

后端在流式通信中承担着多项业务处理工作：

#### 1. 网络背压与缓冲调节
大模型推理通常以每秒数十个 Token 的速度推流，而移动端在弱网环境下接收速度可能受限，复杂页面的 Markdown/LaTeX 解析也可能出现短暂渲染阻塞。如果后端的发送速率远快于前端的接收速率，操作系统 TCP 发送缓冲区填满后，中间件必须具备背压（Backpressure）感知机制，暂停向上游模型拉取数据，防止内存无序积压。

#### 2. 滑动窗口流式安全过滤
内容合规检测如果等到全文生成完再检查，会完全失去流式首字秒级反馈的体验优势；但如果逐字检测，跨 Chunk 的敏感词又容易漏检（例如模型在 Chunk 1 返回“敏”，Chunk 2 返回“感词”）。

![滑动窗口敏感内容流式合规过滤机制](/images/content-rebuild/9982ddec76aa1da4/fig-4.png)

常见的解决方式是采用**滑动窗口缓冲**：
*   后端中继设置一个字符缓冲窗口，窗口长度需大于等于词库中最长敏感词的字符数（例如 8 个字符）。
*   新到达的 Chunk 暂存在缓冲区内，当缓冲区内的字符通过 Trie 树或规则库匹配确认安全后，将左侧滑出的安全字符推流给前端。
*   若窗口内命中违规词，后端立即阻断上游模型连接，向客户端发送合规拦截事件，并清空被污染的未确认片段。

#### 3. 元数据剥离与日志落盘
模型服务返回的流不仅包含自然语言，还包含系统字段（如 `finish_reason`、`prompt_tokens`、`completion_tokens`）。正常传输结束时，后端中继从最后一帧提取用量数据存入审计日志；若中途中断或超时，由于尾帧无法送达，后端还需结合本地已传输的 Token 计数核算实际消耗，不能无条件全额释放或漏记已发生的计算配额。清洗后仅将业务数据推给前端，控制字段保留在后端。

---

### 全链路统一错误契约

调用模型可能在不同阶段出现不同类型的故障。系统需要将底层的技术细节转化为前端可明确处理的业务错误码：

![全链路异常映射与统一契约设计](/images/content-rebuild/9982ddec76aa1da4/fig-5.png)

| 故障来源 | 底层状态或错误 | 归一化业务码 | 前端交互策略 |
| :--- | :--- | :--- | :--- |
| **模型服务** | `429 Too Many Requests` (速率或并发超限) | `MODEL_RATE_LIMIT` | 解析响应中的 `retry_after` 秒数，展示倒计时，避免用户立即重复狂点。 |
| **模型服务** | `context_length_exceeded` (超出上下文上限) | `CONTEXT_OVERFLOW` | 提示会话过长，引导用户开启新会话或由后端裁剪旧历史（参见 [/llm-100/app-system/context-budget.md](/llm-100/app-system/context-budget.md)）。 |
| **内容安全引擎** | 命中敏感策略拦截 | `CONTENT_VIOLATION` | 停止流式输出，移除受影响的部分，给出合规提示，保持输入框可用。 |
| **业务系统** | 用户账户余额或免费额度耗尽 | `INSUFFICIENT_QUOTA` | 弹出额度不足提示，保留用户已编辑的内容，避免重复输入。 |
| **网络层** | 网关 504 超时（模型服务响应过慢） | `UPSTREAM_TIMEOUT` | 自动退避重试一次；若仍失败，展示重试按钮。 |

**流式协议中的错误表达差异**：
*   **连接建立前出错**（如鉴权失败、配额不足）：此时尚未发送 HTTP 200，直接返回常规 HTTP 状态码（如 401、403、429）及 JSON 错误体。
*   **流式传输中途出错**（如生成到第 5 秒模型报 500 或合规拦截）：此时 HTTP 200 响应头已发出，协议已切换至 `text/event-stream`，不能再修改 HTTP 状态码，必须通过标准 SSE 错误事件下发异常通知：

```http
event: error
data: {"code": "MODEL_RATE_LIMIT", "message": "服务繁忙，请稍后重试", "retry_after": 5}
```

---

### 可验证实验：端到端中断与流式中继

下面给出一个基于 Node.js 后端与原生 JavaScript 的最小可运行实验代码，验证中断信号的级联传递机制。

![中断实验验证与控制台抓包时序](/images/content-rebuild/9982ddec76aa1da4/fig-6.png)

#### 1. 前端实现：绑定 AbortController

```javascript
// client.js
let currentAbortController = null;

async function sendPrompt(userText) {
  // 1. 若前一个请求未完成，先触发中断
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();
  const { signal } = currentAbortController;

  try {
    console.log("[前端] 发起流式请求...");
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: userText }),
      signal // 将中断信号与 fetch 绑定
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(`[${errData.code}] ${errData.message}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      console.log("[前端收到分片]:", chunk);
    }
    console.log("[前端] 传输正常结束");
  } catch (err) {
    if (err.name === "AbortError") {
      console.warn("[前端] 请求已主动中断");
    } else {
      console.error("[前端异常]:", err.message);
    }
  } finally {
    currentAbortController = null;
  }
}

function stopGeneration() {
  if (currentAbortController) {
    currentAbortController.abort();
  }
}
```

#### 2. 后端实现：级联传递中断信号

```javascript
// server.js
const http = require("http");

const server = http.createServer(async (req, res) => {
  if (req.method === "POST" && req.url === "/api/chat") {
    const upstreamAbortController = new AbortController();

    // 核心：监听客户端的主动连接断开事件
    req.on("close", () => {
      if (!res.writableEnded) {
        console.log("[后端] 监听到客户端断开连接，向模型下发中止信号");
        upstreamAbortController.abort(); // 中止调用上游模型
      }
    });

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive"
    });

    try {
      // 模拟向上游模型发起生成请求
      await simulateModelInference(upstreamAbortController.signal, (token) => {
        res.write(`data: ${JSON.stringify({ text: token })}\n\n`);
      });
      res.write("data: [DONE]\n\n");
      res.end();
      console.log("[后端] 模型生成完毕并已推送到客户端");
    } catch (err) {
      if (err.name === "AbortError") {
        console.log("[后端] 上游模型调用已中止，停止后续消耗");
      } else {
        console.error("[后端异常]:", err.message);
      }
      res.end();
    }
  }
});

// 模拟模型每 200ms 吐出一个 Token
async function simulateModelInference(signal, onToken) {
  const tokens = ["大", "模", "型", "应", "用", "架", "构", "协", "同", "机", "制"];
  for (let i = 0; i < tokens.length; i++) {
    if (signal.aborted) {
      const abortErr = new Error("Model inference aborted");
      abortErr.name = "AbortError";
      throw abortErr;
    }
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, 200);
      signal.addEventListener("abort", () => {
        clearTimeout(timer);
        const e = new Error("AbortError");
        e.name = "AbortError";
        reject(e);
      });
    });
    onToken(tokens[i]);
  }
}

server.listen(3000, () => console.log("Server running on port 3000"));
```

#### 3. 验证步骤

1. 启动 `node server.js`。
2. 前端发起调用 `sendPrompt("测试")`，终端开始输出 Token。
3. 在第 3 个 Token 输出时，前端调用 `stopGeneration()`。
4. **预期表现**：
   * 前端控制台输出 `[前端] 请求已主动中断`。
   * 后端打印 `[后端] 监听到客户端断开连接，向模型下发中止信号` 与 `[后端] 上游模型调用已中止`。
   * 后端不再输出第 4 个及后续 Token，证明上游循环已退出。

---

### 总结与练习

大模型工程协同的核心原则是**明确信任边界与统一生命周期**：前端管理视图渲染与中断动作；业务后端负责密钥安全、私有数据组装、合规过滤与用量审计；模型服务专注推理计算。只有三端通过中断信号级联和错误契约协同，才能保证生成过程既高效又可控。

关于流式通信协议的技术选型细节，可参考 [/llm-100/app-system/sse-vs-websocket.md](/llm-100/app-system/sse-vs-websocket.md)；在下一讲 [/llm-100/app-system/model-api-request-lifecycle.md](/llm-100/app-system/model-api-request-lifecycle.md) 中，我们将进一步拆解单个模型请求在网络抖动下的指数退避与重试策略。

#### 课后检验

1. **中断验证**：修改实验代码，注释掉后端的 `upstreamAbortController.abort()` 一行。再次触发前端停止，观察后端日志：此时 Token 循环是否还会继续跑完？这一现象在按 Token 计费的商用 API 下会产生什么后果？
2. **跨块敏感词**：假设违规词是“违禁词”，模型分两次吐出（第一个 Chunk 吐出“违”，第二个 Chunk 吐出“禁词”）。设计一段伪代码，说明滑动窗口如何避免“违”字过早发送给前端。
