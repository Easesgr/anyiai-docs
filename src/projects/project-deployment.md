---
title: 大模型服务如何上线：FastAPI 网关、SSE 流式通信与监控追踪
icon: mdi:cloud-outline
description: 讲解大模型应用从本地脚本到在线服务的生产封装，涵盖 FastAPI 异步网关、SSE 流式通信、令牌桶限流、长任务异步解耦与全链路监控。
---

在本地开发环境中，我们通常通过 Jupyter Notebook 或单脚本运行模型调用逻辑，例如在 [AI 知识库系统](/projects/ai-knowledge-base.md) 中测试向量检索，或在 [AI 自动化工作流](/projects/ai-workflow-automation.md) 中执行多步骤 Agent。但在面向多用户提供在线服务时，本地脚本无法直接运行在生产环境。生产服务必须解决网络中断重连、下游模型供应商的速率限制（Rate Limit）、长任务阻塞连接、实例无状态扩展以及延迟异常排查等工程问题。

本文以一个具体的**企业智能助手服务**作为贯穿实例。该服务同时包含两类典型请求：
1. **在线客服问答（短请求）**：要求低延迟，以逐字流式打字效果返回模型生成的回答。
2. **批量文档切分与向量索引（长任务）**：用户上传包含数十页内容的 PDF 文档，解析与向量化耗时可达数分钟，不能长时间挂起客户端 HTTP 连接。

我们将围绕这一服务，实现 API 网关规范封装、Server-Sent Events（SSE）流式推送、并发限流与下游 429 容错、异步状态机解耦、Docker 多阶段镜像打包以及全链路追踪打点。

![大模型生产网关与容器化监控架构总览](/images/content-rebuild/0d7ca9ba4c3f2942/cover.png)

---

### 1. 从本地脚本到在线服务的架构差异

大模型网关本身通常是**网络 I/O 密集型服务**：网关本身不执行矩阵计算，它的主要开销是等待下游推理服务（或外部模型 API）完成首字计算（Prefill）并逐步传输解码内容（Decode）。单次调用往往持续数秒甚至数十秒。

如果使用传统的同步阻塞框架（如传统的 WSGI 同步多进程），一个请求占用一个系统线程。当模型生成耗时 8 秒时，极低的并发就会耗尽工作线程，导致后续所有请求排队超时。

上线大模型网关需要解决四个核心工程问题：
1. **并发模型**：采用基于事件循环的异步 I/O（如 Python `asyncio` 与 ASGI 规范）。在等待外部模型吐字时，事件循环主动让出控制权，使单个工作进程能够维持数千个并发网络连接。
2. **连接超时与网络缓冲**：反向代理（如 Nginx、云负载均衡器）默认设置了连接超时限制（常见为 60 秒）。大模型生成如果采用一次性整段返回，前端会面临长时间白屏，极易因代理超时断连。服务需采用流式下发，尽早吐出首字并维持数据传输。
3. **下游接口限额与抖动**：外部大模型接口普遍限制调用频率（RPM）与每分钟 Token 数（TPM）。网关需具备前置流控与错误退避机制，防止偶发限流击垮业务。
4. **长短任务分离**：短时交互走流式长连接，长时分析走异步任务状态机，实例本身不存储会话与中间结果，保证横向伸缩能力。

![本地执行脚本与生产级 API 网关架构对比](/images/content-rebuild/0d7ca9ba4c3f2942/fig-1.png)

---

### 2. 基于 FastAPI 搭建统一网关：请求校验与统一错误拦截

FastAPI 原生支持异步协程，并依赖 Pydantic 自动完成参数类型校验。网关应参照 [OpenAI-Compatible API](/interview/model-deployment-interview/openai-compatible-api.md) 风格定义输入输出，并在接口外层拦截内部异常，避免将下游原始报错或敏感密钥泄露给调用端。

以下代码展示网关基础框架，包含身份认证与全局异常拦截：

```python
import time
import uuid
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Production Gateway", version="1.0.0")
security = HTTPBearer()

# 模拟有效网关密钥（生产环境应从鉴权中心或加密配置中读取）
VALID_API_KEYS = {"sk-gateway-live-prod-token-9981"}

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen-max")
    messages: List[ChatMessage] = Field(..., min_items=1)
    stream: bool = Field(default=False)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    token = credentials.credentials
    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.get("/health")
async def health_check():
    # 供容器编排与集群健康探针检测，避免误判实例离线
    return {"status": "ok"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 屏蔽内部堆栈细节，返回统一追溯 ID
    error_id = str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-Id", error_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal gateway processing error.",
                "type": "internal_server_error",
                "trace_id": trace_id
            }
        }
    )
```

<ReadMoreLock />

---

### 3. Server-Sent Events (SSE) 流式传输与代理缓冲配置

在交互式客服问答中，等待模型生成完整段落通常需要数秒。通过流式下发，首个 Token 产出后立即可在客户端渲染，显著降低用户的等待感知时间（关于延迟的量化分析见 [大模型推理中的首 Token 延迟和平均生成速度分别受什么影响？](/interview/model-deployment-interview/latency-throughput.md)）。

#### 为什么文本交互优先采用 SSE 而非 WebSocket？

1. **通信模式匹配**：常规对话场景是典型的“单次请求，持续接收下发文本”，无需客户端高频向服务端反向推送数据。SSE 基于单向 HTTP 协议，结构更精简。
2. **协议兼容性好**：SSE 是标准 HTTP 响应，无需进行 WebSocket 协议握手切换（Upgrade），能天然复用现有的 HTTP 鉴权中间件、反向代理与云端日志网关。
3. **客户端原生支持**：浏览器环境自带 `EventSource` 标准 API，具备网络抖动自动重连能力。

#### 避免 Nginx 代理缓冲截断

实现 SSE 时必须设置正确的头部字段：
- `Content-Type: text/event-stream`：告知客户端该响应为事件流数据。
- `Cache-Control: no-cache`：防止中间缓存服务器保留分块内容。
- `X-Accel-Buffering: no`：告知 Nginx 等代理层**关闭输出缓冲**。Nginx 默认会积攒 4KB 至 8KB 数据才一次性刷出，若不关闭缓冲，流式推送在客户端表现依然是整批卡顿。

在长连接推送过程中，若客户端主动断开连接（如用户在界面点击停止生成或关闭网页），网关应通过检测连接状态（如 `request.is_disconnected()`）及时中止生成循环，防止后台算力空转；但需注意，下游模型服务商对已实际完成 Prefill 与部分 Decode 产出的 Token 通常仍会全额计费，中断流式只能避免后续未生成 Token 的额外开销，无法撤销已发生的计费。

```python
import json
import asyncio
from fastapi.responses import StreamingResponse

async def mock_llm_stream_generator():
    """模拟下游大模型推理引擎的逐 Token 异步产出"""
    tokens = ["企业", "智能", "助手", "已上线", "，", "正通过", "SSE", "协议", "提供服务", "。"]
    for token in tokens:
        await asyncio.sleep(0.08)  # 模拟两字之间 80ms 的推理与网络传输间隔
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    
    # 按协议约定发送结束标志
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    if req.stream:
        return StreamingResponse(
            mock_llm_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 通知上游 Nginx 关闭响应缓冲
            }
        )
    
    # 非流式分支：模拟一次性返回
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "企业智能助手已上线。"},
            "finish_reason": "stop"
        }]
    }
```

![SSE 流式通信生命周期与代理缓冲穿透](/images/content-rebuild/0d7ca9ba4c3f2942/fig-2.png)

---

### 4. 令牌桶限流与下游 429 容错退避

生产大模型服务必须兼顾两层限流控制：
1. **网关入口主动流控**：防止突发请求把本地系统或下游接口瞬间打满。
2. **调用下游的被动容错**：当下游接口返回 HTTP 429（Too Many Requests）时，实施退避重试，避免流量雪崩。

#### 令牌桶（Token Bucket）主动限流器

令牌桶以固定的速率 $r$（个/秒）向容量为 $C$ 的桶中投放令牌。请求到达时需索取令牌方可执行：若桶内有余量则直接放行；若不足则予以拒绝。该算法既能平滑平均速率，又允许一定程度的瞬时突发流量：

```python
import time
import asyncio

class TokenBucketLimiter:
    def __init__(self, rate: float, capacity: float):
        """
        :param rate: 每秒填充的令牌数量 (如 10.0 表示 600 RPM)
        :param capacity: 令牌桶最大容量（允许的最大突发数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens_needed: float = 1.0) -> bool:
        async with self.lock:
            now = time.monotonic()
            # 计算上一请求至今应补充的令牌量
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False

# 实例化网关限流器：平摊 5 请求/秒，最大突发容纳 10 个
gateway_limiter = TokenBucketLimiter(rate=5.0, capacity=10.0)

async def check_rate_limit():
    """网关入口依赖：当瞬时突发超出令牌桶容量时直接返回 HTTP 429"""
    if not await gateway_limiter.acquire(1.0):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too Many Requests: gateway rate limit exceeded."
        )
```

#### 带抖动的指数退避重试（Exponential Backoff with Jitter）

当下游模型接口返回 429 时，若所有并发客户端在固定间隔后同时发起重试，会瞬间在下游形成新的流量峰值。标准工程做法是引入带随机扰动的指数退避：

$$t = \min(t_{\max},\; 2^{\text{attempt}} + \text{random\_jitter})$$

```python
import random

async def call_downstream_llm_with_retry(payload: dict, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            # 伪代码：实际调用下游模型 SDK 或 httpx 客户端
            # response = await http_client.post("https://api.example.com/v1/...", json=payload)
            # if response.status_code == 429: raise DownstreamRateLimitException()
            return {"status": "success", "payload": payload}
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail="Downstream model capacity exceeded.")
            # 指数递增并加入 0.1s ~ 0.5s 随机浮动，总等待时间不超过上限
            backoff = min(8.0, (2 ** attempt) + random.uniform(0.1, 0.5))
            await asyncio.sleep(backoff)
```

有关系统吞吐量与性能考核的完整指标定义，可参考 [大模型部署需要关注哪些核心指标？](/interview/model-deployment-interview/core-metrics.md)。

![令牌桶限流与下游 429 退避重试机制](/images/content-rebuild/0d7ca9ba4c3f2942/fig-3.png)

---

### 5. 长耗时文档切分任务的状态机设计

在贯穿例子中，处理知识库长文档（解析 PDF、切片、生成 Embedding 向量并写入数据库）通常需要耗费数分钟。保持 HTTP 连接等待会面临网关超时断连的风险。

针对长耗时作业，工业界通行的做法是**异步解耦状态机**：
1. **接收并响应（HTTP 202）**：客户端提交任务，服务端生成唯一的 `task_id`，初始化状态为 `pending`，立即返回 **HTTP 202 Accepted** 及轮询地址。
2. **后台异步推进**：后台 Worker 读取任务并流转状态（`pending` $\to$ `processing` $\to$ `completed` / `failed`）。
3. **状态轮询或通知**：客户端按固定间隔轮询状态接口，获取处理进度与最终结果。

*注意：FastAPI 内置的 `BackgroundTasks` 运行在当前服务进程内存中。如果生产环境需要处理高可靠长任务，需使用 Redis + Celery 等外部任务队列，防止服务重启导致任务中断丢失。*

```python
from fastapi import BackgroundTasks
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 演示字典（生产环境务必持久化至 Redis 或关系型数据库）
task_database = {}

async def background_document_pipeline(task_id: str, document_url: str):
    try:
        task_database[task_id]["status"] = TaskStatus.PROCESSING
        task_database[task_id]["progress"] = 0.2
        await asyncio.sleep(2.0)  # 模拟文本抽取与分块

        task_database[task_id]["progress"] = 0.7
        await asyncio.sleep(3.0)  # 模拟向量化并写入向量库

        task_database[task_id]["status"] = TaskStatus.COMPLETED
        task_database[task_id]["progress"] = 1.0
        task_database[task_id]["result"] = {"chunks_indexed": 86, "doc_url": document_url}
    except Exception as exc:
        task_database[task_id]["status"] = TaskStatus.FAILED
        task_database[task_id]["error"] = str(exc)

class DocumentTaskRequest(BaseModel):
    document_url: str = Field(..., min_length=1, description="待切分文档下载或访问地址")

@app.post("/v1/tasks/documents", status_code=status.HTTP_202_ACCEPTED)
async def submit_document_task(req: DocumentTaskRequest, bg: BackgroundTasks):
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    task_database[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "progress": 0.0,
        "created_at": int(time.time())
    }
    bg.add_task(background_document_pipeline, task_id, req.document_url)
    return {
        "task_id": task_id,
        "status": "pending",
        "poll_url": f"/v1/tasks/{task_id}"
    }

@app.get("/v1/tasks/{task_id}")
async def query_task_status(task_id: str):
    if task_id not in task_database:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_database[task_id]
```

![长耗时任务的异步解耦状态机流转](/images/content-rebuild/0d7ca9ba4c3f2942/fig-4.png)

---

### 6. Docker 多阶段构建与最小化运行镜像

在部署 Python AI 服务时，若将 C/C++ 编译链（如 `gcc`、`g++`）直接打包在运行镜像中，镜像体积通常会突破 1.5GB。这不仅延长了容器调度与集群冷启动拉取时间，而且编译工具链留存在镜像中还扩大了潜在的安全漏洞攻击面。

通过 **Docker 多阶段构建（Multi-Stage Builds）**，我们将构建阶段与最终运行环境拆解：第一阶段编译依赖并生成二进制 Wheel 包，第二阶段仅复制预编译好的文件，并将运行权限移交给非 root 用户：

```dockerfile
# ==========================================
# 阶段 1: 依赖编译环境 (Builder)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# 安装编译所需的依赖工具，随后清理 apt 缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 将依赖统一安装至 /install 目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# 阶段 2: 生产运行时镜像 (Runtime)
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# 创建非 root 用户提高容器运行安全性
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 仅从 Builder 阶段复制已编译好的 Python 依赖库
COPY --from=builder /install /usr/local

# 复制业务源代码
COPY --chown=appuser:appuser . /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

# 配置健康检查探针，供 Kubernetes 或容器平台监测容器活性
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# 生产环境使用 uvicorn 启动 2 个异步 Worker
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

采用此方式，产出镜像中不包含任何编译器及构建中间产物，体积一般可控制在 300MB 以内，大幅优化了镜像分发效率。

![Docker 多阶段构建与依赖裁剪对比](/images/content-rebuild/0d7ca9ba4c3f2942/fig-5.png)

---

### 7. 全链路 Trace 监控与 Token 消耗统计

在传统 Web 业务中，接口慢通常排查数据库慢查询。但在大模型服务中，一个端到端耗时达 4 秒的请求可能由以下环节叠加产生：
1. 语义路由器判定耗时；
2. 向量数据库检索相关上下文耗时；
3. 下游模型服务商首字排队延迟（Prefill 排队）；
4. 自回归逐字生成（Decode）耗时。

缺少分布式追踪链路（Trace），无法精准定位瓶颈阶段。关于日志埋点与成本审计的深层探讨，可阅读 [部署大模型时如何做日志监控、调用统计和成本控制？](/interview/model-deployment-interview/monitoring-cost.md)。

#### 基于 OpenTelemetry 的打点集成

我们利用 OpenTelemetry 在关键路径上创建嵌套的 `Span`，并在 Span 属性中记录检索条数、模型推理的首 Token 延迟（TTFT）以及 Token 消耗量：

```python
import os
import time
import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# 1. 初始化 OpenTelemetry 追踪配置
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("ai-gateway-service")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# 如果生产环境接入了 LangSmith，可通过设置环境变量激活无侵入自动追踪
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."

async def execute_rag_pipeline_with_trace(query: str, trace_id: str):
    """带有 OpenTelemetry 细粒度 Span 打点的执行流程"""
    with tracer.start_as_current_span("rag_request_pipeline") as root_span:
        root_span.set_attribute("request.trace_id", trace_id)
        root_span.set_attribute("user.query", query)
        
        # 子阶段 1: 向量数据库检索
        with tracer.start_as_current_span("vector_search") as search_span:
            start_search = time.time()
            await asyncio.sleep(0.05)  # 模拟向量相似度搜索
            chunks = ["规则片段 A", "规则片段 B"]
            search_span.set_attribute("search.results_count", len(chunks))
            search_span.set_attribute("search.duration_ms", (time.time() - start_search) * 1000)
        
        # 子阶段 2: 大模型交互与 Token 审计
        with tracer.start_as_current_span("llm_generation") as llm_span:
            start_llm = time.time()
            await asyncio.sleep(0.3)   # 模拟模型生成
            prompt_tokens = 320
            completion_tokens = 95
            
            # 记录 Token 消耗指标，供计费中心统计
            llm_span.set_attribute("llm.model", "qwen-max")
            llm_span.set_attribute("llm.tokens.prompt", prompt_tokens)
            llm_span.set_attribute("llm.tokens.completion", completion_tokens)
            llm_span.set_attribute("llm.tokens.total", prompt_tokens + completion_tokens)
            llm_span.set_attribute("llm.ttft_ms", 75.0)  # 首字延迟
            
        return {"answer": "依据知识库为您解答。", "total_tokens": prompt_tokens + completion_tokens}
```

![OpenTelemetry 与 LangSmith 全链路分布式追踪调用树](/images/content-rebuild/0d7ca9ba4c3f2942/fig-6.png)

---

### 8. 生产交付与验收清单

服务上线前，需逐项验证以下测试用例：

| 验证环节 | 输入条件与操作 | 预期状态流转与输出 | 判定标准 |
| :--- | :--- | :--- | :--- |
| **SSE 流式通信** | 发起 `stream: true` 的 POST 请求至 `/v1/chat/completions` | HTTP 状态码 200，响应头包含 `X-Accel-Buffering: no` 与 `Content-Type: text/event-stream` | 数据逐块以 `data: {...}` 格式下发，末尾有 `data: [DONE]`，无整批缓冲停顿 |
| **网关令牌桶限流** | 并发突发发送 20 个对话请求 | 桶内容量内的前 10 个请求进入执行，超出容量的后续请求立即被拦截 | 拦截请求返回明确的 HTTP 429 错误响应，系统无崩溃且工作进程健康 |
| **长耗时任务状态机** | POST 调用 `/v1/tasks/documents` 提交切分任务 | 立即返回 HTTP 202 及任务 ID；轮询 `/v1/tasks/{id}`，状态由 `pending` $\to$ `processing` $\to$ `completed` | 状态字段推进正常，最终结果字段包含完成切分与向量化的数量统计 |
| **容器化与安全检查** | 执行 `docker run` 并检测运行环境 | 镜像体积小于 350MB；容器内当前执行用户 UID 为 1000（非 root 用户） | 调用 `/health` 探针返回 200，Docker CLI 标记容器状态为 `healthy` |
| **全链路分布式追踪** | 请求时在 HTTP 头部传入 `X-Trace-Id` | APM 监控平台生成以该 ID 为根节点的完整 Span 瀑布树 | 能够清晰查看向量检索耗时、模型首字延迟（TTFT）及消耗的 Token 总量 |

通过掌握流式通信、速率控制、异步任务编排与全链路追踪打点，开发者便完成了将离线模型和检索流程交付为可靠在线服务的关键跨越。更多实战环节请查阅 [AI 项目实战总览](/projects/README.md)。
