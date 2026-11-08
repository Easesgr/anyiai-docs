---
title: 多轮对话历史应该怎么管理？存储模型与状态设计
icon: mdi:book-open-page-variant
description: 详解多轮对话从无状态 API 到有状态业务的存储设计，讲解父指针树数据建模、分支切换、流式写入缓冲与防并发冲突。
---

调用大语言模型的对话接口时，模型服务本身是无状态的。每次发起请求，后端都必须将之前的历史消息作为列表一次性传给模型，模型才能理解前后的上下文。模型在处理完当前请求、断开连接后，并不会在服务端为你保留对话上下文。

但在实际产品中，用户的体验必须是有状态的：用户希望随时翻看几周前的聊天记录、切换不同的会话、在某一轮回答不满意时点击“重新生成”，或者返回去修改某一个问题并展开新的讨论分支。

本讲承接上一讲 [/llm-100/app-system/prompt-template-system.md](/llm-100/app-system/prompt-template-system.md) 中关于提示词模板的动态组装逻辑，向下深入数据存储层，解决会话（Session）与消息（Message）的数据建模、流式写入与缓存协作、分支版本维护以及并发写入控制。至于如何将过长的历史记录进行压缩或滑动裁剪以适应模型上下文窗口，我们将在 [/llm-100/app-system/context-budget.md](/llm-100/app-system/context-budget.md) 中专门讨论。

![多轮对话系统在无状态模型与有状态存储间的状态桥接架构](/images/content-rebuild/96287fc42c559080/cover.png)

上图展示了多轮对话系统的协作关系：应用服务端充当桥梁，向上在数据库中维护有序、可分叉的消息记录，向下将选定的上下文分支整理成模型所需的扁平消息列表。

### 1. 为什么不能只用简单的追加列表？

最直观的想法，是把一个会话里的对话历史当作一条数组，用户发一句追加一条，模型回一句追加一条。但在实际业务中，这种扁平列表会迅速遇到以下四个问题：

1. **重新生成与分支修改无法表达**：如果用户对第 2 轮的回答不满意点击了“重新生成”，系统就得到了第 2 轮的第二个回答。如果直接用新回答覆盖旧回答，用户就无法再查看之前的回复；如果直接把新回答追加到末尾，下次把列表发给模型时，模型会看到同一问下面有两个连着的回答，上下文逻辑就会混乱。
2. **流式生成的半途状态**：按照 [/llm-100/app-system/sse-vs-websocket.md](/llm-100/app-system/sse-vs-websocket.md) 中介绍的流式传输机制，模型的回答是一个字一个字推过来的。如果网络中途断开，或者触发了安全风控拦截，这条消息只生成了一半。存储层必须明确区分这条消息是“正在生成中”、“已完成”还是“已失败”。
3. **工具调用的顺序依赖**：在带有函数调用的交互中，Assistant 角色如果输出了 `tool_calls`，紧随其后的必须是角色为 `tool` 的执行结果，且每条结果都要携带对应的 `tool_call_id`。如果存储顺序打乱或提取错位，模型 API 会直接报错拒绝请求。
4. **并发双击与重复请求**：用户在网络较卡时连续点击发送，或者同时在手机和电脑端提问，如果后端没有并发校验，两条提问会交错写入，导致历史链路上下文错乱。

要解决分支和因果依赖，最通用且易于维护的做法，是将消息建模为带**父节点指针（parent_id）**的树形结构。

### 2. 数据模型设计：Session 与 Message 表

在关系型数据库（如 PostgreSQL 或 MySQL）中，多轮对话通常拆分为两张核心表：会话表（`chat_session`）和消息表（`chat_message`）。

![会话表与消息表的实体关系设计及父指针树形连接](/images/content-rebuild/96287fc42c559080/fig-1.png)

上图展示了两张表的关系：`chat_session` 记录会话维度的全局信息和当前激活的叶子节点，而 `chat_message` 通过 `parent_id` 向上指向上一个消息节点，支持同一会话内长出多个分支。

#### 2.1 会话表（chat_session）

```sql
CREATE TABLE chat_session (
    id VARCHAR(64) PRIMARY KEY,              -- 会话唯一 ID
    user_id VARCHAR(64) NOT NULL,          -- 所属用户或租户 ID
    title VARCHAR(255) NOT NULL,           -- 会话标题
    model_config_snapshot JSONB NOT NULL,  -- 创建时的模型参数配置（模型名、Temperature、System Prompt 等）
    current_leaf_message_id VARCHAR(64),   -- 当前会话正在浏览或激活的最新叶子消息 ID
    is_deleted BOOLEAN DEFAULT FALSE,      -- 会话级软删除标记
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_session_user ON chat_session(user_id, updated_at DESC);
```

这里的 `model_config_snapshot` 记录了会话创建时的配置快照。当后续调整了全局提示词模板或默认模型时，旧会话依然能查阅到当时的配置，便于排查前后回答差异。

#### 2.2 消息表（chat_message）

```sql
CREATE TABLE chat_message (
    id VARCHAR(64) PRIMARY KEY,              -- 消息唯一 ID
    session_id VARCHAR(64) NOT NULL,       -- 归属的会话 ID
    parent_id VARCHAR(64),                  -- 父节点消息 ID，第一条消息的 parent_id 为 NULL
    role VARCHAR(20) NOT NULL,              -- 角色：system | user | assistant | tool
    content TEXT,                           -- 消息文本内容（工具调用时可为 NULL）
    tool_calls JSONB,                       -- Assistant 产出的工具调用参数列表
    tool_call_id VARCHAR(64),               -- 当 role 为 tool 时，对应的工具调用 ID
    token_count INT DEFAULT 0,              -- 该单条消息估算或统计的 Token 数
    status VARCHAR(20) NOT NULL,            -- 状态：pending | streaming | completed | failed
    error_message TEXT,                     -- 失败原因（如超时、风控拦截）
    is_deleted BOOLEAN DEFAULT FALSE,       -- 软删除标记
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES chat_session(id),
    CONSTRAINT fk_parent FOREIGN KEY (parent_id) REFERENCES chat_message(id)
);

CREATE INDEX idx_msg_session_parent ON chat_message(session_id, parent_id);
CREATE INDEX idx_msg_created_at ON chat_message(session_id, created_at ASC);
```

<ReadMoreLock />

通过引入 `parent_id`，消息的先后关系由指针决定，而不再依赖数据库主键自增或时间戳。即使同一问题派生出了三个不同的回答，它们各自向下延伸，互不干扰。

### 3. 读写分层：Redis 缓存与数据库协作

大模型生成往往持续几秒到数十秒，如果前端通过 SSE（Server-Sent Events）接收数据，服务端每收到一个 Token 都去写一次数据库，会导致数据库连接池被占满、产生大量无效写 I/O。

在实际工程中，通常结合 Redis 与关系数据库进行读写分工：

![Redis 高速缓存与 PostgreSQL 数据库的分层读写时序](/images/content-rebuild/96287fc42c559080/fig-2.png)

上图展示了读写的时序配合：Redis 负责承载生成过程中的临时状态与活跃会话读取，数据库负责记录终态和长远持久化。

#### 3.1 写入流程（Write Path）

1. **接收用户提问**：
   - 请求到达后，后端先在 Redis 中尝试获取该会话的互斥锁（配合客户端幂等键防重），未获取到锁则直接拦截并发请求；
   - 加锁成功后，在数据库中插入用户的 `user` 消息（状态设为 `completed`）；
   - 紧接着插入一条空的 `assistant` 占位消息（状态设为 `pending`），其 `parent_id` 指向刚才的 `user` 消息，并记录当前会话的生成状态。
2. **处理流式 Token**：
   - 后端每收到几个 Token，先在内存中缓存；
   - 按一定时间间隔（例如每隔 300~500 毫秒）或字数阈值，将增量内容同步更新到 Redis 临时键中（如 `session:{id}:streaming_buffer`）；
   - 客户端如果在流式传输中断线，重连后可直接从 Redis 临时键中读取已经生成的前半部分文本，而不必等待整个请求跑完。
3. **生成结束与终态持久化**：
   - 模型正常结束时，后端将完整文本内容、Token 消耗统计和终态 `status='completed'` 通过一条 SQL `UPDATE` 更新到数据库的 `assistant` 消息记录，并同步将会话表的 `current_leaf_message_id` 指向该消息；
   - 若生成途中发生超时、断连或风控拦截，则将该记录更新为 `status='failed'` 并写入 `error_message`；
   - 流程结束后，清除 Redis 中的流式临时键，并释放会话互斥锁。

#### 3.2 读取流程（Read Path）

- **活跃对话读取**：可将最近活跃会话的主干消息 ID 列表缓存至 Redis（如 List 或 ZSet），拼接上下文时优先走缓存，降低数据库查询压力。
- **历史翻阅与分支切换**：当用户在侧边栏翻看历史会话，或切换查看历史版本时，后端再查询数据库，通过递归方式拉取指定分支的消息链路。

### 4. 分支对话的实现与回溯查询

支持重新生成和历史提问修改，核心就在于如何操作父指针树。

![重新生成与编辑重发时的分支树裂变及叶子指针迁移机制](/images/content-rebuild/96287fc42c559080/fig-3.png)

上图展示了生成新分支时节点的变化情况：原有的因果链路保留不变，系统只在对应的父节点下创建新的子节点，并将活跃叶子指针指向新节点。

#### 4.1 重新生成（Regenerate）

假设用户对当前最后的 `assistant_1` 不满意：
1. 读取 `assistant_1` 的 `parent_id`，即上一条 `user_1` 的 ID；
2. 新建一条 `assistant_2`，其 `parent_id` 同样设为 `user_1` 的 ID；
3. 生成完毕后，把 `chat_session.current_leaf_message_id` 更新为 `assistant_2`。

此时，`assistant_1` 仍然保留在数据库中。前端只要查询所有 `parent_id = user_1` 的记录，就能统计出当前有 2 个候选回答，并在界面上提供“上一条 / 下一条”（1/2）的切换按钮。

#### 4.2 编辑历史提问（Edit Question）

如果用户修改了前几轮的某一问：
1. 找到被修改问题的 `parent_id`（通常是更早的一条 assistant 消息）；
2. 插入一条新的 `user_new` 消息，其 `parent_id` 指向该上一轮节点；
3. 发起调用生成新的 `assistant_new`，其 `parent_id` 指向 `user_new`；
4. 更新会话的当前叶子节点为 `assistant_new`。

这样，被编辑之前的旧轮次与回答完全不受破坏，用户依然可以切回旧分支。

#### 4.3 递归查询完整上下文（Recursive CTE）

当准备调用模型时，如何把指定叶子节点之上的整条主线抽取出来？在支持通用表表达式（CTE）的关系型数据库（如 PostgreSQL）中，使用一条递归查询就能沿父指针回溯整条链路：

```sql
WITH RECURSIVE conversation_path AS (
    -- 起点：从当前激活的叶子节点开始
    SELECT id, session_id, parent_id, role, content, tool_calls, tool_call_id, created_at, 1 AS depth
    FROM chat_message
    WHERE id = :target_leaf_message_id AND is_deleted = FALSE
    
    UNION ALL
    
    -- 递归回溯：通过 parent_id 逐级向上找父消息
    SELECT m.id, m.session_id, m.parent_id, m.role, m.content, m.tool_calls, m.tool_call_id, m.created_at, cp.depth + 1
    FROM chat_message m
    INNER JOIN conversation_path cp ON m.id = cp.parent_id
    WHERE m.is_deleted = FALSE
)
SELECT * FROM conversation_path
ORDER BY depth DESC; -- 深度越深说明越靠前，倒序排列即为时间正序
```

这条查询会准确滤掉当前分支之外的其他分支回答，只返回从根节点到当前叶子节点的单条正序对话链。

### 5. 并发控制与状态防冲突

模型生成通常持续较长时间。如果不做限制，用户在生成期间再次发送消息，或者网络抖动重发请求，会导致数据错乱。

![并发双击未加锁导致的上下文穿插 vs 分布式锁拦截对比](/images/content-rebuild/96287fc42c559080/fig-4.png)

上图对比了有无控制时的区别：没有锁时，两次发送会交替插入数据库，破坏提问与回答的对应关系；加入锁保护后，重复请求在网关或业务层就被拦截。

#### 5.1 会话状态流转与锁机制

会话可以定义明确的状态转移：
- **空闲（IDLE）**：可以接收新的提问；
- **处理中（PROCESSING）**：正在排队或调用模型前置检查；
- **流式输出中（STREAMING）**：正在接收模型返回的 Token。

在多实例服务部署下，通常利用 Redis 设置带超时的互斥锁：

1. **尝试加锁**：请求到达时执行 `SET lock:session:{session_id} {req_id} NX PX 60000`（设置 60 秒硬超时，防止服务异常退出导致死锁）；
2. **锁失败处理**：如果获取锁失败，说明该会话正在生成，接口直接返回业务提示“正在回答中，请稍候再试”；
3. **客户端幂等键**：前端每次点击发送生成一个唯一的 `client_message_id`。如果用户因网络波动重发相同请求，服务端根据该 ID 命中已有记录，避免同一条消息重复入库。

### 6. 代码演练：父指针回溯与分支隔离

下面的 Python 示例展示了带父指针的消息存储、分支生成以及回溯完整上下文的完整逻辑：

```python
from typing import Dict, List, Optional
import json

class MessageNode:
    def __init__(self, msg_id: str, parent_id: Optional[str], role: str, content: str):
        self.msg_id = msg_id
        self.parent_id = parent_id
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

class ConversationStore:
    def __init__(self):
        # 模拟消息表与会话表的叶子指针
        self.messages: Dict[str, MessageNode] = {}
        self.leaf_pointers: Dict[str, str] = {}  # session_id -> current_leaf_id

    def append_message(self, session_id: str, msg_id: str, role: str, content: str, parent_id: Optional[str] = None) -> str:
        # 若未指定 parent_id，默认接在当前最新叶子节点后面
        if parent_id is None:
            parent_id = self.leaf_pointers.get(session_id)
        
        node = MessageNode(msg_id, parent_id, role, content)
        self.messages[msg_id] = node
        self.leaf_pointers[session_id] = msg_id
        return msg_id

    def trace_context(self, session_id: str, leaf_id: Optional[str] = None) -> List[Dict[str, str]]:
        """从指定叶子节点（或当前激活叶子）反向追溯至根节点，翻转后返回正序上下文"""
        target_id = leaf_id or self.leaf_pointers.get(session_id)
        if not target_id:
            return []

        path = []
        curr_id = target_id
        while curr_id is not None:
            node = self.messages.get(curr_id)
            if not node:
                raise ValueError(f"节点引用损坏: 找不到消息 ID {curr_id}")
            path.append(node.to_dict())
            curr_id = node.parent_id
        
        # 沿父指针向上拿到的是倒序列表，翻转后即为从根到叶的正序对话
        path.reverse()
        return path

# --- 验证测试 ---
if __name__ == "__main__":
    store = ConversationStore()
    s_id = "session_001"

    # 1. 第一轮对话
    store.append_message(s_id, "msg_1", "user", "推荐三本关于分布式系统的经典书。")
    store.append_message(s_id, "msg_2", "assistant", "推荐：《DDIA》、《分布式系统原理与范式》、《深入理解分布式事务》。")

    # 2. 第二轮对话
    store.append_message(s_id, "msg_3", "user", "第一本书的核心作者是谁？")
    store.append_message(s_id, "msg_4", "assistant", "《DDIA》的作者是 Martin Kleppmann。")

    print("=== 原始对话链路 ===")
    print(json.dumps(store.trace_context(s_id), ensure_ascii=False, indent=2))

    # 3. 用户对 msg_4 回答不满意，点击重新生成，产生新节点 msg_5，父节点指向 msg_3
    print("\n>>> 触发重新生成: 针对 msg_3 生成新的回答...")
    store.append_message(s_id, "msg_5", "assistant", "《设计数据密集型应用》（即 DDIA）由英国学者 Martin Kleppmann 编写。", parent_id="msg_3")

    print("=== 切换分支后的上下文 ===")
    print(json.dumps(store.trace_context(s_id), ensure_ascii=False, indent=2))

    # 4. 验证历史分支依然可查：通过旧叶子节点 msg_4 查询完整旧链路
    print("=== 通过历史叶子节点仍可查阅旧分支 ===")
    print(json.dumps(store.trace_context(s_id, leaf_id="msg_4"), ensure_ascii=False, indent=2))
```

运行结果验证了树形存储的核心特性：
- 重新生成后，新创建的 `msg_5` 挂在 `msg_3` 下，会话叶子指针切至 `msg_5`，`msg_4` 的内容被自然旁路，不会混入新的推理输入中；
- 传入 `leaf_id="msg_4"` 依然能完全还原重生成之前的历史视图，数据不会被覆盖。

![代码实验中消息节点指针在内存字典中的回溯与正序反转过程](/images/content-rebuild/96287fc42c559080/fig-5.png)

上图展示了代码中追溯算法的运行逻辑：从叶子节点出发顺着 `parent_id` 向上逐级抓取节点，最后做一次逆序排列，即可输出符合模型 API 格式的正序数组。

### 7. 静态存储与动态上下文的边界

在构建多轮对话系统时，需要区分存储层与组装层的职责边界：

1. **存储持久化不等于输入组装**：存储层的职责是忠实、无损地记录所有轮次的消息和分支。不要在持久化层把老消息物理删除。哪怕某个对话长达几百条、远超当前模型的上下文限制，存储层也要完整保存，供前端翻阅和审计。
2. **树形结构的上限考量**：对于普通聊天，单会话的消息节点大多在几十到几百条之间，单次 CTE 查询耗时微乎其微。但如果是长时间运行的自主 Agent 任务，会话节点可能累积上万条，此时应定期生成断点快照（Checkpoint），避免递归链路过长影响回溯性能。

![对话历史静态持久化与上下文动态预算裁切的职责边界拓扑](/images/content-rebuild/96287fc42c559080/fig-6.png)

如图所示，静态持久化负责记录完整的分支历史，而调用模型时则需要经过预算控制与裁剪，把历史消息压缩或截断在模型 Token 窗口之内。动态裁剪与预算装配的具体策略，我们将在下一讲 [/llm-100/app-system/context-budget.md](/llm-100/app-system/context-budget.md) 中详细解析。
