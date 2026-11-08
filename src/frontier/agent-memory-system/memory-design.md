---
title: Agent 记忆系统如何设计？写入过滤、混合检索、冲突更新与衰减淘汰机制
icon: mdi:cog-outline
tag:
- Agent 记忆系统
- 记忆设计
- 检索机制
- 遗忘策略
description: 详解 Agent 记忆系统在写入、检索、更新与淘汰各阶段的运行机制，结合编程助手场景与 Python 代码，解决记忆污染、事实冲突与长期陈旧失效问题。
---

![Agent 记忆引擎架构全景封面图](/images/content-rebuild/a45a3672d5b6035b/cover.png)

在上一篇[《为什么记忆系统不能什么都存？隐私、噪声和上下文污染如何控制？》](/frontier/agent-memory-system/memory-risks.md)中，我们分析了把对话流水无节制存入数据库带来的上下文污染、隐私泄露与模型决策漂移。要让 Agent 具备跨会话的连贯能力，关键不在于“存了多少数据”，而在于如何建立一套具备状态仲裁和自洁能力的记忆生命周期系统。

如果我们只把对话文本简单切块存入向量数据库，系统在实际使用中很快会遇到四个具体问题：
1. **写入噪声与隐私风险**：临时口令、闲聊废话和敏感隐私一起入库；
2. **检索失真**：纯稠密向量在精确数字、版本号和长尾术语上容易泛化漂移；
3. **事实互斥与冲突**：用户更新了自己的信息（如“我换用了 Python 3.12”），但库中旧事实（“Python 3.10”）依然存在，模型生成时产生矛盾；
4. **记忆膨胀与失效**：数据库只增不减，随着时间推移，早期的单次临时指令与长期习惯混在一起，拖慢检索速度并干扰决策。

本文通过一个贯穿始终的个人编程助手案例，讲解记忆系统在写入、检索、更新与遗忘四个环节的具体机制与可运行的代码实现。

---

## 贯穿场景与生命周期状态机

假设我们正在维护一个编程助手 Agent。用户小李在第 1 天告诉助手：“我平时习惯用 Python 3.10，手机号是 13812345678”；第 5 天小李升级了环境，告诉助手：“我的 Python 版本换成了 3.12”；在接下来的两个月里，小李频繁让助手编写 Python 脚本，但再也没有提及过手机号。

一套合格的记忆系统应当表现出以下行为：
- 手机号在写入时自动完成掩码脱敏；
- 第 5 天写入新环境事实后，旧的 Python 3.10 自动归档失效，防止两个互斥版本同时出现在上下文中；
- 经过 70 天后，Python 偏好因被多次查阅使用而保持高活性，而从未被再次访问的手机号因活性衰减被自动剔除。

要让系统可控，每条记忆条目都必须处于受控的状态转移之中。

![记忆系统全生命周期状态机图](/images/content-rebuild/a45a3672d5b6035b/fig-1.png)

记忆条目的生命周期包含四个阶段：

1. **入库审核态（Ingesting / Validating）**：接收交互流水。首先通过规则匹配敏感隐私（PII）与写入价值打分，未通过校验的废弃，符合条件的进入提取阶段；
2. **提取持久态（Active）**：由轻量模型或规则提取结构化事实（主键、主体、谓词、客体），写入存储并附带版本号与初始权重，进入活跃状态；
3. **检索装配态（Context Assembly）**：接收查询请求，通过语义向量与关键词双路召回候选集，结合时效性衰减完成动态重排与主键冲突消解后注入 Prompt；
4. **衰减归档态（Archived / Evicted）**：后台根据条目的最后访问时间与累计使用频次计算活性衰减，低于临界值的条目转入归档或从活跃索引中物理移出。

如果模型抽取结构化三元组超时或解析失败，系统应降级为基于规则抽取的弱结构化暂存队列，并记录指标，不得阻塞用户主交互流程。

<ReadMoreLock />

---

## 写入管线：过滤脱敏与结构化抽取

写入管线负责把原始对话转换为规范的记忆单元。

![写入管线过滤与抽取流程图](/images/content-rebuild/a45a3672d5b6035b/fig-2.png)

### 1. 规则脱敏与安全拦截
对话文本在进入任何复杂处理前，先经过正则引擎与敏感词词典。对于手机号、身份证号、银行卡号等符合特征的字段，执行掩码脱敏（例如把 `13812345678` 转换为 `138****5678`）；对于 API Key、数据库密码等敏感信息，直接阻断写入并记录审计日志。

### 2. 重要度阈值判定
并非所有对话都值得保留。例如“把刚才的代码行缩进改成 4 格”属于当前任务的单次操作指令，“我平时习惯使用 Python 3.12”才属于持久偏好。我们定义重要度分数 $S_{\text{imp}} \in [0, 1]$，结合启发式规则（如问答偏好指示词）或轻量分类模型评估。设定入库阈值 $\theta_{\text{write}} = 0.5$（可根据实际场景调整），低于该阈值的对话直接丢弃，不进入长期存储。

### 3. 结构化实体抽取
纯文本段落在做冲突覆写时极难精准匹配，因此需要将通过筛选的文本转化为具备明确键名的事实条目：

```json
{
  "primary_key": "user#preferred_python_version",
  "subject": "user",
  "predicate": "preferred_python_version",
  "object": "3.12",
  "confidence": 0.95,
  "source_turn_id": "chat_session_8921_turn_4",
  "timestamp": 1711929600
}
```

通过确定主体（`subject`）与谓词（`predicate`）构成唯一的逻辑主键，这是后续执行版本覆写与去重更新的基础。

---

## 检索管线：混合召回与动态重排

单纯依赖稠密向量（Dense Embedding）检索，容易在精确版本号、变量名和数字上出现语义漂移；而单纯依靠词频匹配（如 BM25）则无法理解同义表述。因此，成熟的做法通常采用**语义向量 + 关键词 + 时间衰减**的混合打分策略。

![多路混合检索与动态打分架构图](/images/content-rebuild/a45a3672d5b6035b/fig-3.png)

### 混合打分公式

设候选记忆条目为 $m$，当前查询为 $q$，综合检索分数为：

$$S_{\text{score}}(m, q) = w_d \cdot S_{\text{dense}}(m, q) + w_s \cdot S_{\text{sparse}}(m, q) + w_t \cdot S_{\text{recency}}(m)$$

其中各路权重满足 $w_d + w_s + w_t = 1$：
- **$S_{\text{dense}}(m, q)$**：查询与记忆正文的向量余弦相似度，映射至 $[0, 1]$；
- **$S_{\text{sparse}}(m, q)$**：BM25 打分归一化后的分值，范围为 $[0, 1]$；
- **$S_{\text{recency}}(m)$**：时间衰减分值，定义为 $S_{\text{recency}}(m) = \exp(-\lambda_r \cdot \Delta t)$，其中 $\Delta t$ 为当前时间与该条目更新时间的差值（单位为天），$\lambda_r$ 为时间衰减率参数。

### 逐步算例

设参数 $w_d = 0.5$，$w_s = 0.3$，$w_t = 0.2$，衰减系数 $\lambda_r = 0.05$。当前查询为：`“我的 Python 运行环境是什么版本？”`。

系统召回了三条候选事实（时间差单位为天）：

| 候选条目内容 | $S_{\text{dense}}$ | 归一化 $S_{\text{sparse}}$ | 时间差 $\Delta t$ | $S_{\text{recency}} = \exp(-0.05 \cdot \Delta t)$ | 最终综合得分 $S_{\text{score}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 条目 A: `用户首选 Python 3.12` | 0.88 | 0.70 | 2 | $\exp(-0.1) \approx 0.905$ | $0.5 \times 0.88 + 0.3 \times 0.70 + 0.2 \times 0.905 \approx 0.831$ |
| 条目 B: `用户在三年前安装了 Python 2.7` | 0.82 | 0.85 | 1000 | $\exp(-50) \approx 0.000$ | $0.5 \times 0.82 + 0.3 \times 0.85 + 0.2 \times 0.000 \approx 0.665$ |
| 条目 C: `用户正在排查 Python 构建脚本` | 0.75 | 0.40 | 1 | $\exp(-0.05) \approx 0.951$ | $0.5 \times 0.75 + 0.3 \times 0.40 + 0.2 \times 0.951 \approx 0.685$ |

条目 B 虽然因关键词“Python”重复出现导致 BM25 分值较高，但由于时间过于久远，时间衰减项趋近于 0，总分被拉低；条目 A 兼顾了语义匹配与时效性，最终排在首位。

---

## 冲突更新机制：主键因果仲裁

当系统同时检索到“用户常用 Python 3.10”与“用户常用 Python 3.12”时，把两份互相矛盾的事实一起送给大模型，会导致模型产生混乱。

![实体冲突检测与因果版本仲裁流程图](/images/content-rebuild/a45a3672d5b6035b/fig-4.png)

冲突仲裁发生在两个阶段：**写入落库时**与**检索装配后的上下文去重阶段**。系统依据实体主键（Subject-Predicate Key）执行确定性仲裁：

### 仲裁元数据与状态流转
每条记录包含逻辑主键 `primary_key`、更新时钟 `timestamp`、版本号 `version` 与置信度 `confidence`。

当新事实 $M_{\text{new}}$ 拟写入，且发现已存在相同 `primary_key` 的旧事实 $M_{\text{old}}$ 时：
- **因果覆写**：若 $M_{\text{new}}$ 的时间戳晚于 $M_{\text{old}}$，将 $M_{\text{old}}$ 状态更新为 `archived`（移出常规索引），写入 $M_{\text{new}}$ 并设置 `version = M_old.version + 1`；
- **多值追加**：如果谓词属于多值属性（如“用户掌握的技术栈”，可以同时包含 Python 和 Rust），则将 Object 纳入主键规则构成复合键 `(subject, predicate, object)`，避免单值覆写；
- **低信丢弃**：若新记录时间戳陈旧（乱序到达），或置信度低于阈值，则拒绝覆写并写入审计日志。

在检索召回候选池中，若依然命中同主键的多条历史版本，在组装入 Prompt 之前执行一次去重过滤，只保留版本号最新的一条。

---

## 遗忘与降级：考虑访问频次的动态衰减

如果记忆衰减只看创建或修改的时间流逝（$\Delta t$），会导致重要的低频事实被错误清理。例如用户的“花生严重过敏”，可能半年内都不会再次提及，但该事实不能被轻易遗忘；相反，某些临时事实如果从未被使用过，应当较快淘汰。

![带访问频次修正的动态权重衰减与淘汰机制图](/images/content-rebuild/a45a3672d5b6035b/fig-5.png)

### 频次修正衰减函数

引入访问次数（$N_{\text{access}}$）对半衰期进行自适应调整，计算条目在当前时刻的活性分值 $A(t)$：

$$A(t) = S_{\text{base}} \cdot \exp\left( - \frac{\Delta t}{\tau \cdot \left(1 + \ln(1 + N_{\text{access}})\right)} \right)$$

- **$S_{\text{base}}$**：条目的基础重要度常数（$0 < S_{\text{base}} \le 1$）；
- **$\Delta t$**：自上次被检索命中并装配使用以来流逝的天数；
- **$\tau$**：基准时间衰减尺度（即未修正时活性衰减至 $1/e \approx 36.8\%$ 的特征时间常数，例如设为 30 天）；
- **$N_{\text{access}}$**：该条目被实际装配进入 Prompt 的累计次数。

### 对比算例

设定基准时间衰减尺度 $\tau = 30$ 天。考虑两条同样经历了 $\Delta t = 60$ 天未被引用的条目：

1. **高频核心事实（条目 1）**：$S_{\text{base}} = 0.95$，$N_{\text{access}} = 20$。
   修正因子为 $1 + \ln(1 + 20) \approx 1 + 3.045 = 4.045$。有效衰减时间常数被延长至 $30 \times 4.045 \approx 121.3$ 天。
   $$A(t)_1 = 0.95 \cdot \exp\left(-\frac{60}{121.3}\right) \approx 0.95 \cdot 0.610 \approx 0.579$$
   经过 60 天后，其活性分值仍保持在可用水线上。

2. **单次临时事实（条目 2）**：$S_{\text{base}} = 0.60$，$N_{\text{access}} = 0$。
   修正因子为 $1 + \ln(1) = 1.0$。有效衰减时间常数保持 30 天。
   $$A(t)_2 = 0.60 \cdot \exp\left(-\frac{60}{30}\right) = 0.60 \cdot \exp(-2) \approx 0.60 \times 0.135 \approx 0.081$$
   经过 60 天后，其活性跌破 $0.1$。

### 淘汰机制
后台定时任务定期遍历所有处于 `Active` 状态的记忆条目：
- 当 $A(t) < \theta_{\text{evict}}$（例如 $0.12$），将条目标记为 `evicted`，从召回索引中剔除，释放上下文检索资源；
- 当单租户条目数超出限制时，优先淘汰 $A(t)$ 最低的非核心标签数据。

---

## 最小闭环工程验证

以下 Python 脚本实现了一个完整的最小记忆引擎闭环，涵盖 PII 过滤脱敏、主键因果覆写、频次修正衰减打分以及检索上下文组装：

![最小闭环运行验证与交互输出示意图](/images/content-rebuild/a45a3672d5b6035b/fig-6.png)

```python
import math
import time
import re
from typing import Dict, List, Optional, Tuple

class MemoryEngine:
    def __init__(self, tau: float = 30.0, evict_threshold: float = 0.12):
        self.tau = tau
        self.evict_threshold = evict_threshold
        # 存储主键映射: primary_key -> dict
        self.active_store: Dict[str, dict] = {}
        self.archive_store: List[dict] = []

    def _sanitize_pii(self, text: str) -> str:
        # 手机号脱敏规则：中间 4 位转为星号
        return re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', text)

    def write_fact(self, subject: str, predicate: str, obj: str, 
                   importance: float, timestamp: float, confidence: float = 1.0) -> bool:
        # 1. 过滤判断：重要度低于阈值直接丢弃
        if importance < 0.5:
            return False

        clean_obj = self._sanitize_pii(obj)
        pk = f"{subject.lower()}#{predicate.lower()}"
        
        # 2. 冲突消解与因果覆写
        if pk in self.active_store:
            existing = self.active_store[pk]
            # 时间序仲裁：新时间戳覆写旧事实
            if timestamp >= existing["timestamp"]:
                existing["status"] = "archived"
                self.archive_store.append(existing)
                new_version = existing["version"] + 1
            else:
                return False  # 陈旧乱序数据，拒绝更新
        else:
            new_version = 1

        # 3. 写入新实体
        record = {
            "pk": pk,
            "subject": subject,
            "predicate": predicate,
            "object": clean_obj,
            "importance": importance,
            "timestamp": timestamp,
            "last_accessed_at": timestamp,
            "access_count": 0,
            "version": new_version,
            "confidence": confidence,
            "status": "active"
        }
        self.active_store[pk] = record
        return True

    def compute_activation(self, record: dict, current_time: float) -> float:
        delta_days = max(0.0, (current_time - record["last_accessed_at"]) / 86400.0)
        # 频次修正有效衰减时间常数: tau * (1 + ln(1 + N))
        frequency_factor = 1.0 + math.log(1.0 + record["access_count"])
        effective_half_life = self.tau * frequency_factor
        return record["importance"] * math.exp(-delta_days / effective_half_life)

    def search_and_assemble(self, query_terms: List[str], current_time: float, top_k: int = 2) -> str:
        candidates: List[Tuple[float, dict]] = []
        
        for pk, record in list(self.active_store.items()):
            score_act = self.compute_activation(record, current_time)
            
            # 衰减淘汰判定
            if score_act < self.evict_threshold:
                record["status"] = "evicted"
                self.archive_store.append(record)
                del self.active_store[pk]
                continue
                
            # 匹配度模拟打分：词项命中 + 活性加权
            content = f"{record['subject']} {record['predicate']} {record['object']}"
            term_match = sum(1.0 for term in query_terms if term.lower() in content.lower())
            
            if term_match > 0:
                final_rank = term_match * 0.5 + score_act * 0.5
                candidates.append((final_rank, record))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:top_k]
        
        # 更新召回条目的访问时间与使用频次
        for _, rec in selected:
            rec["access_count"] += 1
            rec["last_accessed_at"] = current_time

        # 4. 上下文装配
        if not selected:
            return ""
        facts_prompt = "已知用户记忆事实：\n" + "\n".join(
            [f"- {r['subject']} 的 {r['predicate']} 是 {r['object']} (版本: v{r['version']})" for _, r in selected]
        )
        return facts_prompt

# --- 验证贯穿场景 ---
if __name__ == "__main__":
    engine = MemoryEngine(tau=30.0, evict_threshold=0.12)
    t0 = 1710000000.0  # 初始基准时间
    
    # 1. 第 0 天写入初始事实
    engine.write_fact("User", "language", "Python 3.10", importance=0.8, timestamp=t0)
    engine.write_fact("User", "phone", "13812345678", importance=0.9, timestamp=t0)
    
    # 2. 第 5 天发生环境变更：用户更新为 Python 3.12
    t1 = t0 + 5 * 86400
    engine.write_fact("User", "language", "Python 3.12", importance=0.85, timestamp=t1)
    
    # 3. 模拟第 5 天查询环境事实（语言命中，access_count 变为 1）
    print("--- 立即查询 (第 5 天) ---")
    prompt_ctx = engine.search_and_assemble(["language"], current_time=t1)
    print(prompt_ctx)
    
    # 4. 模拟第 25 天再次检索使用环境事实（语言命中，access_count 变为 2）
    t_mid = t0 + 25 * 86400
    engine.search_and_assemble(["language"], current_time=t_mid)
    
    # 5. 模拟时间跨越至第 70 天，查询全部用户记忆
    t2 = t0 + 70 * 86400
    print("\n--- 经历长周期流逝查询 (第 70 天) ---")
    prompt_ctx_later = engine.search_and_assemble(["User"], current_time=t2)
    print(prompt_ctx_later)
```

执行上述代码后，可以观察到三个关键现象：
1. `Python 3.10` 在第 5 天被归档，版本自动升为 `v2` 且内容覆写为 `Python 3.12`，上下文中只保留唯一有效版本；
2. 手机号码写入时完成了脱敏替换（`138****5678`）；
3. 到了第 70 天：手机号因从未被访问（$N_{\text{access}} = 0$），活性分值降为 $0.9 \times \exp(-70/30) \approx 0.087 < 0.12$，触发淘汰机制并被自动移出；而语言偏好由于在第 5 天和第 25 天被两次查询使用（$N_{\text{access}} = 2$），有效衰减周期延长至约 63 天，距上次访问流逝 45 天时的活性为 $0.85 \times \exp(-45/62.96) \approx 0.416 > 0.12$，成功保留在最终上下文中。

---

## 适用边界与系统选型

记忆系统的设计应根据具体任务复杂度来选择，避免不必要的组件冗余：

1. **轻量单次任务（如单轮文档总结、格式转换）**：不需要引入长期记忆模块，直接依靠模型上下文窗口即可完成，外挂记忆系统只会增加网络 I/O 开销与延迟。
2. **通用会话助手与客服 Agent**：适合采用轻量级键值存储搭配规则抽取器，以用户 ID 为主键管理状态。关于系统架构解耦，可参考[AI 系统设计](/architecture/ai-system-design.md)与[大模型应用系统怎么设计？](/fundamentals/llm-basics/app-system-design.md)。
3. **多模态长生命周期助手**：需要兼顾会话情景流（Episodic Memory）与语义画像。在这类系统中，还需结合[企业知识库问答系统](/interview/ai-system-design-interview/knowledge-base-qa.md)与[Agent 工具调用平台设计](/interview/ai-system-design-interview/agent-tool-platform.md)进行状态协同，并在大模型调用超时或解析异常时参考[AI 系统如何处理模型调用失败、超时和降级？](/interview/ai-system-design-interview/failure-handling.md)建立兜底机制。

关于记忆概念划分的更多理论基础，可阅读本专题起点[《AI Agent 的记忆系统是什么？短期记忆、长期记忆和用户画像有什么区别？》](/frontier/agent-memory-system/what-is-memory.md)；若需准备面试中的概念梳理，可对照查阅[Agent 中的 Memory 有哪些类型？](/interview/agent-interview/agent-memory.md)。
