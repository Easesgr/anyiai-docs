---
title: 多跳复杂查询如何规划？子任务分解、异构路由与状态编排
icon: mdi:brain
tag:
- Agentic RAG
- 查询规划
- 答案校验
- 系统设计
description: 解析多跳查询如何通过有向依赖图拆分子任务，结合异构数据源动态路由、拓扑调度与证据冲突仲裁，解决跨实体依赖与事实不一致问题。
---

在检索增强生成（RAG）系统中，用户的提问往往并非单次检索就能回答。例如这样一个问题：“对比星河动力最新完成的一轮融资，与蓝箭航天最新估值所处的投资阶段差异。”

如果直接拿这句话去向量数据库中检索，检索模块并不知道“星河动力最新一轮融资”具体是哪一轮、金额多少，也无法直接定位蓝箭航天的最新估值。单次检索召回的文本片段，大概率只包含两家公司的通用介绍或早期新闻，关键的事实数据会被噪声淹没。

这类涉及跨实体对比、未知中间变量和多数据源聚合的提问，被称为**多跳复杂查询（Multi-hop Complex Query）**。解决这类问题的核心思路不是让检索模型硬猜，而是**将大问题拆解为带依赖关系的子任务图，为每个子任务分发适合的数据引擎，按拓扑顺序调度执行，并在最后对多源证据进行冲突仲裁**。

![复杂多跳查询规划机制总览](/images/content-rebuild/e71c7ec41bffdb8b/cover.png)

---

## 一、多跳复杂问题的三类依赖模式

多跳复杂问题之所以难以直接检索，是因为其所需事实分散在不同实体、时间窗口或存储系统之中。在设计规划逻辑前，通常先识别出三类基础的任务依赖模式：

1. **水平并列依赖（Independent Branches）**：需要横向拉取多个独立实体的相同或不同属性。例如“对比 2023 年 A 公司与 B 公司的研发费用占比”。两个公司的财报数据互不影响，可以完全并行检索，最后汇总比对。
2. **链式跨实体依赖（Sequential Multi-hop）**：后续查询的检索关键词直接依赖前序步骤的产出。例如“找到甲公司母公司投资的那家具身智能企业，并查询其团队规模”。在第一步查出“母公司名称”、第二步查出“具身智能企业名称”之前，系统根本无法构造第三步的检索关键词。
3. **约束过滤依赖（Filter Refinement）**：先查出全局约束条件（如政策发布日期、版本发布时间线），再根据该约束缩小第二阶段的检索或过滤范围。

![多跳复杂问题的三类拓扑结构对比](/images/content-rebuild/e71c7ec41bffdb8b/fig-1.png)

在实际生产中，复杂问题往往是上述模式的组合。如果系统只支持单次检索或线性两阶段检索，就无法稳定处理带依赖分支的任务。

---

<ReadMoreLock />

## 二、查询规划：构建有向无环图（DAG）

为了让程序可调度，查询规划器（Query Planner）通常采用**子任务分解（Sub-question Decomposition）**范式。规划器借助大语言模型的结构化输出能力（如 JSON Schema），将原始自然语言输入转换为一个**有向无环图（Directed Acyclic Graph, DAG）**。

图中的每个节点是一个独立的执行任务，节点中声明自身 ID、检索目标、依赖的前置节点 ID 以及产出的上下文键名。

以下是一次规范化拆解的示例：

```json
{
  "plan_id": "plan_20260330_001",
  "original_query": "对比 DeepSeek 母公司投资的具身智能企业与宇树科技在最新融资阶段上的差异",
  "tasks": [
    {
      "task_id": "step_1",
      "query": "DeepSeek 的母公司或主要控股关联方是哪家机构？",
      "dependencies": [],
      "target_source": "dense_vector",
      "output_key": "parent_company"
    },
    {
      "task_id": "step_2",
      "query": "{parent_company} 参与投资的具身智能或人形机器人初创公司是哪家？",
      "dependencies": ["step_1"],
      "target_source": "web_search",
      "output_key": "invested_robot_company"
    },
    {
      "task_id": "step_3",
      "query": "宇树科技当前的最新融资轮次是什么？",
      "dependencies": [],
      "target_source": "structured_sql",
      "output_key": "unitree_round"
    },
    {
      "task_id": "step_4",
      "query": "{invested_robot_company} 当前的最新公开融资轮次是什么？",
      "dependencies": ["step_2"],
      "target_source": "structured_sql",
      "output_key": "target_robot_round"
    },
    {
      "task_id": "step_5",
      "query": "对比 {invested_robot_company}（轮次：{target_robot_round}）与宇树科技（轮次：{unitree_round}）的融资进展差异",
      "dependencies": ["step_3", "step_4"],
      "target_source": "reasoning_engine",
      "output_key": "final_comparison"
    }
  ]
}
```

![基于 Plan-and-Solve 的子任务拓扑拆解过程](/images/content-rebuild/e71c7ec41bffdb8b/fig-2.png)

从该结构可以看出：
- `step_1` 与 `step_3` 无任何前置依赖，系统启动时即可同时分发。
- `step_2` 必须等待 `step_1` 产出具体实体，将其填入 `{parent_company}` 占位符后才能组装出真实检索词。
- `step_5` 依赖 `step_3` 与 `step_4` 的输出，负责最终的整合与逻辑对比。

通过将隐式依赖显式化为依赖图，调度器能够明确各子任务的拓扑顺序与可并发边界，避免盲目串行导致的延迟叠加，或参数未就绪时的无效并发。

---

## 三、异构数据源动态路由

在真实的软件架构中，知识并不全都保存在同一个向量库里。多跳子任务面临的查询诉求不同，适合的数据引擎也各异：

1. **稠密向量数据库（Dense Vector DB）**：适合处理长文本的主题概念检索、非结构化文档的语义模糊匹配。但在精确型号、数值统计和绝对匹配上表现较弱。
2. **全文检索倒排索引（如 BM25）**：擅长匹配专有名词、产品料号、特定错误码或特定人名，能有效弥补向量嵌入对生僻专有名词不敏感的缺陷。
3. **结构化关系型数据库（SQL / OLAP）**：适合处理明确的数值统计、区间筛选、聚合排序（如“统计 Q3 各部门预算消耗”）。将此类问题交给向量库检索往往会产生幻觉，而交给 SQL 引擎则可以实现确定性计算。
4. **外部检索工具（Web Search API）**：用于获取超出私有知识库更新截止时间的动态信息，如当月突发事件、近期融资披露等。

![异构数据源动态路由分发机制](/images/content-rebuild/e71c7ec41bffdb8b/fig-3.png)

路由决策通常有两类实现方式：
- **规则与模式匹配（Rule-based Router）**：子任务若命中特定数据字典、表名结构或固定关键词，直接分发至对应数据源，执行开销极低且完全可预测。
- **模型分类判断（LLM Function Calling / Router Agent）**：由轻量级语言模型根据子任务的意图描述，从注册的工具列表（Tool Registry）中挑选最匹配的数据源接口。

路由阶段的目标是让结构化计算由数据库负责，让语义理解由文档索引负责，避免工具错位带来的误差累积。

---

## 四、执行调度：拓扑排序、并发与动态槽位填充

子任务依赖图确定后，调度器需要解决两个核心问题：如何最大化执行效率，以及前序结果如何安全传递给后续任务。

如果全部按顺序串行执行，多个网络 IO 累加会导致响应时间极长；如果全部并发执行，依赖前序实体的子任务会因为缺失参数而报错。

标准的解决方案是维护一个**就绪任务队列（Ready Queue）**与**共享上下文状态（State Store）**：
1. 调度器对所有任务做拓扑分析，找出所有依赖已满足（入度为 0）的节点投入并发执行。
2. 当某个节点执行成功后，将其输出写回共享上下文，并将自身标记为已完成。
3. 下游节点的依赖项全部就绪后，调度器从上下文中取出前置实体的具体值，替换下游任务提示词或查询词中的参数占位符（即槽位填充 Slot Filling），将其解锁并推入就绪队列。

以下是一个状态调度器的 Python 原型示例：

```python
import asyncio
from typing import Dict, Any, List

class TaskNode:
    def __init__(self, task_id: str, query: str, deps: List[str], target: str, output_key: str):
        self.task_id = task_id
        self.query = query
        self.deps = deps
        self.target = target
        self.output_key = output_key
        self.status = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
        self.result = None

class PlanOrchestrator:
    def __init__(self, tasks: List[TaskNode]):
        self.tasks = {t.task_id: t for t in tasks}
        self.context: Dict[str, Any] = {}

    async def execute_task(self, task: TaskNode) -> str:
        # 槽位填充：将模板中的 {var} 替换为上下文中保存的前置真实提取项
        resolved_query = task.query
        for key, val in self.context.items():
            resolved_query = resolved_query.replace(f"{{{key}}}", str(val))
        
        # 模拟调用异构数据源检索接口（实际工程中按 task.target 分发）
        await asyncio.sleep(0.05)
        
        # 示意返回值
        mock_registry = {
            "step_1": "幻方量化 / 深度求索",
            "step_2": "智元机器人",
            "step_3": "B2 轮",
            "step_4": "A 轮",
            "step_5": f"{self.context.get('invested_robot_company')}处于{self.context.get('target_robot_round')}，宇树科技处于{self.context.get('unitree_round')}。"
        }
        return mock_registry.get(task.task_id, "ok")

    async def run(self):
        completed_tasks = set()
        total_tasks = len(self.tasks)

        while len(completed_tasks) < total_tasks:
            # 筛选当前依赖均已满足的待执行任务
            ready_tasks = [
                t for t in self.tasks.values()
                if t.status == "PENDING" and all(dep in completed_tasks for dep in t.deps)
            ]
            
            if not ready_tasks:
                if any(t.status == "PENDING" for t in self.tasks.values()):
                    raise RuntimeError("检测到死锁或未满足的前置依赖，流程中止")
                break

            # 标记为运行中并并发发起调用
            for t in ready_tasks:
                t.status = "RUNNING"
            
            results = await asyncio.gather(*(self.execute_task(t) for t in ready_tasks))
            
            # 消费执行结果，更新上下文与状态机
            for task, res in zip(ready_tasks, results):
                task.result = res
                task.status = "COMPLETED"
                completed_tasks.add(task.task_id)
                self.context[task.output_key] = res

        return self.context.get("final_comparison")
```

![执行调度状态机：无依赖并发与串行依赖注入时序](/images/content-rebuild/e71c7ec41bffdb8b/fig-4.png)

在生产工程中，调度器还需要实现网络超时熔断、幂等性重试以及脏数据兜底机制，以防止个别子任务异常导致全链路卡死。

---

## 五、多源证据聚合与冲突仲裁

当异构数据源返回的信息汇集到最终合成节点时，另一个严峻问题会出现：**不同数据源之间的事实可能互相矛盾**。

例如在查询某企业核心财务指标时：
- 内部主数据库（SQL）记录显示：“2024Q1 净利润 12.4 亿元”；
- 媒体快讯（Web Search）文章提到：“该公司 Q1 调整后净利润达到 15 亿元”；
- 较早上传的分析师研报（Vector DB）中写着：“预计 Q1 净利润约 10.8 亿元”。

如果直接把这三段文本一股脑塞给最终回答模型，模型容易自相矛盾或随机采信某一段。系统需要在编排层建立**基于元数据证据的仲裁规则**：

| 维度 | 业务主数据库（SQL/DW） | 权威官方文档 / 研报归档 | 外部网络动态检索 | 开放向量索引切片 |
| :--- | :--- | :--- | :--- | :--- |
| **适用场景** | 数值聚合、交易流水、主数据 | 深度业务解读、口径说明 | 突发时讯、近期公开变更 | 概念解释、语义模糊检索 |
| **时间戳判定** | 以数据库系统提交时间戳为准 | 以文档元数据发布时间为准 | 以网页抓取与发布时间为准 | 依赖切片入库维护时间 |
| **冲突处理建议** | 业务主数据时效满足时优先作为统计基准，但需防范同步延迟 | 需核对统计口径与审计披露状态，与时点数据交叉验证 | 按发布源权威签名与时间戳动态裁决，警惕未证实传言 | 缺乏时间戳与权威签名时降低置信权重，仅作补充参考 |

![跨数据源证据聚合与事实冲突仲裁流水线](/images/content-rebuild/e71c7ec41bffdb8b/fig-5.png)

解决冲突的工程原则通常包括：
1. **以元数据时间戳判定有效区间**：优先采纳时间戳更新、且数据源签名权威的记录。
2. **保留分歧并显式归因（Attribution）**：如果两个可信源在同一指标上出现口径分歧且系统无法自动抹平，生成模块应当如实陈述：“据企业内部报表记载为 A，而公开市场快讯口径为 B”，而不是由模型随意捏造一个中间数字。

---

## 六、链路可观测性与异常熔断

多任务规划机制将单次模型交互转变成了一个多步骤的计算流程。此时，系统的故障排查难度显著提高。

如果第一阶段的实体抽取节点把“星河动力”误识别为“星际荣耀”，后续所有基于该实体的检索都会在错误的方向上浪费 Token 和调用时间。因此，生产系统必须引入细粒度的**链路追踪（Trace）与阻断机制**：

1. **Span 树形调用追踪**：借助 LangSmith、OpenTelemetry 等工具，将 DAG 中的每个子任务记录为一个独立的 Trace Span。记录其输入的查询词、上下文变量、命中数据源、召回片段、模型耗时与 Token 开销。
2. **空结果与低置信度短路（Circuit Breaking）**：当下游节点所需的核心实体提取结果为空，或置信度评分低于预设阈值时，调度器应及时中断该依赖分支，避免下游盲目空转。
3. **人工接入点（Human-in-the-loop）**：在关键合规或金融测算场景中，规划器分解出的 DAG 可以先推送到交互界面供分析人员审阅或补充槽位，确认无误后再执行批量拉取。

![全链路 Trace 监控与子任务状态机演变视图](/images/content-rebuild/e71c7ec41bffdb8b/fig-6.png)

---

## 七、适用边界与工程权衡

查询规划与多步调度提供了处理高难度问题的能力，但这套机制也伴随着清晰的工程代价：

1. **响应延迟增加**：即使利用拓扑排序最大化并发，多阶段的模型推理、结构化解析与多源 IO 调用，也会使端到端耗时从单次检索的数百毫秒上升到数秒甚至数十秒。
2. **对模型指令遵循能力的强依赖**：规划的质量高度依赖模型输出严谨 JSON 的能力。如果使用的模型在长上下文下容易破坏 JSON Schema，或经常编造不存在的依赖 ID，整个系统就会频繁死锁报错。
3. **架构选型建议**：
   - 对于意图明确、单次检索即可覆盖的事实问答，优先使用标准单阶段检索配合重排序（Re-ranking）；
   - 仅当问题包含**多个未知实体的级联依赖**、**跨结构化与非结构化系统的混合计算**时，引入带 DAG 分解与状态调度的查询规划方案才是性价比合理的选择。
