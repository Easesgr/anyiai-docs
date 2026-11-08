---
title: 如何让大模型回答有据可查并精确定位原文？
icon: mdi:link-variant
tag:
- 高频场景题
- 来源追溯
- 可信AI
description: 解析大模型回答溯源机制，涵盖物理元数据绑定、XML注入、网关NLI两阶段校验、前端流式防截断状态机与引用质量量化评估。
---

在金融合规、法律咨询、内部制度查询与政企知识库等严谨场景中，大模型生成的回答仅有语义上的“大致正确”是远远不够的。业务人员需要核验事实出处，点击引用角标后必须能精准定位到对应文档的某一页甚至高亮段落。

如果仅依赖大模型在回答末尾自由输出“参考文献”，经常会出现引文不存在、张冠李戴等现象。本文梳理一套完整的端到端引用溯源体系，涵盖数据准备、Prompt 约束、后置网关校验、流式渲染及离线评估。

![溯源引用校验端到端架构概览](/images/content-rebuild/295b7243df257634/cover.png)

---

## 面试标准口述：虚假引用成因与工程应对

### 30 秒口述答案

> 仅靠 Prompt 提示“请注明引用出处”，模型经常出现三类错误：**编号漂移**（引用了并不存在的编号）、**张冠李戴**（切片存在，但切片内容根本不支持该句论断）以及**常识污染**（使用预训练权重回答，却附带召回文档的编号）。
>
> 工程上解决该问题需要构建四段链路：
> 1. **数据层**：分块时不仅保存文本，还固化文档 ID、页码、字符偏移量及 PDF 边框坐标（bbox）；
> 2. **注入层**：在 Prompt 中用结构化标签（如 XML `<evidence id="...">`）包裹参考切片，约束模型仅在有证据支撑的句子后输出指定角标；
> 3. **网关层**：在输出端进行两阶段拦截。第一阶段用正则校验引用 ID 存在性，第二阶段利用轻量 NLI（自然语言推理）模型核验“证据原文”是否真的蕴含该“生成单句”；
> 4. **前端层**：针对流式传输（SSE）下引用标记被切碎的问题，用状态机滑动窗口平滑解析，并调用查看器高亮原文。

![自由生成导致虚假引用与结构化约束对比](/images/content-rebuild/295b7243df257634/fig-1.png)

---

## 贯穿示例：企业休假政策问答

为了让整套机制落地，我们以一条员工制度查询为例：

*   **用户提问**：“员工没休完的年假最晚能延期到什么时候？逾期怎么补偿？”
*   **检索召回的两个切片**：
    *   切片 A（`chunk_a7f9`）：出自《员工手册.pdf》第 14 页，内容为“员工年假在入职满一年后的次月 1 日起算，当年度未休完的年假至多顺延至次年 3 月 31 日。”
    *   切片 B（`chunk_3b81`）：出自《薪酬细则.pdf》第 3 页，内容为“未顺延且未休完的年假，按员工日工资标准的 200% 折算补偿。”

我们看这套链路如何保证输出不仅写对答案，还能点对位置、查对原文。

---

## 数据准备：切片级物理定位元数据

要在前端实现“点击角标直接翻到对应 PDF 并在那一行画黄色高亮”，切片（Chunk）在写入向量库或检索库时，就不能只存一段文本和文件名，必须保留解析引擎提取出的物理版面坐标。

建议在入库 Payload 中固化如下字段：

```json
{
  "doc_id": "corp_policy_2024_v2.pdf",
  "chunk_id": "chunk_a7f9",
  "page_idx": 14,
  "char_start": 4820,
  "char_end": 5340,
  "bbox": [72.0, 180.5, 540.0, 320.0],
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "text": "员工年假在入职满一年后的次月1日起算，当年度未休完的年假至多顺延至次年3月31日..."
}
```

*   **`page_idx` 与 `bbox`**：`bbox` 通常表示为 `[x0, y0, x1, y1]`，供 PDF.js 等阅读器直接定位并绘制高亮框；
*   **`char_start` 与 `char_end`**：供纯文本或 Markdown 渲染器做字符级选取；
*   **`content_hash`**：切片正文的 SHA-256 摘要。当底库文档更新重新切片时，可通过哈希检测引用是否指向了旧版本或已变更内容。

![Chunk物理溯源元数据数据结构映射图](/images/content-rebuild/295b7243df257634/fig-2.png)

<ReadMoreLock />

---

## 上下文注入：基于 XML 标签的显式证据块约束

纯文本提示词（如“请在每句话后面写[1]、[2]”）极易引起序号错乱。在主流大模型中，使用结构化标签（如 XML）包裹切片，并赋予明确的切片 ID，能大幅提高模型对格式约束的遵循率。

### Prompt 注入模板

```xml
<system>
你是一个专业事实问答助手。回答必须完全基于提供的参考资料。

【约束规则】
1. 凡是包含事实陈述的句子，必须在句尾紧跟引用标记，格式为：[^chunk_id]。
2. 严禁自行发明引用编号，引用的 chunk_id 必须存在于 <context> 列表中。
3. 若参考资料未提及某问题，直接说明信息缺失，严禁依赖模型预训练常识回答并伪造引用。
</system>

<context>
  <evidence id="chunk_a7f9" doc="员工手册.pdf" page="14">
    员工年假在入职满一年后的次月1日起算，当年度未休完的年假至多顺延至次年3月31日。
  </evidence>
  <evidence id="chunk_3b81" doc="薪酬细则.pdf" page="3">
    未顺延且未休完的年假，按员工日工资标准的200%折算补偿。
  </evidence>
</context>

<query>员工没休完的年假最晚能延期到什么时候？逾期怎么补偿？</query>
```

### 预期输出效果

> 员工未休完的年假最晚可顺延至次年 3 月 31 日 [^chunk_a7f9]。对于逾期未休完且未顺延的部分，公司将按照员工日工资标准的 200% 进行折算补偿 [^chunk_3b81]。

通过让模型输出具体的 `chunk_id` 而不是自增整数 `[1]`、`[2]`，能直接把模型生成文本与后端元数据对接，避免多轮对话或分块重排序导致的索引混淆。

![XML标签上下文注入与生成链路图](/images/content-rebuild/295b7243df257634/fig-3.png)

---

## 后验校验：两阶段网关拦截与校验

即使有严格的 Prompt 约束，模型仍可能出现幻觉。输出网关层需要设置自动化检验程序。

![两阶段引用后验与NLI校验流程图](/images/content-rebuild/295b7243df257634/fig-4.png)

### 阶段一：ID 存在性过滤

第一阶段属于轻量规则过滤：检查回答中出现的切片 ID，是否全部属于本次输入给模型的候选集。

```python
import re

def validate_citation_ids(response_text: str, valid_chunk_ids: set[str]) -> tuple[str, set[str]]:
    citation_pattern = r"\[\^(chunk_[a-zA-Z0-9]+)\]"
    citations = re.findall(citation_pattern, response_text)
    
    # 找出不存在于参考库中的虚构 ID
    hallucinated_ids = set(citations) - valid_chunk_ids
    if hallucinated_ids:
        # 降级策略：剥离非法引用的标记，避免前端点击产生悬空报错
        cleaned_text = re.sub(
            citation_pattern,
            lambda m: m.group(0) if m.group(1) in valid_chunk_ids else "",
            response_text
        )
        return cleaned_text, hallucinated_ids
    return response_text, set()
```

### 阶段二：命题级自然语言推理（NLI）蕴含校验

通过阶段一仅证明该 ID 在输入中出现过，并不能证明该切片真的支持对应的陈述。例如模型可能引用了《员工手册》的切片 `chunk_a7f9`，句子却写着“逾期补偿 500%”。

此时可将回答切分为单句（Hypothesis，假设），提取其绑定的切片正文（Premise，前提），输入自然语言推理模型（如轻量级 DeBERTa-v3 或专用 NLI 模型），计算三分类概率：

1. **蕴含（Entailment）**：前提能够推导出假设；
2. **矛盾（Contradiction）**：假设与前提冲突；
3. **中立（Neutral）**：前提信息不足以判断假设真伪。

| 校验判定 | 典型处置策略 |
| :--- | :--- |
| **强蕴含**（$P(\text{Entailment}) \ge 0.8$） | 检验通过，保留引用并允许前端高亮跳转。 |
| **矛盾冲突**（$P(\text{Contradiction}) \ge 0.5$） | 判定为事实冲突。高安全合规场景阻断该句并替换兜底话术；常规场景剥离角标并添加警示标记。 |
| **证据不足 / 中立**（其他情况，含弱蕴含或 $P(\text{Neutral})$ 占优） | 证据不足以充分支撑该句论断，网关剔除该处的引用角标，保留正文，防止误导。 |

> **工程性能折中**：在流式生成场景下，逐句运行交叉编码器（Cross-Encoder）会引入额外计算耗时。生产中通常采用异步校验流水线：前端先流式显示文字，角标显示校验中的占位状态；或仅对高风险、低置信度的回答抽取关键词进行快速校验。

---

## 流式渲染：SSE 切割与前端状态机解析

在 Server-Sent Events (SSE) 流式返回时，大模型输出的 Token 不会恰好按标记边界对齐。引用标记 `[^chunk_a7f9]` 往往会被拆成多个数据包：

*   数据包 1：`3月31日 [^ch`
*   数据包 2：`unk_a7f`
*   数据包 3：`9] 员工应当...`

如果前端每收到一个包就直接送入 Markdown 渲染器，未闭合的 `[^ch` 会破坏 Markdown 语法树，导致页面样式闪烁、文字乱跳。

![流式Token切割与状态机缓冲解析机制](/images/content-rebuild/295b7243df257634/fig-5.png)

### 状态机缓冲解析实现

前端通过维护一个字符缓冲区（Buffer），在匹配到完整标记后再行替换：

```typescript
class CitationStreamParser {
  private buffer: string = "";
  private insideCitation: boolean = false;

  public processChunk(incomingToken: string): { displayText: string; newCitations: string[] } {
    this.buffer += incomingToken;
    let displayText = "";
    const newCitations: string[] = [];

    while (this.buffer.length > 0) {
      if (!this.insideCitation) {
        const startIndex = this.buffer.indexOf("[^");
        if (startIndex === -1) {
          // 没有遇到引用起始符，但要防止结尾刚好是个孤立的 '['
          const safeLength = this.buffer.endsWith("[") ? this.buffer.length - 1 : this.buffer.length;
          displayText += this.buffer.slice(0, safeLength);
          this.buffer = this.buffer.slice(safeLength);
          break;
        } else {
          // 输送 [^ 之前的内容，并将状态切换为标记内
          displayText += this.buffer.slice(0, startIndex);
          this.buffer = this.buffer.slice(startIndex);
          this.insideCitation = true;
        }
      } else {
        const endIndex = this.buffer.indexOf("]");
        if (endIndex === -1) {
          // 引用标记尚未闭合；若缓冲区超过上限（如 64 字符）则判定为非引用文本，回退释放并重置
          if (this.buffer.length > 64) {
            displayText += this.buffer;
            this.buffer = "";
            this.insideCitation = false;
          }
          break;
        } else {
          // 捕获到闭合标记，检查格式
          const fullTag = this.buffer.slice(0, endIndex + 1);
          const match = fullTag.match(/^\[\^(chunk_[a-zA-Z0-9]+)\]$/);
          if (match) {
            newCitations.push(match[1]);
            displayText += ` <cite class="citation-badge" data-id="${match[1]}"></cite> `;
          } else {
            displayText += fullTag; // 格式不符则原样输出文本
          }
          this.buffer = this.buffer.slice(endIndex + 1);
          this.insideCitation = false;
        }
      }
    }
    return { displayText, newCitations };
  }
}
```

当用户点击解析后的 `<cite>` 角标时，前端利用初始化时接收到的切片元数据映射表，调用查看器接口（如 `viewer.navigateTo(page_idx, bbox)`），在原文档画布中绘制半透明遮罩高亮。

---

## 引用质量评估：精确率与召回率算例

引用质量评估通常从“有没有乱标”和“该标的是否都标了”两个维度进行统计。

![引用质量评估算例与混淆分类图](/images/content-rebuild/295b7243df257634/fig-6.png)

### 指标定义

设模型回答拆解为多个独立句子（Claims）：

1. **引用精确率（Citation Precision）**：
   在所有被模型打上了引用的句子中，经人工或 NLI 核验，**确实被该引用切片充分支持（Entailed）**的比例。
   $$P_{\text{cite}} = \frac{\text{标了引用且被证据支持的句子数}}{\text{模型打上引用的句子总数}}$$
   *精确率低代表存在胡乱标注或张冠李戴。*

2. **引用召回率（Citation Recall）**：
   在回答中所有包含事实陈述的句子中，**正确标注了支持出处**的比例。
   $$R_{\text{cite}} = \frac{\text{标了引用且被证据支持的句子数}}{\text{包含事实判断的句子总数}}$$
   *召回率低代表存在凭空叙述事实却漏标出处。*

### 逐步算例

假设模型输出一段包含 4 句话的回答：
*   $S_1$：“公司设立在深圳南山区。”标注了引用 $[\text{Chunk}_1]$。核验：$\text{Chunk}_1$ 明确写有此内容。**（有效支持）**
*   $S_2$：“注册资本为 5000 万元。”标注了引用 $[\text{Chunk}_1]$。核验：$\text{Chunk}_1$ 未提及注册资本。**（标注错误，非充分支持）**
*   $S_3$：“核心团队均毕业于清华北大。”**未标注引用**。核验：该句属于事实陈述，但遗漏了出处。**（漏标出处）**
*   $S_4$：“欢迎随时垂询。”客套话，不包含事实论断，不计入事实陈述集。

代入计算：
*   打上引用的句子集合包含 $S_1, S_2$，共 2 句；其中有效支持的为 $S_1$（1 句）。
    $$P_{\text{cite}} = \frac{1}{2} = 50\%$$
*   包含事实陈述的句子集合包含 $S_1, S_2, S_3$，共 3 句；有效标注且支持的为 $S_1$（1 句）。
    $$R_{\text{cite}} = \frac{1}{3} \approx 33.3\%$$

---

## 面试追问与避坑指南

### 追问 1：模型输出引用角标 `[^chunk_1]`，但切片 1 并不支持该结论，输出网关层如何做降级处理？

**回答要点**：
1. **文本切分与提取**：网关接收完整回复后，利用标点或规则将正文切为单句，定位到包含 `[^chunk_1]` 的具体命题；
2. **运行 NLI 蕴含判定**：以切片 1 原文为前提，以该句为主张，调用 NLI 分类器。若判定为“中立”或“矛盾”：
   *   **常规展示场景（轻度降级）**：剔除 `[^chunk_1]` 引用标记，保留生成文本，或将角标样式替换为灰色的“待核实”，防止用户点击跳入无关文档；
   *   **严谨合规场景（重度拦截）**：若判定为矛盾，直接阻断该句输出，替换为安全兜底话术（如：“关于该事项，参考文档中无明确依据”）；
3. **埋点沉淀**：将该次请求的上下文、问题及生成内容写入异常日志，作为 RAG 微调或 Prompt 优化的 Bad Case 样本。

### 追问 2：流式返回 Markdown 时，引用标记跨多个 Token 被切碎，前端如何保证页面不崩溃、不高频重绘？

**回答要点**：
1. **缓冲队列（Buffer）**：前端收到每个 SSE 数据片时，不立刻交给 DOM 渲染，而是推入解析状态机；
2. **截断特征拦截**：当扫描到引用起始特征 `[^`（并在缓冲区末尾暂存可能未拼完整的 `[`）时挂起后续文字，进入引用解析模式，直到检测到 `]` 闭合符；若累加超过合理上限（如 64 字符）仍未闭合，则判定为普通文本并回退释放；
3. **虚拟占位替换**：解析器将合法的完整标记转换为带固定宽度的 HTML 自定义标签（如 `<span class="cite-badge" ...>`），再统一送入 Markdown 解析库，避免 Markdown 引擎将孤立的标点误识别为链接或代码块。

### 常见错误认知

*   ❌ **错误认知**：“只要把检索到的前 5 个切片挂在问答界面的侧边栏，就是做好了答案溯源。”
    *   *实际情况*：侧边栏展示仅是给出了候选集，用户依旧不知道回答里的每一句话源自哪一篇、哪一页。这没有解决张冠李戴的事实核验成本。
*   ❌ **错误认知**：“在 Prompt 里要求‘回答时必须写出第几页第几行’，大模型就能准确报出行号。”
    *   *实际情况*：自回归大模型对文本在屏幕或纸质上的物理排版没有内置坐标感知。它输出的“第 3 页第 5 行”往往是词表预测出的虚构数字。页码与版面坐标必须在解析入库阶段提取，并由系统显式注入给模型。
