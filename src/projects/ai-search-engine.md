---
title: 如何构建实时 AI 搜索引擎：从查询重写到来源归因
icon: mdi:magnify
description: 介绍实时 AI 搜索引擎的端到端实现：包含多轮查询重写、异步网络抓取、网页正文清洗、上下文预算裁剪与内联引用校验，附代码与验收方案。
---

在构建基于私有静态知识库的检索增强系统（如 [/projects/ai-knowledge-base.md](/projects/ai-knowledge-base.md)）时，文档通常已经预先切块、向量化并持久化于本地向量库中。然而，面对开放域动态资讯、突发事件与实时行情，静态知识库无法涵盖最新的互联网变化，大模型预训练权重也存在固定的知识截止时间。

实时 AI 搜索引擎的作用，是在接收到用户的自然语言提问后，在线检索互联网、抓取异构网页、提取有效证据，并在数秒内生成带有精确来源引用的回答。

以一个典型的多轮体育赛事问答为例：用户在第一轮询问赛事对阵，在第二轮追问“他拿了昨晚的冠军吗？”。系统需要将代词“他”消解为具体球员，将“昨晚”结合系统时间换算为真实日期，向搜索引擎发起检索，解析网页，并在回答“阿尔卡拉斯直落三盘卫冕成功 [1]”时，标明来源证据。

![实时 AI 搜索引擎全链路架构图](/images/content-rebuild/aecc1d43b5369679/cover.png)

---

### 系统处理流程与输入输出规范

商业搜索引擎通常只返回百字左右的片段（Snippet），这些片段往往碎片化且包含 SEO 堆砌词，直接作为上下文容易导致模型产生事实幻觉。因此，完整的检索问答流程通常分为五个连续阶段：

1. **查询理解与重写**：接收多轮历史对话与系统当前时间戳，消除代词和相对时间，输出 1 到 3 个独立的关键词搜索表达式。
2. **在线检索与调度**：调用开源元搜索引擎（如 SearXNG）或专用搜索 API，获取相关候选网页的 URL 列表。
3. **异步并发抓取与正文清洗**：并发拉取候选页面 HTML，剥离导航栏、广告和脚本，提取纯文本段落。
4. **证据分块与上下文裁剪**：在给定的模型上下文 Token 预算内，对段落按单网页采样配额和预算上限进行裁剪装填，构建候选证据池。
5. **带引用的文本生成与后置校验**：模型依据证据池回答问题并在句末标注引用序号（如 `[1]`），系统在后处理阶段核验陈述与证据的词法支持度。

![AI 搜索引擎全链路处理流程](/images/content-rebuild/aecc1d43b5369679/fig-1.png)

下文依次给出各阶段的具体实现与代码。

<ReadMoreLock />

---

### 阶段一：多轮对话与模糊查询的搜索词重写

在多轮交互中，用户输入经常出现代词省略和相对时间。例如：

> 用户第 1 轮：2024 年温网男单决赛是谁和谁打？  
> 助手第 1 轮：2024 年温布尔登网球锦标赛男单决赛在阿尔卡拉斯与德约科维奇之间进行。  
> 用户第 2 轮：他拿了昨晚的冠军吗？  

如果直接拿“他拿了昨晚的冠军吗？”调用搜索 API，检索系统无法确定主体与时间。必须通过大语言模型将其重写为包含明确主体与时间范围的检索词。

![多轮对话指代消解与查询重写机制对比](/images/content-rebuild/aecc1d43b5369679/fig-2.png)

#### 输入输出定义
- **输入**：系统基准时间戳（如 `2024-07-15 10:00:00`）、最近 3 到 5 轮历史对话列表、用户当前自然语言提问。
- **输出**：JSON 对象，包含思考说明和 1 到 3 个适合通用搜索引擎的关键词列表。

#### 实现代码

```python
import json
from datetime import datetime

REWRITE_SYSTEM_PROMPT = """你是一个搜索词改写助手。请根据对话历史和当前基准时间，分析用户的最新输入。
1. 消除代词指代（如“他”、“这家公司”等换成具体实体）。
2. 将相对时间（如“昨天”、“本周”、“去年”）换算为具体时间或年份。
3. 生成 1 到 3 个适合网络搜索引擎的关键词组合，多个关键词用空格分隔。
4. 严格输出合法的 JSON 格式，不要输出其他解释文字。

输出格式示例：
{
  "reasoning": "用户询问阿尔卡拉斯在2024年7月14日温网男单决赛的结果",
  "queries": ["2024 温网 男单决赛 冠军 结果", "阿尔卡拉斯 德约科维奇 温网决赛 战报"]
}
"""

def build_rewrite_prompt(history: list[dict], current_query: str) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    return (
        f"【基准时间】: {now_str}\n"
        f"【对话历史】:\n{history_text}\n\n"
        f"【当前提问】: {current_query}"
    )
```

重写之后，模型生成的关键词能够独立承载语义，避免了搜索接口因上下文缺失返回不相关网页。

---

### 阶段二：搜索引擎调用与异步网页抓取

获取搜索词后，系统向搜索服务请求候选网页。在实际部署中，可以使用自建的开源元搜索引擎 SearXNG，也可以调用商业搜索 API。

为了保证事实的准确性，系统不能止步于搜索引擎提供的百字摘要，而需要并发拉取排名前列的网页原始 HTML。

![搜索引擎调度与异步网页抓取](/images/content-rebuild/aecc1d43b5369679/fig-3.png)

#### 实现代码

```python
import asyncio
import httpx

async def search_searxng(query: str, searxng_url: str = "http://localhost:8080") -> list[dict]:
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
        "language": "zh-CN"
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{searxng_url}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        # 提取前 5 个结果的 URL 和标题
        return [
            {"url": item["url"], "title": item["title"], "snippet": item.get("content", "")}
            for item in data.get("results", [])[:5]
        ]

async def fetch_single_page(client: httpx.AsyncClient, url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = await client.get(url, headers=headers, timeout=4.0, follow_redirects=True)
        if resp.status_code == 200:
            return {"url": url, "html": resp.text, "error": None}
        return {"url": url, "html": "", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"url": url, "html": "", "error": str(e)}

async def batch_fetch_pages(urls: list[str]) -> list[dict]:
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [fetch_single_page(client, url) for url in urls]
        return await asyncio.gather(*tasks)
```

每个外部请求必须配置严格的超时限制（如 4 秒），避免个别响应缓慢的第三方站点拖垮整体问答的端到端响应时间。

---

### 阶段三：网页正文提取与噪声剔除

抓取到的原始 HTML 体积通常在几十 KB 到数 MB 不等，包含导航栏、版权声明、页脚链接、CSS 样式、JavaScript 脚本和广告弹窗。如果直接拼入提示词，会挤占宝贵的上下文窗口，并干扰模型对核心事实的理解。

![网页正文提取与噪声剔除前后结构对比](/images/content-rebuild/aecc1d43b5369679/fig-4.png)

#### 清洗流程与实现

正文抽取采用基于 DOM 树密度分析的工具库（如 `trafilatura`），执行以下清洗步骤：
1. 剔除 `<script>`、`<style>`、`<noscript>`、`<svg>` 等非文本标签。
2. 依据容器标签名、类名及 ID 模式，过滤页眉、页脚、侧边栏和评论区。
3. 计算文本密度与超链接密度，保留正文段落。
4. 输出去除多余空行的紧凑纯文本。

```python
import trafilatura

def extract_clean_text(raw_html: str, url: str) -> str:
    if not raw_html:
        return ""
    extracted = trafilatura.extract(
        raw_html,
        url=url,
        include_links=False,
        include_images=False,
        output_format="txt",
        no_fallback=False
    )
    if not extracted:
        return ""
    lines = [line.strip() for line in extracted.split("\n") if line.strip()]
    return "\n".join(lines)
```

经过清洗后，异构网页统一转换为以自然段落组织的无格式文本。

---

### 阶段四：Token 预算控制下的证据裁剪与装填

清洗后的 5 个网页文本总字数可能仍达到数万字，超出单次生成的最佳证据长度。在向生成模型提交提示词前，必须依据 Token 预算进行段落分块与筛选。

在中文大语言模型中，1 个汉字通常编码为 1 到 1.5 个 Token（具体转换比例取决于模型词表，此处取工程经验估算值 1.3 作为安全参考上限）。在没有加载本地完整分词器时，可用字符数乘以 1.3 估算预算占用。

![动态上下文裁剪与预算控制机制](/images/content-rebuild/aecc1d43b5369679/fig-5.png)

#### 裁剪与装填规则
1. **段落拆分**：将清洗后的网页拆分为 100 到 300 字的独立段落。
2. **单网页限额**：为避免某个长网页垄断全部上下文，同一 URL 最多选取前 2 个关键段落。
3. **预算装填**：设定证据池预算（例如 2500 Token），按顺序贪心填充，并赋予全局唯一的证据编号（如 `[1]`, `[2]`）。

```python
def pack_evidence_passages(
    cleaned_pages: list[dict],
    max_tokens: int = 2500
) -> list[dict]:
    evidence_pool = []
    counter = 1
    current_tokens = 0
    
    for page in cleaned_pages:
        text = page.get("cleaned_text", "")
        if not text:
            continue
        # 筛选有实际内容的段落
        paragraphs = [p for p in text.split("\n") if len(p) >= 30]
        
        # 单网页限制最多采纳 2 段
        for p in paragraphs[:2]:
            est_tokens = int(len(p) * 1.3)
            if current_tokens + est_tokens > max_tokens:
                break
            evidence_pool.append({
                "id": counter,
                "url": page["url"],
                "title": page["title"],
                "text": p
            })
            counter += 1
            current_tokens += est_tokens
            
    return evidence_pool
```

经过裁剪，组装出既不超限、来源又相对均衡的事实证据集合。

---

### 阶段五：内联引用生成与后置校验

AI 搜索与常规生成的最大区别是回答具备可追溯性。回答中的每个关键事实断言后，都必须附带对应证据的编号（如 `[1]`）。模型输出完成后，系统还要执行引用核验，筛查编造引用的情况。

![内联引用生成与归因校验机制](/images/content-rebuild/aecc1d43b5369679/fig-6.png)

#### 引用生成的提示词约束

```markdown
你是一个严格基于检索证据回答问题的研究助手。请仅根据下列参考资料回答用户问题。

【回答规则】：
1. 每个关键事实陈述句末尾必须标注对应的资料编号，格式为 [数字]，例如：“阿尔卡拉斯以 3-0 战胜德约科维奇 [1]。”
2. 严禁编造资料中不存在的事实。若资料不足以回答，明确说明“检索资料未提及”。
3. 如果单个断言来自多个资料，合并标注，如 [1][2]。
4. 正文结束后，附上【参考来源】列表，包含编号、标题与 URL。

【参考资料】：
[1] 标题: 2024温网男单决赛战报
URL: https://example.com/wimbledon-final
内容: 2024年7月14日，卡洛斯·阿尔卡拉斯以6-2、6-2、7-6战胜德约科维奇，蝉联温网男单冠军。

[2] 标题: 网球大满贯记录统计
URL: https://example.com/records
内容: 本次决赛后德约科维奇大满贯冠军总数依然为24座。

【用户问题】：
阿尔卡拉斯卫冕温网了吗？比分是多少？
```

#### 引用支持度的后置核验（Post-hoc Verification）

部分生成结果可能出现“假引用”：即模型在句末标注了 `[1]`，但对应资料中根本没有包含相关词汇。虽然精准的语义蕴含核验需要更重型的自然语言推理（NLI）模型，但在工程流水线中，可以使用字符 2-gram（Bi-gram）特征交集进行轻量级的词法支持度初筛。

```python
import re

def extract_bigrams(text: str) -> set[str]:
    """提取文本的连续双字符片段，过滤纯空白字符，用于词法重合度计算"""
    clean_text = re.sub(r'\s+', '', text)
    if len(clean_text) < 2:
        return set(clean_text)
    return {clean_text[i:i+2] for i in range(len(clean_text) - 1)}

def verify_citations(answer: str, evidence_pool: list[dict]) -> dict:
    evidence_map = {item["id"]: item["text"] for item in evidence_pool}
    # 匹配句子和紧跟其后的引用标注，如：比赛由阿尔卡拉斯获胜。[1]
    sentence_pattern = re.compile(r'([^。！？\n]+[。！？]?)\s*((?:\[\d+\])+)')
    matches = sentence_pattern.findall(answer)
    
    audit_details = []
    for sentence, cite_block in matches:
        cite_ids = [int(n) for n in re.findall(r'\[(\d+)\]', cite_block)]
        sent_grams = extract_bigrams(sentence)
        if not sent_grams:
            continue
            
        for cid in cite_ids:
            if cid not in evidence_map:
                audit_details.append({"status": "INVALID_ID", "id": cid, "sentence": sentence})
                continue
                
            source_text = evidence_map[cid]
            source_grams = extract_bigrams(source_text)
            
            # 计算断言句与证据源段落的二元字符覆盖重合比率
            matched = sent_grams & source_grams
            overlap_ratio = len(matched) / len(sent_grams)
            
            # 若重合度低于 0.25，标记为弱支撑告警，提示可能存在幻觉引用
            if overlap_ratio < 0.25:
                audit_details.append({
                    "status": "LOW_SUPPORT",
                    "id": cid,
                    "sentence": sentence,
                    "overlap": round(overlap_ratio, 2)
                })
            else:
                audit_details.append({"status": "VERIFIED", "id": cid})
                
    return {"total_citations": len(matches), "audit_details": audit_details}
```

该校验不能替代深层语义理解，但能在毫秒级计算内有效过滤掉编号越界、段落完全无关的明显错误，并为前端展示提供可信度提示。

---

### 事实冲突与权威度处理策略

在开放域互联网中，不同网页针对同一事件的陈述可能出现矛盾。例如在突发事件初期，不同新闻源报道的统计数据存在偏差。处理事实冲突的工程原则包括：

1. **显式展示分歧**：在系统提示词中明确要求，若检索到的多个来源对同一数值或结论陈述不一致，必须在回答中客观呈现各方分歧（例如：“据来源 [1] 报道……；而来源 [2] 则报道……”），禁止自行捏造一致数据。
2. **时效性优先**：提取网页的发布时间元数据（如 HTML 中的 `article:published_time`）。针对时效性要求高的查询，将更新时间更近的网页排在前列。
3. **域名权重初筛**：建立可配置的权威信源域名列表。在证据池装填时，优先保留来自官方机构和权威发布平台的段落。

---

### 系统验收与测试用例

系统搭建完成后，应依据以下关键指标和测试用例进行验收测试：

| 测试项目 | 输入案例 | 预期处理结果 | 自动化验收标准 |
| :--- | :--- | :--- | :--- |
| **多轮指代消解** | 历史: “苹果公司发布了 Vision Pro”<br>提问: “它的售价是多少？” | 重写为含“苹果 Vision Pro”、“官方售价”的独立检索词 | 重写搜索词中不包含代词“它”，且通过接口能查到具体价格 |
| **相对时间锚定** | 提问: “昨晚欧冠谁赢了？”（假设系统日期为 2024-06-02） | 重写为包含前一日实际日期（如 2024年6月1日）与欧冠决赛的相关检索词 | 搜索参数包含换算后的具体日期字符串，非字面“昨晚” |
| **HTML 噪声清洗** | 包含导航栏、侧边栏广告、版权信息的复杂 HTML 页面 | 抽取后仅保留正文段落 | 结果无 `<script>`、`<nav>` 等标签，且样板 DOM 特征词过滤率达标 |
| **内联引用完整性** | “阿尔卡拉斯卫冕温网了吗？” | 回答文本包含对应证据编号（如 `[1]`），且附带对应来源列表 | 正则匹配校验通过，无 `INVALID_ID`，后置引用词法覆盖率高于阈值 |
