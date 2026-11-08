---
title: Temperature 和 Top-P 参数有什么作用？如何选择合适的值？
icon: mdi:thermostat
tag:
- 采样参数
- Temperature
- Top-P
description: 详解自回归采样中 Temperature 与 Top-P 的数学原理、执行流程、联动边界与各业务场景调参方法，附可验证模拟代码。
---

自回归大语言模型在生成文本时，每个解码步骤输出的并不是最终文字，而是词表中每一个 Token 的未归一化打分（即 Logits）。如何把这一组数值转换成最终选出的下一个 Token，取决于解码采样算法。在实际开发和调用 API 时，Temperature（采样温度）和 Top-P（核采样）是控制生成文本随机性与连贯性的两个最关键参数。

![大模型解码采样机制核心链路概览](/images/content-rebuild/b4d00a1ee9eca435/cover.png)

---

## 1. 面试 30 秒口述答案

如果面试官提问“Temperature 和 Top-P 分别控制什么？在项目中如何选择合适的值？”，可以按如下三个要点清晰口答：

1. **职责与原理分工**：
   - **Temperature（温度）** 调控**概率分布的平坦程度**。通过在 Softmax 前将 Logits 除以参数 $T$，改变高低得分之间的相对悬殊度。$T$ 越低，高分词优势被放大，输出越确定；$T$ 越高，词与词之间的概率差距被抹平，输出越丰富多样。但 Temperature **绝不改变词的大小排序**。
   - **Top-P（核采样）** 负责**动态过滤低概率长尾词**。它将候选词按概率从大到小排列并累加，只保留累积概率刚达到阈值 $P$ 的最小子集，过滤掉其余噪词。保留的候选集合大小会随着模型置信度自适应变化。
2. **执行先后顺序**：
   底层计算流程是**“Logits 经 Temperature 缩放 $\to$ 转为 Softmax 概率 $\to$ 依 Top-P 截断长尾 $\to$ 剩余词重归一化 $\to$ 离散随机抽样”**。
3. **调参实践策略**：
   日常调优建议使用控制变量法，通常**固定 Top-P（如 0.90），优先调节 Temperature**。严谨任务（如代码编写、数学推导、SQL 生成）选用低温（$0.0 \sim 0.2$）；结构化提取或 RAG 事实问答选用中低温（$0.2 \sim 0.4$）；开放对话选用中温（$0.6 \sim 0.7$）；创意文学写作选用高温（$0.8 \sim 1.0$）。避免两者同时大幅变动，以免生成效果失去可控性。

![Temperature平滑分布与Top-P动态核截断关系图](/images/content-rebuild/b4d00a1ee9eca435/fig-1.png)

---

## 2. 贯穿示例与 Temperature 的数学行为

为了直观理解，我们假设模型在输入“今天天气真”之后预测下一个词。词表中得分最高的前三个词及其原始 Logits 如下：
- “好”：$z_1 = 4.0$
- “晴”：$z_2 = 2.0$
- “冷”：$z_3 = 1.0$

### 数学公式与保序性质

引入采样温度 $T > 0$ 后，第 $i$ 个 Token 被选取的概率计算公式为：

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^V \exp(z_j / T)}$$

由于函数 $f(z) = z / T$ 在 $T > 0$ 时严格单调递增，Softmax 同样严格保序，因此：**调节 Temperature 不会改变候选词之间的高低大小关系**。只要原始得分 $z_a > z_b$，无论 $T$ 设为多大或多小，计算出的概率永远满足 $P(w_a) > P(w_b)$。

Temperature 改变的是两个 Token 之间的**相对概率比值**：

$$\frac{P(w_a)}{P(w_b)} = \exp\left(\frac{z_a - z_b}{T}\right)$$

- 当 $z_a > z_b$ 且 $T \to 0^+$ 时，$\frac{z_a - z_b}{T} \to +\infty$，概率比值趋近无穷大。头部分数最高的词会占据压倒性优势。
- 当 $T \to +\infty$ 时，$\frac{z_a - z_b}{T} \to 0$，概率比值趋近于 1。词与词之间的得分差异被彻底抹平。

### 三种典型状态及数值算例

1. **$T \to 0^+$（趋近贪心搜索）**：分母中最大项占据主导地位，最大得分词概率趋向 1.0，其余词概率趋近 0，退化为确定性输出。
2. **$T = 1.0$（标准 Softmax）**：不对模型打分做任何人工缩放，保持训练目标下的概率分布。
3. **$T \to \infty$（均匀分布）**：所有 Token 的概率都趋于 $\frac{1}{V}$，模型输出完全退化为随机噪声。

使用上述贯穿例子的数值（保留四位小数）观察不同温度下的概率演化：

| 候选词 | 原始得分 $z_i$ | $T=0.5$（低温） | $T=1.0$（基准） | $T=2.0$（高温） |
| :--- | :--- | :--- | :--- | :--- |
| **好** | 4.0 | $\frac{\exp(8.0)}{\sum} \approx 0.9796$ | $\frac{\exp(4.0)}{\sum} \approx 0.8438$ | $\frac{\exp(2.0)}{\sum} \approx 0.6285$ |
| **晴** | 2.0 | $\frac{\exp(4.0)}{\sum} \approx 0.0179$ | $\frac{\exp(2.0)}{\sum} \approx 0.1142$ | $\frac{\exp(1.0)}{\sum} \approx 0.2312$ |
| **冷** | 1.0 | $\frac{\exp(2.0)}{\sum} \approx 0.0024$ | $\frac{\exp(1.0)}{\sum} \approx 0.0420$ | $\frac{\exp(0.5)}{\sum} \approx 0.1403$ |

算例表明：降温让分布急剧收拢，“好”的选中概率达到近 98%；升温则把尾部词的概率显著拉起，使“冷”等低分词也有被抽样的机会。

<ReadMoreLock />

![Temperature不同取值下的Logits除法与Softmax概率分布演化](/images/content-rebuild/b4d00a1ee9eca435/fig-2.png)

---

## 3. Top-P（核采样）动态截断机制

### 为什么单纯调节 Temperature 不够？

在开放式生成场景中，如果只依靠调高 Temperature 激发创造力，词表中数以万计的无关词（如错别字、语法畸形词）仍然保留着微小的非零概率。随着生成文本长度增加，多项式采样迟早会偶发命中这些低概率长尾噪词，造成乱码或语句断裂。

早期的方法是使用固定截断数量的 **Top-K 采样**（只保留概率最高的前 $K$ 个词）。但固定的 $K$ 无法适应模型置信度的动态变化：
- **高确定性上下文**：例如输入“中华人民共和国的首都”，下一个词几乎非“北”莫属。若设固定 $K=50$，剩下的 49 个候选词大部分是不合理的噪音。
- **开放语境上下文**：例如输入“走进展厅，迎面摆放着”，合理的展品名词可能有上百种。若设固定 $K=3$，会误杀大量合理的词汇，导致输出单一刻板。

### Top-P 的数学定义与动态伸缩

Top-P（核采样，Nucleus Sampling）通过**累积概率质量**动态决定候选集大小。

设模型输出概率降序排列为：$P(w_{(1)}) \ge P(w_{(2)}) \ge \dots \ge P(w_{(V)})$。给定阈值 $p \in (0, 1]$，定义最小候选集合 $V^{(p)}$ 满足：

$$V^{(p)} = \{ w_{(1)}, w_{(2)}, \dots, w_{(k)} \}$$

其中截断索引 $k$ 是让累积概率刚达到或超过 $p$ 的最小正整数：

$$k = \min \left\{ m \in \{1, \dots, V\} \,\Big|\, \sum_{i=1}^m P(w_{(i)}) \ge p \right\}$$

截断后，集合外的低概率词直接丢弃，对集合内的词执行**重归一化（Re-normalization）**，使其概率和重新为 1：

$$P'(w_{(i)}) = \frac{P(w_{(i)})}{\sum_{j=1}^k P(w_{(j)})}, \quad 1 \le i \le k$$

### 动态核伸缩直观对比

假设我们固定阈值 $p = 0.90$：
1. **陡峭分布（确定性极高）**：若头部词概率为 $[0.95, 0.03, 0.01, \dots]$，第一项累积和即为 $0.95 \ge 0.90$。截断点触发，$k=1$，候选池自动缩减为 1 个词，完全屏蔽长尾噪声。
2. **平坦分布（选择多样化）**：若各词概率为 $[0.25, 0.22, 0.20, 0.18, 0.10, 0.03, 0.02]$，累积求和过程为：$0.25 \to 0.47 \to 0.67 \to 0.85$（未达标），加上第 5 项后累积达 $0.95 \ge 0.90$。截断点为 $k=5$，候选池自动扩容到前 5 个词，保留多样性。

![Top-P在平坦与陡峭分布下的候选集动态伸缩机制](/images/content-rebuild/b4d00a1ee9eca435/fig-3.png)

---

## 4. 解码管线执行顺序与代码实现

在推理框架（如 vLLM、llama.cpp、Hugging Face Transformers）中，采样参数的处理流程具备严格的时序：

```
原始 Logits 输入
      │
      ▼
[1. Temperature 缩放]
      │   Logits' = Logits / T
      ▼
[2. Softmax 概率转换]
      │   P = softmax(Logits')
      ▼
[3. Top-K 截断 (若启用)]
      │   保留前 K 项，其余置零并重归一化
      ▼
[4. Top-P 核截断]
      │   降序累加至累积概率 >= p，截断后续项
      ▼
[5. 概率重归一化]
      │   P' = P / sum(P)
      ▼
[6. 多项式离散采样]
      │   依照 P' 随机抽样产生最终 Token ID
```

> **工程实现注记**：部分追求极致吞吐的推理引擎（如 vLLM 或 llama.cpp 的 GPU 内核），为了减少内存反复搬运与数值误差，会在 Logits 空间直接进行 Top-K 降序选拔，并对被截断的位置赋值 $-\infty$，最后只对过滤后的子集做单次 Softmax 抽样。无论在 Logits 空间还是概率空间过滤，数学结果严格等价。

以下是用 Python/NumPy 编写的可运行模拟实现：

```python
import numpy as np

def sample_next_token(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> int:
    logits = np.array(logits, dtype=np.float64)
    
    # 1. 边界防御：温度趋近于 0 时执行确定性贪心解码
    if temperature <= 1e-5:
        return int(np.argmax(logits))
    
    # 2. Temperature 缩放
    scaled_logits = logits / temperature
    
    # 3. 数值稳定 Softmax (减去最大值防止指数溢出)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # 4. Top-K 过滤 (可选)
    if 0 < top_k < len(probs):
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.ones(len(probs), dtype=bool)
        mask[top_k_indices] = False
        probs[mask] = 0.0
        probs = probs / np.sum(probs)
    
    # 5. Top-P (核采样) 过滤
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # 找到累积概率刚达到或超过 top_p 的截止位置
        cutoff_index = int(np.searchsorted(cumulative_probs, top_p))
        valid_indices = sorted_indices[:cutoff_index + 1]
        
        # 其余截断并重归一化
        truncated_probs = np.zeros_like(probs)
        truncated_probs[valid_indices] = probs[valid_indices]
        probs = truncated_probs / np.sum(truncated_probs)
        
    # 6. 多项式离散抽样
    token_id = np.random.choice(len(probs), p=probs)
    return int(token_id)
```

![Logits从模型输出到最终Token抽样的完整过滤与重归一化时序](/images/content-rebuild/b4d00a1ee9eca435/fig-4.png)

---

## 5. 业务场景调参推荐与实践法则

不同业务场景对文本生成的“准确严谨”与“发散创新”要求不同：

| 业务任务类型 | 推荐 Temperature | 推荐 Top-P | 选参因果解释与预期行为 |
| :--- | :--- | :--- | :--- |
| **代码编写 / SQL 生成 / 数学计算** | $0.0 \sim 0.2$ | $0.90 \sim 0.95$ | 语法结构严密，容错率低。低温大幅抑制低概率噪词，倾向输出确定性高的规范代码与推导。$T=0$ 等价于纯贪心解码。 |
| **RAG 事实问答 / 信息抽取 / 会议纪要** | $0.2 \sim 0.4$ | $0.85 \sim 0.90$ | 输出必须忠于参考文档，避免发散臆造事实；微小的温度保持语句通顺不生硬。 |
| **通用智能客服 / 多轮对话机器人** | $0.6 \sim 0.7$ | $0.90$ | 需要兼顾拟人感和会话多样性，同时严控上下文意图不跑偏。 |
| **文学写作 / 广告文案 / 创意头脑风暴** | $0.8 \sim 1.0$ | $0.92 \sim 0.95$ | 鼓励词汇丰富度与新颖搭配，略微放宽 Top-P 引入更多偶发词汇。 |

### 单变量调节法则

在优化实际项目生成效果时，**切忌同时大幅调节 Temperature 和 Top-P**。如果两个参数同时改变，开发者无法判断文本变差究竟是因为分布过于平坦（Entropy 过高），还是因为长尾过滤过严/过宽。

推荐的调试步骤：
1. **锚定 Top-P**：将其固定在基准线（如 `0.90`）。
2. **步进调节 Temperature**：以 0.1 为步长调整温度。输出机械呆板时逐步提高 $T$；输出出现逻辑混乱或事实错误时逐步调低 $T$。
3. **微调长尾阈值**：如果在合适的创造力水平下，模型偶尔蹦出罕见错字或断句生硬，保持 $T$ 不变，将 Top-P 从 0.90 下调至 0.80 左右，进一步切断危险尾部。基础概念对比可参考 [Temperature、Top-k、Top-p 是什么？](/fundamentals/inference-and-generation/temperature-topk-topp.md) 与 [采样参数对比](/interview/llm-interview/sampling-params.md)。

![不同业务任务类型在Temperature与Top-P参数坐标系中的推荐落点](/images/content-rebuild/b4d00a1ee9eca435/fig-5.png)

---

## 6. 面试深度追问与常见误区

### 追问 1：如果同时启用了 Top-K=50 和 Top-P=0.9，系统是如何进行协同过滤的？

**解答**：
底层框架依次执行两道串联滤网，**取二者交集，以更严苛的条件为准**：
1. 先执行 Top-K：无论分布如何，强制截取概率最高的前 50 个候选词，将其余数万个词置零并重归一化；
2. 再执行 Top-P：在这 50 个词内部按概率从大到小累加。一旦累积和达到 0.9，立即截断后续项。
3. **结论**：最终候选集合大小 $\le \min(K, |V^{(p)}|)$。当语义分散时，候选池至多保留 50 个；当语义高度集中时，候选池可能仅剩 1~2 个。两者配合可防范极端发散场景。

### 追问 2：调大 Temperature 会不会导致原本排名第二的词超越第一名？

**解答**：
**绝对不会**。因为在 $T > 0$ 时，映射 $z_i \mapsto z_i / T$ 是严格保序变换。若原始分数 $z_1 > z_2$，则缩放后必然有 $z_1 / T > z_2 / T$。经过单调递增的指数函数 $\exp(\cdot)$ 和 Softmax 计算后，必然有 $P(w_1) > P(w_2)$。Temperature 改变的是概率差别的绝对差距，但不会逆转候选词的原始名次。

### 追问 3：在程序中把 Temperature 设为 0 会触发除零异常吗？推理系统如何兼容？

**解答**：
直接执行数值计算 $z_i / 0$ 会导致浮点除零（产生 `Inf` 或 `NaN`）。主流推理服务（如 vLLM 或 OpenAI API）在入口层内置了分支兼容：一旦检测到 `temperature == 0`（或低于极小阈值），系统会**直接短路跳过后续的除法、Softmax、Top-P 截断及随机采样**，直接调用 `argmax(logits)`，以确定性贪心模式输出当前得分最大的 Token。而在 Hugging Face Transformers 中，若开启采样则要求 `temperature` 为严格正数，切换为贪心解码需显式设置 `do_sample=False`。

### 常见高频误区辨析

- **误区一：“Temperature 调高能让模型学到更多知识”**：
  这是错误的。模型在推理阶段所有权重早已冻结，Temperature 无法引入新知识，它只是单纯提高了模型选中低置信度候选词的随机概率。盲目将温度设得过高（如 $> 1.5$）只会导致语句支离破碎和无序乱码。
- **误区二：“把 Top-P 设得很低（如 0.2）就能杜绝大模型幻觉”**：
  这是错误的。幻觉的成因在于模型参数内部的关联性认知偏差（参见 [大模型幻觉解析](/interview/llm-interview/hallucination.md)）。如果模型在某个问题上产生错误认知，其预测的错误 Token 本身 Logit 就会排在第一名。此时哪怕将 Top-P 压缩到极致，模型依然会自信地输出错误事实。

![Top-K与Top-P级联过滤时的交集筛选与尾部剔除机制](/images/content-rebuild/b4d00a1ee9eca435/fig-6.png)

---

## 7. 关联学习与知识拓展

深入理解自回归采样机制，是构建高可靠智能体与模型落地调优的基础。建议读者沿着以下知识脉络继续延伸：

- **预训练与自回归原理**：关于模型生成文本的端到端机制，请参阅 [大模型到底是什么？](/interview/llm-interview/what-is-llm.md)。
- **采样参数对比**：关于 Top-K 与贪心采样的更多细则，请参阅 [Temperature、Top-k、Top-p 分别控制什么？](/interview/llm-interview/sampling-params.md) 与 [提示工程采样配置指南](/llm-100/prompt-engineering/Temperature_Top-p_Top-k_分别控制什么.md)。
- **底层推理加速**：关于自回归生成阶段每一步 Logits 计算所涉及的显存缓存机制，请参阅 [KV Cache 原理解析](/interview/llm-interview/kv-cache.md)。
