---
title: 多模态大模型入门指南：从图文检索到视觉语言生成
icon: mdi:image-multiple-outline
tag:
- 多模态基础
- 多模态
- 大模型基础
description: 梳理多模态大模型从双塔向量对齐到自回归视觉语言模型的核心机制、计算流程与工程选型边界。
---

![多模态基础学习指南与知识地图封面](/images/content-rebuild/997fd640b974a0a8/cover.png)

单模态语言模型只能处理离散的文本符号序列。但在实际工程中，大量输入是连续的物理世界信号：图片中的排版、表格的行列跨度、图表的趋势曲线以及各类照片。多模态技术的目标，就是建立一套数学转换方法，将图像、音频等信号转化为大语言模型可以参与计算的向量序列。

本模块梳理从图像向量化、图文对齐到图文自回归生成的完整链路，帮助开发者建立清晰的技术全貌与选型标准。

---

## 贯穿示例：同一张图片的两种处理范式

以一张“草地上奔跑的金毛犬叼着红色飞盘”的照片为例，面对不同的业务诉求，存在两种截然不同的架构设计：

1. **检索与打标签任务（判别式）**：
   - 输入：图像 + 候选文字列表（“狗在玩耍”、“猫在睡觉”、“汽车行驶”）。
   - 输出：各候选标签的匹配概率。
   - 机制：图像与文本分别通过独立编码器提取为一个定长全局向量，通过向量余弦相似度计算匹配分值。这是 CLIP 的典型工作方式。
2. **看图描述与推理任务（生成式）**：
   - 输入：图像 + 提示词（“请指出狗嘴里叼着什么颜色的物品，并描述草坪光照条件”）。
   - 输出：逐字生成的自然语言回答。
   - 机制：图像不能只压缩成单一向量，而是被切分成数十到数百个局部图像块（Patches），每个块被编码并投影成一个“视觉 Token”。这些视觉 Token 与提示词的文本 Token 拼接成一长串序列，送入 Transformer 解码器逐字自回归生成文本。这是 LLaVA、Qwen-VL 等视觉语言模型（VLM）的典型工作方式。

---

## 模块内容导引与知识递进

本模块共包含 7 篇核心文章，按技术层次从底层表征、架构组装逐步推进到工程落地与物理局限：

![多模态知识体系演进拓扑图](/images/content-rebuild/997fd640b974a0a8/fig-1.png)

### 第一阶段：模态表征与跨模态对齐
- [/fundamentals/multimodal-basics/what-is-multimodal.md](/fundamentals/multimodal-basics/what-is-multimodal.md)：从离散文本符号与连续高维信号的物理差异出发，梳理早期特征拼接与现代对齐技术的演进历程。
- [/fundamentals/multimodal-basics/clip.md](/fundamentals/multimodal-basics/clip.md)：详解双塔对比学习（Contrastive Language-Image Pretraining）的损失函数、温度系数缩放与跨模态零样本分类机制。

### 第二阶段：视觉离散化与生成架构组装
- [/fundamentals/multimodal-basics/visual-token.md](/fundamentals/multimodal-basics/visual-token.md)：解析图像网格切分为 Patch、线性映射为视觉 Embedding、添加位置编码的数学过程。
- [/fundamentals/multimodal-basics/multimodal-how.md](/fundamentals/multimodal-basics/multimodal-how.md)：详细梳理视觉编码器（Vision Encoder）、跨模态投影层（Linear/MLP/Resampler）与语言解码器（LLM Backbone）的级联数据流与拼接推理机制。

### 第三阶段：垂直业务落地与工程约束
- [/fundamentals/multimodal-basics/ocr-doc-understanding.md](/fundamentals/multimodal-basics/ocr-doc-understanding.md)：对比传统专用 OCR 流水线、版面还原模型与端到端视觉大模型在格式规范文档中的召回率与计算开销。
- [/fundamentals/multimodal-basics/multimodal-rag.md](/fundamentals/multimodal-basics/multimodal-rag.md)：探讨复杂 PDF 页面截图解析、文本与图像混合切块、多路召回与跨模态重排序工程设计。
- [/fundamentals/multimodal-basics/multimodal-limitations.md](/fundamentals/multimodal-basics/multimodal-limitations.md)：分析 Patch 固定分辨率切分造成的细微文字丢失、缺乏深度线索导致的几何误判，以及大量视觉 Token 挤占 KV Cache 显存的现实瓶颈。

---

## 核心机制对比：CLIP 双塔与自回归 VLM

理解多模态系统的关键分水岭，在于分清“嵌入空间匹配度打分”与“视觉引导的自回归序列生成”的结构差异：

![判别式双塔对齐与生成式大模型架构对比](/images/content-rebuild/997fd640b974a0a8/fig-2.png)

### 1. 判别式双塔架构（以 CLIP 为例）

在 CLIP 中，图像编码器 $f_I$（如 ViT 或 ResNet）将整张图片映射为向量 $z_I \in \mathbb{R}^d$，文本编码器 $f_T$ 将提示文本映射为向量 $z_T \in \mathbb{R}^d$。

两向量经 L2 范数归一化后计算余弦相似度，并乘以一个可学习的缩放参数（Logit Scale）$\tau$：

$$S(I, T) = \tau \cdot \left( \frac{z_I}{\|z_I\|_2} \cdot \frac{z_T}{\|z_T\|_2} \right)$$

- **数据形态**：整幅图像的信息被压缩进一个定长向量（如 512 维或 768 维）。
- **计算优势**：图像与文本可以离线预先计算并存入向量数据库，检索时只需做高维向量内积运算，速度极快。
- **能力边界**：由于缺乏局部序列交互，全局向量会丢失大量细粒度空间拓扑与微小目标细节，无法生成自由文本回答，只能做匹配与分类。

### 2. 生成式视觉语言大模型（以 LLaVA 架构为例）

在端到端生成式模型中，输入图片首先被划分为 $N$ 个固定尺寸的 Patch（例如 $14 \times 14$ 像素为一个 Patch）。

视觉主干网络提取这 $N$ 个块的局部特征矩阵 $X_v \in \mathbb{R}^{N \times d_v}$。随后通过投影矩阵 $W \in \mathbb{R}^{d_v \times d_{llm}}$，将视觉特征对齐到语言模型的词表嵌入维度：

$$H_v = X_v W \in \mathbb{R}^{N \times d_{llm}}$$

- **序列拼接**：若用户输入的提示词包含 $L$ 个文本 Token，其嵌入矩阵为 $H_t \in \mathbb{R}^{L \times d_{llm}}$。系统将视觉嵌入与文本嵌入按顺序拼接：

$$H_{input} = [H_v ; H_t] \in \mathbb{R}^{(N + L) \times d_{llm}}$$

- **自回归推理**：将整个拼接序列输入标准 Transformer 解码器。在多头注意力机制中，后续生成的每一个文本 Token 都能跨模态关注到所有视觉 Token 与先前的文本 Token。
- **工程代价**：每个视觉 Token 都会在推理期间占用解码器的注意力上下文与 KV Cache。例如输入一张图片产生 576 个视觉 Token，即使只提问一个字，也需要先执行这 576 个 Token 的 Prefill 计算并在自回归阶段持续占用显存缓存。

---

## 落地选型决策：专用 OCR 还是多模态 VLM？

在设计文档问答与复杂报表解析系统时，并非所有场景都适合直接调用通用视觉语言大模型：

![文档理解与多模态系统落地选型决策树](/images/content-rebuild/997fd640b974a0a8/fig-3.png)

### 场景 1：票据、合同与高密度数值提取
- **业务特征**：发票金额、身份证号、银行对账单，对单个字符的准确率要求极高（差错容忍度低），且需要回溯字符在原始页面中的具体坐标边界（Bounding Box）。
- **推荐方案**：采用成熟的**专用 OCR 引擎 + 规则提取 / 版面分析模型（如 LayoutLM）**。
- **原因**：专用 OCR 输出字符边界框与明确置信度，能直接编写校验规则（如借贷平账校验）。若直接用纯生成式 VLM，模型容易受到统计语言先验影响，在模糊像素处“脑补”或错位生成数值。

### 场景 2：开放式排版图表分析与多模态知识问答
- **业务特征**：包含折线图趋势分析、海报图文整体含义理解、多模态图文混排摘要等无法通过简单正则表达式抽取的语义任务。
- **推荐方案**：采用**端到端视觉语言模型（VLM） + 输出结构化解析（JSON Schema / Pydantic） + 后置规则校验**。
- **原因**：VLM 具备对整幅图排版拓扑和语义关系的泛化理解能力，能够直接回答自然语言提问；通过后置规则引擎拦截关键数值，可平衡语义推断能力与数据可靠性。
