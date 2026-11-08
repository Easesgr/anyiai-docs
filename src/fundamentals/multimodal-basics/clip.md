---
title: CLIP 是什么？图像和文本如何对齐到同一向量空间
icon: mdi:image-text
tag:
- 多模态基础
- CLIP
- 图文对齐
description: 解析 CLIP 的双塔架构与对比学习原理，通过计算算例与官方代码说明图文如何对齐到统一向量空间，以及零样本分类与跨模态检索的具体实现。
---

在[多模态模型是什么？](/fundamentals/multimodal-basics/what-is-multimodal.md)中，我们了解到不同模态的数据可以通过编码器映射到同一个语义向量空间中。CLIP（Contrastive Language-Image Pre-training，对比语言-图像预训练）正是将这一想法落地的关键模型。

一句话概括 CLIP 的核心功能：**它把图像和文本分别转化为两个固定长度的特征向量，使得内容相关的图文向量夹角尽可能小，无关的图文向量夹角尽可能大。** 这种能力让模型不需要重新微调训练，就能用自然语言提示直接对图片做零样本分类，或者通过文字检索图像。

![CLIP双塔对比学习与统一嵌入空间概览](/images/content-rebuild/641e811ba57f914e/cover.png)

上图展示了 CLIP 的基本流程：一张照片和一段描述分别穿过图像编码器与文本编码器，最终投影到同一个超球面上计算余弦相似度。为了看清它为什么这样设计，我们需要先了解传统视觉分类遇到了什么问题。

## 传统图像分类的问题：固定标签切断了语言关联

在 CLIP 出现之前，主流的计算机视觉分类模型（如 ResNet）使用的是封闭类别的监督学习。模型在特征提取层之后接入一个固定维度为 $K$ 的全连接分类头，再通过 Softmax 函数输出每个类别的预测概率：

$$P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{f}(\mathbf{x}))}{\sum_{k=1}^K \exp(\mathbf{w}_k^T \mathbf{f}(\mathbf{x}))}$$

这种经典模式在落地时会遇到三个明显限制：

1. **标签彼此孤立，丢失语义关联**：分类损失函数将标签看作互斥的 One-hot 向量。在欧氏空间中，任意两个不同类别的 One-hot 向量距离都是 $\sqrt{2}$，内积均为 0。模型无法从标签本身得知“金毛寻回犬”和“拉布拉多”属于近亲，而与“微波炉”毫无关系。标签背后的自然语言常识被完全切断了。
2. **输出空间锁死，新增类别代价高**：模型的输出维度 $K$ 是固定的。如果要新增第 $K+1$ 个类别，就必须修改分类头的权重矩阵，重新收集对应标注图片并进行微调训练，无法直接利用新类别的文字名称。
3. **人工标注成本高昂**：清洗并人工标注几百万张图片成本极高，而且学术数据集中的图片往往缺少真实网络环境中的复杂背景与多样性。

![传统固定类别分类头与CLIP开放词表对齐机制的结构对比](/images/content-rebuild/641e811ba57f914e/fig-1.png)

如上图所示，CLIP 改变了思路：它不再让图像去匹配一个孤立的数字编号，而是让图像直接去匹配一段自然语言文本，用文本编码器生成的向量充当“动态分类权重”，从而打破了固定类别的限制。

## 双塔架构：视觉与文本编码器如何独立计算

为了同时处理图像和文本，CLIP 采用了被称为“双塔（Two-Tower）”的双编码器结构。两个编码器各司其职，在计算上完全独立：

- **视觉编码器（Visual Encoder, $f_v$）**：负责将输入图片 $I$ 转换为特征向量 $\mathbf{z}_v$。官方提供了两种主流结构：卷积网络 ResNet（如 ResNet-50、ResNet-101）以及视觉 Transformer（如 ViT-B/32、ViT-B/16、ViT-L/14）。以 ViT-B/32 为例，输入的图片被切分成 $32 \times 32$ 像素的 Patch 序列，经过自注意力层提取特征后，取 `[CLS]` 标记对应的隐藏状态或使用注意力池化输出全局视觉向量。
- **文本编码器（Text Encoder, $f_t$）**：负责将输入的自然语言描述 $T$ 转换为特征向量 $\mathbf{z}_t$。CLIP 使用标准的 Transformer 解码器架构（带有因果自注意力掩码），词表大小为 49,152，输入文本序列长度上限为 77 个 [Token](/interview/llm-interview/token-basics.md)。句子两端分别包裹 `[SOS]`（起始符）与 `[EOS]`（终止符），最终提取 `[EOS]` 处的激活向量作为整个句子的全局文本表示。

<ReadMoreLock />

![CLIP双编码器特征提取与线性投影流向](/images/content-rebuild/641e811ba57f914e/fig-2.png)

由于视觉编码器和文本编码器输出的原始向量维度可能不同（例如视觉主干输出 768 维，文本主干输出 512 维），CLIP 分别使用线性投影矩阵 $W_v$ 和 $W_t$ 将两者的特征统一映射到维度为 $d$ 的空间（例如 $d = 512$），并进行 $L_2$ 范数归一化：

$$\mathbf{u}_i = \frac{W_v f_v(I_i)}{\|W_v f_v(I_i)\|_2}, \quad \mathbf{v}_i = \frac{W_t f_t(T_i)}{\|W_t f_t(T_i)\|_2}$$

归一化后，两个向量的内积直接等于它们在高维超球面上的**余弦相似度**（Cosine Similarity）：

$$\text{sim}(I_i, T_j) = \mathbf{u}_i \cdot \mathbf{v}_j = \cos(\theta_{i,j}) \in [-1, 1]$$

双塔解耦带来了显著的工程优势：在构建海量图片搜索引擎时，系统可以提前对库存图片进行离线编码，将算好的归一化向量存入向量数据库。当用户输入检索文字时，只需要对这一句文字做一次文本编码，随后在数据库中执行向量内积搜索即可，无需在图像和文本之间运行耗时的成对跨注意力（Cross-Attention）计算。

## 对比学习机制：如何用批次内的图文对计算损失

CLIP 并没有让模型逐字生成一段图片的描述，而是把它转变成一个判别对比任务。在训练时，系统从 4 亿个公开网页图文对中采出一个包含 $N$ 组图文对的批次（Batch）：$\{(I_1, T_1), (I_2, T_2), \dots, (I_N, T_N)\}$。

![批次内相似度矩阵计算与正负样本配对机制](/images/content-rebuild/641e811ba57f914e/fig-3.png)

两塔分别对这 $N$ 张图片和 $N$ 段文本编码归一化后，得到图像特征矩阵 $U \in \mathbb{R}^{N \times d}$ 与文本特征矩阵 $V \in \mathbb{R}^{N \times d}$。将它们做矩阵相乘，便得到一个 $N \times N$ 的相似度矩阵：

1. **正样本对**：对角线上的 $N$ 个元素 $(i, i)$，对应原本就成对匹配的图片与文本描述。
2. **负样本对**：非对角线上的 $N(N-1)$ 个元素 $(i, j) \; (i \neq j)$，代表批次内互相错位的图片和文本，作为天然的负样本监督信号。

为了调节相似度数值的动态范围并控制 Softmax 概率的平滑程度，CLIP 设置了一个可学习的缩放参数 $\tau$（代码中通过优化参数 $\text{logit\_scale}$ 实现，其中缩放因子为 $\exp(\text{logit\_scale}) = \frac{1}{\tau}$）：

$$S_{i, j} = \frac{\mathbf{u}_i \cdot \mathbf{v}_j}{\tau} = \mathbf{u}_i \cdot \mathbf{v}_j \cdot \exp(\text{logit\_scale})$$

训练损失由两个对称的 InfoNCE 交叉熵共同组成：

**图像到文本（Image-to-Text）损失**：以每一张图片为基准（按行计算），使其与正确文本的匹配得分在同行中最高：

$$\mathcal{L}_{\text{i2t}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i, i})}{\sum_{j=1}^N \exp(S_{i, j})}$$

**文本到图像（Text-to-Image）损失**：以每一段文本为基准（按列计算），使其与正确图片的匹配得分在同列中最高：

$$\mathcal{L}_{\text{t2i}} = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(S_{j, j})}{\sum_{i=1}^N \exp(S_{i, j})}$$

**最终总损失**为两者的平均值：$\mathcal{L} = \frac{1}{2} (\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}})$。

### 逐步计算示意（$N=2$ 最小算例）

我们假设批次大小 $N=2$，缩放因子设为 $\frac{1}{\tau} = 10$。假设两组图文编码归一化后的余弦相似度矩阵为：

$$\text{CosSim} = \begin{bmatrix} 0.8 & 0.1 \\ 0.2 & 0.7 \end{bmatrix}$$

乘上缩放系数 10 得到 Logits 矩阵 $S$：

$$S = \begin{bmatrix} 8.0 & 1.0 \\ 2.0 & 7.0 \end{bmatrix}$$

1. **计算图像到文本损失（按行归一化）**：
   - 第 1 行（图片 1）：$P(T_1 \mid I_1) = \frac{\exp(8.0)}{\exp(8.0) + \exp(1.0)} = \frac{2980.96}{2980.96 + 2.72} \approx 0.99909$
   - 第 2 行（图片 2）：$P(T_2 \mid I_2) = \frac{\exp(7.0)}{\exp(2.0) + \exp(7.0)} = \frac{1096.63}{7.39 + 1096.63} \approx 0.99331$
   - $\mathcal{L}_{\text{i2t}} = -\frac{1}{2}(\ln(0.99909) + \ln(0.99331)) \approx -\frac{1}{2}(-0.00091 - 0.00671) = 0.00381$

2. **计算文本到图像损失（按列归一化）**：
   - 第 1 列（文本 1）：$P(I_1 \mid T_1) = \frac{\exp(8.0)}{\exp(8.0) + \exp(2.0)} = \frac{2980.96}{2980.96 + 7.39} \approx 0.99753$
   - 第 2 列（文本 2）：$P(I_2 \mid T_2) = \frac{\exp(7.0)}{\exp(1.0) + \exp(7.0)} = \frac{1096.63}{2.72 + 1096.63} \approx 0.99753$
   - $\mathcal{L}_{\text{t2i}} = -\frac{1}{2}(\ln(0.99753) + \ln(0.99753)) \approx 0.00247$

3. **总损失**：$\mathcal{L} = \frac{0.00381 + 0.00247}{2} = 0.00314$。

在反向传播驱动下，模型会不断拉近对角线正样本的距离，同时推开非对角线负样本的距离。

### 为什么对比学习依赖超大 Batch Size？
在损失函数的分母中，负样本数量直接取决于批次大小 $N$。如果 $N$ 设置很小（例如 32）：
- 负样本太少，高维空间中模型只需要掌握非常粗浅的特征（比如只看画面主色调）就能把正负样本区分开，无法逼迫编码器提取精细的语义特征；
- 偶发出现的假负样本（即偶然分入同一批次的两组图文恰好语义相近）会带来更大的梯度扰动。

因此，CLIP 在预训练中采用了高达 $32,768$ 的超大 Batch Size，在单次更新中为每张图片提供了数万个候选负样本，以此保证学到的特征具备稳健的语义分辨力。

## 零样本预测的五个步骤：如何用文本做分类

预训练完成后的 CLIP 可以直接执行**零样本分类（Zero-shot Classification）**，其本质是把分类问题转换成了图文匹配问题。

![CLIP零样本分类工作流程分解](/images/content-rebuild/641e811ba57f914e/fig-4.png)

针对包含 $K$ 个分类标签的下游数据集，零样本分类的执行步骤如下：

1. **构建提示词模板**：如果直接输入孤立词汇（如 `cat`），由于训练语料多为完整句子，单个词缺乏语境且容易产生歧义。通常使用自然语言模板将其封装为完整句子，例如 `"a photo of a {class}"`。
2. **离线编码文本分类库**：将 $K$ 个扩展后的文本提示输入文本编码器并做 $L_2$ 归一化，构成一个尺寸为 $K \times d$ 的特征矩阵 $W_{\text{zero-shot}}$。这一步可以提前离线计算完成。
3. **编码输入图片**：待预测图片经过预处理后输入视觉编码器，提取并归一化为 $1 \times d$ 的向量 $\mathbf{u}$。
4. **矩阵相乘计算相似度**：计算向量内积并乘以温度缩放常数（官方代码中通常固定乘以 100.0）：$\text{logits} = 100.0 \times \mathbf{u} W_{\text{zero-shot}}^T \in \mathbb{R}^{1 \times K}$。
5. **Softmax 归一化输出**：对 Logits 应用 Softmax，得到该图像在各类别上的概率分布，选取概率最高的标签作为预测结果。

使用官方 `clip` 库复现零样本预测的完整代码如下：

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# 1. 加载模型与预处理流水线
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. 准备候选分类标签并使用模板封装
classes = ["cat", "dog", "car", "airplane"]
text_prompts = [f"a photo of a {c}" for c in classes]
text_tokens = clip.tokenize(text_prompts).to(device)  # 形状: [4, 77]

# 3. 构造单张测试图像（这里使用单色图像示例）
image = preprocess(Image.new("RGB", (224, 224), color="red")).unsqueeze(0).to(device)

with torch.no_grad():
    # 提取特征
    image_features = model.encode_image(image)       # 形状: [1, 512]
    text_features = model.encode_text(text_tokens)   # 形状: [4, 512]

    # 执行 L2 归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算余弦相似度并转换为概率分布
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = similarity.cpu().numpy()[0]

for cls_name, prob in zip(classes, probs):
    print(f"{cls_name}: {prob * 100:.2f}%")
```

## 典型应用场景与能力边界

CLIP 构建的对齐向量空间为多模态领域提供了许多基础能力，但也存在明确的适用边界。

![CLIP工程落地三类核心场景示意](/images/content-rebuild/641e811ba57f914e/fig-5.png)

### 三类常见工程应用
1. **跨模态以文搜图**：文本与图片共享同一空间，用户输入检索词后，计算查询向量与全库图片向量的内积，即可完成毫秒级匹配。这也是现代[多模态 RAG 怎么做？](/fundamentals/multimodal-basics/multimodal-rag.md)构建图像召回的基础。
2. **数据集无监督清洗**：利用预设的提示词（如 `"a low quality blurry photo"`），可以快速评估图片与低质量描述的相似度，自动剔除爬虫抓取的无效样本。
3. **多模态下游任务的特征提取器**：Stable Diffusion 使用 CLIP 的文本编码器将生成提示词转化为引导向量；早期开源多模态大模型（如 LLaVA）也使用 CLIP 的视觉主干来提取全图基础特征。

![CLIP能力边界与典型失效模式结构分解](/images/content-rebuild/641e811ba57f914e/fig-6.png)

### CLIP 的能力边界与常见失效场景
CLIP 本质上是一个**判别式度量模型**，它并不包含可以自回归输出单词的语言生成器，这决定了它在以下任务中表现受限：

1. **不能主动生成连续文本**：CLIP 只能给“已有文本”和“已有图片”之间的匹配程度打分，无法在看到图片后输出长段描述或回答复杂问题。如果需要自由问答，必须接入大语言模型，具体机制可参考[多模态大模型怎么工作？](/fundamentals/multimodal-basics/multimodal-how.md)。
2. **词袋效应与空间关系识别较弱**：CLIP 对句子中的关键词十分敏感，但对语序和物体绑定关系的理解不够严密。例如，对于 `"a red cube on a blue cylinder"`（蓝色圆柱上的红色立方体）与 `"a blue cube on a red cylinder"`（红色圆柱上的蓝色立方体），CLIP 往往给出非常相近的得分，难以区分主体与客体的颜色归属。
3. **细粒度计数与几何定位能力不足**：因为视觉编码器最终会把整张图片汇总成一个一维向量，物体的坐标、绝对数量和边界信息在池化过程中大部分被压缩掉了。它很难准确判断画面中究竟有 3 只猫还是 4 只猫，也无法输出检测框。
4. **文本长度限制为 77 个 Token**：固定的上下文窗口限制了它处理长段研报或密集说明文字的能力。对于包含文字排版的复杂图表，需要借助更专门的解析架构，可参考[OCR、图表理解和文档理解有什么区别？](/fundamentals/multimodal-basics/ocr-doc-understanding.md)。

为了让大模型在理解图像时不丢失空间坐标与细节，多模态架构并没有止步于单个整图向量，而是将图像拆解为保留空间结构的离散信息单位，这就是我们下一篇要讨论的主题：[视觉 Token 是什么？](/fundamentals/multimodal-basics/visual-token.md)。
