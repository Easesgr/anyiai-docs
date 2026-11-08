---
title: PDF 解析为什么容易乱码与串行？文本、表格和版面如何正确还原？
icon: mdi:file-pdf-box
tag:
- 文档解析
- PDF
- 版面识别
- 表格提取
description: 系统分析 PDF 内部排版指令与坐标体系，解释乱码、串行、表格崩解的原因，并给出几何重构、视觉版面分析与降级评测方案。
---

在构建基于文档的知识检索系统（如企业知识库或 RAG）时，PDF 是最常见的数据来源，也是最容易解析失败的格式。在前一篇[文档解析的基础认知](/frontier/document-parsing/what-is-file-parsing.md)中，我们梳理了常见文档格式的解析思路。本文聚焦 PDF 格式，解释其底层排版逻辑、文字与表格容易损坏的原因，并介绍工程中通用的重构与版面分析方法。

我们以一份典型的双栏技术文献为例：页面顶部是居中标题，中间是左右双栏正文，左栏嵌入了一个无边框财务数据表，右栏包含一个带分数线的公式。如果直接使用基础提取工具读取该页，往往会遇到三种典型故障：提取出的文本变成无序乱码；左右两栏文本被按行交错拼在一起；表格的列对齐彻底崩解为单行孤立数字。

![深入 PDF 解析机制全景图](/images/content-rebuild/cc0cc3778b2bef7f/cover.png)

要解决这些问题，必须先理解 PDF 的底层存储机制：它从一开始就不是为了“存储文章语义结构”而设计的，而是为了“在任何打印机和屏幕上精准重绘画笔”。

## 页面绘图指令与乱码根源

PDF（Portable Document Format）脱胎于 PostScript 页面描述语言。文件内部并没有“段落”、“标题”或“表格”等 DOM 节点概念，只有一系列按顺序排列的绘图操作符（Content Stream Operators）。

### 文本绘制操作符与坐标系
在 PDF 的内容流中，文本必须包裹在 `BT`（Begin Text Object）与 `ET`（End Text Object）之间。解析时最常涉及的操作符包括：
- `Tf`：指定字体对象编号与字号，例如 `/F1 12 Tf`。
- `Tm` / `Td`：设置文本变换矩阵或移动文本原点，决定字符绘制的绝对基线（Baseline）位置。
- `Tj` / `TJ`：输出字符。`Tj` 接收一个字符代码串，`TJ` 则允许在字符之间插入相对位移数值，用于微调字间距（Kerning）。

PDF 默认采用以磅（Point，1/72 英寸）为单位的笛卡尔坐标系。与网页浏览器或图像像素坐标系（原点在左上角、Y 轴向下）不同，PDF 默认坐标原点 $(0, 0)$ 位于页面的**左下角**，X 轴向右递增，Y 轴向上递增。

一段典型的文本流指令如下：

```text
BT
/F1 12 Tf
1 0 0 1 72 712 Tm
(Architecture) Tj
ET
```

这段指令仅告诉阅读器：切换到 `F1` 字体、12 磅大小，将画笔移动到基线坐标 $(72, 712)$ 处，画出对应字符的外形。它没有声明该词属于哪一个段落，也没有说明后面是否应该换行。

### 为什么屏幕能看懂，提取出来却是乱码？

很多开发者遇到过这种现象：PDF 在浏览器或阅读器中显示完全正常，但用代码提取或手动复制时，得到的却是一堆无意义符号、问号或不可见空白字符。

这是因为在 PDF 规范中，**字形呈现（Glyph Rendering）与字符语义（Text Unicode）是两套独立的机制**。

排版工具生成 PDF 时，为了压缩体积并避免跨设备字体缺失，通常会执行“字体子集化”（Font Subsetting）。软件只抽取文档用到的字形，并为这些字形分配临时的局部索引代码（Character Code）。例如，文档中出现的汉字“安”，其在字体子集中的索引可能是 `0x0001`，而不再是标准的 Unicode 编码。

![PDF 文字绘制与 ToUnicode 字符映射查找流](/images/content-rebuild/cc0cc3778b2bef7f/fig-1.png)

当 PDF 阅读器渲染页面时，它只需要读取字符代码 `0x0001`，找到字体内部对应的矢量轮廓数据并绘制在屏幕上；屏幕上的字形肉眼清晰可见。

但如果解析器想把 `0x0001` 还原成可编辑的 UTF-8 字符“安”，就必须依赖字体对象字典中附带的 `/ToUnicode` 映射表（CMap 流）。
- **映射表完整**：解析器根据 CMap 将 `0x0001` 转换为 `U+5B89`，提取出正确的“安”字。
- **映射表缺失或错误**：如果制作 PDF 的软件未内嵌 `/ToUnicode` 表，或者使用了私有自定义映射，解析器就只能直接输出原始字节码（例如输出为乱码控制符或落在 Unicode 私有使用区 PUA 的无效编码）。

在这种情况下，任何只读取底层指令流的纯文本抽取库都会失效，工程上必须将该页面渲染为图像，转由 OCR 模型读取文字。

<ReadMoreLock />

## 几何重构：从字符碎片到自然阅读流

即便 `/ToUnicode` 映射完全正常，解析器从内容流中拿到的也只是一堆离散的字符及其外接矩形框（Bounding Box: $x_0, y_0, x_1, y_1$）。要将这些碎片拼接为连续的段落，需要处理三类几何问题。

### 1. 浮点基线抖动与行聚类
排版软件在输出文本时，由于微调基准线或不同字体混排，同一物理行上的字符，其纵向坐标可能会出现微小的浮点数偏差（例如某个字符的底部 Y 坐标是 712.02，相邻字符是 711.95）。如果直接按 Y 坐标相等来判断行，整行文本就会被切割得支离破碎。

常用的行聚类规则基于容差阈值 $\epsilon_y$：

$$|y_{top, A} - y_{top, B}| \le \epsilon_y \quad \text{且} \quad |y_{bottom, A} - y_{bottom, B}| \le \epsilon_y$$

阈值 $\epsilon_y$ 通常设定为当前文字字号的 20% 到 30%。解析器将所有纵向重叠度处于容差范围内的字符划分到同一个物理行集合中。

### 2. 空格推断与连字符恢复
PDF 指令流中通常不会显式写入空格字符（`U+0020`）。排版软件往往只是把下一个字符的起始 X 坐标往右移动了一段距离。因此，解析器必须计算相邻字符之间的水平间距：

$$\Delta x = x_{0, \text{next}} - x_{1, \text{prev}}$$

当 $\Delta x$ 超过设定阈值 $\theta_{\text{space}}$（通常取当前字体字号或字符宽度的 0.25 到 0.5 倍）时，解析器在两者之间主动补入一个空格。

对于英文文献行末的连字符断行（如行末是 `docu-`，下一行开头是 `ment`），解析器需基于词典或拼写规则，剔除连字符并将两段词素合并回 `document`，避免后续分词检索时词干失效。

### 3. 多栏排版的串行错乱与中缝检测
学术论文和报刊通常采用双栏或三栏排版。如果解析算法不感知版面结构，盲目将页面上所有文字按照全局 Y 坐标从上到下排序，就会出现典型的“串行”灾难：左栏第 1 行与右栏第 1 行被拼成了一句，左栏第 2 行又与右栏第 2 行连在一起。

![双栏 PDF 串行错乱与投影切分示意图](/images/content-rebuild/cc0cc3778b2bef7f/fig-2.png)

解决双栏串行的基础方法是**投影切分法**：
1. **X 轴投影直方图**：统计整页字符在 X 轴上的分布。如果是标准双栏排版，页面中间会出现一段没有文字投影的空白区间，即中缝（Gutter Gap）。
2. **垂直分割视口**：当中缝宽度超过设定的栏间距阈值时，算法建立垂直分割线，将页面划分为左、右独立的子视口。
3. **视口内拓扑排序**：在每个子视口内部，分别执行“自上而下、自左向右”的文本排序，最后按照人类自然阅读习惯（先读完左栏全部内容，再读右栏）将两组文本拼接起来。

## 表格提取机制：有线表与无线表

表格不仅包含文字，还依赖单元格网格表达行列交叉的语义。PDF 表格在工程上通常划分为两类：

![有线表与无线表解析机制对比](/images/content-rebuild/cc0cc3778b2bef7f/fig-3.png)

### 有线表：矢量线段与形态学交点提取
有线表（Bordered Table）拥有完整的可见边框。边框可能以 PDF 矢量线条指令（如矩形 `re`、移动路径 `m`、画线 `l` 和描边 `S`）存在，也可能是光栅图像中的线条。

提取流程包含以下步骤：
1. **提取线条集合**：如果是矢量 PDF，遍历内容流过滤出所有的水平线段集 $H$ 和垂直线段集 $V$；如果是扫描图像，则通过形态学水平和垂直卷积核（Morphology Kernel）提取线条。
2. **计算交点求网格**：计算水平线与垂直线的交点坐标 $(x_i, y_j)$，由四个闭合交点围成的最小矩形即为一个候选单元格。
3. **单元格跨度计算**：检查相邻的候选网格之间是否有实际线条阻断。若没有线条阻断，则将网格合并，并标记对应的跨行（rowspan）或跨列（colspan）属性。
4. **文本空间投射**：将前面提取出的字符外接矩形映射到各单元格区域内，完成单元格文本填充。

### 无线表：投影直方图与行列聚类
无线表（Borderless Table）没有任何连续物理线条，数据仅靠列对齐与留白区分，提取难度显著提高：
- **列边界探测**：计算表格区域内字符在 X 轴的投影密度直方图。投影为 0 的谷底区间即为列间隙，由此划定各列边界。
- **行边界对齐**：按字符 Y 轴投影聚类得到基准行。需要注意单元格内文本换行（Cell Wrapping）问题，不能把单元格内的自然物理换行误判为新的数据行，需结合整行各列的垂直位置与行高综合判断从属关系。

## 公式与图表：为什么纯文本抽取会失效？

在科技文献中，公式和图表往往承载了核心结论，但这两类元素在底层完全被解构为离散图形。

![公式矢量元素离散化与 LaTeX 重建](/images/content-rebuild/cc0cc3778b2bef7f/fig-4.png)

### 公式离散化机制
一个简单的除法公式 $\frac{a+b}{c-d}$，在 PDF 内容流中会被打散为：
- 字符 `a`、`+`、`b` 的位置指令；
- 一条用矢量画线指令绘制的水平细线（并非减号字符 `-`，也没有 Unicode 编码）；
- 字符 `c`、`-`、`d` 的位置指令；
- 字符在内容流中的书写顺序完全可能颠倒，分母可能先于分子出现。

积分号 $\int$、根号 $\sqrt{}$ 或大型矩阵括号，往往是由专用数学字体中的多个字形拼贴而成。纯文本提取器拿到的只是一堆毫无语义联系的散乱字符和无法识别的符号片段。

### 现代工程处理路径
1. **目标检测定位**：使用版面分析模型（如 YOLO 或 Mask R-CNN）定位页面中的独立公式区域（Display Formula）和行内公式区域（Inline Formula），得到区域坐标框（BBox）。
2. **图像切片与视觉重建**：将公式区域渲染为高分辨率图像，输入到专用的公式识别视觉模型（如 LaTeX-OCR、UniMERNet 等），由模型端到端输出规范的 LaTeX 字符串。
3. **图表（Figure）归档**：对于架构图、折线图等复杂图表，解析器直接将其外接框区域切图保存为独立图像文件，在 Markdown 正文中插入图片占位符；若需要图表内容摘要，再将其送入多模态大模型进行语义描述。

## 扫描件与双层 PDF 的处理细节

纸质扫描件或手机拍照生成的 PDF，页面实质是一张位图对象（Image XObject），内容流中没有文字操作符。为了支持检索，通常会生成“双层 PDF”（Sandwich PDF）。

![双层 PDF 物理层与隐形文字层对齐机制](/images/content-rebuild/cc0cc3778b2bef7f/fig-5.png)

双层 PDF 包含两层信息：
1. **图像可见层**：原汁原味的扫描图像，用户在屏幕上看到的内容。
2. **隐形文字层**：由 OCR 引擎识别后生成的文本指令。这些文本使用特殊的渲染模式参数 `3 Tr`（Neither fill nor stroke text，即完全透明不绘制），叠放在对应的图像坐标上方。

### 常见工程陷阱与对齐治理
- **坐标漂移与错位**：如果扫描件存在微小的倾斜，而 OCR 引擎未进行倾斜校正（Deskew），生成的透明文本框就会与图像字符发生偏离，导致用户选词或定位失败。工程上必须在 OCR 处理前执行仿射变换矫正页面角度。
- **重复抽取（Duplicate Extraction）**：某些文档原本就包含部分矢量文本，后续又被误跑了一次 OCR，导致同一位置既有底层矢量文字，又叠了一层 OCR 隐形文字。提取时需对文字框计算交并比（IoU），对重叠度高于阈值（如 0.85）的重复文本进行去重。
- **低置信度字符过滤**：纸张折痕、阴影或污渍常被 OCR 误识别为孤立的下划线 `_`、竖线 `|` 或句点。解析程序需要结合 OCR 引擎给出的每个词的置信度得分（Confidence Score），将低于阈值（如 $<0.5$）的孤立噪点剔除。

## 规则解析与视觉版面分析的技术选型

纯基于坐标几何规则的算法（如 `pdfminer`、`pdfplumber`）运行极快（单页通常耗时 10~50 毫秒），但在面对多图环绕、复杂跨栏、图文混排等非标准版面时，规则容易顾此失彼。现代工程方案通常引入视觉深度学习模型进行辅助。

![基于视觉版面分析的多模态混合解析流水线](/images/content-rebuild/cc0cc3778b2bef7f/fig-6.png)

### 主流方案对比

| 方案类型 | 代表技术 / 模型 | 核心机制与优势 | 局限性与工程挑战 |
| :--- | :--- | :--- | :--- |
| **纯规则几何解析** | `pdfminer.six`, `pdfplumber`, `PyMuPDF` | 解析矢量坐标与线条，通过投影与间距阈值还原文本和简单表格。资源消耗极低，速度极快。 | 遇到无边框表格、分栏混排、复杂排版容易串行；ToUnicode 损坏时无法处理。 |
| **目标检测 / 版面分割** | LayoutLMv3, PP-Structure, YOLO-Layout | 将页面渲染为图像，预测各个区域的类别与边框（Title, Text, Table, Figure 等）。 | 只划分区块，区块内部阅读顺序仍需重排；小字号或密集文本仍需配合后处理。 |
| **端到端文档模型** | Nougat, Donut, GOT-OCR | 直接输入文档图像，自回归生成带有 Markdown/LaTeX 语法的文本序列。 | 推理耗费 GPU 算力；表格数据较长时存在幻觉或截断风险；单页耗时较长。 |
| **混合架构（Hybrid）** | MinerU, Marker | 先用视觉模型划分版面结构，若原 PDF 包含矢量文字则按坐标提取，若缺失或损坏则回退 OCR。 | 流程链路长，工程依赖复杂，需要平衡 GPU 显存调度与 CPU 吞吐。 |

## 状态机降级逻辑与评测指标

在生产环境中，没有哪一种单一方案能低成本且高质量地覆盖所有 PDF。推荐采用分层检测与状态机降级策略：

```text
输入单页 PDF 
  │
  ├──> 1. 尝试原生矢量快速解析 (PyMuPDF / pdfplumber)
  │      │
  │      ├──> 文本乱码率 > 阈值 (ToUnicode 损坏) ──┐
  │      │                                         │
  │      └──> 检测到复杂多栏/穿透表格/公式 ────────┤
  │                                                │
  │                                                ▼
  └──> 2. 正常产出 Markdown <────────── 降级至视觉版面分析与 OCR 流水线
```

### 质量评测指标
要量化衡量 PDF 解析质量，通常采用两套指标：

#### 1. 文本准确率：字符错误率（CER）
基于编辑距离（Levenshtein Distance）衡量提取文本与参考标注（Ground Truth）的差异：

$$\text{CER} = \frac{S + D + I}{N}$$

其中 $S$ 为替换字符数，$D$ 为删除字符数，$I$ 为插入字符数，$N$ 为参考标注的标准总字符数。CER 越低代表文本还原质量越高，通常需要控制在 2% 以下。

#### 2. 表格结构准确率：TEDS
表格语义依赖行、列拓扑，单纯比对字符无法反映结构错位。学术界与工业界通用 TEDS（Tree-Edit-Distance-based Similarity）指标。表格被统一转换为 HTML 语法树（如 `<table>`, `<tr>`, `<td>`），计算预测树与真实标注树之间的树编辑距离：

$$\text{TEDS}(T_a, T_b) = 1 - \frac{\text{EditDist}(T_a, T_b)}{\max(|T_a|, |T_b|)}$$

其中 $\text{EditDist}(T_a, T_b)$ 为转换所需的最小节点增删改加权代价，$|T|$ 为树的节点总数。TEDS 取值范围在 $[0, 1]$ 之间，越接近 1 说明表格结构恢复越精准。

### 最小可运行验证代码：双栏判定与乱码检测

下面的 Python 示例展示了如何使用 `PyMuPDF`（fitz）提取带有坐标的文本块，识别明显的中央中缝进行双栏重排，并在乱码率过高时发出降级信号：

```python
import unicodedata
from typing import List, Dict, Any
import fitz  # PyMuPDF

def is_garbled_char(char: str) -> bool:
    """判断字符是否落在 Unicode 私有使用区 (PUA) 或为替换字符"""
    cp = ord(char)
    # Unicode 私有使用区 (PUA) 范围: E000-F8FF, F0000-FFFFD, 100000-10FFFD
    if (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0xFFFFD) or (0x100000 <= cp <= 0x10FFFD):
        return True
    if char == "\ufffd" or unicodedata.category(char) == "Co":
        return True
    return False

def parse_page_with_fallback(page: fitz.Page, gutter_threshold: float = 20.0) -> Dict[str, Any]:
    """
    提取单页文本，处理基础双栏排序并监控乱码率。
    """
    # 提取结构化文本块: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0]  # 只保留纯文本块
    
    all_chars = "".join(b[4] for b in text_blocks)
    total_chars = len(all_chars)
    if total_chars == 0:
        return {"status": "FALLBACK_OCR", "reason": "EMPTY_PAGE", "text": ""}
    
    # 1. 乱码率统计
    garbled_count = sum(1 for c in all_chars if is_garbled_char(c))
    garbled_ratio = garbled_count / total_chars
    if garbled_ratio > 0.05:
        # 乱码率超过 5%，判定 ToUnicode 映射可能失效，触发降级
        return {"status": "FALLBACK_OCR", "reason": f"HIGH_GARBLED_RATIO: {garbled_ratio:.2%}", "text": ""}
    
    # 2. 双栏几何重构：检测是否存在明显的中央中缝
    page_width = page.rect.width
    mid_x = page_width / 2.0
    
    left_column = []
    right_column = []
    spanned_blocks = []  # 跨栏块（如居中大标题、通栏表格等）
    
    gutter_half = gutter_threshold / 2.0
    for b in text_blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        # 左栏块完全位于中缝左边界以左，右栏块完全位于中缝右边界以右，其余归为跨栏块
        if x1 <= mid_x - gutter_half:
            left_column.append(b)
        elif x0 >= mid_x + gutter_half:
            right_column.append(b)
        else:
            spanned_blocks.append(b)
            
    # PyMuPDF 导出的 block 坐标原点在页面左上角（Y 轴向下），按顶部 Y 坐标（b[1]）升序即为自上而下阅读顺序
    left_column.sort(key=lambda b: b[1])
    right_column.sort(key=lambda b: b[1])
    spanned_blocks.sort(key=lambda b: b[1])
    
    # 3. 按阅读顺序组装：先顶部通栏，再左栏，后右栏，最后底部通栏
    ordered_text = ""
    top_spanned = [b for b in spanned_blocks if b[1] < page.rect.height * 0.25]
    bottom_spanned = [b for b in spanned_blocks if b[1] >= page.rect.height * 0.25]
    
    for b in top_spanned:
        ordered_text += b[4] + "\n"
    for b in left_column:
        ordered_text += b[4] + "\n"
    for b in right_column:
        ordered_text += b[4] + "\n"
    for b in bottom_spanned:
        ordered_text += b[4] + "\n"
        
    return {"status": "SUCCESS", "text": ordered_text.strip(), "garbled_ratio": garbled_ratio}
```

通过上述机制解析出结构化的 Markdown 文本后，文档解析的第一步才算完成。在将这些文本写入检索数据库之前，还需要处理标题层级继承、表格与公式的分块切分，以及文档元数据的挂载。后续处理步骤请参考下一篇：[从解析结果到知识库切块构建](/frontier/document-parsing/parsing-to-knowledge-base.md)。
