---
title: 微调数据应该怎么构造？从格式规范到清洗去重
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 微调数据决定了大模型的指令依从性与输出规范。本文详解单轮三元组与多轮ChatML格式、清洗去重管线及负例控制，附可运行脚本与实操练习。
---

在进入模型训练之前，我们首先需要搞清楚微调数据在工程上究竟要解决什么问题。在上一讲 [/llm-100/finetuning/pretraining-vs-finetuning.md](/llm-100/finetuning/pretraining-vs-finetuning.md) 中，我们已经明确：预训练赋予了模型通用的语言概率分布与基础世界知识，而监督微调（Supervised Fine-Tuning, SFT）的核心目标则是调整模型的交互行为、建立对指令格式的依从性，并对齐业务所需的输入输出结构。

监督微调并不是向模型参数中强行写入实时业务事实的可靠途径。如果训练数据充斥噪声、字段混乱或指令高度雷同，微调后的模型不仅无法稳定输出指定格式，反而容易学到错误的推理模板甚至加剧事实幻觉。本文以构建一个“SQL 诊断与调优助手”为例，从单轮三元组、多轮 ChatML 结构、质量评估、清洗去重流水线到负例样本设计，逐步拆解微调数据的构建标准。

![微调数据构造与清洗工程流水线架构图](/images/content-rebuild/0e2f9731bc5dcc71/cover.png)

上图展示了微调数据工程的完整流向：收集到的原始语料经过格式规范化、规则过滤、SimHash 去重与安全样本配比后，输出为训练器能够稳定解析的序列集合。

---

### 单轮数据规范：Instruction 与 Input 的职责切分

单任务微调中最常见的组织方式是 Alpaca 风格的三元组或二元组结构。很多初学者容易混淆任务指令（Instruction）与输入材料（Input）的边界，导致拼出的样本逻辑混乱。

这三个字段的明确分工如下：
- **Instruction（任务要求）**：描述希望模型执行的操作规则，即“做什么”。例如：“为以下慢查询 SQL 提供基于覆盖索引的优化建议”。
- **Input（输入实体或上下文）**：完成该操作所依赖的外部材料，即“基于什么材料做”。例如具体的 SQL 语句和表结构 DDL。如果任务本身自包含（例如“解释什么是数据库事务的隔离级别”），则该字段应设置为空字符串。
- **Output（预期输出）**：模型针对上述指令和上下文应当生成的标准解答，也就是模型训练时的监督目标。

![Alpaca格式单轮指令三元组结构与模板映射示意图](/images/content-rebuild/0e2f9731bc5dcc71/fig-1.png)

如上图所示，当数据被喂给分词器前，需要通过 Prompt 模板拼接为平铺文本。以下是典型的组装逻辑：

```python
def format_alpaca_sample(instruction: str, input_text: str = "") -> str:
    """
    将结构化字段拼接为统一的提示词文本
    """
    clean_inst = instruction.strip()
    clean_input = input_text.strip()
    
    if clean_input:
        return (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{clean_inst}\n\n"
            f"### Input:\n{clean_input}\n\n"
            f"### Response:\n"
        )
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{clean_inst}\n\n"
        f"### Response:\n"
    )
```

如果随意把待处理的 SQL 塞进 Instruction，或者把核心任务规则混在 Input 中，会导致模型在推理阶段无法清晰分辨“指令规则”与“操作对象”，降低模型对不同提示词句式的泛化依从能力。

<ReadMoreLock />

---

### 多轮对话数据：ChatML 规范与特殊定界 Token

当业务需要多轮交互（如助手先诊断问题、追问索引状态，用户补充后再给出方案）时，数据通常采用基于角色（Role）的列表形式组织（如 ChatML 或 ShareGPT 格式）。

#### 1. 结构化表示示例

```json
{
  "messages": [
    {"role": "system", "content": "你是一名专业的数据库工程师，只回答与 SQL 优化和索引设计相关的问题。"},
    {"role": "user", "content": "MySQL 中联合索引 (a, b, c)，在查询 WHERE b=2 AND c=3 时能走索引吗？"},
    {"role": "assistant", "content": "不能直接利用联合索引定位起点。因为该索引先按字段 a 排序，缺失 a 条件时无法利用 B+ 树的有序性完成根节点定位。"},
    {"role": "user", "content": "那如果查询是 WHERE a=1 AND c=3 呢？"},
    {"role": "assistant", "content": "可以利用索引定位到 a=1 的记录范围。对于条件 c=3，MySQL 5.6 引入的索引下推（ICP）可在存储引擎层遍历时先按 c 过滤，减少回表次数。"}
  ]
}
```

#### 2. 特殊定界 Token 的封装逻辑

语言模型在底层只接收一维的整数 ID 序列。为了让注意力计算能够准确识别发言人角色与语句起止，分词器引入了不可拆分的特殊定界符，最典型的便是 ChatML 的 `<|im_start|>` 和 `<|im_end|>`。

![ChatML多轮对话格式与特殊定界Token封装机制](/images/content-rebuild/0e2f9731bc5dcc71/fig-2.png)

如上图所示，每一轮发言都会被定界符包裹：
```text
<|im_start|>system
你是一名专业的数据库工程师...<|im_end|>
<|im_start|>user
MySQL 中联合索引...<|im_end|>
<|im_start|>assistant
不能直接利用联合索引...<|im_end|>
<|im_start|>user
那如果查询是...<|im_end|>
<|im_start|>assistant
可以利用索引定位到...<|im_end|>
```

**多轮数据校验必须遵守的两条硬性约束**：
1. **角色交替顺序**：消息序列必须满足 `system (可选且仅首位) -> user -> assistant -> user -> assistant` 的严格交替。如果出现连续两个 `user` 或连续两个 `assistant`，模型的自回归预测上下文就会混乱。
2. **因果依赖不可中断**：计算第 $N$ 轮 Assistant 的回复损失时，前 $N-1$ 轮对话全部作为输入上下文。严禁为了缩减长度而强行删去中间轮次（例如留下第 1 轮提问和第 3 轮回答），因果链条一旦断裂，模型将无法学到正确的上下文回溯逻辑。

在模型训练时，通常只有 `assistant` 区域的输出会计算交叉熵损失，用户输入和系统提示词区域会被掩码忽略，这一技术的具体计算过程见 [/llm-100/finetuning/sft-instruction-tuning.md](/llm-100/finetuning/sft-instruction-tuning.md)。

---

### 数据质量控制：多样性、难度与长度分布

监督微调中，数据质量的重要性远高于单纯的数据集行数。构建数据集时，需要从以下三个维度进行把控：

![微调数据集质量控制的三大核心维度](/images/content-rebuild/0e2f9731bc5dcc71/fig-3.png)

上图直观呈现了高质量数据集在任务意图、逻辑难度和序列长度上的分布要求。

#### 1. 指令多样性（Diversity）
如果微调集中 90% 的指令都是“请分析以下 SQL 的性能：”，模型面对“这条查询为什么变慢了？”、“帮我重写下面的语句以减少回表”等句式变体时，遵循能力就会明显下滑。构建数据时，应确保动词短语（如分析、改写、提取、校验、对比）和业务场景（单表聚合、多表连接、子查询展开）分布均衡。

#### 2. 难度梯度与多约束条件
如果数据集全是一句话即答的简单题目，模型在面对复合要求时容易丢掉部分约束。应设计不同难度的样本梯度：
- *基础单步任务*：“给出改写后的 SQL 语句。”
- *复合约束任务*：“给出改写后的 SQL 语句，并在注释中标注新创建的联合索引列顺序，禁止修改现有 WHERE 条件的等值逻辑，最终以 Markdown 代码块返回。”

引入显式负向约束（“禁止……”、“不要……”）能促使模型学会在生成过程中维持多维检查条件。

#### 3. 长度分布控制
- **末端被硬截断的危害**：如果数据预处理设置的 `max_length` 过小，导致很多样本的 `output` 在结尾被生硬截断，模型就会在反向传播中学习到“句子没写完直接输出终止符”的错误概率，导致线上回答频繁出现半截话。
- **异常长样本的填充开销**：如果少数异常数据（如整段未经清理的几千行报错日志）拉长了批次序列，且没有采取动态分桶处理，会导致同一个批次内的其他短文本被大量的填充符（padding）填满，徒增显存与计算耗费。批处理分桶的具体打包策略可参阅 [/interview/llm-interview/transformer-large-dataset.md](/interview/llm-interview/transformer-large-dataset.md)。

---

### 清洗去重实操：规则过滤与 SimHash 指纹

原始数据进入训练集前，需要经过分阶段的清洗流水线。

![微调数据清洗与过滤端到端流水线流程图](/images/content-rebuild/0e2f9731bc5dcc71/fig-4.png)

如上图所示，流水线依次处理格式校验、长度与字符规则过滤、SimHash 指纹去重，最后完成输出。

#### 1. 脏数据对模型造成的具体破坏

![格式混乱与事实幻觉数据导致模型推理退化的对比分析](/images/content-rebuild/0e2f9731bc5dcc71/fig-5.png)

如果数据中包含以下缺陷，将直接体现在微调后的模型表现上：
- **无用套话残留**：“好的，作为一名AI，我为您解答……”这类固定开头会让模型失去专业角色调性。
- **输出格式漂移**：同一个查询场景，一会儿直接给纯 SQL，一会儿包在代码块里，一会儿夹杂长篇文字。这会导致下游自动化解析脚本频繁报错。
- **错误先验注入**：如果样本中包含“MySQL 默认存储引擎是 MyISAM”等事实错误，自回归损失会强行改变模型底层的参数连接，使模型在相关问题上出现稳定输出错误。

#### 2. 自包含清洗与 SimHash 近似去重代码

对于大规模数据去重，逐条对比耗时过长。SimHash 将文本特征投影为 64 位二进制指纹，指纹间的汉明距离（不同 bit 位的数量）可以近似衡量文本特征夹角。在计算指纹时，需将任务指令与输入上下文合并计算，防止“同一通用指令但处理不同数据输入”的有效样本被误删。当汉明距离小于设定的阈值（通常为 3）时，可判定两条提示文本高度相似并予以剔除。

```python
import re
import hashlib
from typing import List, Dict

def clean_text_heuristics(text: str) -> str:
    """去除 HTML 标签、控制字符并标准化空白符"""
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 替换不可见控制字符为空格
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    # 合并多余空白符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_sample(instruction: str, output: str) -> bool:
    """启发式规则校验"""
    if len(instruction) < 5 or len(output) < 5:
        return False
    if len(instruction) > 2048 or len(output) > 4096:
        return False
    
    # 检测异常标点符号重复（防止退化为死循环的语料进入训练）
    for punct in ["。", ".", "!", "?", "\n"]:
        if output.count(punct * 5) > 0:
            return False
            
    # 过滤明显的套话模板残留
    dirty_phrases = ["作为一个AI", "作为一个人工智能模型", "我无法访问互联网实时信息"]
    for phrase in dirty_phrases:
        if phrase in output and len(output) < 60:
            return False
            
    return True

def get_simhash_signature(text: str, num_bits: int = 64) -> int:
    """
    生成 64 位 SimHash 二进制局部敏感哈希指纹
    同时支持英文单词与中文 2-gram 切分
    """
    # 提取连续英文单词与中文字符
    words = re.findall(r'[a-zA-Z0-9]+|[一-龥]', text.lower())
    if not words:
        return 0
    
    # 生成简易 n-gram 词袋
    tokens = []
    for i in range(len(words)):
        tokens.append(words[i])
        if i < len(words) - 1:
            tokens.append(f"{words[i]}_{words[i+1]}")
            
    v = [0] * num_bits
    for token in tokens:
        # 取 MD5 前 16 位字符作为 64 位整型哈希
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:16], 16)
        for i in range(num_bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
            
    fingerprint = 0
    for i in range(num_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(sig1: int, sig2: int) -> int:
    """计算两个指纹之间的汉明距离"""
    return bin(sig1 ^ sig2).count('1')

def clean_and_deduplicate_dataset(raw_dataset: List[Dict[str, str]], 
                                   simhash_threshold: int = 3) -> List[Dict[str, str]]:
    """
    执行整套清洗和去重流程
    """
    cleaned_samples: List[Dict[str, str]] = []
    seen_signatures: List[int] = []
    
    for item in raw_dataset:
        inst = clean_text_heuristics(item.get("instruction", ""))
        inp = clean_text_heuristics(item.get("input", ""))
        out = clean_text_heuristics(item.get("output", ""))
        
        if not is_valid_sample(inst, out):
            continue
            
        # 需拼接 instruction 与 input，防止同指令不同输入的有效样本被误删
        dedup_text = f"{inst}\n{inp}".strip() if inp else inst
        sig = get_simhash_signature(dedup_text)
        is_duplicate = False
        
        for exist_sig in seen_signatures:
            if hamming_distance(sig, exist_sig) <= simhash_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            seen_signatures.append(sig)
            sample = {"instruction": inst, "output": out}
            if inp:
                sample["input"] = inp
            cleaned_samples.append(sample)
            
    return cleaned_samples
```

关于机器学习与特征工程的基础概念，可参考 [/fundamentals/ml-basics/data-features-labels.md](/fundamentals/ml-basics/data-features-labels.md)。

---

### 负例设计与安全拒答的配比

在很多业务场景中，当输入缺少必要前置条件或涉及恶意注入时，模型不应生硬瞎猜，而应主动指出缺陷或给出规范拒绝。

![安全拒答样本配比的天平平衡尺度示意图](/images/content-rebuild/0e2f9731bc5dcc71/fig-6.png)

如上图所示，负例样本在训练集中需要保持合理的配比，避免产生过度拒答。

#### 1. 负例样本的核心应用场景
1. **前置条件缺失时的诚实回应**：当给定的表结构中根本没有查询所用的字段时，模型应指出“缺少相关字段的类型与索引信息，无法给出优化方案”，而不是随机编造索引名。
2. **指令注入防范与安全边界**：面对提示词注入（Prompt Injection，试图利用用户输入强行覆盖既定系统角色指令）或违规越狱提问，模型应中立且清晰地拒绝执行违规操作。

#### 2. 过度拒答的形成原因与控制建议
如果训练集中塞入过多形式单一的机械拒答语（如成千上万条“抱歉，我无法回答该问题”），模型在梯度更新中容易将敏感词（如“攻击”、“漏洞”、“慢查询”）与拒绝输出直接强关联。这会导致用户询问正规技术防护（如“怎样排查 SQL 慢查询漏洞？”）时，模型也直接触发拒答，破坏了系统的可用性。

**工程建议**：
- **控制配比**：在常规指令微调任务中，拒答与安全样本占比通常建议维持在 **1% 到 5%** 之间。
- **说明原因而非直接拒绝**：拒答样本必须交代“为什么无法回答”以及“还需要补充什么条件”，避免模型沦为机械的复读机。

---

### 课后动手练习

**目标**：编写一个多轮对话数据校验函数 `validate_chat_messages(messages: list) -> tuple[bool, str]`，验证一组 ChatML 格式的对话记录是否可以直接交付训练。

**输入与验证规则**：
1. 整个列表必须非空。
2. 若包含 `role: "system"`，该消息必须且仅能出现在首位（索引 0）。
3. 消息角色必须满足严格交替：剔除开头的 `system` 之后，发言序列必须严格按照 `user -> assistant -> user -> assistant` 依次递进，且最后一条消息的角色必须是 `assistant`（训练时需要目标输出）。
4. 每一条消息的 `content` 不能为空字符串或纯空格。

**可直接运行的检验代码**：

```python
def validate_chat_messages(messages: list) -> tuple[bool, str]:
    if not messages:
        return False, "消息列表为空"
        
    # 检查 system 角色位置
    for idx, msg in enumerate(messages):
        if msg.get("role") == "system" and idx != 0:
            return False, "system 角色只能出现在首位"
            
    start_idx = 1 if messages[0].get("role") == "system" else 0
    dialogue = messages[start_idx:]
    
    if not dialogue:
        return False, "缺少有效的问答轮次"
    if dialogue[-1].get("role") != "assistant":
        return False, "最后一轮必须是 assistant 的回复"
        
    expected_role = "user"
    for msg in dialogue:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        
        if not content:
            return False, f"角色 {role} 的发言内容不能为空"
        if role != expected_role:
            return False, f"角色交替错误：期望 {expected_role}，实际收到 {role}"
            
        expected_role = "assistant" if expected_role == "user" else "user"
        
    return True, "校验通过"

# 验证用例
valid_sample = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "怎么查慢日志？"},
    {"role": "assistant", "content": "开启 slow_query_log 即可。"}
]

invalid_sample = [
    {"role": "user", "content": "第一问"},
    {"role": "user", "content": "第二问"}  # 连续 user 角色
]

assert validate_chat_messages(valid_sample)[0] is True
assert validate_chat_messages(invalid_sample)[0] is False
print("所有校验逻辑测试通过！")
```

当数据准备完毕并通过格式校验后，下一步便是将序列加载到训练器中进行前向传播与梯度更新。关于标签掩码（Label Masking）与参数微调的具体计算方式，请前往下一讲：[/llm-100/finetuning/sft-instruction-tuning.md](/llm-100/finetuning/sft-instruction-tuning.md)。
