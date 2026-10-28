---
title: AI 工程师入门学习路线
icon: mdi:rocket-launch
tag:
  - 学习路线
  - AI 工程师
  - 入门
---

# AI 工程师入门学习路线

> 从零基础到能找到 AI 应用工程师岗位，建议 3-4 个月

---

## 阶段 1：基础打底（2-3 周）

### 必学

- [机器学习基础](/fundamentals/ml-basics/) - 监督学习、损失函数、过拟合
- [深度学习基础](/fundamentals/dl-basics/) - 神经网络、反向传播、CNN/RNN

### 实战

用 PyTorch 实现一个简单的图像分类器（CIFAR-10），感受训练、验证、评估的全流程。

---

## 阶段 2：大模型核心（3-4 周）

### 必学

- [Transformer 架构](/fundamentals/transformer-arch/) - 注意力机制、位置编码
- [大模型基础](/fundamentals/llm-basics/) - 预训练、SFT、RLHF
- [Token 与分词](/fundamentals/token-and-tokenizer/) - BPE、WordPiece
- [Embedding](/fundamentals/embedding/) - 向量表示

### 实战

从 Hugging Face 加载开源模型（如 Qwen2.5），跑一次文本生成，感受 Token、Temperature 等参数的实际效果。

---

## 阶段 3：Prompt 工程（1-2 周）

### 必学

- [Prompt 面试题](/interview/prompt-interview/) - 结构化 Prompt、Few-shot、CoT

### 实战

做一个"智能简历解析"工具：
- 用 Prompt 设计角色和任务
- 用 Few-shot 控制输出格式
- 用 Chain-of-Thought 提升推理准确率

---

## 阶段 4：方向选择（1-2 周）

根据兴趣选择深入方向：

- **应用方向** → 进入 [大模型应用开发](/learning-path/llm-app-dev.html)
- **RAG 方向** → 做一个知识库问答系统
- **Agent 方向** → 做一个带工具调用的对话机器人
- **求职方向** → 刷 [面试题](/interview/)

---

## 学习节奏建议

| 阶段 | 周期 | 每日投入 |
|------|------|----------|
| 基础打底 | 2-3 周 | 2-3 小时 |
| 大模型核心 | 3-4 周 | 2-3 小时 |
| Prompt 工程 | 1-2 周 | 1-2 小时 |
| 方向选择 | 1-2 周 | 1-2 小时 |

**核心原则**：每个阶段至少做完一个项目，不要只收藏不看。能讲出来的知识才是真的学会了。

---

## 求职目标

学完这套路线，可以投递：

- AI 应用工程师
- Prompt 工程师
- LLM 应用开发工程师
