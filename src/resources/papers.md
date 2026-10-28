---
title: 论文资源
icon: mdi:file-document-outline
tag:
  - AI 论文
  - LLM
  - RAG
---

# AI 必读论文与阅读路径

论文不是按时间顺序背标题。推荐按问题阅读：模型如何建模序列、如何预训练、如何对齐、如何接入外部知识、如何调用工具与评估。

## Transformer 与预训练

| 论文 | 解决的问题 | 入口 |
| --- | --- | --- |
| Attention Is All You Need | 用注意力替代循环结构，建立 Transformer | [arXiv](https://arxiv.org/abs/1706.03762) |
| BERT | 双向预训练如何服务理解任务 | [arXiv](https://arxiv.org/abs/1810.04805) |
| Language Models are Few-Shot Learners | 大规模自回归模型的上下文学习 | [arXiv](https://arxiv.org/abs/2005.14165) |
| LLaMA | 开放权重基础模型的训练配方 | [arXiv](https://arxiv.org/abs/2302.13971) |

## 对齐与高效微调

| 论文 | 解决的问题 | 入口 |
| --- | --- | --- |
| InstructGPT | 人类反馈如何提升指令遵循 | [arXiv](https://arxiv.org/abs/2203.02155) |
| LoRA | 如何用低秩更新降低微调成本 | [arXiv](https://arxiv.org/abs/2106.09685) |
| Direct Preference Optimization | 如何绕开复杂 RL 流程进行偏好优化 | [arXiv](https://arxiv.org/abs/2305.18290) |

## RAG 与检索

| 论文 | 解决的问题 | 入口 |
| --- | --- | --- |
| Retrieval-Augmented Generation | 将参数化模型与外部检索结合 | [arXiv](https://arxiv.org/abs/2005.11401) |
| DPR | 用双编码器做密集段落检索 | [arXiv](https://arxiv.org/abs/2004.04906) |
| ColBERT | 更细粒度的 late interaction 检索 | [arXiv](https://arxiv.org/abs/2004.12832) |

## Agent、工具与推理

| 论文 | 解决的问题 | 入口 |
| --- | --- | --- |
| Chain-of-Thought Prompting | 通过中间推理提高复杂任务表现 | [arXiv](https://arxiv.org/abs/2201.11903) |
| ReAct | 将推理轨迹与行动交替结合 | [arXiv](https://arxiv.org/abs/2210.03629) |
| Toolformer | 让语言模型学习何时调用工具 | [arXiv](https://arxiv.org/abs/2302.04761) |

## 怎样把论文读成能力

1. 先写下论文要解决的具体问题和假设。
2. 看方法图与实验设置，确认输入、输出、数据集和指标。
3. 复现实验中的一个最小部分，或在自己的任务上做对照实验。
4. 记录失效条件和未回答的问题；论文结论通常只在特定设置下成立。
5. 对快速发展的方向，结合后续工作、开源实现和官方文档更新理解。

## 论文阅读优先级

| 读者目标 | 建议先读 |
| --- | --- |
| 想理解模型原理 | Attention、BERT、GPT-3、LLaMA |
| 想做微调与对齐 | InstructGPT、LoRA、DPO |
| 想做知识库 | DPR、RAG、ColBERT |
| 想做工具型 Agent | CoT、ReAct、Toolformer |

每条路线先精读两三篇代表作即可；不要试图一次读完所有引用链。

## 读实验时重点看什么

- Baseline 是否公平，是否与强基线比较；
- 数据集与指标是否贴近你的真实任务；
- 消融实验是否真正说明关键设计有效；
- 成本、算力、失败案例和限制是否被披露；
- 开源实现是否复现了论文中的关键设置。

## 延伸资源

- [arXiv](https://arxiv.org/)：论文预印本与版本更新。
- [Papers with Code](https://paperswithcode.com/)：任务、论文、代码与排行榜索引。
- [Hugging Face Papers](https://huggingface.co/papers)：社区讨论与实现线索。

它们适合发现资料；最终判断仍应回到论文原文、补充材料与官方实现。
