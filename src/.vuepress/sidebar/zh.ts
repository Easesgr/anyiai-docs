import { sidebar } from "vuepress-theme-hope";

export const zhSidebar = sidebar({
  "/learning-path/": [
    {
      "text": "学习路线",
      "link": "/learning-path/",
      "icon": "mdi:map",
      "children": [
        {
          "text": "AI 开发入门",
          "link": "/learning-path/ai-engineer.md"
        },
        {
          "text": "模型原理与部署",
          "link": "/learning-path/from-principles-to-ai-engineering.md"
        },
        {
          "text": "大模型应用开发",
          "link": "/learning-path/llm-app-dev.md"
        },
        {
          "text": "RAG 实战",
          "link": "/learning-path/rag-practice.md"
        },
        {
          "text": "Agent 开发",
          "link": "/learning-path/agent-engineer.md"
        },
        {
          "text": "系统设计面试",
          "link": "/learning-path/ai-interview-prep.md"
        },
        {
          "text": "算法工程师面试",
          "link": "/learning-path/algorithm-interview.md"
        }
      ]
    }
  ],
  "/fundamentals/": [
    {
      "text": "AI 基础",
      "link": "/fundamentals/",
      "icon": "mdi:brain",
      "children": [
        {
          "text": "Transformer 概览",
          "link": "/fundamentals/transformer-arch.md"
        },
        {
          "text": "机器学习",
          "link": "/fundamentals/ml-basics/",
          "icon": "mdi:cog-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "机器学习入门",
              "link": "/fundamentals/ml-basics/what-is-ml.md"
            },
            {
              "text": "数据、特征与标签",
              "link": "/fundamentals/ml-basics/data-features-labels.md"
            },
            {
              "text": "损失函数与梯度下降",
              "link": "/fundamentals/ml-basics/loss-gradient-descent.md"
            },
            {
              "text": "线性回归与逻辑回归",
              "link": "/fundamentals/ml-basics/linear-logistic-regression.md"
            },
            {
              "text": "训练、验证与测试集",
              "link": "/fundamentals/ml-basics/train-val-test.md"
            },
            {
              "text": "过拟合与欠拟合",
              "link": "/fundamentals/ml-basics/overfitting-underfitting.md"
            },
            {
              "text": "分类评估",
              "link": "/fundamentals/ml-basics/evaluation-classification.md"
            },
            {
              "text": "回归评估",
              "link": "/fundamentals/ml-basics/evaluation-regression.md"
            },
            {
              "text": "决策树与集成学习",
              "link": "/fundamentals/ml-basics/tree-models.md"
            },
            {
              "text": "聚类与降维",
              "link": "/fundamentals/ml-basics/clustering-dimensionality.md"
            }
          ]
        },
        {
          "text": "深度学习",
          "link": "/fundamentals/dl-basics/",
          "icon": "mdi:brain",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "深度学习入门",
              "link": "/fundamentals/dl-basics/what-is-dl.md"
            },
            {
              "text": "神经网络",
              "link": "/fundamentals/dl-basics/neural-network-basics.md"
            },
            {
              "text": "激活函数",
              "link": "/fundamentals/dl-basics/activation-functions.md"
            },
            {
              "text": "前向与反向传播",
              "link": "/fundamentals/dl-basics/forward-backward-propagation.md"
            },
            {
              "text": "梯度消失与爆炸",
              "link": "/fundamentals/dl-basics/gradient-vanishing-exploding.md"
            },
            {
              "text": "优化器",
              "link": "/fundamentals/dl-basics/optimizers.md"
            },
            {
              "text": "归一化",
              "link": "/fundamentals/dl-basics/normalization.md"
            },
            {
              "text": "Dropout 与正则化",
              "link": "/fundamentals/dl-basics/dropout-regularization.md"
            },
            {
              "text": "卷积神经网络",
              "link": "/fundamentals/dl-basics/cnn.md"
            },
            {
              "text": "RNN、LSTM 与 GRU",
              "link": "/fundamentals/dl-basics/rnn-lstm-gru.md"
            },
            {
              "text": "注意力机制",
              "link": "/fundamentals/dl-basics/attention-mechanism.md"
            },
            {
              "text": "Transformer 入门",
              "link": "/fundamentals/dl-basics/transformer.md"
            }
          ]
        },
        {
          "text": "Token 与分词",
          "link": "/fundamentals/token-and-tokenizer/",
          "icon": "mdi:translate",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Token 入门",
              "link": "/fundamentals/token-and-tokenizer/what-is-token.md"
            },
            {
              "text": "分词算法",
              "link": "/fundamentals/token-and-tokenizer/how-tokenizer-works.md"
            },
            {
              "text": "中文分词",
              "link": "/fundamentals/token-and-tokenizer/chinese-tokenization.md"
            },
            {
              "text": "Token 预算与计费",
              "link": "/fundamentals/token-and-tokenizer/token-budget.md"
            },
            {
              "text": "分词对应用的影响",
              "link": "/fundamentals/token-and-tokenizer/tokenization-impact.md"
            }
          ]
        },
        {
          "text": "文本向量",
          "link": "/fundamentals/embedding/",
          "icon": "mdi:vector-square",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Embedding 入门",
              "link": "/fundamentals/embedding/what-is-embedding.md"
            },
            {
              "text": "语义相似度",
              "link": "/fundamentals/embedding/embedding-similarity.md"
            },
            {
              "text": "RAG 中的向量检索",
              "link": "/fundamentals/embedding/embedding-in-rag.md"
            },
            {
              "text": "稀疏与稠密向量",
              "link": "/fundamentals/embedding/sparse-dense-vectors.md"
            },
            {
              "text": "向量模型选型",
              "link": "/fundamentals/embedding/embedding-model-selection.md"
            },
            {
              "text": "向量检索优化",
              "link": "/fundamentals/embedding/embedding-optimization.md"
            }
          ]
        },
        {
          "text": "Transformer 架构",
          "link": "/fundamentals/transformer-arch/",
          "icon": "mdi:chart-box-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Transformer 与 RNN、CNN",
              "link": "/fundamentals/transformer-arch/transformer-vs-rnn-cnn.md"
            },
            {
              "text": "Q、K、V",
              "link": "/fundamentals/transformer-arch/q-k-v-explained.md"
            },
            {
              "text": "注意力缩放",
              "link": "/fundamentals/transformer-arch/why-divide-sqrt-dk.md"
            },
            {
              "text": "多头注意力",
              "link": "/fundamentals/transformer-arch/multi-head-attention.md"
            },
            {
              "text": "位置编码",
              "link": "/fundamentals/transformer-arch/positional-encoding.md"
            },
            {
              "text": "编码器与解码器",
              "link": "/fundamentals/transformer-arch/encoder-decoder-variants.md"
            },
            {
              "text": "BERT 与 GPT",
              "link": "/fundamentals/transformer-arch/bert-vs-gpt.md"
            },
            {
              "text": "计算复杂度",
              "link": "/fundamentals/transformer-arch/transformer-complexity.md"
            }
          ]
        },
        {
          "text": "大模型基础",
          "link": "/fundamentals/llm-basics/",
          "icon": "mdi:robot-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "大模型入门",
              "link": "/fundamentals/llm-basics/what-is-llm.md"
            },
            {
              "text": "文本表示",
              "link": "/fundamentals/llm-basics/token-to-context.md"
            },
            {
              "text": "上下文窗口",
              "link": "/fundamentals/llm-basics/context-window.md"
            },
            {
              "text": "预训练",
              "link": "/fundamentals/llm-basics/pretraining.md"
            },
            {
              "text": "SFT 与 LoRA",
              "link": "/fundamentals/llm-basics/finetuning-sft-lora.md"
            },
            {
              "text": "采样参数",
              "link": "/fundamentals/llm-basics/inference-params.md"
            },
            {
              "text": "模型幻觉",
              "link": "/fundamentals/llm-basics/hallucination.md"
            },
            {
              "text": "提示词",
              "link": "/fundamentals/llm-basics/prompt-basics.md"
            },
            {
              "text": "工具调用",
              "link": "/fundamentals/llm-basics/function-calling.md"
            },
            {
              "text": "应用系统设计",
              "link": "/fundamentals/llm-basics/app-system-design.md"
            }
          ]
        },
        {
          "text": "训练与微调",
          "link": "/fundamentals/training-and-finetuning/",
          "icon": "mdi:tune-vertical",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "训练流程",
              "link": "/fundamentals/training-and-finetuning/training-pipeline.md"
            },
            {
              "text": "预训练目标",
              "link": "/fundamentals/training-and-finetuning/pretraining-deep.md"
            },
            {
              "text": "提示词、RAG 与微调",
              "link": "/fundamentals/training-and-finetuning/prompt-rag-finetuning.md"
            },
            {
              "text": "SFT 监督微调",
              "link": "/fundamentals/training-and-finetuning/sft.md"
            },
            {
              "text": "微调数据",
              "link": "/fundamentals/training-and-finetuning/finetuning-data.md"
            },
            {
              "text": "LoRA 与 QLoRA",
              "link": "/fundamentals/training-and-finetuning/lora-qlora.md"
            },
            {
              "text": "RLHF 与 DPO",
              "link": "/fundamentals/training-and-finetuning/rlhf-dpo.md"
            },
            {
              "text": "微调排错",
              "link": "/fundamentals/training-and-finetuning/finetuning-pitfalls.md"
            }
          ]
        },
        {
          "text": "推理与生成",
          "link": "/fundamentals/inference-and-generation/",
          "icon": "mdi:lightning-bolt-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "推理流程",
              "link": "/fundamentals/inference-and-generation/inference-pipeline.md"
            },
            {
              "text": "采样策略",
              "link": "/fundamentals/inference-and-generation/temperature-topk-topp.md"
            },
            {
              "text": "KV Cache",
              "link": "/fundamentals/inference-and-generation/kv-cache.md"
            },
            {
              "text": "流式输出",
              "link": "/fundamentals/inference-and-generation/streaming.md"
            },
            {
              "text": "结构化输出与工具调用",
              "link": "/fundamentals/inference-and-generation/structured-output.md"
            },
            {
              "text": "上下文长度",
              "link": "/fundamentals/inference-and-generation/context-length.md"
            },
            {
              "text": "推理加速",
              "link": "/fundamentals/inference-and-generation/inference-acceleration.md"
            },
            {
              "text": "生成问题排查",
              "link": "/fundamentals/inference-and-generation/generation-pitfalls.md"
            }
          ]
        },
        {
          "text": "多模态",
          "link": "/fundamentals/multimodal-basics/",
          "icon": "mdi:image-multiple-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "多模态入门",
              "link": "/fundamentals/multimodal-basics/what-is-multimodal.md"
            },
            {
              "text": "CLIP 图文对齐",
              "link": "/fundamentals/multimodal-basics/clip.md"
            },
            {
              "text": "视觉 Token",
              "link": "/fundamentals/multimodal-basics/visual-token.md"
            },
            {
              "text": "多模态生成",
              "link": "/fundamentals/multimodal-basics/multimodal-how.md"
            },
            {
              "text": "OCR 与文档理解",
              "link": "/fundamentals/multimodal-basics/ocr-doc-understanding.md"
            },
            {
              "text": "多模态 RAG",
              "link": "/fundamentals/multimodal-basics/multimodal-rag.md"
            },
            {
              "text": "多模态的限制",
              "link": "/fundamentals/multimodal-basics/multimodal-limitations.md"
            }
          ]
        },
        {
          "text": "评测指标",
          "link": "/fundamentals/evaluation-metrics/",
          "icon": "mdi:clipboard-check-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "评测入门",
              "link": "/fundamentals/evaluation-metrics/what-is-evaluation.md"
            },
            {
              "text": "分类指标",
              "link": "/fundamentals/evaluation-metrics/classification-metrics.md"
            },
            {
              "text": "生成质量指标",
              "link": "/fundamentals/evaluation-metrics/generation-metrics.md"
            },
            {
              "text": "大模型基准",
              "link": "/fundamentals/evaluation-metrics/llm-benchmarks.md"
            },
            {
              "text": "RAG 评测",
              "link": "/fundamentals/evaluation-metrics/rag-evaluation.md"
            },
            {
              "text": "应用评测",
              "link": "/fundamentals/evaluation-metrics/app-evaluation.md"
            }
          ]
        }
      ]
    }
  ],
  "/llm-100/": [
    {
      "text": "大模型 100 讲",
      "link": "/llm-100/",
      "icon": "mdi:school-outline",
      "children": [
        {
          "text": "模型基础",
          "link": "/llm-100/basics/",
          "icon": "mdi:book-open-page-variant-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "参数、数据与算力",
              "link": "/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md"
            },
            {
              "text": "Token",
              "link": "/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md"
            },
            {
              "text": "分词算法对比",
              "link": "/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md"
            },
            {
              "text": "Embedding",
              "link": "/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md"
            },
            {
              "text": "Transformer 架构",
              "link": "/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md"
            },
            {
              "text": "注意力计算",
              "link": "/llm-100/basics/Attention_机制到底在算什么.md"
            },
            {
              "text": "自注意力与多头注意力",
              "link": "/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md"
            },
            {
              "text": "RoPE 与 ALiBi",
              "link": "/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md"
            },
            {
              "text": "三类模型架构",
              "link": "/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md"
            },
            {
              "text": "GPT、BERT 与 T5",
              "link": "/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md"
            }
          ]
        },
        {
          "text": "模型训练",
          "link": "/llm-100/training/",
          "icon": "mdi:tune-vertical",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "预训练",
              "link": "/llm-100/training/what-is-pretraining.md"
            },
            {
              "text": "自回归语言模型",
              "link": "/llm-100/training/autoregressive-lm.md"
            },
            {
              "text": "MLM 与 CLM",
              "link": "/llm-100/training/mlm-vs-clm.md"
            },
            {
              "text": "损失计算",
              "link": "/llm-100/training/loss-calculation.md"
            },
            {
              "text": "梯度下降与反向传播",
              "link": "/llm-100/training/gradient-descent-backprop.md"
            },
            {
              "text": "训练数据与扩展定律",
              "link": "/llm-100/training/why-massive-data.md"
            },
            {
              "text": "训练显存开销",
              "link": "/llm-100/training/why-high-vram.md"
            },
            {
              "text": "FP16 与 BF16",
              "link": "/llm-100/training/mixed-precision.md"
            },
            {
              "text": "混合精度排错",
              "link": "/llm-100/training/混合精度训练_FP16_BF16_有什么区别.md"
            },
            {
              "text": "并行训练",
              "link": "/llm-100/training/parallel-strategies.md"
            },
            {
              "text": "ZeRO 与 FSDP",
              "link": "/llm-100/training/zero-deepspeed-fsdp.md"
            }
          ]
        },
        {
          "text": "微调与对齐",
          "link": "/llm-100/finetuning/",
          "icon": "mdi:tune-vertical",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "预训练与微调",
              "link": "/llm-100/finetuning/pretraining-vs-finetuning.md"
            },
            {
              "text": "微调数据构造",
              "link": "/llm-100/finetuning/finetuning-data-construction.md"
            },
            {
              "text": "SFT 指令微调",
              "link": "/llm-100/finetuning/sft-instruction-tuning.md"
            },
            {
              "text": "参数高效微调",
              "link": "/llm-100/finetuning/adapter-prefix-ptuning.md"
            },
            {
              "text": "LoRA",
              "link": "/llm-100/finetuning/lora-efficient-finetuning.md"
            },
            {
              "text": "QLoRA",
              "link": "/llm-100/finetuning/qlora-cost-reduction.md"
            },
            {
              "text": "RLHF",
              "link": "/llm-100/finetuning/why-rlhf.md"
            },
            {
              "text": "DPO 与 RLHF",
              "link": "/llm-100/finetuning/dpo-vs-rlhf.md"
            },
            {
              "text": "过拟合与遗忘",
              "link": "/llm-100/finetuning/finetuning-overfitting.md"
            },
            {
              "text": "微调效果评估",
              "link": "/llm-100/finetuning/evaluate-finetuned-model.md"
            }
          ]
        },
        {
          "text": "提示词",
          "link": "/llm-100/prompt-engineering/",
          "icon": "mdi:message-text-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "提示词与条件概率",
              "link": "/llm-100/prompt-engineering/Prompt_为什么能影响大模型输出.md"
            },
            {
              "text": "消息角色",
              "link": "/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md"
            },
            {
              "text": "零样本与少样本",
              "link": "/llm-100/prompt-engineering/Zero-shot_Few-shot_Chain-of-Thought_有什么区别.md"
            },
            {
              "text": "思维链",
              "link": "/llm-100/prompt-engineering/CoT_思维链提示为什么有效.md"
            },
            {
              "text": "提示词结构",
              "link": "/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.md"
            },
            {
              "text": "JSON 输出",
              "link": "/llm-100/prompt-engineering/如何让大模型稳定输出_JSON.md"
            },
            {
              "text": "提示词注入",
              "link": "/llm-100/prompt-engineering/Prompt_注入攻击是什么_如何防范.md"
            },
            {
              "text": "采样参数",
              "link": "/llm-100/prompt-engineering/Temperature_Top-p_Top-k_分别控制什么.md"
            },
            {
              "text": "输出一致性",
              "link": "/llm-100/prompt-engineering/为什么同一个_Prompt_每次输出不一样.md"
            },
            {
              "text": "提示词设计与评测",
              "link": "/llm-100/prompt-engineering/面试高频_如何设计一个高质量_Prompt.md"
            }
          ]
        },
        {
          "text": "RAG 检索",
          "link": "/llm-100/rag/",
          "icon": "mdi:database",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "RAG 入门",
              "link": "/llm-100/rag/what-is-rag.md"
            },
            {
              "text": "RAG 与微调",
              "link": "/llm-100/rag/rag-vs-finetuning.md"
            },
            {
              "text": "文档切分",
              "link": "/llm-100/rag/chunk-strategy.md"
            },
            {
              "text": "向量模型选型",
              "link": "/llm-100/rag/embedding-selection.md"
            },
            {
              "text": "向量数据库",
              "link": "/llm-100/rag/vector-db-in-rag.md"
            },
            {
              "text": "BM25",
              "link": "/llm-100/rag/bm25-in-rag.md"
            },
            {
              "text": "混合检索",
              "link": "/llm-100/rag/retrieval-methods.md"
            },
            {
              "text": "重排序",
              "link": "/llm-100/rag/rerank-importance.md"
            },
            {
              "text": "回答错误排查",
              "link": "/llm-100/rag/rag-wrong-answers.md"
            },
            {
              "text": "检索效果优化",
              "link": "/llm-100/rag/optimize-rag.md"
            }
          ]
        },
        {
          "text": "Agent 基础",
          "link": "/llm-100/agent-basics/",
          "icon": "mdi:cpu-64-bit",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Agent 与工作流",
              "link": "/llm-100/agent-basics/Agent_是什么_它和普通_ChatBot_有什么区别.md"
            },
            {
              "text": "Agent 组成",
              "link": "/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.md"
            },
            {
              "text": "工具调用",
              "link": "/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md"
            },
            {
              "text": "函数调用与 MCP",
              "link": "/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md"
            },
            {
              "text": "ReAct",
              "link": "/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md"
            },
            {
              "text": "循环终止",
              "link": "/llm-100/agent-basics/Agent_为什么容易陷入死循环.md"
            },
            {
              "text": "工作记忆",
              "link": "/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.md"
            },
            {
              "text": "短期与长期记忆",
              "link": "/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md"
            },
            {
              "text": "多 Agent 协作",
              "link": "/llm-100/agent-basics/多_Agent_协作到底解决什么问题.md"
            },
            {
              "text": "Agent 系统设计",
              "link": "/llm-100/agent-basics/面试高频_如何设计一个可落地的_Agent_系统.md"
            }
          ]
        },
        {
          "text": "Agent 工程",
          "link": "/llm-100/agent-engineering/",
          "icon": "mdi:cog-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "框架选型",
              "link": "/llm-100/agent-engineering/agent-framework-selection.md"
            },
            {
              "text": "LangGraph 工作流",
              "link": "/llm-100/agent-engineering/langgraph-controlled-workflow.md"
            },
            {
              "text": "Agent 状态机",
              "link": "/llm-100/agent-engineering/agent-state-machine.md"
            },
            {
              "text": "规划、执行与反思",
              "link": "/llm-100/agent-engineering/planner-executor-reflection.md"
            },
            {
              "text": "任务拆解",
              "link": "/llm-100/agent-engineering/agent-task-decomposition.md"
            },
            {
              "text": "工具接入",
              "link": "/llm-100/agent-engineering/agent-tool-integration.md"
            },
            {
              "text": "工具调用排错",
              "link": "/llm-100/agent-engineering/agent-tool-failure.md"
            },
            {
              "text": "权限与隔离",
              "link": "/llm-100/agent-engineering/agent-permissions-isolation.md"
            },
            {
              "text": "执行追踪",
              "link": "/llm-100/agent-engineering/agent-execution-tracing.md"
            },
            {
              "text": "运行稳定性",
              "link": "/llm-100/agent-engineering/agent-reliability.md"
            }
          ]
        },
        {
          "text": "应用系统",
          "link": "/llm-100/app-system/",
          "icon": "mdi:server",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "应用架构",
              "link": "/llm-100/app-system/llm-app-architecture.md"
            },
            {
              "text": "前后端协作",
              "link": "/llm-100/app-system/frontend-backend-model.md"
            },
            {
              "text": "模型 API 调用",
              "link": "/llm-100/app-system/model-api-request-lifecycle.md"
            },
            {
              "text": "SSE 与 WebSocket",
              "link": "/llm-100/app-system/sse-vs-websocket.md"
            },
            {
              "text": "提示词模板",
              "link": "/llm-100/app-system/prompt-template-system.md"
            },
            {
              "text": "对话历史",
              "link": "/llm-100/app-system/conversation-history.md"
            },
            {
              "text": "上下文预算",
              "link": "/llm-100/app-system/context-budget.md"
            },
            {
              "text": "应用缓存",
              "link": "/llm-100/app-system/llm-application-caching.md"
            },
            {
              "text": "面试助手",
              "link": "/llm-100/app-system/ai-interview-assistant.md"
            },
            {
              "text": "企业知识库问答",
              "link": "/llm-100/app-system/enterprise-knowledge-qa.md"
            }
          ]
        },
        {
          "text": "部署与推理",
          "link": "/llm-100/deployment/",
          "icon": "mdi:cloud-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "推理性能瓶颈",
              "link": "/llm-100/deployment/inference-bottlenecks.md"
            },
            {
              "text": "KV Cache",
              "link": "/llm-100/deployment/kv-cache.md"
            },
            {
              "text": "PagedAttention",
              "link": "/llm-100/deployment/paged-attention.md"
            },
            {
              "text": "vLLM",
              "link": "/llm-100/deployment/vllm-inference.md"
            },
            {
              "text": "量化方法",
              "link": "/llm-100/deployment/quantization-methods.md"
            },
            {
              "text": "硬件与显存估算",
              "link": "/llm-100/deployment/hardware-estimation.md"
            },
            {
              "text": "本地部署工具",
              "link": "/llm-100/deployment/local-serving-tools.md"
            },
            {
              "text": "高并发服务",
              "link": "/llm-100/deployment/high-concurrency-serving.md"
            },
            {
              "text": "API 与私有部署",
              "link": "/llm-100/deployment/api-vs-self-hosting.md"
            },
            {
              "text": "应用成本优化",
              "link": "/llm-100/deployment/llm-cost-optimization.md"
            }
          ]
        },
        {
          "text": "评测与安全",
          "link": "/llm-100/evaluation/",
          "icon": "mdi:shield-check-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "评测基准",
              "link": "/llm-100/evaluation/llm-evaluation-benchmarks.md"
            },
            {
              "text": "生成质量评估",
              "link": "/llm-100/evaluation/generation-quality-metrics.md"
            },
            {
              "text": "幻觉成因",
              "link": "/llm-100/evaluation/hallucination-causes.md"
            },
            {
              "text": "减少幻觉",
              "link": "/llm-100/evaluation/reducing-hallucinations.md"
            },
            {
              "text": "模型安全风险",
              "link": "/llm-100/evaluation/llm-security-risks.md"
            },
            {
              "text": "内容安全护栏",
              "link": "/llm-100/evaluation/content-safety-guardrails.md"
            },
            {
              "text": "日志与监控",
              "link": "/llm-100/evaluation/llm-observability.md"
            },
            {
              "text": "RAG、微调与 Agent",
              "link": "/llm-100/evaluation/rag-finetuning-agent.md"
            },
            {
              "text": "应用平台设计",
              "link": "/llm-100/evaluation/llm-platform-design.md"
            },
            {
              "text": "课程复习",
              "link": "/llm-100/evaluation/learning-roadmap.md"
            }
          ]
        }
      ]
    }
  ],
  "/frontier/": [
    {
      "text": "前沿专题",
      "link": "/frontier/",
      "icon": "mdi:fire",
      "children": [
        {
          "text": "Agent 记忆",
          "link": "/frontier/agent-memory-system/",
          "icon": "mdi:brain",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "记忆类型",
              "link": "/frontier/agent-memory-system/what-is-memory.md"
            },
            {
              "text": "隐私与噪声",
              "link": "/frontier/agent-memory-system/memory-risks.md"
            },
            {
              "text": "记忆读写与更新",
              "link": "/frontier/agent-memory-system/memory-design.md"
            }
          ]
        },
        {
          "text": "Agent Skills",
          "link": "/frontier/agent-skills/",
          "icon": "mdi:cpu-64-bit",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Skills 入门",
              "link": "/frontier/agent-skills/what-is-agent-skills.md"
            },
            {
              "text": "编写可复用 Skill",
              "link": "/frontier/agent-skills/reusable-skill.md"
            },
            {
              "text": "复杂任务封装",
              "link": "/frontier/agent-skills/skill-encapsulation.md"
            }
          ]
        },
        {
          "text": "Agentic RAG",
          "link": "/frontier/agentic-rag/",
          "icon": "mdi:database",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Agentic RAG 入门",
              "link": "/frontier/agentic-rag/what-is-agentic-rag.md"
            },
            {
              "text": "多步检索",
              "link": "/frontier/agentic-rag/multi-step-retrieval.md"
            },
            {
              "text": "复杂查询规划",
              "link": "/frontier/agentic-rag/complex-query-planning.md"
            }
          ]
        },
        {
          "text": "AI 编程",
          "link": "/frontier/ai-coding-agent/",
          "icon": "mdi:code-tags",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "编程 Agent 入门",
              "link": "/frontier/ai-coding-agent/what-is-coding-agent.md"
            },
            {
              "text": "代码修改流程",
              "link": "/frontier/ai-coding-agent/coding-agent-workflow.md"
            },
            {
              "text": "验证与回滚",
              "link": "/frontier/ai-coding-agent/reliable-coding-agent.md"
            }
          ]
        },
        {
          "text": "AI 工作流",
          "link": "/frontier/ai-workflow/",
          "icon": "mdi:cog-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "工作流入门",
              "link": "/frontier/ai-workflow/what-is-ai-workflow.md"
            },
            {
              "text": "节点设计",
              "link": "/frontier/ai-workflow/workflow-nodes.md"
            },
            {
              "text": "校验、重试与人工介入",
              "link": "/frontier/ai-workflow/stable-workflow.md"
            }
          ]
        },
        {
          "text": "上下文工程",
          "link": "/frontier/context-engineering/",
          "icon": "mdi:brain",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "上下文工程入门",
              "link": "/frontier/context-engineering/what-is-context-eng.md"
            },
            {
              "text": "上下文组织",
              "link": "/frontier/context-engineering/high-quality-context.md"
            }
          ]
        },
        {
          "text": "文档解析",
          "link": "/frontier/document-parsing/",
          "icon": "mdi:file-document-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "文件格式与解析",
              "link": "/frontier/document-parsing/what-is-file-parsing.md"
            },
            {
              "text": "PDF 版面还原",
              "link": "/frontier/document-parsing/pdf-parsing.md"
            },
            {
              "text": "清洗与入库",
              "link": "/frontier/document-parsing/parsing-to-knowledge-base.md"
            }
          ]
        },
        {
          "text": "GraphRAG",
          "link": "/frontier/graphrag/",
          "icon": "mdi:graph-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "GraphRAG 入门",
              "link": "/frontier/graphrag/what-is-graphrag.md"
            },
            {
              "text": "实体与关系提取",
              "link": "/frontier/graphrag/entity-extraction.md"
            },
            {
              "text": "应用场景与选型",
              "link": "/frontier/graphrag/graphrag-scenarios.md"
            }
          ]
        },
        {
          "text": "本地模型",
          "link": "/frontier/local-llm/",
          "icon": "mdi:server",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "本地部署选型",
              "link": "/frontier/local-llm/what-is-local-llm.md"
            },
            {
              "text": "本地运行工具",
              "link": "/frontier/local-llm/local-llm-tools.md"
            },
            {
              "text": "应用接入",
              "link": "/frontier/local-llm/local-llm-integration.md"
            }
          ]
        },
        {
          "text": "长上下文",
          "link": "/frontier/long-context-app/",
          "icon": "mdi:arrow-left-right",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "长上下文入门",
              "link": "/frontier/long-context-app/what-is-long-context.md"
            },
            {
              "text": "常见问题与成本",
              "link": "/frontier/long-context-app/long-context-pitfalls.md"
            },
            {
              "text": "长文档问答",
              "link": "/frontier/long-context-app/doc-qa-design.md"
            }
          ]
        },
        {
          "text": "Loop Engineering",
          "link": "/frontier/loop-engineering/",
          "icon": "mdi:refresh",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "反馈循环入门",
              "link": "/frontier/loop-engineering/Loop_Engineering_到底是什么.md"
            },
            {
              "text": "循环组成",
              "link": "/frontier/loop-engineering/一个_Loop_由哪些部分组成.md"
            },
            {
              "text": "结果验证",
              "link": "/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md"
            },
            {
              "text": "循环与 Agent",
              "link": "/frontier/loop-engineering/Loop_Engineering_和_Agent_有什么关系.md"
            },
            {
              "text": "代码自修复",
              "link": "/frontier/loop-engineering/最简单的_Loop_写代码_跑测试_自动修复.md"
            },
            {
              "text": "内容写作循环",
              "link": "/frontier/loop-engineering/Loop_Engineering_在内容创作里怎么用.md"
            },
            {
              "text": "循环终止与成本",
              "link": "/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md"
            }
          ]
        },
        {
          "text": "MCP 协议",
          "link": "/frontier/mcp-protocol/",
          "icon": "mdi:power-plug-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "MCP 入门",
              "link": "/frontier/mcp-protocol/what-is-mcp.md"
            },
            {
              "text": "服务端、客户端与工具",
              "link": "/frontier/mcp-protocol/mcp-server-client-tool.md"
            },
            {
              "text": "MCP 工具接入",
              "link": "/frontier/mcp-protocol/mcp-integration.md"
            }
          ]
        },
        {
          "text": "多轮对话",
          "link": "/frontier/multi-turn-dialogue/",
          "icon": "mdi:message-reply-text-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "多轮对话难点",
              "link": "/frontier/multi-turn-dialogue/why-multi-turn-hard.md"
            },
            {
              "text": "对话上下文",
              "link": "/frontier/multi-turn-dialogue/context-management.md"
            },
            {
              "text": "意图追踪与澄清",
              "link": "/frontier/multi-turn-dialogue/intent-tracking.md"
            }
          ]
        },
        {
          "text": "Vibe Coding",
          "link": "/frontier/vibe-coding/",
          "icon": "mdi:music-note",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Vibe Coding 入门",
              "link": "/frontier/vibe-coding/what-is-vibe-coding.md"
            },
            {
              "text": "适用场景",
              "link": "/frontier/vibe-coding/vibe-coding-scenarios.md"
            },
            {
              "text": "代码验证与排错",
              "link": "/frontier/vibe-coding/vibe-coding-pitfalls.md"
            }
          ]
        },
        {
          "text": "工作流编排",
          "link": "/frontier/workflow-orchestration/",
          "icon": "mdi:chart-box-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "流程编排入门",
              "link": "/frontier/workflow-orchestration/what-is-workflow.md"
            },
            {
              "text": "节点、状态与分支",
              "link": "/frontier/workflow-orchestration/workflow-concepts.md"
            },
            {
              "text": "任务拆解实例",
              "link": "/frontier/workflow-orchestration/workflow-examples.md"
            }
          ]
        }
      ]
    }
  ],
  "/interview/": [
    {
      "text": "面试题",
      "link": "/interview/",
      "icon": "mdi:fire",
      "children": [
        {
          "text": "大模型",
          "link": "/interview/llm-interview/",
          "icon": "mdi:robot-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "大模型基础",
              "link": "/interview/llm-interview/what-is-llm.md"
            },
            {
              "text": "Transformer 与大模型",
              "link": "/interview/llm-interview/transformer-vs-llm.md"
            },
            {
              "text": "Decoder-only",
              "link": "/interview/llm-interview/decoder-only.md"
            },
            {
              "text": "ChatGLM 与 GPT",
              "link": "/interview/llm-interview/chatglm-vs-gpt.md"
            },
            {
              "text": "Token 计算",
              "link": "/interview/llm-interview/token-basics.md"
            },
            {
              "text": "分词算法",
              "link": "/interview/llm-interview/tokenization.md"
            },
            {
              "text": "Embedding 与权重共享",
              "link": "/interview/llm-interview/embedding.md"
            },
            {
              "text": "Word2Vec 训练与评估",
              "link": "/interview/llm-interview/word2vec-evaluation.md"
            },
            {
              "text": "Word2Vec 到 BERT",
              "link": "/interview/llm-interview/word2vec-to-bert.md"
            },
            {
              "text": "Q、K、V 计算",
              "link": "/interview/llm-interview/attention-qkv.md"
            },
            {
              "text": "QK 点积与缩放",
              "link": "/interview/llm-interview/self-attention-qk.md"
            },
            {
              "text": "点积与加性注意力",
              "link": "/interview/llm-interview/attention-dot-vs-add.md"
            },
            {
              "text": "注意力掩码",
              "link": "/interview/llm-interview/attention-masking.md"
            },
            {
              "text": "编码器与解码器注意力",
              "link": "/interview/llm-interview/decoder-encoder-mha-diff.md"
            },
            {
              "text": "多头注意力",
              "link": "/interview/llm-interview/why-multi-head-attention.md"
            },
            {
              "text": "MHA、MQA 与 GQA",
              "link": "/interview/llm-interview/mha-mqa-gqa.md"
            },
            {
              "text": "位置编码推导",
              "link": "/interview/llm-interview/transformer-positional-encoding.md"
            },
            {
              "text": "位置编码与 RoPE",
              "link": "/interview/llm-interview/positional-encoding.md"
            },
            {
              "text": "LLaMA 旋转位置编码",
              "link": "/interview/llm-interview/llama-rope.md"
            },
            {
              "text": "FFN 与 SwiGLU",
              "link": "/interview/llm-interview/ffn-swiglu.md"
            },
            {
              "text": "LayerNorm 与 BatchNorm",
              "link": "/interview/llm-interview/layer-normalization.md"
            },
            {
              "text": "LayerNorm 与 RMSNorm",
              "link": "/interview/llm-interview/layernorm-rmsnorm.md"
            },
            {
              "text": "残差连接",
              "link": "/interview/llm-interview/transformer-residual.md"
            },
            {
              "text": "LSTM 与梯度消失",
              "link": "/interview/llm-interview/lstm-gradient-vanishing.md"
            },
            {
              "text": "序列到序列",
              "link": "/interview/llm-interview/transformer-seq2seq.md"
            },
            {
              "text": "ViT 图像建模",
              "link": "/interview/llm-interview/vit-vision-transformer.md"
            },
            {
              "text": "ViLT 多模态融合",
              "link": "/interview/llm-interview/vilt-vision-transformer.md"
            },
            {
              "text": "预训练目标",
              "link": "/interview/llm-interview/pretraining-objective.md"
            },
            {
              "text": "训练与对齐流程",
              "link": "/interview/llm-interview/training-pipeline.md"
            },
            {
              "text": "梯度裁剪",
              "link": "/interview/llm-interview/gradient-clipping.md"
            },
            {
              "text": "大规模数据训练",
              "link": "/interview/llm-interview/transformer-large-dataset.md"
            },
            {
              "text": "RLHF 流程",
              "link": "/interview/llm-interview/rlhf-pipeline.md"
            },
            {
              "text": "RLHF 与 DPO",
              "link": "/interview/llm-interview/rlhf-dpo.md"
            },
            {
              "text": "ORPO 对齐",
              "link": "/interview/llm-interview/orpo-alignment.md"
            },
            {
              "text": "LoRA 数学原理",
              "link": "/interview/llm-interview/lora-math-principle.md"
            },
            {
              "text": "LoRA 超参数",
              "link": "/interview/llm-interview/lora-hyperparams.md"
            },
            {
              "text": "LoRA 与 QLoRA",
              "link": "/interview/llm-interview/lora-qlora.md"
            },
            {
              "text": "上下文窗口",
              "link": "/interview/llm-interview/context-window.md"
            },
            {
              "text": "KV Cache 显存",
              "link": "/interview/llm-interview/kv-cache.md"
            },
            {
              "text": "Temperature 与 Top-p",
              "link": "/interview/llm-interview/sampling-temperature-topp.md"
            },
            {
              "text": "解码与采样策略",
              "link": "/interview/llm-interview/sampling-params.md"
            },
            {
              "text": "训练与推理显存",
              "link": "/interview/llm-interview/transformer-memory.md"
            },
            {
              "text": "推理性能瓶颈",
              "link": "/interview/llm-interview/transformer-bottleneck.md"
            },
            {
              "text": "模型评估",
              "link": "/interview/llm-interview/transformer-evaluation.md"
            },
            {
              "text": "涌现能力",
              "link": "/interview/llm-interview/emergent-abilities.md"
            },
            {
              "text": "幻觉与缓解",
              "link": "/interview/llm-interview/hallucination.md"
            },
            {
              "text": "MoE 架构",
              "link": "/interview/llm-interview/moe.md"
            }
          ]
        },
        {
          "text": "提示词",
          "link": "/interview/prompt-interview/",
          "icon": "mdi:message-text-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "提示词工程",
              "link": "/interview/prompt-interview/what-is-prompt.md"
            },
            {
              "text": "提示词结构",
              "link": "/interview/prompt-interview/prompt-structure.md"
            },
            {
              "text": "消息角色",
              "link": "/interview/prompt-interview/prompt-types.md"
            },
            {
              "text": "角色与约束设计",
              "link": "/interview/prompt-interview/prompt-design.md"
            },
            {
              "text": "零样本与少样本",
              "link": "/interview/prompt-interview/shot-types.md"
            },
            {
              "text": "上下文学习",
              "link": "/interview/prompt-interview/few-shot-learning.md"
            },
            {
              "text": "思维链原理",
              "link": "/interview/prompt-interview/chain-of-thought.md"
            },
            {
              "text": "思维链的限制",
              "link": "/interview/prompt-interview/chain-of-thought-2.md"
            },
            {
              "text": "结构化输出",
              "link": "/interview/prompt-interview/structured-output.md"
            },
            {
              "text": "提示词失效排查",
              "link": "/interview/prompt-interview/prompt-failure.md"
            },
            {
              "text": "减少幻觉",
              "link": "/interview/prompt-interview/reduce-hallucination.md"
            },
            {
              "text": "提示词注入防御",
              "link": "/interview/prompt-interview/prompt-injection.md"
            },
            {
              "text": "多轮上下文",
              "link": "/interview/prompt-interview/multi-turn-context.md"
            },
            {
              "text": "RAG 提示词",
              "link": "/interview/prompt-interview/rag-prompt.md"
            },
            {
              "text": "提示词优化与微调",
              "link": "/interview/prompt-interview/prompt-vs-finetuning.md"
            },
            {
              "text": "Spring AI 工具调用",
              "link": "/interview/prompt-interview/tool-calling-spring-ai.md"
            }
          ]
        },
        {
          "text": "RAG",
          "link": "/interview/rag-interview/",
          "icon": "mdi:database",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "RAG 基础",
              "link": "/interview/rag-interview/what-is-rag.md"
            },
            {
              "text": "LangChain RAG",
              "link": "/interview/rag-interview/what-is-rag-langchain.md"
            },
            {
              "text": "RAG 系统模块",
              "link": "/interview/rag-interview/rag-pipeline.md"
            },
            {
              "text": "RAG 执行流程",
              "link": "/interview/rag-interview/rag-complete-workflow.md"
            },
            {
              "text": "生产容错与降级",
              "link": "/interview/rag-interview/rag-full-pipeline.md"
            },
            {
              "text": "文档预处理",
              "link": "/interview/rag-interview/rag-document-pipeline.md"
            },
            {
              "text": "多格式文档解析",
              "link": "/interview/rag-interview/rag-multiformat-parsing.md"
            },
            {
              "text": "表格、代码与图片",
              "link": "/interview/rag-interview/rag-content-types.md"
            },
            {
              "text": "文档切分原理",
              "link": "/interview/rag-interview/chunk-strategy.md"
            },
            {
              "text": "切分策略对比",
              "link": "/interview/rag-interview/rag-chunk-strategies.md"
            },
            {
              "text": "切块大小与重叠",
              "link": "/interview/rag-interview/rag-chunk-size-overlap.md"
            },
            {
              "text": "向量检索原理",
              "link": "/interview/rag-interview/embedding-in-rag.md"
            },
            {
              "text": "Embedding 工程调优",
              "link": "/interview/rag-interview/rag-embedding.md"
            },
            {
              "text": "相似度算法",
              "link": "/interview/rag-interview/rag-similarity-algorithms.md"
            },
            {
              "text": "向量数据库",
              "link": "/interview/rag-interview/vector-database.md"
            },
            {
              "text": "pgvector 与专用数据库",
              "link": "/interview/rag-interview/vector-db-vs-traditional.md"
            },
            {
              "text": "向量增量更新",
              "link": "/interview/rag-interview/vector-db-incremental.md"
            },
            {
              "text": "元数据过滤",
              "link": "/interview/rag-interview/rag-metadata-filtering.md"
            },
            {
              "text": "检索方式对比",
              "link": "/interview/rag-interview/retrieval-methods.md"
            },
            {
              "text": "Top-K 选择",
              "link": "/interview/rag-interview/rag-topk.md"
            },
            {
              "text": "相似度阈值",
              "link": "/interview/rag-interview/rag-similarity-threshold.md"
            },
            {
              "text": "BM25 混合召回",
              "link": "/interview/rag-interview/hybrid-retrieval.md"
            },
            {
              "text": "RRF 结果融合",
              "link": "/interview/rag-interview/rag-hybrid-retrieval.md"
            },
            {
              "text": "重排计算原理",
              "link": "/interview/rag-interview/rerank.md"
            },
            {
              "text": "重排模型选型",
              "link": "/interview/rag-interview/rag-rerank.md"
            },
            {
              "text": "查询改写",
              "link": "/interview/rag-interview/query-rewrite.md"
            },
            {
              "text": "查询增强与过滤",
              "link": "/interview/rag-interview/contextual-query-enhancement.md"
            },
            {
              "text": "检索上下文组织",
              "link": "/interview/rag-interview/rag-prompt-design.md"
            },
            {
              "text": "来源引用",
              "link": "/interview/rag-interview/rag-source-citation.md"
            },
            {
              "text": "幻觉成因",
              "link": "/interview/rag-interview/rag-hallucination.md"
            },
            {
              "text": "RAG 评测",
              "link": "/interview/rag-interview/rag-evaluation.md"
            },
            {
              "text": "检索排错",
              "link": "/interview/rag-interview/rag-troubleshooting.md"
            },
            {
              "text": "Advanced RAG",
              "link": "/interview/rag-interview/advanced-rag.md"
            },
            {
              "text": "Modular RAG",
              "link": "/interview/rag-interview/modular-rag.md"
            },
            {
              "text": "Spring AI 模块化检索",
              "link": "/interview/rag-interview/spring-ai-modular-rag.md"
            },
            {
              "text": "Re-Reading",
              "link": "/interview/rag-interview/re-reading-spring-ai.md"
            },
            {
              "text": "RAG 与微调对比",
              "link": "/interview/rag-interview/rag-vs-finetuning.md"
            },
            {
              "text": "RAG 与微调协同",
              "link": "/interview/rag-interview/rag-vs-finetuning-2.md"
            }
          ]
        },
        {
          "text": "Agent",
          "link": "/interview/agent-interview/",
          "icon": "mdi:cpu-64-bit",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "Agent 基础",
              "link": "/interview/agent-interview/what-is-agent.md"
            },
            {
              "text": "Agent 与 Chain",
              "link": "/interview/agent-interview/langchain-agent-vs-chain.md"
            },
            {
              "text": "Agent 执行循环",
              "link": "/interview/agent-interview/agent-loop.md"
            },
            {
              "text": "ReAct 原理",
              "link": "/interview/agent-interview/react.md"
            },
            {
              "text": "ReAct 工程实现",
              "link": "/interview/agent-interview/react-principle.md"
            },
            {
              "text": "工具调用与函数调用",
              "link": "/interview/agent-interview/tool-calling.md"
            },
            {
              "text": "工具参数兼容",
              "link": "/interview/agent-interview/tool-schema-compatibility.md"
            },
            {
              "text": "MCP 基础",
              "link": "/interview/agent-interview/mcp.md"
            },
            {
              "text": "MCP 架构",
              "link": "/interview/agent-interview/mcp-architecture.md"
            },
            {
              "text": "MCP 传输模式",
              "link": "/interview/agent-interview/mcp-modes.md"
            },
            {
              "text": "MCP 调用流程",
              "link": "/interview/agent-interview/mcp-workflow.md"
            },
            {
              "text": "MCP 与函数调用",
              "link": "/interview/agent-interview/mcp-vs-function-calling.md"
            },
            {
              "text": "MCP 与 Skills",
              "link": "/interview/agent-interview/mcp-vs-skills.md"
            },
            {
              "text": "MCP 安全",
              "link": "/interview/agent-interview/mcp-security.md"
            },
            {
              "text": "A2A 协议",
              "link": "/interview/agent-interview/a2a-protocol.md"
            },
            {
              "text": "任务规划",
              "link": "/interview/agent-interview/agent-planning.md"
            },
            {
              "text": "记忆分层",
              "link": "/interview/agent-interview/agent-memory.md"
            },
            {
              "text": "上下文管理",
              "link": "/interview/agent-interview/context-management.md"
            },
            {
              "text": "执行失败恢复",
              "link": "/interview/agent-interview/agent-error-handling.md"
            },
            {
              "text": "多 Agent 协作",
              "link": "/interview/agent-interview/multi-agent.md"
            },
            {
              "text": "子 Agent 委派",
              "link": "/interview/agent-interview/multi-agent-collaboration.md"
            },
            {
              "text": "框架选型",
              "link": "/interview/agent-interview/agent-frameworks.md"
            },
            {
              "text": "Agentic RAG",
              "link": "/interview/agent-interview/agentic-rag.md"
            },
            {
              "text": "Agent 安全",
              "link": "/interview/agent-interview/agent-security.md"
            }
          ]
        },
        {
          "text": "AI 编程",
          "link": "/interview/ai-coding-interview/",
          "icon": "mdi:code-tags",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "编程助手原理",
              "link": "/interview/ai-coding-interview/core-principles.md"
            },
            {
              "text": "项目上下文",
              "link": "/interview/ai-coding-interview/project-context.md"
            },
            {
              "text": "仓库读写与检索",
              "link": "/interview/ai-coding-interview/code-repo-access.md"
            },
            {
              "text": "工具与沙箱",
              "link": "/interview/ai-coding-interview/tool-design.md"
            },
            {
              "text": "代码 Agent 流程",
              "link": "/interview/ai-coding-interview/code-agent-flow.md"
            }
          ]
        },
        {
          "text": "推理部署",
          "link": "/interview/model-deployment-interview/",
          "icon": "mdi:cloud-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "部署指标",
              "link": "/interview/model-deployment-interview/core-metrics.md"
            },
            {
              "text": "延迟与吞吐",
              "link": "/interview/model-deployment-interview/latency-throughput.md"
            },
            {
              "text": "KV Cache 显存",
              "link": "/interview/model-deployment-interview/kv-cache-deploy.md"
            },
            {
              "text": "vLLM 与分页注意力",
              "link": "/interview/model-deployment-interview/vllm-paged-attention.md"
            },
            {
              "text": "连续批处理",
              "link": "/interview/model-deployment-interview/continuous-batching.md"
            },
            {
              "text": "推理量化",
              "link": "/interview/model-deployment-interview/quantization.md"
            },
            {
              "text": "推理并行策略",
              "link": "/interview/model-deployment-interview/parallel-strategies.md"
            },
            {
              "text": "显存估算",
              "link": "/interview/model-deployment-interview/memory-estimation.md"
            },
            {
              "text": "服务性能优化",
              "link": "/interview/model-deployment-interview/optimization.md"
            },
            {
              "text": "OpenAI 兼容接口",
              "link": "/interview/model-deployment-interview/openai-compatible-api.md"
            },
            {
              "text": "限流与降级",
              "link": "/interview/model-deployment-interview/rate-limiting.md"
            },
            {
              "text": "监控与成本",
              "link": "/interview/model-deployment-interview/monitoring-cost.md"
            }
          ]
        },
        {
          "text": "系统设计",
          "link": "/interview/ai-system-design-interview/",
          "icon": "mdi:server",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "RAG、Agent 与工作流",
              "link": "/interview/ai-system-design-interview/rag-agent-workflow.md"
            },
            {
              "text": "多模型网关",
              "link": "/interview/ai-system-design-interview/multi-model-gateway.md"
            },
            {
              "text": "会话与上下文",
              "link": "/interview/ai-system-design-interview/session-context-management.md"
            },
            {
              "text": "超时与流式中断",
              "link": "/interview/ai-system-design-interview/failure-handling.md"
            },
            {
              "text": "权限与隐私",
              "link": "/interview/ai-system-design-interview/security-privacy.md"
            },
            {
              "text": "日志、监控与评测",
              "link": "/interview/ai-system-design-interview/logging-monitoring.md"
            },
            {
              "text": "工具调用平台",
              "link": "/interview/ai-system-design-interview/agent-tool-platform.md"
            },
            {
              "text": "企业知识库问答",
              "link": "/interview/ai-system-design-interview/knowledge-base-qa.md"
            },
            {
              "text": "多租户问答",
              "link": "/interview/ai-system-design-interview/multi-tenant-platform.md"
            },
            {
              "text": "智能客服",
              "link": "/interview/ai-system-design-interview/smart-customer-service.md"
            },
            {
              "text": "内容生成平台",
              "link": "/interview/ai-system-design-interview/content-generation-platform.md"
            },
            {
              "text": "面试助手",
              "link": "/interview/ai-system-design-interview/ai-interview-assistant.md"
            }
          ]
        },
        {
          "text": "场景与排错",
          "link": "/interview/high-freq-scenarios/",
          "icon": "mdi:star-outline",
          "collapsible": true,
          "expanded": false,
          "children": [
            {
              "text": "回答不准确",
              "link": "/interview/high-freq-scenarios/inaccurate-answer.md"
            },
            {
              "text": "检索未命中",
              "link": "/interview/high-freq-scenarios/rag-retrieval-failure.md"
            },
            {
              "text": "检索正确但回答错误",
              "link": "/interview/high-freq-scenarios/rag-answer-wrong.md"
            },
            {
              "text": "原文引用与溯源",
              "link": "/interview/high-freq-scenarios/answer-traceability.md"
            },
            {
              "text": "文档清洗与入库",
              "link": "/interview/high-freq-scenarios/doc-cleaning.md"
            },
            {
              "text": "混排文档检索",
              "link": "/interview/high-freq-scenarios/rag-multiformat.md"
            },
            {
              "text": "模糊问题与追问",
              "link": "/interview/high-freq-scenarios/ambiguous-query.md"
            },
            {
              "text": "工具失败与死循环",
              "link": "/interview/high-freq-scenarios/agent-failure.md"
            },
            {
              "text": "超时降级",
              "link": "/interview/high-freq-scenarios/timeout-degradation.md"
            },
            {
              "text": "API 成本优化",
              "link": "/interview/high-freq-scenarios/cost-optimization.md"
            },
            {
              "text": "提示词注入防御",
              "link": "/interview/high-freq-scenarios/prompt-injection-defense.md"
            },
            {
              "text": "MVP 方案取舍",
              "link": "/interview/high-freq-scenarios/mvp-approach.md"
            },
            {
              "text": "业务系统集成",
              "link": "/interview/high-freq-scenarios/business-integration.md"
            },
            {
              "text": "上线反馈与优化",
              "link": "/interview/high-freq-scenarios/feedback-optimization.md"
            }
          ]
        }
      ]
    }
  ],
  "/architecture/": [
    {
      "text": "工程架构",
      "link": "/architecture/",
      "icon": "diagram-project",
      "children": [
        {
          "text": "应用架构",
          "link": "/architecture/ai-system-design.md"
        },
        {
          "text": "模型网关",
          "link": "/architecture/model-gateway.md"
        },
        {
          "text": "多模型路由",
          "link": "/architecture/multi-model-routing.md"
        },
        {
          "text": "流式响应",
          "link": "/architecture/streaming.md"
        },
        {
          "text": "上下文管理",
          "link": "/architecture/context-management.md"
        },
        {
          "text": "Token 成本",
          "link": "/architecture/token-cost.md"
        },
        {
          "text": "异步任务与排队",
          "link": "/architecture/async-tasks.md"
        },
        {
          "text": "长任务执行",
          "link": "/architecture/long-task-execution.md"
        },
        {
          "text": "权限与隔离",
          "link": "/architecture/security-permission.md"
        },
        {
          "text": "容灾与降级",
          "link": "/architecture/high-availability.md"
        },
        {
          "text": "日志与链路追踪",
          "link": "/architecture/logging-trace.md"
        },
        {
          "text": "自动化评测",
          "link": "/architecture/ai-evaluation.md"
        }
      ]
    }
  ],
  "/projects/": [
    {
      "text": "项目实践",
      "link": "/projects/",
      "icon": "rocket",
      "children": [
        {
          "text": "单文档问答",
          "link": "/projects/ai-doc-qa.md"
        },
        {
          "text": "多文档知识库",
          "link": "/projects/ai-knowledge-base.md"
        },
        {
          "text": "论文阅读助手",
          "link": "/projects/ai-paper-reader.md"
        },
        {
          "text": "AI 搜索引擎",
          "link": "/projects/ai-search-engine.md"
        },
        {
          "text": "PrepRAG 系统设计",
          "link": "/projects/rag-project-experience.md"
        },
        {
          "text": "PrepRAG 后端实现",
          "link": "/projects/rag-implementation.md"
        },
        {
          "text": "PrepRAG 面试复盘",
          "link": "/projects/rag-interview-qa.md"
        },
        {
          "text": "长文写作助手",
          "link": "/projects/ai-writing-assistant.md"
        },
        {
          "text": "视频脚本生成",
          "link": "/projects/ai-video-script.md"
        },
        {
          "text": "简历优化助手",
          "link": "/projects/ai-resume-assistant.md"
        },
        {
          "text": "编程与调试助手",
          "link": "/projects/ai-coding-assistant.md"
        },
        {
          "text": "模拟面试助手",
          "link": "/projects/ai-interview-assistant.md"
        },
        {
          "text": "LangGraph 自动化",
          "link": "/projects/ai-workflow-automation.md"
        },
        {
          "text": "服务部署",
          "link": "/projects/project-deployment.md"
        }
      ]
    }
  ],
  "/resources/": [
    {
      "text": "工具与资源",
      "link": "/resources/",
      "icon": "mdi:compass-outline",
      "children": [
        {
          "text": "论文",
          "link": "/resources/papers.md"
        },
        {
          "text": "技术书",
          "link": "/resources/tech-books.md"
        },
        {
          "text": "开源项目",
          "link": "/resources/ai-opensource.md"
        },
        {
          "text": "模型平台",
          "link": "/resources/model-platform.md"
        },
        {
          "text": "API 平台",
          "link": "/resources/api-platform.md"
        },
        {
          "text": "向量数据库",
          "link": "/resources/vector-db.md"
        },
        {
          "text": "RAG 框架",
          "link": "/resources/rag-framework.md"
        },
        {
          "text": "Agent 框架",
          "link": "/resources/agent-framework.md"
        },
        {
          "text": "MCP Server",
          "link": "/resources/mcp-server.md"
        },
        {
          "text": "AI 编程工具",
          "link": "/resources/ai-coding-tools.md"
        },
        {
          "text": "通用 AI 工具",
          "link": "/resources/ai-tools.md"
        }
      ]
    }
  ],
  "/about/": [
    {
      "text": "关于本站",
      "link": "/about/",
      "icon": "circle-info",
      "children": [
        {
          "text": "作者介绍",
          "link": "/about/author-intro.md"
        },
        {
          "text": "常见问题",
          "link": "/about/faq.md"
        },
        {
          "text": "学习资料",
          "link": "/about/resource-pack.md"
        },
        {
          "text": "PDF 领取方式",
          "link": "/about/pdf-download.md"
        },
        {
          "text": "交流与关注",
          "link": "/about/join-community.md"
        },
        {
          "text": "贡献与勘误",
          "link": "/about/contributing-guide.md"
        },
        {
          "text": "更新日志",
          "link": "/about/update-history.md"
        }
      ]
    }
  ]
});
