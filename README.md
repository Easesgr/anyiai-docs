<p align="center">
<img width="1448" height="1086" alt="logo" src="https://github.com/user-attachments/assets/927b5b93-052b-4eb8-9eb1-75c58f6ee4f0" />
</p>

<h1 align="center">Anyi Ai Learn</h1>

<p align="center">一个专注于 AI 领域的学习导航站，旨在帮助开发者系统性地掌握 AI 技术栈</p>

<p align="center">
  <a href="https://www.anyiai.cc">🌐 在线访问</a> ·
  <a href="#内容板块">📚 内容板块</a> ·
  <a href="#本地开发">💻 本地开发</a> ·
  <a href="#关注公众号">📱 关注公众号</a>
</p>

---

## 简介

Anyi Ai Learn 是一个免费开源的 AI 学习资源站点，内容覆盖从机器学习基础到大模型应用开发的完整知识体系。每篇文章围绕一个具体问题展开，讲清楚"是什么、为什么、怎么做"。

> 📢 资料文章正在持续整理中，目前已完成 AI 基础和大模型面试题板块，其他板块陆续更新中。

## 内容板块

### AI 基础 ✅

10 大模块，80+ 篇文章，从零构建完整的 AI 知识体系：

| 模块 | 文章数 | 核心内容 |
|------|--------|----------|
| 机器学习基础 | 10 篇 | 经典算法、特征工程、模型评估 |
| 深度学习基础 | 12 篇 | 神经网络、CNN、RNN、注意力机制 |
| Transformer 架构 | 8 篇 | Self-Attention、位置编码、BERT vs GPT |
| 大模型基础 | 10 篇 | 预训练、微调、Prompt、幻觉问题 |
| Token 与分词 | 5 篇 | BPE、中文分词、Token 预算 |
| Embedding | 6 篇 | 语义相似度、向量检索、模型选型 |
| 训练与微调 | 8 篇 | SFT、LoRA、RLHF、DPO |
| 推理与生成 | 8 篇 | KV Cache、流式输出、采样参数 |
| 多模态基础 | 7 篇 | CLIP、视觉 Token、多模态 RAG |
| 评测指标 | 6 篇 | BLEU、ROUGE、BERTScore、MMLU |

### AI 面试指南 ✅

18 篇大模型面试高频问题，覆盖基础概念、架构设计、训练流程和推理优化。

### AI 应用开发 🔄

Prompt Engineering、RAG、Agent、Function Calling 等核心开发技能。

### 前沿专题 🔄

MCP 协议、Agentic RAG、GraphRAG、上下文工程等前沿技术。

### 资源导航 🔄

API 平台、模型平台、向量数据库、Agent 框架等工具与资源。

## 技术栈

- **框架**：VuePress 2
- **主题**：vuepress-theme-hope
- **包管理**：pnpm
- **部署**：GitHub Actions → GitHub Pages

## 本地开发

```bash
# 安装依赖
pnpm install

# 启动开发服务器
pnpm docs:dev

# 构建生产版本
pnpm docs:build

# 清除缓存后启动
pnpm docs:clean-dev
```

## 项目结构

```
anyi-docs/
├── src/
│   ├── .vuepress/
│   │   ├── config.ts          # VuePress 配置
│   │   ├── theme.ts           # 主题配置
│   │   ├── navbar/            # 导航栏配置
│   │   ├── sidebar/           # 侧边栏配置
│   │   ├── styles/            # 自定义样式
│   │   ├── components/        # 自定义组件
│   │   ├── readmore.config.ts # 阅读锁定配置
│   │   └── public/            # 静态资源
│   ├── fundamentals/          # AI 基础
│   ├── interview/             # AI 面试指南
│   ├── ai-app/                # AI 应用开发
│   ├── frontier/              # 前沿专题
│   ├── resources/             # 资源导航
│   └── about/                 # 关于
├── package.json
└── README.md
```

## 内容风格

- **说人话**：不堆公式，不拽术语，用大白话讲清楚原理
- **有问题导向**：每篇文章围绕一个具体问题展开
- **有体系**：模块之间有逻辑关系，学习路径清晰
- **能落地**：不只是理论，还有实战建议和选型指南

## 关注公众号

扫描下方二维码关注微信公众号 **安逸Ai**，获取更多 AI 学习资料和更新通知。

<img width="1280" height="1280" alt="qrcode_for_gh_ffeb6e481fc7_1280" src="https://github.com/user-attachments/assets/4c634d49-8c12-4593-9751-020ba3b4400b" />

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

MIT License
