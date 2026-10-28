---
title: AI 工具
icon: mdi:wrench-outline
tag:
  - AI 工具
  - 生成式 AI
  - 效率工具
---

# AI 工具导航：按任务选择，而非追热点

工具更新速度很快。选工具时先问三件事：输入数据是否允许上传、输出能否商用、是否能接入现有工作流。下面列的是稳定、官方入口明确的常见选择，而不是“万能工具排行榜”。

## 通用对话与研究

- [ChatGPT](https://chatgpt.com/)：通用写作、分析、数据处理、代码与多模态协作。
- [Claude](https://claude.ai/)：长文档阅读、写作、分析与代码任务。
- [Gemini](https://gemini.google.com/)：Google 生态、多模态和长上下文任务。
- [Perplexity](https://www.perplexity.ai/)：带来源链接的检索型问答；重要结论仍应回到原始来源核验。

适合：草稿、信息整理、会议准备与研究辅助。不要把未核实的输出直接当事实发布。

## 图像与设计

- [Adobe Firefly](https://firefly.adobe.com/)：适合 Adobe 工作流和商业设计协作。
- [Midjourney](https://www.midjourney.com/)：概念图、风格探索与视觉灵感。
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)：希望本地或自定义工作流的开发者。
- [Canva AI](https://www.canva.com/ai-image-generator/)：社媒、演示和轻设计素材。

重点确认：商用许可、人物肖像/品牌素材授权、生成内容标识以及团队资产管理。

## 音频、语音与视频

- [ElevenLabs](https://elevenlabs.io/docs)：语音合成、配音与语音 Agent。
- [OpenAI 音频能力](https://platform.openai.com/docs/guides/audio)：语音转写、生成和实时语音应用。
- [Runway](https://runwayml.com/)：视频生成与创意编辑。
- [Descript](https://www.descript.com/)：文字驱动的音视频剪辑与转写。

重点确认：声音克隆授权、使用者同意、版权归属、存储期限和敏感录音处理方式。

## 文档、知识库与自动化

- [Notion AI](https://www.notion.so/product/ai)：团队文档与知识工作流。
- [Zapier AI](https://zapier.com/ai)：SaaS 自动化与应用连接。
- [n8n](https://docs.n8n.io/)：可自托管的工作流自动化。
- [Dify](https://docs.dify.ai/)：可视化 LLM 应用、知识库与工作流。

自动化前先确定：谁能触发、谁能批准、失败如何重试、产生的外部写操作如何审计。

## 个人建议

先建立一个稳定的组合：一个对话/研究工具、一个创作工具、一个自动化工具。把常用 Prompt、素材规范、人工复核点和输出模板沉淀下来，比不断更换工具更能提高效率。

## 内容生产的推荐工作流

~~~text
需求与受众 → 资料检索/原始来源 → 大纲 → 初稿 → 人工核实
        → 视觉/音视频制作 → 合规检查 → 发布 → 数据与反馈复盘
~~~

AI 可以加速草稿、变体和格式转换，但事实、引用、品牌语气和发布责任仍应由人承担。

## 数据与版权检查

| 使用场景 | 必查事项 |
| --- | --- |
| 上传企业文档 | 是否用于训练、保留多久、管理员能否关闭数据共享 |
| 生成商业图片 | 平台条款、素材来源、人物和商标风险 |
| 克隆声音 | 被克隆者授权、可撤销机制、生成内容标识 |
| 自动发送消息 | 触发条件、审批人、退订/撤销与审计记录 |

## 避免工具碎片化

团队应该规定少量核心工具和统一素材库；将账号、权限、模板、成本归因和离职交接纳入管理。个人工具再强，无法被团队复用就很难形成长期效率。
