export const learningStages = [
  {
    id: 'L1', label: '接口调用', title: '模型 API 调用',
    description: '发送请求、解析响应，处理超时、重试与 Token 计费。',
    outcome: '练习：封装一个模型 API 客户端。',
    href: '/llm-100/app-system/model-api-request-lifecycle.html',
    linkLabel: '阅读课程', cover: '/images/content-rebuild/b976b18f3a2e241d/cover.png',
    topics: ['Token 与采样', '结构化输出', '超时与重试'],
  },
  {
    id: 'L2', label: '模型原理', title: 'Transformer 架构',
    description: '注意力、位置编码与张量计算，以及推理的计算开销。',
    outcome: '练习：计算一次推理的显存开销。',
    href: '/fundamentals/transformer-arch/',
    linkLabel: '阅读课程', cover: '/images/content-rebuild/914738cd43afa552/cover.png',
    topics: ['注意力机制', 'KV Cache', '训练与微调'],
  },
  {
    id: 'L3', label: '应用实践', title: 'RAG 知识问答',
    description: '文档切分、检索与重排，搭建带来源引用的问答流程。',
    outcome: '练习：为知识库问答添加原文引用。',
    href: '/llm-100/rag/what-is-rag.html',
    linkLabel: '阅读课程', cover: '/images/content-rebuild/1ba02e09ddd23a13/cover.png',
    topics: ['知识库检索', '工具调用', 'Agent 状态机'],
  },
  {
    id: 'L4', label: '系统架构', title: '模型网关',
    description: '统一模型接口，配置限流、超时与链路追踪。',
    outcome: '练习：为模型服务配置超时和限流。',
    href: '/architecture/model-gateway.html',
    linkLabel: '阅读课程', cover: '/images/content-rebuild/cba0a54969a72542/cover.png',
    topics: ['模型网关', '并发与限流', '全链路追踪'],
  },
  {
    id: 'L5', label: '效果评测', title: '应用评测',
    description: '构建评测集，比较回答质量并排查错误。',
    outcome: '练习：用评测集比较两版应用的效果。',
    href: '/architecture/ai-evaluation.html',
    linkLabel: '阅读课程', cover: '/images/content-rebuild/d9846430d8cb799a/cover.png',
    topics: ['评测集设计', '质量门禁', '故障复盘'],
  },
];

export const learningDirections = [
  {
    id: 'engineering', label: '应用开发', description: '模型调用、RAG 与 Agent',
    routes: [
      { title: 'AI 开发入门', description: '模型 API、结构化输出与错误处理。', href: '/learning-path/ai-engineer.html', stage: 'L1 → L3' },
      { title: 'RAG 实战', description: '文档切分、混合检索、重排与评测。', href: '/learning-path/rag-practice.html', stage: 'L3 → L5' },
      { title: 'Agent 开发', description: '工具调用、状态编排与失败恢复。', href: '/learning-path/agent-engineer.html', stage: 'L3 → L4' },
    ],
  },
  {
    id: 'principles', label: '原理与架构', description: '计算机制与服务设计',
    routes: [
      { title: '模型原理与部署', description: 'Prefill、Decode、KV Cache 与硬件选型。', href: '/learning-path/from-principles-to-ai-engineering.html', stage: 'L2 → L4' },
      { title: '大模型应用开发', description: '后端接口、会话管理、监控与容错。', href: '/learning-path/llm-app-dev.html', stage: 'L1 → L4' },
    ],
  },
  {
    id: 'interview', label: '面试准备', description: '系统设计与算法考点',
    routes: [
      { title: '系统设计面试', description: '性能估算、权限隔离与故障处理。', href: '/learning-path/ai-interview-prep.html', stage: 'L4 → L5' },
      { title: '算法工程师面试', description: '注意力推导、显存计算、LoRA 与 DPO。', href: '/learning-path/algorithm-interview.html', stage: 'L2 → L5' },
    ],
  },
];
