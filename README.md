<p align="center">
  <img src="src/.vuepress/public/logo.svg" alt="Anyi AI" width="240">
</p>

<h1 align="center">Anyi Ai Learn</h1>

<p align="center">一个专注于 AI 领域的学习导航站，旨在帮助开发者系统性地掌握 AI 技术栈</p>

<p align="center">
  <a href="https://www.anyiai.cc">🌐 在线访问</a> ·
  <a href="https://github.com/Easesgr/anyiai-docs">📦 GitHub</a> ·
  <a href="#网站预览">🖼️ 网站预览</a> ·
  <a href="#内容板块">📚 内容板块</a> ·
  <a href="#本地开发">💻 本地开发</a> ·
  <a href="#关注公众号">📱 关注公众号</a>
</p>

---

## 简介

Anyi Ai Learn 是一个免费开源的 AI 学习资源站点，内容覆盖从机器学习基础到大模型应用开发的完整知识体系。每篇文章围绕一个具体问题展开，讲清楚"是什么、为什么、怎么做"。

内容、配图和网站源码在此仓库持续更新，欢迎纠错、补充学习资料和参与建设。

当前版本包含 **502 个内容页面、3,310 张重构配图**，覆盖基础、课程、面试、专题、项目和架构。首页提供分阶段课程轮播和学习入口，全站采用统一的墨绿阅读样式；配图使用带人物的彩色马克笔插画与「@安逸i」署名。

## 网站预览

以下截图来自 [www.anyiai.cc](https://www.anyiai.cc/)，更新于 2026-11-08。

### 首页

![Anyi AI 首页](docs/screenshots/home.png)

### 学习路线

![按阶段进入课程的学习路线](docs/screenshots/learning-path.png)

### 文章阅读

![正文、插画与文章目录](docs/screenshots/reading.png)

### 资源导航

![AI 工具与资源导航](docs/screenshots/resources.png)

## 内容板块

| 板块 | 内容 | 在线阅读 |
|------|------|----------|
| AI 基础 | 机器学习、深度学习、Transformer、大模型、Token、Embedding、训练、推理、多模态与评估 | [进入](https://www.anyiai.cc/fundamentals/) |
| AI 经典面试题 | 大模型、Prompt、RAG、Agent、AI 编程、系统设计、部署与高频场景 | [进入](https://www.anyiai.cc/interview/) |
| 大模型 100 讲 | 大模型原理、提示词工程、RAG、Agent、微调、应用系统与部署 | [进入](https://www.anyiai.cc/llm-100/) |
| 前沿专题 | MCP、Agent Skills、Agentic RAG、GraphRAG、上下文工程与 AI 工作流 | [进入](https://www.anyiai.cc/frontier/) |
| 学习路线 | 从原理到 AI 工程、AI 面试备战等目标导向的学习路径 | [进入](https://www.anyiai.cc/learning-path/) |
| 实战项目 | 知识库问答、AI 编程助手、面试助手、搜索与工作流实践 | [进入](https://www.anyiai.cc/projects/) |
| 系统架构 | 对话状态、任务队列、Token 成本、权限、可观测性与部署 | [进入](https://www.anyiai.cc/architecture/) |
| 资源导航 | API 平台、模型平台、向量数据库、RAG 与 Agent 框架、AI 工具、论文与书籍 | [进入](https://www.anyiai.cc/resources/) |

## 技术栈

- **框架**：VuePress 2
- **主题**：vuepress-theme-hope
- **包管理**：pnpm
- **构建**：Vite
- **图片管理**：Git LFS 保存完整插画，缩略图与站点 Logo 直接保存在 Git 中
- **线上部署**：腾讯云 COS 静态网站与 CDN；GitHub Actions 校验源码

## 本地开发

使用 Node.js 24、pnpm 10.33.0 和 [Git LFS](https://git-lfs.com/)。完整图片约 8.35 GB；请预留依赖、LFS 对象和生产输出所需空间。

```bash
# 安装 Git LFS 后，先克隆源码，再单独拉取插画
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Easesgr/anyiai-docs.git
cd anyiai-docs
git lfs pull

# 安装指定版本的 pnpm
npm install --global pnpm@10.33.0

# 安装依赖
pnpm install --frozen-lockfile

# 启动开发服务器
pnpm docs:dev

# 构建生产版本
pnpm docs:build

# 清除缓存后启动
pnpm docs:clean-dev
```

生产构建输出位于 `src/.vuepress/dist/`。

只阅读 Markdown 或修改文章时，可以暂不执行 `git lfs pull`。运行完整站点、构建可部署版本或导出 PDF 前需要拉取图片；LFS 指针文件不能作为图片发布。

### PDF 导出

```bash
# 首次使用时安装浏览器
pnpm exec playwright install chromium

# 查看可导出的专题
pnpm docs:pdf --list

# 导出全部专题，默认输出至 output/pdf/
pnpm docs:pdf
```

### 部署配置

推送 `main` 后，`.github/workflows/deploy-docs.yml` 会检查 LFS 指针、安装锁定依赖并校验页面编译。自动校验不下载全部高清插画，也不发布其输出。

当前完整站点使用腾讯云 COS/CDN：本地先执行 `git lfs pull` 和 `pnpm docs:build`，将 `src/.vuepress/dist/` 中的内容上传到 COS 桶根目录，再刷新 CDN 缓存。先上传新资源，确认成功后替换 HTML，首页最后替换。

工作流保留手动 GitHub Pages 部署选项，默认关闭。只有在 Actions 手动运行并启用 `deploy_github_pages` 时，才会拉取原图并尝试发布；发布前检查 1 GB 大小限制。完整站点超过该限制，应使用 COS/CDN。对于精简后的 Fork，可在 Settings → Pages 中选择 `gh-pages` 分支。

Fork 后请修改 `src/.vuepress/theme.ts` 中的 `hostname`、`repo`、作者信息，以及 `src/.vuepress/config.ts` 中的网站信息。当前站点使用根路径 `/`；如果部署到 GitHub Pages 的项目子路径，还需调整 `base` 和静态资源路径。

可选配置：`GOOGLE_SITE_VERIFICATION` 用于构建时输出 Google 验证标签；`BAIDU_PUSH_TOKEN` 通过 GitHub Actions 仓库 Secret 配置，用于手动部署后提交更新页面。不配置百度 Token 时，工作流会跳过提交。

本地环境文件、明文阅读码、缓存、浏览器记录、原始素材和 PDF 导出不纳入版本控制。阅读组件使用的哈希配置会随前端发布，可按需修改 `src/.vuepress/readmore.config.ts`。

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
│   ├── interview/            # AI 经典面试题
│   ├── llm-100/              # 大模型 100 讲
│   ├── frontier/              # 前沿专题
│   ├── learning-path/        # 学习路线
│   ├── projects/             # 实战项目
│   ├── architecture/         # AI 系统架构
│   ├── resources/             # 资源导航
│   └── about/                 # 关于
├── docs/screenshots/         # 网站实景截图
├── scripts/                 # PDF 导出、内容校验与可选的重构工具
├── config/image-branding.json # 图片署名规则
├── .gitattributes           # 高清插画的 Git LFS 规则
├── .github/workflows/        # 源码校验与手动部署
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

<p align="center">
  <img src="src/.vuepress/public/images/wechat-qrcode.png" alt="安逸Ai 公众号" width="200">
</p>

## 贡献

欢迎提交 Issue 和 Pull Request。新增或调整文章时，请同步维护导航和侧边栏，并在提交前运行 `pnpm docs:build`。

内容重构、图片署名与首页预览工具见 [CONTENT_REBUILD.md](CONTENT_REBUILD.md)。这些是可选的维护工具，依赖单独配置的 AnyiOS 环境；普通阅读、开发和构建无需接入模型服务。生成记录、原始回执和本地配置保存在被 Git 忽略的 `output/` 中。

## 许可证

[MIT License](LICENSE)
