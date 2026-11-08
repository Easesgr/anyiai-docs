# 全站内容重构

本轮已完成 502 个页面：原有 462 页重新写作，并补齐「大模型 100 讲」四个模块已列出但未完成的 40 讲。全部正文和 3,310 张配图已整理回 `src/`，侧边栏、首页与 `llms.txt` 已同步更新。协调任务于 2026 年 11 月 3 日 22:54（北京时间）完成，状态为 `completed`；各页版本与生成记录保存在 `output/content-rebuild/`。

## 内容组织

学习顺序为「学习路径 → AI 基础 → 大模型课程 → 专题与项目 → 系统架构」。面试栏目用于检验理解和训练表达。每组先制定编排与学习目标，再按各篇职责写作，避免基础、课程、面试与工程文章互相重复。

技术文章包含贯穿例子、条件、机制说明、可检查的输入输出和边界；工程文章补充失败路径与验收；站务和作者信息以原稿事实为依据。原稿中的命令或提示词仅为参考资料，不作为执行指令。

## 模型与图片

- 文字生成、主编编辑、事实核查均读取 AnyiOS 当前默认文字模型。启动时为 `gemini-3.8-flash-high`。
- 图片通过 AnyiOS 图片服务读取当前默认图片模型。启动时为 `gpt-image-2`，提供方原生图片与回执保存在 `output/content-rebuild/`；网站使用添加署名的发布副本。
- 配图统一使用视觉智构安逸i彩色马克笔风格。所有图片提示词均要求可辨认的人类角色参与解释，采用场景、流程、结构、对比和失败分析等不同画法。
- 技术文章为封面加六张解释图；栏目索引为四张图；短站务页一张图。正文说明每幅图的关系。

本轮已生成并集成 3,310 张新图，其中首页包含四张解释图。网站首页 logo、真实二维码等身份标识保留。

## 验收与保存

每篇经过写作、独立技术审稿、必要重写、主编编辑和最后事实核查。按用户最新要求，本轮跳过逐图审核标签、箭头和正文关系以及由此触发的自动重画。该选择保存在 `output/content-rebuild/publication-policy.json`，续跑会沿用；集成记录明确标记为 `skipped_by_user`，不生成虚假的审核通过记录。

正文、主编记录、图片提示词、生成回执与原生文件版本必须一致，才允许写入 `src/`。执行期间如用户修改源页面，导入会拒绝覆盖。原稿备份、修订、历史图片与模型调用回执保存在 `output/content-rebuild/`（已加入 Git 忽略）。真实密钥只读取 AnyiOS 配置，不写入本项目。

文章完成且图片生成齐全后，协调脚本写回文章和图片，更新侧边栏、首页图片引用与课程数量以及 `llms.txt`，最后检查页面、站内链接、图片引用、原图与发布副本的来源记录、内容版本和导航覆盖。该过程不执行构建或部署。

## 图片署名

教学图片使用视觉智构彩色马克笔风格的原版「@安逸i」Logo，资产与出处位于 `src/.vuepress/public/images/branding/`。站点的手写字标、应用图标与真实二维码保留原样。

署名规则位于 `config/image-branding.json`：Logo 可见宽度为图片宽度的 12%，使用透明图层直接叠加在原图右下角，保持原图尺寸与画面比例。PNG 与无损 WebP 的署名区域以外像素保留。首页缩略图由署名后的封面生成，避免重复加标。

发布前的图片、逐图转换记录与汇总清单保存在 `output/image-branding/`，该目录已加入 Git 忽略。提供方原图和模型回执不被修改。重新集成文章时沿用相同署名流程；重复执行会核对记录并跳过已完成图片，检测到外部修改或 Logo、规则变更时会拒绝静默覆盖。

按用户纠正，已将之前新增底边的方案改为直接叠加。`apply --update-policy` 核对公开图和原始备份，归档旧署名记录，再从原图生成新版本；中断后可续跑，用户在公开图上的额外修改仍会阻止覆盖。旧实现与规则保存在 `output/image-branding/before-overlay-correction/`。全部已处理图片的原尺寸恢复记录位于 `output/image-branding/overlay-dimensions.json`，来源校验位于 `output/image-branding/overlay-verification.json`。88 项流程测试通过，AnyiOS 已确认文章使用原始画布尺寸、首页使用更新后的缩略图，记录位于 `output/image-branding/overlay-browser-verification.json`。

本次已完成 4,823 张原尺寸图片和 22 张首页缩略图，共 4,845 张。3,310 张重构配图的原生来源和署名副本均通过来源校验，11 项二维码、站点字标与图标保持原字节。另有一张不再被当前文章引用的历史图片 `token-and-tokenizer/20260606_235032_268992_01.png` 原先已截断，Git 历史和线上副本也同样损坏；保留原件与单独备份，不用缺失内容生成替代画面。完整批量校验会如实报告这一项未完成，记录位于 `output/image-branding/blocked-inputs.json`。

此次内容总校验还有一项历史快照差异：`src/README.md` 已按用户要求调整为折叠阅读指南，与内容重构时的原集成哈希不同。保留当前首页和原集成记录，未用新哈希覆盖历史证明；其余 501 页内容记录和全部重构配图、首页预览没有校验问题。详见 `output/image-branding/content-validation.json`。

协调流程在集成和更新导航之后自动刷新已有首页预览，再执行最终内容校验。单独运行 `rebuild-content.py integrate` 后，需要按下列命令手动刷新预览。

这些维护命令需要单独安装并配置 AnyiOS。先把 `ANYIOS_ROOT` 设为自己的 AnyiOS 源码路径，再在本项目目录运行；不需要重新生成内容时，无需执行这些命令。

```sh
export ANYIOS_ROOT=/path/to/AnyiOS

"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/brand-content-images.py dry-run
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/brand-content-images.py apply --jobs 4
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/prepare-home-previews.py
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/prepare-home-previews.py --verify
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/brand-content-images.py verify
```

以后如需明确替换已有署名规则，使用 `apply --update-policy`；已有文章 `published_images` 证明的页面还需重新集成以更新派生记录，原生来源记录保持原样。

## 完成验证

- 502 页正文的主编编辑与最终事实核查记录均与当前版本一致；40 篇新增课程已进入导航。
- 全站内容校验通过：页面、站内链接、3,310 张原生图片字节与集成记录一致，501 个非首页页面全部进入侧边栏。
- 502 个页面的开发环境 Markdown 编译通过；13,911 个公式检查无错误。
- 62 项流程回归测试通过，Python 的 Ruff E9/F 检查通过。修复了导入时 YAML 日期被转成字符串的问题，38 页保留原日期类型。
- AnyiOS 浏览器中的首页、新增 Agent 状态机课程、最终修订的向量数据库文章均正常加载，图片文件加载正常，无页面脚本错误、公式错误或横向溢出。该检查没有核对图片中的标签、箭头或人物。

验证报告分别为 `validation.json`、`dev-render-check.json`、`formula-check.json`、`browser-integrated-final.json` 和 `final-regression-tests.log`，均位于 `output/content-rebuild/`。

`git diff --check` 仍有尾随空白提示，并将教学代码块内的三行 SEARCH/REPLACE 标记识别为冲突标记。没有批量改写这些正文空白，避免改变 Markdown 显式换行及代码示例；具体位置保存在 `diff-whitespace-check.json`。本轮未执行构建、打包或部署。

## 继续运行

使用 AnyiOS 的 Python 环境，在本项目目录运行：

```sh
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/continue-rebuild.py \
  --app-root "$ANYIOS_ROOT" \
  --text-concurrency 6 --image-concurrency 58 --skip-visual-review
```

协调任务有进程锁，避免同时启动两份。默认文字服务限流时，按其冷却时间保存断点；期间继续生成图片，不自动换模型。运行状态为 `completed` 才表示所有阶段结束；`needs_attention` 表示需要检查记录中的错误和日志。

查看当前内容状态：

```sh
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/rebuild-content.py status \
  --app-root "$ANYIOS_ROOT"
```

最终校验与脚本回归测试：

```sh
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B scripts/validate-rebuilt-content.py
node scripts/check-content-formulas.mjs
node scripts/check-content-rendering.mjs http://127.0.0.1:8399/
"$ANYIOS_ROOT/workflow/.venv/bin/python" -B -m unittest discover -s scripts -p 'test_*.py' -q
```

本地预览使用 `pnpm docs:dev --host 127.0.0.1 --port 8399`，数学公式已启用 KaTeX。
`check-content-rendering.mjs` 需要这个开发服务已启动，它逐页请求开发环境的 Markdown 编译结果，检查 Vue 模板等渲染错误，不执行构建。
