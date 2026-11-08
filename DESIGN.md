---
name: 安逸 AI 学习站
description: 墨绿、浅灰与手写字标组成的 AI 学习阅读系统
colors:
  forest: '#147459'
  forest-deep: '#0e5943'
  ink: '#17231e'
  muted: '#54625a'
  aluminum: '#edf0ee'
  paper: '#fbfcfa'
  white: '#ffffff'
  soft-green: '#dfeae3'
  rule: '#cdd7d0'
  amber: '#d6aa37'
  resource: '#dfe9e2'
typography:
  display:
    fontFamily: 'Noto Heading, Manrope, PingFang SC, sans-serif'
    fontSize: 'clamp(2.25rem, 3.4vw, 3rem)'
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: '-0.035em'
  headline:
    fontFamily: 'Manrope, PingFang SC, sans-serif'
    fontSize: 'clamp(1.5rem, 2vw, 1.875rem)'
    fontWeight: 700
    lineHeight: 1.4
    letterSpacing: '-0.025em'
  body:
    fontFamily: 'Manrope, PingFang SC, Microsoft YaHei, sans-serif'
    fontSize: '1.0625rem'
    fontWeight: 400
    lineHeight: 1.95
  label:
    fontFamily: 'Manrope, PingFang SC, sans-serif'
    fontSize: '0.88rem'
    fontWeight: 700
    lineHeight: 1.6
rounded:
  tag: '4px'
  control: '6px'
  compact: '8px'
  surface: '14px'
spacing:
  xs: '4px'
  sm: '8px'
  md: '16px'
  lg: '24px'
  xl: '32px'
  section: '32px'
  spacious: '48px'
components:
  button-primary:
    backgroundColor: '{colors.forest}'
    textColor: '{colors.white}'
    rounded: '{rounded.control}'
    padding: '0.65rem 1.2rem'
    height: '44px'
  button-primary-hover:
    backgroundColor: '{colors.forest-deep}'
  button-secondary:
    textColor: '{colors.ink}'
    rounded: '{rounded.control}'
    padding: '0.65rem 1.2rem'
    height: '44px'
  stage-active:
    backgroundColor: '{colors.forest}'
    textColor: '{colors.white}'
    size: '48px'
  course-preview:
    backgroundColor: '{colors.white}'
    rounded: '{rounded.surface}'
    padding: '12px'
  resource-invitation:
    backgroundColor: '{colors.resource}'
    textColor: '{colors.ink}'
    rounded: '{rounded.surface}'
    padding: '1.5rem 1.75rem'
---

# Design System: 安逸 AI 学习站

## Overview

**Creative North Star: "墨绿课程接线台"**

以安静的阅读背景承载清楚的学习结构。墨绿用于可进入的课程、当前阶段与主动操作，浅灰区分导航和正文，细线连接知识节点。实际教学图提供色彩与人物，界面为图文解释留出空间。

Logo 延续原来的手写 Anyi Ai 字形、小鸟与爱心，仅调整配色及背景。用户已否定几何三节点 Logo；后续品牌扩展应从旧字标轮廓出发。

教学图片的署名采用视觉智构原版彩色马克笔「@安逸i」Logo，保留彩色人物与手写字形，以透明图层直接叠加在每张原图的右下角。可见宽度为原图的 12%，边距为短边的 1.2%，图片保持原有尺寸与比例。首页缩略图从已署名封面生成，具体规则见 `config/image-branding.json`，原图与记录保存在 `output/image-branding/`。

**Key Characteristics:**

- 平面浅灰与墨绿，使用细线和留白组织内容。
- 标题、路线、课程预览与长文保持统一字阶。
- 学习操作优先，引流固定在导航、资料区与文章结尾。
- 彩色马克笔插画保留原生文件与人物；界面避免占用图文的注意力。

## Colors

墨绿强调操作与选中状态，深色文字承担阅读；暖金只作为小面积的连接标记及 Logo 细节。

### Primary

- **课程墨绿**：链接、主要按钮、活动节点与当前路线。
- **深墨绿**：悬停按钮、文字链接及需要更强对比的操作。

### Secondary

- **连接暖金**：阶段间连接点、课程目标标记，以及原字标的小鸟和爱心。

### Neutral

- **森林墨色**：标题、正文和强信息。
- **柔和灰绿**：说明、元信息、辅助文字。
- **铝灰背景**：首页、导航与侧栏。
- **阅读纸面**：长文主区域。
- **白色**：课程预览和二维码底。
- **淡绿与分隔线**：轻量容器、标签和内容边界。

**The Contrast Rule.** 选中项使用白字与墨绿底，不能被正文的深色 strong 样式覆盖。颜色只辅助表达状态，同时保留文字、形状或选中属性。

## Typography

**Display Font:** 自托管 Noto Heading（Noto Sans SC 标题字符子集），搭配 Manrope 与中文系统字体。

**Body Font:** 自托管 Manrope，中文优先使用 PingFang SC、Microsoft YaHei。正文使用真正的 400 字重，界面和标题使用现有 500 / 700 字重，避免用 Medium 字体冒充 Regular。字体及 OFL 许可放在 public/fonts，页面不依赖第三方字体请求。Regular 文件来自 Google Fonts 的 Manrope v23。

正文保持宽松行高，文章标题可以自然换行。首页的大标题承担主要层级，技术标签使用正常无衬线字体；等宽字体仅用于代码。

### Hierarchy

- **Display**：首页两行主标题；手机按实际宽度缩放。
- **Headline**：首页章节与知识方向标题。
- **Article title**：文章 H1 使用固定 2.125rem，手机 1.75rem，行高 1.45 / 1.5；长标题自然换行。
- **Article headings**：H2 1.5rem、H3 1.25rem、H4 1.0625rem，依靠间距与字重呈现层级。
- **Body**：文章正文 1.0625rem，手机 1rem，行高 1.95 / 1.9；表格 0.9375rem，代码 0.875rem，图注与目录 0.8125rem。
- **Label**：阶段、按钮、专题入口；辅助说明适当缩小，保持明确对比。

## Layout

首页内容容器最大 1408px，桌面两侧内边距 40px，手机 20px。首屏使用左右结构：文字与行动位于左侧，较大的课程图解和五阶段切换位于右侧；719px 以下上下堆叠。课程预览使用整幅图片，短标题与课程链接位于图片下方。

标题与两项主操作下方增加三列插画入口：AI 基础、项目实践、面试题。入口使用真实课程封面和短名称，以图片填充首屏，不增加介绍段落；手机保持三列，使用紧凑间距。

导航高度为桌面和移动端 64px。文章采用固定侧栏、居中的主阅读区与宽屏目录，正文容器最大 760px。侧栏宽度 16.5rem，手机侧栏 17rem，由主题按钮展开。页面中的表格、公式和代码允许自身横向滚动，页面整体不应横向溢出。

首页采用紧凑的信息密度：首屏上下间距各 32px，章节间距 32px，手机缩至 24px。学习路线缩为一组短链接，图解精选横向展示真实课程封面。六个课程文件夹在桌面和平板为三列，手机为两列；桌面文件夹舞台高 280px，窄屏 224px、手机随图片宽度在 188–302px 之间调整，展开后的插画与相邻内容保留间距。原首页长文收进原生阅读指南折叠区。学习路线页继续使用方向索引与列表，路线按钮与列表行分别以 60px、64px 为起点。主要按钮至少 44px。正文保留 1.95 行高。

## Elevation & Depth

平面为默认状态。背景色与细线区分阅读区和导航，不为每个区域添加悬浮阴影。下拉菜单和验证对话框使用柔和阴影：0 16px 48px rgb(23 35 30 / 12%)。选中的阶段节点带轻微下投影，表示当前操作焦点。课程文件夹中的图片纸张使用 0 5px 16px rgb(23 35 30 / 12%)，表达前后叠放关系。

## Shapes

按钮和输入框采用紧凑小圆角，课程预览与资料容器采用较大的统一圆角。圆形用于有实际阶段含义的 L1–L5 节点；细线承担连接关系。Logo 使用既有手写轮廓，不以规则几何形状替代品牌字形。

## Components

### Buttons

主要按钮为墨绿白字，次要按钮为透明底、细边线。悬停加深或淡绿填充，键盘焦点使用 2px 墨绿描边并留出 5px 距离。箭头使用统一线宽的 SVG。

### Navigation

按用户要求恢复旧版七栏：AI 基础、经典面试题、大模型100讲、前沿专题、学习路线、资源导航、关于。使用原有 MDI 分类图标、完整下拉目录与精简的知识点名称；每个下拉菜单提供该分类总目录入口。导航 Logo 为 140×46px，窄桌面和手机使用 124×42px。桌面保留搜索和领取 PDF，960–1199px 将搜索收为图标按钮，960–1023px 的 PDF 入口收进「关于」，959px 以下显示折叠菜单，手机菜单内保留 PDF 入口。下拉菜单可滚动，高度不超过视口剩余空间。导航保持可见，回到首页无需全屏 Logo 加载层。

栏目标题是可直接进入分类目录的真实链接；旁边的独立箭头按钮负责展开子目录，提供 `aria-expanded` 与 `aria-controls`。桌面移入标题可预览目录，方向下键进入第一项，Escape 关闭并返回展开按钮；手机采用同样的「标题进入、箭头展开」规则，选择后关闭导航。通过主题的自定义组件别名替换默认导航与下拉组件，避免把链接嵌进按钮。菜单随路由变化关闭，避免文章的延迟渲染打断刚打开的菜单。

侧栏与面包屑使用简短的知识点名称，例如「学习路线」「注意力掩码」「模型网关」，完整文章标题保留在正文。重构脚本保留人工整理的导航短名称。首页、路线和资料入口直接说明内容与操作，去掉口号、反问和重复的阶段评价。

### Stage selector

五阶段使用真实课程数据。点击或左右方向键切换时，标题、图片和课程链接同时更新。首页仅显示短标题与入口，完整路线页保留解释和练习目标。使用 tablist、tab、tabpanel 和 aria-selected，非活动项不占普通 Tab 顺序。

右侧课程预览默认从 L3 开始，每 6.5 秒轮播，提供当前页、上一张、下一张和播放控制，按钮至少 44px。鼠标停在轮播区时继续播放，点击阶段、翻页或触屏选择后重新计时；重复选择当前阶段也给出完整阅读间隔。离屏或页面隐藏时暂停计时，恢复后重新计时。只有主动暂停或键盘进入课程后停止自动播放，需主动继续；普通鼠标焦点不触发永久暂停。减少动态效果偏好关闭自动轮播，保留手动选择。

课程切换使用短暂水平位移、轻微模糊和透明度变化；默认内容始终可见。退出中的预览立即设为 inert，防止快速切换后误入旧课程。减少动态效果偏好关闭这些位移。

### Illustrated homepage

延续旧版图片轮播与文件夹展开。图集以 48 秒一轮平移，按钮、鼠标悬停、键盘操作、离屏和页面隐藏均可停止运动；键盘进入后转换为稳定的横向列表，重复副本不进入辅助技术与键盘顺序。减少动态效果模式隐藏副本和播放按钮，提供静态可滚动图集。

课程文件夹由三张现有教学图组成，默认轻微叠放并露出封面。鼠标移入或键盘聚焦时，当前文件夹中的图片向上弹出并扇形展开，移开后收回；进入 380ms、收回 220ms，使用现有 ease-out，阴影从 0 5px 16px 增至 0 12px 24px。每个文件夹独立响应。触屏和减少动态效果模式始终静态展开，点击直接进入目录。原图保留，首页使用 640px 或 960px 的 WebP 预览，来源、哈希和转换参数记录于 images/home-previews/manifest.json，由 scripts/prepare-home-previews.py 可重复生成。

### Learning directions

桌面为方向索引与课程行列表；手机变成横向短标签及单列路线。活动方向采用高对比配色，每条路线展示目标和实际入口。

### Article surfaces

正文采用 760px 阅读宽度、17px Regular 字重和清楚的段落节奏。标题体量收紧，标签与作者、阅读时长集中在标题下方。侧栏使用 14px 短名称，当前文章保留淡绿选中状态。图片按完整比例展示，图注与正文之间保留明确间距；表格、浅灰绿代码区、公式和引用共用同一组线条与文字颜色。

文章目录通过布局的 TOC 插槽渲染为原生 details / summary。1440px 以上位于右侧并默认展开，长标题完整换行，当前锚点有颜色和字重标记；窄屏位于正文上方并默认收起，选择章节后自动收起。锚点跳转按导航和目录摘要的实际高度留出位置，防止标题被固定区域遮挡。目录支持键盘展开、链接导航和独立滚动。保留文章路径、内容、教学图片、学习路线导航及文章结尾资料入口。

### Resource invitation

淡绿底承载公众号说明、四个真实关键词与二维码。首页使用宽版，文章末尾使用紧凑版，移动端二维码置于文字下方。领取按钮进入资料说明页，不暗示文件会直接下载。

### Verification dialog

使用同一墨绿配色。打开后焦点进入验证码输入框，Tab 保持在对话框内，Escape 关闭并恢复焦点；空输入和错误码提供明确反馈。

## Do's and Don'ts

### Do:

- **Do** 延续原手写字标、小鸟与爱心的轮廓。
- **Do** 使用真实课程、真实二维码与准确的资料领取方式。
- **Do** 让图解承担解释任务，让导航和正文保持稳定。
- **Do** 同时维护桌面、手机、焦点状态与减少动态效果模式。
- **Do** 用简短的栏目与知识点名称导航，完整标题用于正文。

### Don't:

- **Don't** 再用三节点几何字母替代用户要求保留的品牌字标。
- **Don't** 将新页面重新叠加为彩虹渐变或玻璃卡片；首页图片轮播必须可暂停，并尊重减少动态效果偏好。
- **Don't** 添加虚构的用户量、学习进度或效果保证。
- **Don't** 把资料领取入口混入正文段落，打断文章阅读。
- **Don't** 用口号、反问或「从……到……」的长句充当导航和首页标题。
