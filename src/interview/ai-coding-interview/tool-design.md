---
title: AI 编程中如何设计工具调用与执行沙箱？
icon: mdi:wrench-outline
tag:
- AI编程面试
- 工具设计
- 封装
description: 详解 AI 编程场景下的工具接口设计、长输出截断、PTY 终端会话保持与安全沙箱防护机制，帮助开发者掌握生产级代码 Agent 的实现细节。
---

在构建代码代理（Coding Agent）时，语言模型本身只能生成文本，无法直接读写开发者的磁盘或运行测试命令。宿主环境（Host）必须通过一套工具接口（Tools / Functions）向模型开放有限的操作能力。

在真实工程中，代码代理执行失败往往不是因为模型的推理能力不足，而是工具接口定义模糊、编辑定位容易漂移、命令输出打爆了上下文窗口，或者缺乏执行沙箱带来了安全隐患。本文围绕接口设计、上下文截断与安全隔离三个核心维度，梳理代码代理工具系统的设计要点。

![AI 编程工具调用与安全沙箱整体架构](/images/content-rebuild/a44ecd46f02f890e/cover.png)

---

## 核心结论口述：面试中如何两分钟讲清工具设计

如果在面试中被问到“如何设计 AI 编程场景下的工具调用与沙箱”，可以按以下结构清晰作答：

> 编程代理的工具设计核心在于四件事：**局部准确定位、上下文预算防护、结构化执行反馈与多层安全隔离**。
>
> 1. **文件修改工具**：避免让模型全量重写大文件，也尽量不依赖绝对行号。主流做法是采用带上下文的“字符串块精确匹配替换”（`old_str` 替换为 `new_str`），并在执行前校验目标块在文件中的唯一性，减少模型对行号计算漂移导致的修改错误。
> 2. **代码检索工具**：文件树和文本搜索（如集成 ripgrep）必须设置结果条数上限与 Token 预算，默认忽略构建产物与依赖目录，防止一次搜索输出几千行撑爆上下文。
> 3. **终端执行工具**：通过伪终端（PTY）保持持久会话，使多步执行能够延续环境变量与工作目录；设置硬超时防止交互式命令卡死；对终端输出采用“保留头部参数与尾部报错、折叠中间日志”的 Head-Tail 截断策略。
> 4. **执行反馈通道**：工具执行报错时不能只返回“执行失败”，而要返回包含退出码、标准错误堆栈、文件与行列号的结构化 JSON，帮助模型在下一步自回归推断中快速定位原因。
> 5. **安全沙箱机制**：提示词约束无法抵御恶意指令注入。必须在系统层实现静态危险命令快速过滤、破坏性操作人工确认（Human-in-the-Loop），以及基于容器的隔离（只读根目录、限制网络出向、非 root 权限）。

在上一篇 [/interview/ai-coding-interview/code-repo-access.md](/interview/ai-coding-interview/code-repo-access.md) 中，我们讨论了代码仓库的索引与检索机制。接下来，我们以一个具体的贯穿案例，拆解工具调用的完整执行链条。

### 贯穿场景：修复一个线上鉴权报错

假设我们正在维护一个 Node.js 服务，测试套件报出错误：
```text
TypeError: Cannot read properties of undefined (reading 'token')
    at AuthService.validateToken (src/auth.ts:42:18)
```
代码代理要完成这次修复，通常经历以下动作：
1. 调用 `view_file` 查看 `src/auth.ts` 第 35~55 行代码；
2. 调用 `edit_file` 将可能引起空指针的属性访问修改为可选链或判空；
3. 调用 `run_terminal_command` 运行 `npm test -- -t auth` 验证测试是否通过；
4. 若单测未通过，分析终端返回的结构化堆栈信息，发起下一步修改。

下面我们逐一分析这几类工具的接口细节与设计权衡。

---

## 文件读写工具设计：参数模式与局部精确替换

文件操作通常包含查看（`view_file`）、创建（`create_file`）和修改（`edit_file`）。其中，修改已有代码是出错频率最高的操作。

![文件读写工具调用与字符串锚点替换机制](/images/content-rebuild/a44ecd46f02f890e/fig-1.png)

### 1. `view_file`：基于行号窗口的分页读取

大型工程文件动辄数千行，一次性读取全文不仅消耗大量 Token，还会稀释有效上下文。合理的查看工具应当要求限制读取范围：

* **参数定义**：
  * `path`：目标文件的相对路径。
  * `start_line`：起始行号（从 1 开始）。
  * `limit`：读取行数（推荐默认 100~200 行，上限 400 行）。
* **输出格式**：宿主读取文件后，在每行前面显式添加行号（如 `0042|   const token = user.token;`）。这些行号供模型定位参考，但并不强制后续修改必须以行号为基准。

### 2. `edit_file`：三种修改模式对比

修改已有代码通常有三种技术路线：

| 方案 | 实现方式 | 优点 | 常见问题 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **全量覆盖** | 模型输出修改后的整份文件 | 实现简单，无需定位算法 | 输出 Token 消耗巨大，生成速度慢，超长文件易被最大输出截断 | 极小的配置文件（如几十行的 JSON/YAML） |
| **行号范围替换** | 传入起始行、结束行与新代码 | 传输 Token 少 | 模型在多轮长对话中极难精确计算当前行号偏移，容易误删相邻行 | 基于 AST 严格重构的受控脚本工具 |
| **字符串块匹配替换** | 传入 `old_str` 与 `new_str` | 不依赖行号计算，Token 消耗小，保留局部上下文 | 若 `old_str` 在文件中有多处完全相同的内容，需要扩大上下文范围以确保唯一 | **主流代码代理的首选实现** |

在字符串块替换模式下，工具的 JSON Schema 定义如下：

```json
{
  "name": "edit_file",
  "description": "在指定文件中将唯一的 old_str 替换为 new_str。old_str 应包含目标代码及其上下 2-3 行未改动代码作为唯一定位锚点。",
  "parameters": {
    "type": "object",
    "properties": {
      "path": { "type": "string" },
      "old_str": { "type": "string", "description": "待替换的原始代码块" },
      "new_str": { "type": "string", "description": "替换后的新代码块" }
    },
    "required": ["path", "old_str", "new_str"]
  }
}
```

#### 宿主端的前置校验（Precondition Checks）
宿主在真正写入磁盘前，必须执行三项防错检查：
1. **唯一性检查**：统计 `old_str` 在目标文件中的命中次数。若命中了 0 次，拒绝写入，提示模型“未找到匹配代码，请先核对最新内容”；若命中了 2 次及以上，同样拒绝写入，提示“存在多处重复内容，请扩充 `old_str` 的上下文字符串以确保唯一”。
2. **格式与换行符容差**：Windows（CRLF）与 Linux（LF）换行符不一致，或者制表符与空格混用，可能导致纯文本精确匹配失败。宿主在严格匹配失败后，可尝试在换行符归一化或首尾空白容差模式下二次查找。
3. **原子写入**：修改应先写入临时文件，校验成功后再原子重命名替换，防止因进程中断导致原文件被截断写坏。

<ReadMoreLock />

---

## 探索与搜索工具设计：Token 预算与结果过滤

当面对陌生的代码仓库时，代码代理需要遍历目录树或使用正则搜索符号定义。如果没有预算限制，一次宽泛的搜索就可能返回上万条匹配项，直接耗尽上下文预算。

![代码探索工具的结果预算截断机制](/images/content-rebuild/a44ecd46f02f890e/fig-2.png)

### 1. `file_tree`：广度限制与忽略规则

* **路径白名单/黑名单**：底层必须内置忽略规则，强制排除 `.git/`、`node_modules/`、`dist/`、`target/`、`__pycache__/` 等构建缓存与三方依赖目录。
* **深度约束**：提供 `max_depth` 参数（通常建议默认 2 或 3 层），防止一次性展开深层复杂树形结构。
* **条目截断**：设置单次调用的返回条目上限（如 150 项）。一旦超过上限，截断并输出提示：“目录节点较多，已截断显示剩余内容。请指定具体子目录参数继续探索”。

### 2. `grep_search`：搜索结果预算保护

搜索工具底层通常调用 `ripgrep` 等高效文本搜索工具。为了保护上下文窗口，宿主应做两层保护：

* **输入参数细化**：除了必填的 `query` 外，支持 `path`（搜索子目录）和 `include_pattern`（如 `*.ts`、`*.py`），引导模型缩小搜索范围。
* **结果汇总与截断**：单次搜索返回的匹配行数设定硬上限（例如 80 行）。同一文件内命中过多时只取前几行，并在末尾附加明确的折叠提示。

```python
def format_grep_results(raw_matches: list[str], max_lines: int = 80) -> str:
    """对 grep 搜索结果施加行数预算限制，返回格式化后的文本。"""
    total = len(raw_matches)
    if total <= max_lines:
        return "\n".join(raw_matches)
    
    retained = raw_matches[:max_lines]
    dropped_count = total - max_lines
    notice = f"\n... [搜索结果过多，已折叠剩余 {dropped_count} 条匹配。建议补充 include_pattern 过滤文件类型，或缩小搜索路径。]"
    return "\n".join(retained) + notice
```

---

## 终端执行工具设计：持久化 PTY 与长日志截断

终端执行工具（`run_terminal_command`）是代码代理验证假设、运行测试和编译项目的关键手段。直接使用无状态的命令执行方式（如 Python 的 `subprocess.run(shell=True)`）会遇到以下问题：
* 每次命令在全新的子进程中运行，`cd /app/backend` 或 `export ENV=dev` 在命令结束后立即失效；
* 缺乏交互式伪终端支持，部分 CLI 工具因检测不到终端环境（non-TTY）改变输出缓冲行为。

![终端工具的持久化伪终端会话与长输出 Head-Tail 截断策略](/images/content-rebuild/a44ecd46f02f890e/fig-3.png)

### 1. 基于伪终端（PTY）的会话保持

在生产环境中，通常为每个任务分配一个有状态的伪终端会话（如 Linux 的 `pty` 结合轻量包装，或基于 `tmux` 守护后台进程）：

* **环境状态继承**：上一条命令切换的工作目录、设置的临时环境变量，在当前会话的后续命令中继续有效。
* **控制字符过滤**：终显输出往往夹杂 ANSI 转义字符（如颜色高亮 `\x1b[32m`、光标定位符等）。宿主返回给模型前，应使用正则清洗掉无意义的控制序列，避免噪声干扰自回归预测。

### 2. 硬超时与进程组清理

模型有时会误触发等待用户输入的命令（如交互式 `git rebase`、`vi`）或启动常驻服务（如 `npm run dev`），导致进程永久挂起：

* **硬超时设定**：单次命令执行设定最大等待时间（例如 60~120 秒）。
* **终止整个进程组**：超时触发时，不能仅向父进程发信号，必须向整个进程组发送 `SIGTERM`；若超时数秒仍未退出，则升级为 `SIGKILL`，防止衍生子进程遗留变为孤儿进程占用端口。

### 3. 长日志的 Head-Tail 截断策略

测试套件执行失败或编译崩溃时，日志往往包含成千上万行。**很多实现采用“只保留前 100 行”或“只保留后 100 行”的做法，这两种策略都有明显缺陷**：
* 只留头部：丢失了日志底部的崩溃点和异常堆栈跟踪（Stack Trace）；
* 只留尾部：丢失了顶部的命令入参、环境变量检查和初次失败的测试用例名称。

推荐的做法是 **Head-Tail 双端保留策略**：保留前部启动信息与尾部报错信息，折叠中间重复的通过日志或输出：

```python
def truncate_terminal_output(raw_output: str, head_lines: int = 50, tail_lines: int = 100) -> str:
    """保留终端输出的头部与尾部，折叠中间冗余日志。"""
    lines = raw_output.splitlines()
    total = len(lines)
    if total <= (head_lines + tail_lines):
        return raw_output
    
    head_part = "\n".join(lines[:head_lines])
    tail_part = "\n".join(lines[-tail_lines:])
    dropped = total - head_lines - tail_lines
    
    divider = f"\n\n... [中间折叠 {dropped} 行标准输出日志，仅保留前 {head_lines} 行环境参数与后 {tail_lines} 行错误堆栈] ...\n\n"
    return head_part + divider + tail_part
```

---

## 执行反馈通道：用结构化错误引导自愈

当工具执行出现非零退出码或异常时，如果宿主仅向模型返回一句非结构化的文本（如“命令执行失败，请重试”），模型无法获取有效的错误上下文，往往会在下一轮继续输出相似的错误调用。

![工具执行的自由文本报错与结构化因果反馈对比](/images/content-rebuild/a44ecd46f02f890e/fig-4.png)

自回归语言模型依赖输入中的具体报错信息来调整推断方向。宿主应将执行结果封装为确定性的结构化载荷（JSON 格式）。

### 结构化反馈载荷设计

无论命令成功还是失败，均采用标准格式返回：

```json
{
  "status": "error",
  "exit_code": 1,
  "command": "npm test -- -t auth",
  "stdout_summary": "Running 1 test suite... FAIL src/auth.test.ts",
  "stderr": "TypeError: Cannot read properties of undefined (reading 'token')\n    at AuthService.validateToken (src/auth.ts:42:18)",
  "diagnostics": [
    {
      "file": "src/auth.ts",
      "line": 42,
      "column": 18,
      "message": "Cannot read properties of undefined (reading 'token')"
    }
  ],
  "suggestion": "单测在 src/auth.ts 第 42 行因未捕获的 TypeError 失败。请检查该处的对象判空或调用 view_file 确认该行逻辑。"
}
```

### 为什么结构化反馈更有效？

1. **明确受损位置**：直接将文件名、行号与错误类型作为独立字段呈现，模型无需在包含进度条、编译警告的杂乱文本中自行正则推测报错来源。
2. **提供确定性分支凭据**：退出码（`exit_code`）能帮模型明确区分“命令根本不存在”（如 127）与“测试断言失败”（如 1）。前者需要安装依赖或调整 PATH，后者则需要进入源代码排查业务逻辑。

关于应用层工具协议的更完整规范，可以参考 [/frontier/mcp-protocol/mcp-server-client-tool.md](/frontier/mcp-protocol/mcp-server-client-tool.md) 与 [/llm-100/agent-engineering/agent-tool-integration.md](/llm-100/agent-engineering/agent-tool-integration.md)；关于复杂智能体系统的整体设计可进一步查阅 [/architecture/ai-system-design.md](/architecture/ai-system-design.md) 及 [/fundamentals/llm-basics/app-system-design.md](/fundamentals/llm-basics/app-system-design.md)。

---

## 安全边界与沙箱机制：纵深防御的三个层级

代码代理拥有生成并执行 Shell 命令的能力。如果缺乏隔离，提示词注入（Prompt Injection）或模型误判可能会对生产环境或开发主机造成毁灭性破坏（如误删系统文件、读取本地 SSH 秘钥并外传）。

安全保障不能寄希望于在 System Prompt 里写“请不要执行危险命令”。提示词属于软约束，容易被模型多轮对话遗忘或注入绕过。必须在宿主系统层建立刚性防御。

![纵深安全防御的三层隔离体系](/images/content-rebuild/a44ecd46f02f890e/fig-5.png)

### 1. 静态危险模式拦截（低成本粗筛）

在命令传递给执行器前，宿主可使用正则或 AST 解析工具检查高危模式：
* 破坏性系统命令：如 `rm -rf /`、`mkfs`、格式化磁盘等；
* 凭证窥探：如 `cat ~/.ssh/id_rsa`、`cat ~/.aws/credentials`；
* 典型外连反弹 Shell：如通过 `nc -e /bin/sh` 反向连接外部主机。

> **边界说明**：黑名单式过滤无法防御通过 Base64 编码、多层变量拼接混淆后的指令。它只能作为第一道低成本粗筛，阻断无心错误，绝不能作为唯一的安全边界。

### 2. 人机协同审批机制（Human-in-the-Loop）

针对不同破坏性级别的操作实施权限分级：
* **只读操作（自动放行）**：`view_file`、`grep_search`、只读测试；
* **局部文件写入（提示记录 / 自动放行）**：受管仓库目录内的 `edit_file`、`create_file`；
* **高危外部操作（必须用户手动确认）**：`git push`、修改仓库外的文件、执行包含网络下载依赖的外部脚本。

### 3. 容器沙箱与操作系统级隔离（硬防线）

生产级代码代理的终端命令应当运行在独立的容器或轻量虚拟机（如 Docker、gVisor、Firecracker）中：
* **非 Root 运行**：容器内以普通非特权用户（如 UID 1000）运行进程；
* **只读根文件系统**：除项目工作区目录与临时目录（`/tmp`）外，系统其余目录均挂载为只读（`ro`）；
* **网络出向限制（Egress Filtering）**：默认关闭非必要的公网访问，仅在模型明确需要下载三方包时临时开放受信任的包管理镜像源，防止敏感数据被通过 HTTP/DNS 隧道渗漏。

---

## 面试演练与常见误区复盘

![面试高频误区与技术归因总结](/images/content-rebuild/a44ecd46f02f890e/fig-6.png)

### 常见追问 1：如果大模型输出的 `edit_file` 匹配不到 `old_str`，系统该如何设计降级路径？

* **不推荐回答**：“直接把报错吞掉，让模型盲猜重试，或者强制覆盖整个文件。”
* **可落地回答**：
  1. 宿主立即中断修改操作，保持磁盘文件不被污染。
  2. 宿主对比当前文件的实际文本与 `old_str`，检查是否属于换行符格式（CRLF / LF）不一致或缩进空格数量不匹配。在确认无歧义的前提下，可由宿主进行容差规范化匹配；
  3. 若实质内容不匹配，工具向模型返回结构化错误信息，明确指出：“未找到匹配代码。当前目标位置附近的代码内容为 [...]，请根据最新代码重新提供唯一的 `old_str` 锚点。”引导模型基于真实上下文再次调整。

### 常见追问 2：为什么不直接让模型生成 `git diff` / Unified Diff 格式的补丁？

* **分析**：不是因为模型读不懂 Diff，而是因为 Unified Diff 语法要求在每一段修改前精准给出修改前后的起始行号和变更行数（例如 `@@ -42,7 +42,9 @@`）。
* 自回归模型在逐 Token 生成时，必须在还没写出具体代码改动之前，就准确预测出这几个数字。一旦行数统计出现 1 行偏差，标准的 `patch` 或 `git apply` 命令就会由于校验失败直接抛错（`patch does not apply`）。
* 相比之下，子串精确替换（`old_str` 替换为 `new_str`）不需要提前预测数字元数据，更符合自回归语言模型的生成特性，在工程上鲁棒性更强。

### 常见认知误区

1. **误区一：工具定义得越丰富，代理表现越好**
   * **纠偏**：向模型注入数十个功能重叠的工具会显著增加 Prompt 开销，并增加模型在工具选择时的决策混淆。成熟的代码代理通常只保留 5~8 个核心原子工具（读、查、改、跑命令等），保持工具语义互相独立。
2. **误区二：在 Prompt 里严格声明“请遵守安全准则”就足够安全**
   * **纠偏**：一旦代理在分析包含恶意代码的仓库时读入了含有注入诱导的内容（间接提示词注入），模型极易背离预设规则。安全边界必须由外部宿主程序硬编码保护，配合沙箱和人工审批，不能托付给模型自觉。

在下一篇 [/interview/ai-coding-interview/code-agent-flow.md](/interview/ai-coding-interview/code-agent-flow.md) 中，我们将进一步探讨如何利用状态机和编排循环，将这些原子工具组织成端到端的问题修复流。
