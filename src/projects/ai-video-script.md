---
title: AI视频脚本生成器：如何用结构化输出保证分镜与时长准确？
icon: mdi:video-outline
description: 通过 Pydantic 建模、Tool Calling 与语速节奏校验，解决短视频分镜脚本格式漂移与声画脱节问题，实现自动化导出分镜表与口播时间轴。
---

在短视频制作中，剧本不是一段普通的散文，而是一组必须按时间轴推进的结构化指令。前文在 [/projects/ai-writing-assistant.md](/projects/ai-writing-assistant.md) 中讨论了文本续写与内容构思，但在视频生产链条中，下游的拍摄人员、剪辑软件或自动化文生视频模型，需要精确知道每一个镜头的序号、景别、运镜方式、画面描述、台词旁白以及单镜头时长。

如果直接让大语言模型（LLM）自由生成分镜，很容易遇到两个典型问题：一是格式不稳定，比如大模型遗漏括号、自创未定义的运镜词汇；二是声画节奏失衡，例如一个时长 2 秒的镜头里塞入了 35 个汉字的旁白，配音根本无法正常念完。本文以一个“60秒手冲咖啡教程”短视频脚本为例，讲解如何使用 Pydantic 建立严格的数据模型、利用 Tool Calling 锁定数据传输通道，并在应用层加入台词语速校验与错误重试循环。

![短视频分镜头脚本各核心字段与视听维度的映射关系](/images/content-rebuild/c7044ea7c3413ce1/cover.png)

### 短视频分镜的数据结构设计

一个能直接交付给拍摄和后期团队的分镜头节点，需要包含 6 个核心要素：

1. **镜头序号（`shot_id`）**：从 1 开始递增的整数，用于标定时间线顺序。
2. **景别（`shot_size`）**：必须使用通用术语，如特写、近景、中景、全景、远景，方便摄影机位摆放。
3. **运镜方式（`camera_movement`）**：如固定、推镜、拉镜、摇镜、移镜、跟镜头，作为拍摄或 AI 视频生成工具的运动指令。
4. **画面视觉描述（`visual_description`）**：具体的画面主体、环境与动作，避免使用“气氛感拉满”等抽象形容词。
5. **台词与音效（`audio_script`）**：对白、旁白或关键音效标记。
6. **预估时长（`duration_seconds`）**：该镜头的持续秒数，必须与台词长度相匹配。

缺少任何一个字段，下游程序就无法计算总时长，也无法拆分时间轨道。

### 结构化输出方案对比：Prompt、JSON Mode 与 Tool Calling

让大模型输出符合上述结构的数据，常见有三种方案：

![三种结构化输出机制的可靠性与约束层级对比](/images/content-rebuild/c7044ea7c3413ce1/fig-1.png)

#### 1. 纯 Prompt 提示词约束
在提示词中写明“请严格输出如下 JSON 格式，不要返回任何多余字符”。
* **问题所在**：大模型自回归生成文本时，很容易在开头加上“好的，这是为您生成的脚本：”，或在 JSON 尾部漏掉逗号、括号。应用层调用 `json.loads()` 时经常抛出语法解析错误。

#### 2. JSON Mode（语法级约束）
在 API 请求中开启 JSON Mode（例如 OpenAI 的 `response_format={"type": "json_object"}`）。
* **效果与局限**：模型底层的解码器会限制输出符号，确保生成的内容一定是合法的 JSON 语法树。但它**不校验具体的业务字段**。模型仍然可能遗漏 `duration_seconds` 字段，或者把本应是数字的字段输出为字符串。

#### 3. Tool Calling 与受约束解码（工具调用与模式约束）
将数据结构定义转为 JSON Schema，注册到 API 的 `tools` 参数中，强制模型调用该工具：
* **常规 Tool Calling**：API 提供专门的函数参数传输通道，模型经过微调会主动对齐字段名。这大幅降低了字段漏填的概率，但如果不加额外约束，在极少数复杂嵌套场景下仍可能出现类型偏移。
* **受约束解码（Structured Outputs / 文法掩码）**：若服务商支持严格模式（如开启 `strict: true`）或本地部署时使用文法引导解码（Guided Decoding），引擎在每一步解码时会通过掩码屏蔽不符合 Schema 的 Token，彻底杜绝字段缺失或类型错误。
* **工程现实**：即使数据结构通过 Schema 严格锁死，静态 Schema 也无法校验“单镜头语速是否合理”或“各镜头时长加起来是否等于目标总时长”。因此稳妥的架构是：**使用 Tool Calling 约束通信协议，在应用层使用 Pydantic 进行类型运行时验证，并编写自定义规则校验视听节奏，遇错时回传提示重试**。

### 使用 Pydantic 定义分镜与脚本 Schema

我们使用 Python 的 `pydantic`（v2）来定义分镜头和整部脚本的结构规范：

![Pydantic Schema 对分镜头字段与枚举值的严格校验逻辑](/images/content-rebuild/c7044ea7c3413ce1/fig-2.png)

```python
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator

class ShotSize(str, Enum):
    CLOSE_UP = "特写"
    MEDIUM_CLOSE_UP = "近景"
    MEDIUM_SHOT = "中景"
    FULL_SHOT = "全景"
    WIDE_SHOT = "远景"

class CameraMovement(str, Enum):
    STATIC = "固定"
    PUSH = "推镜"
    PULL = "拉镜"
    PAN = "摇镜"
    TRACK = "移镜"
    FOLLOW = "跟镜头"

class VideoShot(BaseModel):
    shot_id: int = Field(..., description="镜头序号，从1开始自增")
    shot_size: ShotSize = Field(..., description="镜头景别")
    camera_movement: CameraMovement = Field(..., description="运镜方式")
    visual_description: str = Field(..., min_length=5, max_length=200, description="画面视觉具象细节")
    audio_script: str = Field(..., description="本镜头的旁白、对白或音效提示")
    duration_seconds: float = Field(..., ge=1.0, le=30.0, description="单镜头时长（秒），支持一位小数")

class VideoScript(BaseModel):
    title: str = Field(..., max_length=50, description="视频标题")
    target_audience: str = Field(..., description="目标受众定位")
    total_target_duration: int = Field(..., ge=5, le=300, description="目标总时长（秒）")
    shots: List[VideoShot] = Field(..., min_length=1, description="分镜头列表")

    @field_validator("shots")
    @classmethod
    def verify_sequential_ids(cls, v: List[VideoShot]) -> List[VideoShot]:
        for idx, shot in enumerate(v, start=1):
            if shot.shot_id != idx:
                raise ValueError(f"镜头序号不连续：列表第 {idx} 项的 shot_id 却为 {shot.shot_id}")
        return v
```

<ReadMoreLock />

### 节奏控制：语速比与时长平衡算法

脚本格式完全合法后，仍需检验台词与时长的匹配度。例如一段 2.0 秒的镜头，如果台词写了 30 个字，无论真人还是 TTS（文本转语音引擎）都无法在 2 秒内吐字清晰地播报完毕。

我们设定如下节奏规则：
1. **舒适语速基准**：中文常规口播语速约为每秒 3.0 到 4.5 个字。
2. **单镜头拦截边界**：
   * **语速上限**：单字速率计算方式为 `字数 / 时长`。若每秒字数超过 5.2 字，视为语速过快，触发拦截（纯空镜头或音效镜头允许字数为 0）。
   * **语速下限**：若镜头时长大于 4.0 秒且包含台词，每秒字数不得低于 1.5 字，避免无意义的过长画面静止拖沓。
3. **总时长容差**：各镜头累加的实际总时长，与设定的目标总时长偏差不得超过 10%（最小容差设为 2.0 秒）。

![台词语速与镜头时长配比的节奏校验模型](/images/content-rebuild/c7044ea7c3413ce1/fig-3.png)

基于该规则编写节奏校验器：

```python
from typing import Tuple, List

def validate_script_rhythm(script: VideoScript) -> Tuple[bool, List[str]]:
    errors = []
    total_actual_duration = sum(s.duration_seconds for s in script.shots)
    target = script.total_target_duration
    
    # 1. 校验总时长偏差
    tolerance = max(2.0, target * 0.10)
    if abs(total_actual_duration - target) > tolerance:
        errors.append(
            f"总时长偏差超标：各镜头累计 {total_actual_duration:.1f} 秒，" 
            f"目标为 {target} 秒，允许偏差范围为 ±{tolerance:.1f} 秒。"
        )
    
    # 2. 校验单镜头语速
    for shot in script.shots:
        text = shot.audio_script.strip()
        char_count = len(text)
        if char_count == 0:
            continue  # 允许空镜头或纯动作镜头无台词
            
        wps = char_count / shot.duration_seconds
        if wps > 5.2:
            errors.append(
                f"镜头 {shot.shot_id} 语速过快：台词共 {char_count} 字，" 
                f"时长仅 {shot.duration_seconds} 秒，平均语速 {wps:.1f} 字/秒（上限为 5.2 字/秒）。请精简台词或增加镜头时长。"
            )
        elif wps < 1.5 and shot.duration_seconds > 4.0:
            errors.append(
                f"镜头 {shot.shot_id} 节奏拖沓：台词仅 {char_count} 字，" 
                f"时长为 {shot.duration_seconds} 秒，语速为 {wps:.1f} 字/秒（下限为 1.5 字/秒）。请补充台词或缩短镜头时长。"
            )
            
    return len(errors) == 0, errors
```

### 校验失败捕获与错误信息回传重试循环

如果模型返回的数据未通过 Pydantic 校验或节奏校验，应用层不应直接报错崩溃，而是应当将具体的错误信息反馈给模型，让其在上一轮输出的基础上定向修改。

![校验失败捕获与错误信息回传修复的时序流程](/images/content-rebuild/c7044ea7c3413ce1/fig-4.png)

实现带有状态与错误回传的生成函数：

```python
import json
from openai import OpenAI
from pydantic import ValidationError

client = OpenAI()

def generate_video_script_with_retry(
    theme: str, 
    duration_seconds: int, 
    max_retries: int = 3
) -> VideoScript:
    
    tools = [{
        "type": "function",
        "function": {
            "name": "deliver_video_script",
            "description": "交付格式合格且视听节奏匹配的短视频分镜头脚本",
            "parameters": VideoScript.model_json_schema()
        }
    }]
    
    messages = [
        {
            "role": "system", 
            "content": "你是一名专业短视频编导。请根据用户需求规划分镜头剧本，并调用 deliver_video_script 交付。"
                       "确保台词与镜头时长相符，中文口播语速保持在每秒 3.0 到 4.5 个字之间。"
        },
        {
            "role": "user", 
            "content": f"创作主题：{theme}。视频目标总时长：{duration_seconds} 秒。"
        }
    ]
    
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "deliver_video_script"}},
            temperature=0.3
        )
        
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        
        if not tool_calls:
            messages.append(choice.message)
            messages.append({
                "role": "user",
                "content": "未检测到结构化工具调用，请严格调用 deliver_video_script 提交脚本。"
            })
            continue
            
        tool_call = tool_calls[0]
        call_id = tool_call.id
        raw_args = tool_call.function.arguments
        
        # 步骤 1：Pydantic Schema 校验
        try:
            script_obj = VideoScript.model_validate_json(raw_args)
        except ValidationError as val_err:
            err_msg = f"Schema 结构验证未通过：{val_err.errors()}"
            messages.append(choice.message)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": err_msg})
            continue
            
        # 步骤 2：业务视听节奏校验
        rhythm_ok, rhythm_errors = validate_script_rhythm(script_obj)
        if rhythm_ok:
            return script_obj
        
        feedback = "脚本已解析成功，但视听节奏检查未通过：\n" + "\n".join(rhythm_errors) + "\n请针对上述问题调整对应镜头的时长或台词字数后重新提交。"
        messages.append(choice.message)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": feedback})
        
    raise RuntimeError(f"尝试 {max_retries} 次交互后，仍未获得符合全部约束的视频脚本。")
```

### 多通道导出：拍摄表与配音时间轴

校验合格的 `VideoScript` 对象可以根据下游消费方需求，转换成 Markdown 拍摄分镜表或供 TTS 合成的台词时间轴：

![结构化脚本向 Markdown 拍摄表与配音时序的双向导出](/images/content-rebuild/c7044ea7c3413ce1/fig-5.png)

```python
def export_markdown_shooting_board(script: VideoScript) -> str:
    lines = [
        f"# 拍摄分镜表：{script.title}",
        f"**受众**：{script.target_audience} | **设计总时长**：{script.total_target_duration} 秒\n",
        "| 镜号 | 景别 | 运镜 | 时长(秒) | 画面描述 | 台词与音效 |",
        "| :---: | :---: | :---: | :---: | :--- | :--- |"
    ]
    for s in script.shots:
        clean_visual = s.visual_description.replace("\n", " ")
        clean_audio = s.audio_script.replace("\n", " ")
        lines.append(f"| {s.shot_id} | {s.shot_size.value} | {s.camera_movement.value} | {s.duration_seconds} | {clean_visual} | {clean_audio} |")
    return "\n".join(lines)

def export_tts_voiceover_cues(script: VideoScript) -> str:
    output = []
    accumulated_time = 0.0
    for s in script.shots:
        start_time = accumulated_time
        end_time = accumulated_time + s.duration_seconds
        output.append(f"[{start_time:05.1f}s -> {end_time:05.1f}s] (镜头 {s.shot_id}) {s.audio_script}")
        accumulated_time = end_time
    return "\n".join(output)
```

### 测试与验收流程

为了验证整个生成流水线在实际使用中的表现，我们在开发环境中运行三组具有代表性的题材测试：

![不同题材视频脚本端到端测试与格式容错验收流程](/images/content-rebuild/c7044ea7c3413ce1/fig-6.png)

#### 测试用例矩阵（开发验证示意）

| 测试题材 | 目标时长 | 镜头预期 | 常见违背场景 | 验收判定 |
| :--- | :--- | :--- | :--- | :--- |
| **科技快讯** | 30 秒 | 5~7 个 | 信息密度过大，模型单镜头输出 30 字却只分配 2 秒（语速达 15 字/秒） | 节奏校验拦截，反馈具体镜头超标，第 2 轮重试精简台词后通过 |
| **手冲咖啡教程** | 60 秒 | 8~12 个 | 动作步骤繁琐，偶发景别字段使用未经定义的自由词汇（如“俯瞰特写”） | Pydantic 枚举校验拦截非标词，回传错误后第 2 轮修正为标准枚举值 |
| **球鞋快闪广告** | 15 秒 | 5~6 个 | 切换快、含纯音乐/动作镜头，台词字数为 0 | 空台词镜头顺利放行，总时长累计误差在 ±1.5 秒内，一次通过 |

#### 验收步骤

1. **运行生成调用**：执行 `generate_video_script_with_retry("手冲咖啡入门教程", duration_seconds=60)`。
2. **观察交互历史**：检查 `messages` 列表。若触发了 `ValidationError` 或 `validate_script_rhythm` 失败，确认追加的 `role: tool` 消息是否准确指出了问题镜头 ID 和超标数值。
3. **交付物核验**：
   * 输出的 Markdown 表格各列对齐，无非枚举字段。
   * 累加 TTS 时间轴的最终时间戳，与 60 秒目标误差不超过 $\pm 6$ 秒（10% 容差内）。

通过 Tool Calling 确定通信协议，并配合 Pydantic 强类型与业务算法校验回传，开发人员无需再编写冗长脆弱的正则表达式清洗大模型输出。接下来，读者可以进入 [/projects/ai-workflow-automation.md](/projects/ai-workflow-automation.md)，将本文生成的结构化分镜数据无缝接入后续的生图与视频渲染编排节点。
