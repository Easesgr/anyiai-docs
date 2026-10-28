---
title: System Prompt、User Prompt 和 Assistant Message 分别有什么作用？
icon: mdi:forum-outline
tag:
  - Prompt面试
  - System Prompt
  - 消息类型
---

![](/images/prompt-interview/prompt-vs-finetuning_cover.png)

# System Prompt、User Prompt 和 Assistant Message 分别有什么作用？

面试官问这道题，其实就是想看你清不清楚 AI 对话的底层逻辑——谁在下指令、谁在响应、谁在定义规则。

这三者的关系很简单：**System Prompt 是剧本，User Prompt 是台词，Assistant Message 是演员说出来的台词。** 接下来一个个说。

---
## 1️⃣ System Prompt —— AI 的「角色设定卡」

System Prompt 就是给 AI 的"使用说明书"，告诉它：**你是什么角色、以什么风格说话、什么能做什么不能做。**

举个例子，你跟 AI 说"你是一个毒舌但有用的程序员"，这句话一般写在 System Prompt 里。

它的特点是：

- **全局生效**：对话全程都起作用，不会因为聊到第 10 句就消失
- **开发者定义**：不是用户写的，是配置 AI 的人写的
- **优先级最高**：即使用户 Prompt 试图"越界"，System Prompt 的设定会优先

简单理解：System Prompt 就是电影开拍前给演员的**角色卡**——"你演谁、你的性格是什么、你会怎么说话"，演员全程照着这个演。

![System Prompt 组成要素图：角色设定+行为规则+语气风格三层关系](/images/prompt-interview/20260608_165702_924416_01.png)

---
<ReadMoreLock />

## 2️⃣ User Prompt —— 用户的「即时需求」

User Prompt 就是用户每次说的话，是你发给 AI 的那条消息。

比如你在对话框里写："帮我写一封请假邮件"

这就是你的 User Prompt。它的特点：

- **每次对话都变化**：这轮聊这个、下轮聊那个，内容完全不确定
- **用户主导**：用户想聊什么就聊什么
- **触发 AI 响应**：有了 User Prompt，AI 才会有针对性地回复

类比一下：User Prompt 就像你去餐厅点菜——服务员（AI）问你"想吃点什么"，你报出的那道菜。

注意一点：User Prompt 不等于 Input。Input 是喂给模型的原始数据，可能包含历史对话；而 User Prompt 就是当前这轮用户说的话。

![用户输入"帮我写道歉信"映射为 User Prompt 的示意图](/images/prompt-interview/20260608_165702_292992_01.png)

---

## 3️⃣ Assistant Message —— AI 的「表演输出」

Assistant Message 就是 AI 根据 System Prompt + User Prompt 生成的那段回复。

AI 做了什么？它把 System 里的角色设定、规则，和 User 的具体需求**组合起来**，然后生成一个符合要求的回复。

比如：

- System 说："你是毒舌程序员，说话带点吐槽但很专业"
- User 说："帮我看看这段代码哪里报错"
- Assistant 生成："哈？这变量名起得也太随意了吧...好，问题在这，缺少分号。"

这就是 Assistant Message——**演员说完的台词**。

![System + User → AI处理 → Assistant Message 的三段式流程图](/images/prompt-interview/20260608_165703_909504_01.png)

---

## 4️⃣ 为什么要分开？—— 分层的价值

你可能会问：直接把所有指令写在一起不行吗？

还真不行，分开有实际价值：

**1. 可维护性**
想象一下：如果角色设定和具体需求混在一起，改需求时你得找半天。分开后，System Prompt 写死了核心逻辑，每次对话只需要关注 User Prompt 就行。

**2. 安全性**
System Prompt 定义的规则不会被用户 Prompt 轻易覆盖。比如 AI 被设定为"不能提供违法建议"，即使你在 User Prompt 里诱导它，它也会遵守 System 的约束。

**3. 工程实践**
现在 Coze、Dify 这些平台已经把三者拆成独立的输入框了，开发调试的时候非常方便。

类比餐厅：
- System = 菜谱（固定的，厨师必须遵守）
- User = 点菜单（每次不一样，服务员传给厨师）
- Assistant = 端出来的菜（根据菜谱+点单做出来的）

![System Prompt 与 User Prompt 六维度对比表](/images/prompt-interview/20260608_165727_050944_01.png)

---

## 💬 面试怎么答

**基础版（能过的回答）：**

> System Prompt 是 AI 的说明书，定义它的角色、规则和风格，全程生效；User Prompt 是用户每次说的话，代表即时需求，触发 AI 响应；Assistant Message 是 AI 根据前两者生成的回复内容。把它们分开主要是为了可维护性和安全性，System 的设定不会被用户 Prompt 轻易覆盖。

**加分版（让面试官眼前一亮）：**

> 补充一个细节：User Prompt 不完全等于 Input——Input 包含历史上下文，是喂给模型的原始数据，而 User Prompt 是当前这轮对话中用户说的话。在工程实践中，这三者有明确的处理顺序：System 在前、User 在后，拼接成完整的输入给模型。这个顺序不能乱，因为 System 的优先级最高，它定义的规则会约束后续所有响应。

---

## 一句话总结

System Prompt 是剧本，User Prompt 是台词，Assistant Message 是演员说出来的台词——三者各司其职，AI 才能正常"演出"。