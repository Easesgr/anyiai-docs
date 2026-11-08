---
title: 分类模型评估：如何判断模型到底猜得准不准？
icon: mdi:checkbox-marked-circle-outline
tag:
- 机器学习
- 模型评估
- 分类问题
description: 详解混淆矩阵、精确率、召回率、F1分数、ROC-AUC与PR曲线的数学含义、计算方法及不平衡场景选型。
---

在完成[/fundamentals/ml-basics/train-val-test.md](/fundamentals/ml-basics/train-val-test.md)中的数据集划分后，我们需要在测试集上检验分类模型的表现。最直观的想法往往是：统计模型猜对的样本占总样本的比例。但在实际工程中，单凭“准确率”往往会带来严重的误导。

例如在信用卡盗刷检测、疾病早筛或垃圾邮件拦截中，两类错误的后果完全不对称：漏掉一笔真实的盗刷，平台需要承担资金损失；而把正常消费误判为盗刷，只是让用户多确认一次。为了合理评估模型，我们需要一套能够拆解不同错误类型、适应样本倾斜并在连续阈值下观察模型行为的评估指标体系。

![分类评估指标体系概览](/images/content-rebuild/9eca07d2a3e18029/cover.png)

## 混淆矩阵：二分类预测的基本计数单元

所有二分类指标的基础都是**混淆矩阵（Confusion Matrix）**。对于二分类任务，每个样本的真实标签只有正（Positive）和负（Negative）两种，模型给出的判断也只有正和负两种。真实情况与预测结果交叉，将所有测试样本划分为互不重叠的四个象限：

| 真实情况 \ 预测结果 | 预测为正类（Predicted Positive） | 预测为负类（Predicted Negative） |
| :--- | :--- | :--- |
| **真实为正类（Actual Positive）** | **真阳性（True Positive, TP）**：抓取成功 | **假阴性（False Negative, FN）**：漏报（第 II 类错误） |
| **真实为负类（Actual Negative）** | **假阳性（False Positive, FP）**：误报（第 I 类错误） | **真阴性（True Negative, TN）**：正确排除 |

假设测试集中共有 $P$ 个真实正样本、$N$ 个真实负样本，测试集总样本量为 $M = P + N$。这四个基础计数满足恒等关系：

$$P = TP + FN$$
$$N = FP + TN$$
$$M = TP + FP + FN + TN$$

混淆矩阵不仅记录了对了多少、错了多少，还明确指出了错误的流向：是把正例漏掉了（$FN$），还是把负例误伤了（$FP$）。

![混淆矩阵的四象限划分结构](/images/content-rebuild/9eca07d2a3e18029/fig-1.png)

## 基础指标：准确率、精确率、召回率与 F1

基于混淆矩阵的四个计数，我们可以从不同业务视角计算出比例指标。

### 1. 准确率（Accuracy）
准确率衡量所有预测正确的样本占总体样本的比例：

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{TP + TN}{M}$$

准确率适用的前提是：正样本和负样本数量大致相当，且两类预测错误的代价基本对称。

### 2. 精确率（Precision，查准率）
精确率关注的是**模型判定为正的样本集合**中，真正是正样本的比例：

$$\text{Precision} = \frac{TP}{TP + FP}$$

精确率回答的是“模型报警的可信度”。如果系统误报（$FP$）极多，精确率就会下降。在垃圾邮件拦截或自动封禁账号场景中，误杀正常邮件和无辜用户的代价极高，此时系统通常需要优先保障高精确率。

### 3. 召回率（Recall，查全率 / 真正例率 TPR）
召回率关注的是**所有客观存在的真实正样本集合**中，被模型成功抓取出来的比例：

$$\text{Recall} = \frac{TP}{TP + FN}$$

召回率回答的是“目标事件有没有漏网”。如果系统漏报（$FN$）很多，召回率就会非常低。在重大疾病筛查或高危设备故障预警中，漏检一个正例可能会导致严重后果，此时系统会优先追求高召回率。

![精确率与召回率的分母视角差异](/images/content-rebuild/9eca07d2a3e18029/fig-2.png)

<ReadMoreLock />

### 4. F1 分数与调和平均数
在多数情况下，精确率与召回率此消彼长：想抓得更全，往往需要放宽报警门槛，从而增加误报；想报警更准，通常需要收紧门槛，从而增加漏报。为了综合评估二者，我们使用**F1 分数（F1-score）**，即精确率与召回率的**调和平均数（Harmonic Mean）**：

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

为什么不能直接计算算术平均数 $\frac{\text{Precision} + \text{Recall}}{2}$？

算术平均数对极端失衡不敏感。假设一个极端模型将全部样本都判为正类，其召回率（Recall）为 $100\%$，但由于误报极其严重，精确率（Precision）跌到 $1\%$。如果计算算术平均：

$$\frac{1.0 + 0.01}{2} = 50.5\%$$

这看似仍有“及格”水准。而调和平均数的倒数是各数倒数的算术平均，因此数值会强烈受制于较小的那一项。计算 F1 时：

$$F_1 = 2 \cdot \frac{1.0 \times 0.01}{1.0 + 0.01} \approx 0.0198$$

调和平均数立刻给出了接近 0 的惩罚分数。只有当精确率和召回率同时维持在较高水平时，F1 分数才会处于高位。

如果业务对两者的重视程度不同，可以使用更具通用性的 $F_\beta$ 分数：

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

当 $\beta=2$ 时，召回率权重更高；当 $\beta=0.5$ 时，精确率权重更高。

## 类别不平衡陷阱：准确率为何会失效

在很多现实任务中，正负样本的比例极不均衡。例如在信用卡欺诈识别中，欺诈交易可能只占总交易量的 $1\%$ 不到。

假设我们测试一个欺诈检测模型，测试集中共有 1000 笔交易，其中 10 笔是欺诈交易（正类），990 笔是正常交易（负类），正负比例为 $1:99$。

现在对比两个模型：

- **基线模型 A（全部猜正常）**：不做任何计算，直接将所有交易都判定为正常交易（全部预测为负类）。
  - $TP = 0, \ FP = 0, \ FN = 10, \ TN = 990$
  - $\text{Accuracy} = \frac{0 + 990}{1000} = 99.0\%$
  - $\text{Recall} = \frac{0}{0 + 10} = 0.0\%$
- **识别模型 B（具有判别能力）**：成功拦截了 8 笔欺诈，但因风控规则相对敏感，将 20 笔正常交易误判为欺诈。
  - $TP = 8, \ FP = 20, \ FN = 2, \ TN = 970$
  - $\text{Accuracy} = \frac{8 + 970}{1000} = 97.8\%$
  - $\text{Recall} = \frac{8}{8 + 2} = 80.0\%$
  - $\text{Precision} = \frac{8}{8 + 20} \approx 28.57\%$

如果单纯以准确率作为评估标准，放过所有盗刷交易的模型 A（$99.0\%$）竟然胜过了抓出 $80\%$ 盗刷的模型 B（$97.8\%$）。原因在于：在极大基数的负类主导下，$TN$ 的绝对数量淹没了正类的预测变化。因此，在样本不平衡场景中，准确率无法客观反映模型的真实能力。

![类别不平衡场景下准确率指标失效机制对比](/images/content-rebuild/9eca07d2a3e18029/fig-3.png)

## 决策阈值滑动：精确率与召回率的动态权衡

在[/fundamentals/ml-basics/linear-logistic-regression.md](/fundamentals/ml-basics/linear-logistic-regression.md)中，逻辑回归等分类器输出的通常不是直接的离散标签，而是一个连续的概率估计 $\hat{p} = P(Y=1|X)$。最终要输出正还是负，必须设定一个决策阈值（Threshold）$\theta$：

$$\hat{y} = \begin{cases} 1, & \text{若 } \hat{p} \ge \theta \\ 0, & \text{若 } \hat{p} < \theta \end{cases}$$

多数算法库默认设定 $\theta = 0.5$。但 $\theta$ 并不是固定不变的物理常数，而是可以根据业务需求调节的旋钮。

当我们把阈值 $\theta$ 从 0 平滑滑动到 1 时，混淆矩阵的四个计数会连续变动：

1. **$\theta \to 1$（大幅提高门槛）**：只有模型极度确信的样本才会被预测为正类。预测为正的总量（$TP + FP$）急剧萎缩，误报数 $FP$ 趋近 0。此时精确率通常升高，但大量稍不确定的正样本被归入负类，漏报数 $FN$ 骤增，召回率跌向 0。
2. **$\theta \to 0$（大幅放宽门槛）**：几乎所有样本都被预测为正类。真实正样本几乎全部被捕获，漏报数 $FN$ 趋近 0，召回率接近 1；但大量负样本被误判为正类，误报数 $FP$ 激增，精确率大幅跌落至正样本的基础比例。

因此，评估分类模型不能只看单一阈值下的表现，还需要考察模型在不同阈值下的整体判别能力。

![判定阈值滑动引发精确率与召回率的动态权衡](/images/content-rebuild/9eca07d2a3e18029/fig-4.png)

## ROC 曲线与 AUC 值

为了在脱离单一固定阈值的前提下评估分类器的排序能力，我们使用 **ROC 曲线（Receiver Operating Characteristic Curve）**。

### 1. ROC 坐标系与生成机制
ROC 曲线考察的是在不同阈值下，真正例率与假正例率的轨迹变化：
- **纵轴：真正例率（True Positive Rate, TPR）**，等同于召回率：
  $$\text{TPR} = \frac{TP}{P} = \frac{TP}{TP + FN}$$
- **横轴：假正例率（False Positive Rate, FPR）**，即负样本被误判为正的比例：
  $$\text{FPR} = \frac{FP}{N} = \frac{FP}{FP + TN}$$

绘制方法：将测试集中的所有样本按预测概率从大到小排序。将阈值从最高分依次下移跨过每个样本的分数：
- 跨过一个正样本，分母为 $P$ 的 $TP$ 加 1，曲线在纵轴方向向上走一步；
- 跨过一个负样本，分母为 $N$ 的 $FP$ 加 1，曲线在横轴方向向右走一步。

整条曲线从坐标轴原点 $(0,0)$ 开始，最终到达右上角 $(1,1)$。

### 2. AUC 的统计学含义
ROC 曲线下方的面积被称为 **AUC（Area Under the ROC Curve）**。在统计学上，AUC 精确等价于 **Wilcoxon-Mann-Whitney 统计量**：

假设从所有正样本中随机抽取一个正样本 $x^+$，从所有负样本中随机抽取一个负样本 $x^-$。模型对它们的打分分别为 $f(x^+)$ 和 $f(x^-)$。AUC 等于**正样本打分高于负样本打分的概率**：

$$\text{AUC} = P(f(x^+) > f(x^-)) + \frac{1}{2} P(f(x^+) = f(x^-))$$

- **$\text{AUC} = 1.0$**：完美排序。所有正样本的分数都严格高于负样本，存在某个阈值可以把两类完全分开而无任何误判。
- **$\text{AUC} = 0.5$**：等同于随机抛硬币猜测，模型完全没有区分能力。
- **$\text{AUC} < 0.5$**：模型输出了相反的倾向，只要将预测值反转（$1 - \hat{p}$），就能得到一个 AUC 大于 0.5 的有效模型。

### 3. ROC 对类别比例的不变性
ROC 曲线的一个重要性质是**对类别比例的变化不敏感**。观察定义式可知：TPR 的分母严格局限在正样本总数 $P$ 内部，FPR 的分母严格局限在负样本总数 $N$ 内部。如果在测试集中人为注入大量负样本，只要正负样本内部各自的预测分数分布形状不变，TPR 与 FPR 的期望取值就不会改变，ROC 曲线的形状与 AUC 数值将保持稳定。

![ROC 曲线构建过程与 AUC 排序统计意义](/images/content-rebuild/9eca07d2a3e18029/fig-5.png)

## Precision-Recall 曲线：极度偏斜场景的检验工具

ROC-AUC 的“对比例不敏感”并不总是优点。在样本极度不平衡的任务中，这种不敏感有时会掩盖严重的误报缺陷。此时需要借助 **Precision-Recall 曲线（PR 曲线）**。

PR 曲线以召回率（Recall）为横轴，以精确率（Precision）为纵轴。关键差别在于精确率的分母：

$$\text{Precision} = \frac{TP}{TP + FP}$$

精确率的分母同时受正样本预测（$TP$）与负样本误报（$FP$）的共同影响。

假设在一个拥有 100,000 个负样本和 100 个正样本的测试集中（比例 1000:1）：
模型检出了 80 个正样本（$TP = 80$），但同时也产生了 500 个误报（$FP = 500$）。
- 在 ROC 坐标系中：
  $$\text{FPR} = \frac{500}{100000} = 0.005$$
  横轴坐标仅为 $0.005$，曲线依然紧贴左轴，ROC 曲线和 AUC 值依然显得非常好。
- 但在 PR 坐标系中：
  $$\text{Precision} = \frac{80}{80 + 500} \approx 13.79\%$$
  在达到 $80\%$ 召回率时，精确率仅有 $13.79\%$。这意味着系统每报出 100 次警报，有 86 次以上是误报。

PR 曲线不包含庞大的真阴性 $TN$，它直接衡量正样本池被误报稀释的程度。当业务极其看重正类检出质量、且需要严格限制误报数量时，PR 曲线及其面积（PR-AUC 或 Average Precision, AP）是更具参考价值的评估工具。

![极度不平衡下 ROC 曲线与 PR 曲线的灵敏度差异对比](/images/content-rebuild/9eca07d2a3e18029/fig-6.png)

## Python 评估代码实现

借助 `scikit-learn`，我们可以构造一个不平衡测试集，演示各项指标的实际计算流程：

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# 1. 构造正负样本比例约 1:99 的合成数据集
X, y = make_classification(
    n_samples=2000,
    n_features=10,
    weights=[0.99, 0.01],
    random_state=42,
)

# 划分训练集与测试集，保证评估环境独立
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# 2. 拟合逻辑回归模型
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 3. 在测试集上获得预测分类（默认阈值 0.5）与连续预测概率
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# 4. 抽取混淆矩阵元素
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"混淆矩阵: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# 5. 计算标量评估指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_scores)
pr_auc = average_precision_score(y_test, y_scores)

print(f"准确率 (Accuracy):   {acc:.4f}")
print(f"精确率 (Precision):  {prec:.4f}")
print(f"召回率 (Recall):     {rec:.4f}")
print(f"F1 分数 (F1-Score):  {f1:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print(f"PR-AUC (AP):         {pr_auc:.4f}")
```

运行上述代码可以看到：由于负样本占绝大多数，模型的准确率容易维持在 $98\%$ 以上；而精确率、召回率和 PR-AUC 则直接体现了模型在稀有正类上的真实表现。

## 评估指标选型参考

在不同业务场景中，指标的选型应当与业务代价直接挂钩：

1. **正负样本均衡、判错代价基本对称**：直接查看准确率（Accuracy），简单直观。
2. **漏报代价远高于误报代价（医疗罕见病筛查、生产安全排查）**：核心关注召回率（Recall），通常结合业务承受力适当下调判定阈值 $\theta$。
3. **误报代价远高于漏报代价（拦截垃圾邮件、关键业务自动审批）**：核心关注精确率（Precision），通常上调判定阈值 $\theta$ 避免误判。
4. **需要综合考量精确与召回平衡**：使用 F1 分数（或带权重的 $F_\beta$ 分数）。
5. **评估模型整体的特征排序能力，不绑定特定阈值**：查看 ROC-AUC。它不受正负样本比例缩放的影响。
6. **极度不平衡场景下的正类检测（反洗钱、稀有缺陷检测）**：重点查看 PR 曲线与 PR-AUC。它排除了大量真阴性的稀释作用，能直接反映误报对正类报警质量的冲击。

理解了离散标签的评估逻辑后，下一篇我们将进入连续数值预测场景，在[/fundamentals/ml-basics/evaluation-regression.md](/fundamentals/ml-basics/evaluation-regression.md)中讨论均方误差、绝对误差与决定系数等回归评估方法。
