---
title: 回归模型评估：从残差、MAE、RMSE到R²与残差图诊断
icon: mdi:chart-bar
tag:
- 机器学习
- 模型评估
- 回归问题
description: 系统讲解回归任务的误差度量方法，从残差定义、MAE与RMSE的惩罚差异，到MAPE局限、R²方差解释率及残差图诊断模式与Python验证。
---

在分类任务中，模型输出的是离散标签或类别概率，评估核心是判断“类别是否划分正确”（详见 [分类模型评估](/fundamentals/ml-basics/evaluation-classification.md)）。但在预测房屋售价、服务器功耗、传感器读数或交付周期等场景中，预测目标是连续实数。

对连续数值而言，预测值与真实值几乎不可能做到分子级别的严格相等。误差不再是非黑即白的“正确与错误”，而是带有方向和尺度的具体偏离。评估回归模型，既需要标量指标汇总整体偏差量级，也需要残差诊断工具检查误差中是否还残留着未被模型利用的规律。

![回归评估全景封面](/images/content-rebuild/5cc549ae93b829ff/cover.png)

---

## 连续值误差的基础：残差（Residuals）

评估连续预测的基础概念是**残差（Residual）**。设样本 $i$ 的真实观测值为 $y_i$，模型给出的预测值为 $\hat{y}_i$，样本残差 $e_i$ 定义为：

$$e_i = y_i - \hat{y}_i$$

残差的正负符号直接指示了模型的偏差方向：
- 当 $e_i > 0$ 时，真实值大于预测值（$\hat{y}_i < y_i$），模型发生了**低估**；
- 当 $e_i < 0$ 时，真实值小于预测值（$\hat{y}_i > y_i$），模型发生了**高估**；
- 当 $e_i = 0$ 时，预测值与真实值完全重合。

![残差几何定义与偏差方向](/images/content-rebuild/5cc549ae93b829ff/fig-1.png)

残差不是样本点到拟合直线的垂直欧氏距离，而是样本点沿目标变量 $y$ 轴方向到拟合曲面的**垂直投影差**。

如果直接对全体样本的残差求算术平均：

$$\text{Mean Error} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)$$

正残差与负残差会相互抵消。例如预测两个样本，一个高估 100，另一个低估 100，算术平均残差为 0，但模型实际上在每个样本上都发生了严重偏差。因此，要量化误差的真实规模，必须通过绝对值或平方来消除符号抵消。

---

## 核心指标：MAE、MSE 与 RMSE

消除残差正负抵消的两条主要路径，构成了回归评估中最基础的三个标量指标：平均绝对误差（MAE）、均方误差（MSE）与均方根误差（RMSE）。

![MAE与MSE惩罚曲线对比](/images/content-rebuild/5cc549ae93b829ff/fig-2.png)

### 1. 指标定义与单位特征

- **平均绝对误差（MAE, Mean Absolute Error）**：
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
  MAE 对残差取绝对值后再求平均，其单位与原始标签 $y$ 完全相同。若预测目标是房屋售价（万元），MAE = 5.2 即表示“模型平均每个样本偏离真实售价 5.2 万元”。

- **均方误差（MSE, Mean Squared Error）**：
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
  MSE 对残差取平方后再求平均。由于进行了平方运算，MSE 的单位变成了原始单位的平方（如“万元²”），直观可读性下降。但在数学优化中，平方形式在实数域内处处可微，梯度的形式简单，常被直接选为 [线性回归](/fundamentals/ml-basics/linear-logistic-regression.md) 与 [损失函数和梯度下降](/fundamentals/ml-basics/loss-gradient-descent.md) 中的损失函数。

- **均方根误差（RMSE, Root Mean Squared Error）**：
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
  RMSE 对 MSE 执行开方运算，将单位重新拉回与原始数据相同的尺度，既保留了平方机制对大误差的加权惩罚，又恢复了可解释的数值单位。

<ReadMoreLock />

### 2. 算例验证：对异常值的敏感度差异

为什么在同一个测试集上，RMSE 通常大于或等于 MAE？根据数学中的柯西-施瓦茨不等式，恒有 $\text{RMSE} \ge \text{MAE}$；当且仅当所有样本的绝对残差完全相等时，二者才取等号。只要残差之间存在波动，平方机制就会对大误差赋予更高的权重。

我们通过一个具体的算例对比两者的行为变化：

**基准场景（误差均匀，无极端异常值）：**
假设测试集中有 4 个样本，其真实值与预测值如下：
- 样本 1：$y_1 = 10, \hat{y}_1 = 12 \implies e_1 = -2, |e_1| = 2, e_1^2 = 4$
- 样本 2：$y_2 = 20, \hat{y}_2 = 18 \implies e_2 = 2, |e_2| = 2, e_2^2 = 4$
- 样本 3：$y_3 = 30, \hat{y}_3 = 32 \implies e_3 = -2, |e_3| = 2, e_3^2 = 4$
- 样本 4：$y_4 = 40, \hat{y}_4 = 38 \implies e_4 = 2, |e_4| = 2, e_4^2 = 4$

计算指标：
$$\text{MAE}_{base} = \frac{2 + 2 + 2 + 2}{4} = 2.0$$
$$\text{RMSE}_{base} = \sqrt{\frac{4 + 4 + 4 + 4}{4}} = 2.0$$
此时所有样本误差幅度均为 2，MAE 与 RMSE 严格相等。

**扰动场景（混入 1 个大误差样本）：**
在测试集中加入第 5 个样本：真实值 $y_5 = 50$，模型预测值 $\hat{y}_5 = 100$：
- 样本 5：$e_5 = -50, |e_5| = 50, e_5^2 = 2500$

重新计算包含 5 个样本的指标：
$$\text{MAE}_{noise} = \frac{2 + 2 + 2 + 2 + 50}{5} = \frac{58}{5} = 11.6$$
$$\text{RMSE}_{noise} = \sqrt{\frac{4 + 4 + 4 + 4 + 2500}{5}} = \sqrt{\frac{2516}{5}} = \sqrt{503.2} \approx 22.43$$

对比两个指标的变化幅度：
- MAE 从 $2.0$ 上升至 $11.6$（变为原来的 $5.8$ 倍）；
- RMSE 从 $2.0$ 上升至 $22.43$（变为原来的 $11.2$ 倍）。

**指标选型参考：**
1. **MAE 反映典型误差**：绝对值损失对离群样本保持线性增长，不易被极少数毛刺数据大幅拉动。当数据集中不可避免混杂噪音、业务更关心“普通样本的平均偏差”时，MAE 是更稳健的代表。
2. **RMSE 关注极端失误**：平方机制放大了大误差的权重。如果业务场景对大偏差极其敏感（如电力负荷过载预测、高精度装配校准），一次严重误判可能带来严重后果，此时应优先以 RMSE 作为评判标准。

---

## 相对误差指标：MAPE 及其局限

MAE 和 RMSE 都是**绝对尺度指标**。在预测价格时，预测 500 万元的房产产生 10 万元误差（相对误差 2%），与预测 15 万元的二手车产生 10 万元误差（相对误差 66.7%），在 MAE 看来惩罚完全一样。但两者的业务影响截然不同。为了横向对比不同量级业务的预测质量，需要引入**相对百分比误差**。

![MAPE计算逻辑与不对称惩罚](/images/content-rebuild/5cc549ae93b829ff/fig-3.png)

### 1. MAPE 计算公式

**平均绝对百分比误差（MAPE, Mean Absolute Percentage Error）**将每个样本的绝对误差归一化到真实值尺度：

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

MAPE 的输出是无量纲的百分比，允许开发者跨品类对比精度（例如对比高单价商品与低单价快消品的预测水平）。

### 2. MAPE 的两大缺陷

1. **除零崩溃与接近零时的数值不稳定**：
   MAPE 的分母是真实值 $y_i$。当数据集存在 $y_i = 0$（例如某天销售额为零）时，公式发生除零错误，无法计算；当 $y_i$ 非常接近零时，哪怕微小的绝对误差也会被放大成数倍乃至数十倍的百分比，严重扭曲评估结果。
2. **不对称惩罚机制**：
   假设真实值 $y_i = 100$：
   - **模型严重低估**：预测 $\hat{y}_i = 0$，相对误差为 $|100 - 0| / 100 = 100\%$。当预测值非负时，低估带来的相对误差上限被锁定在 $100\%$；
   - **模型严重高估**：预测 $\hat{y}_i = 300$，相对误差为 $|100 - 300| / 100 = 200\%$；若预测 $\hat{y}_i = 1000$，相对误差高达 $900\%$。高估惩罚在数学上没有上限。

因此，以 MAPE 为核心评估导向时，模型如果倾向于保守预测（往偏小的值猜），其指标往往比大胆预测更好看。这会导致模型产生系统性的偏低预测倾向。

为缓解不对称性，工程上常使用**对称平均绝对百分比误差（sMAPE, Symmetric MAPE）**，将分母调整为真实值与预测值的均值：

$$\text{sMAPE} = \frac{1}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2} \times 100\%$$

sMAPE 将单个样本的最大误差限制在 $200\%$ 以内。但在真实值与预测值同时为 0 时，仍需额外设置微小偏置项（$\epsilon$）避免除零。

---

## 决定系数 R²：模型解释了多少方差？

即便计算出 RMSE = 8.5，我们依然很难直接判断：**这个模型算好还是算差？它是否比盲猜均值更有用？**

统计学引入了**决定系数（Coefficient of Determination，记作 $R^2$）**，用来衡量模型相对于“无脑猜均值”基准的改善幅度。

![R方方差分解几何关系](/images/content-rebuild/5cc549ae93b829ff/fig-4.png)

### 1. 基准对比与平方和分解

设真实标签的样本均值为 $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$。

如果我们构建一个最朴素的基准模型——不管输入特征是什么，永远输出固定均值 $\hat{y}_i = \bar{y}$，该基准模型产生的误差平方和被称为**总平方和（$SS_{tot}$, Total Sum of Squares）**：

$$SS_{tot} = \sum_{i=1}^n (y_i - \bar{y})^2$$

$SS_{tot}$ 衡量的是目标标签本身的离散程度。

而我们训练出的模型，其残差平方和被称为**残差平方和（$SS_{res}$, Residual Sum of Squares）**：

$$SS_{res} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

$SS_{res}$ 衡量的是模型未能捕获的残存波动量。

在经典线性最小二乘回归（OLS）中，**如果模型包含常数截距项**，依据正规方程，残差向量与模型拟合值向量严格正交，交叉乘积项为零。此时总平方和可以严格分解为两项之和：

$$SS_{tot} = SS_{reg} + SS_{res}$$

其中 $SS_{reg} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2$ 为**回归平方和**，代表被模型捕获的变化量。需要强调的是，这种严格拆分只在线性回归且带截距项的训练数据上成立；在测试集或非线性模型中，两者的和通常并不严格等于 $SS_{tot}$。

### 2. $R^2$ 的标准定义与取值含义

无论是否为线性模型，$R^2$ 在评估中的通用标准定义统一为：

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

$R^2$ 各取值区间的具体含义如下：
- **$R^2 = 1.0$**：$SS_{res} = 0$，模型预测值与真实值完全一致，所有方差波动均被模型解释。
- **$R^2 = 0.0$**：$SS_{res} = SS_{tot}$，模型的预测效果等同于“永远猜均值”。特征输入没有为模型带来任何有效预测增量。
- **$0 < R^2 < 1.0$**：模型优于均值基准。例如 $R^2 = 0.82$ 表示目标变量中 $82\%$ 的方差波动被模型规律解释，其余 $18\%$ 为不可解释的残差波动。
- **$R^2 < 0$**：**$R^2$ 可以是负数**。当模型在独立测试集上发生严重过拟合、或者回归模型未引入截距项导致预测系统性偏离真实中心时，$SS_{res}$ 可能大幅超过 $SS_{tot}$。这意味着该模型的预测效果比直接盲猜测试集均值还要差，模型已失去实用价值。

---

## 残差图诊断：揭示标量背后的系统性偏差

标量指标（MAE、RMSE、$R^2$）将成百上千个数据点的误差压缩为单个实数，容易掩盖模型的结构性问题。两个 RMSE 完全相同的模型，其实际误差结构可能截然不同。**残差图（Residual Plot）**是排查模型假设缺陷的重要诊断方法。

残差图通常以预测值 $\hat{y}_i$（或某个输入特征 $x_i$）为横坐标，以残差 $e_i = y_i - \hat{y}_i$ 为纵坐标绘制散点图。若拟合合理，残差应当表现为无规律的随机噪声。

![残差诊断四种典型模式](/images/content-rebuild/5cc549ae93b829ff/fig-5.png)

### 1. 理想状态：零线周围随机均匀散布（同方差性）
残差散点均匀、对称地散布在水平线 $e = 0$ 上下两侧，分布宽度随横轴推进保持恒定。这表明：
- 残差期望为 0，模型在各个预测区间均无系统性高估或低估；
- 误差的方差恒定（同方差，Homoscedasticity）；
- 当前模型的假设函数形式与特征基本匹配。

### 2. 缺陷形态一：U 型或倒 U 型弯曲（系统性非线性遗漏）
若残差散点呈现明显的抛物线、弧形或波浪形排布，说明残差并不是纯随机噪声，而是关于输入变量的确定性函数。这表明：
- 模型假设存在欠拟合（参见 [过拟合和欠拟合](/fundamentals/ml-basics/overfitting-underfitting.md)）；
- 真实关系存在二次项、指数项等高阶非线性规律，但模型采用了简单线性拟合，导致非线性趋势残留在误差中。

### 3. 缺陷形态二：漏斗状或喇叭口扩散（异方差性，Heteroscedasticity）
散点在预测值较小的一侧紧密集中，随着预测值增大，残差散布范围像漏斗一样向外扩张（或反向收窄）。这被称为**异方差性**：
- **成因**：误差波动随目标变量绝对大小变化而变化。例如在居民消费支出预测中，低收入群体的支出波动范围较窄，而高收入群体的消费选择多样，支出波动幅度大幅增加。
- **后果**：违背了经典回归的同方差假设，普通最小二乘估计的标准误将产生偏差，模型在高值区间的预测不确定性极高。
- **处理思路**：可对目标变量做对数变换（$\log(y)$）或 Box-Cox 变换，将随数值扩大的乘性波动转换为相对恒定的加性波动；或者改用加权最小二乘法（WLS）。

### 4. 缺陷形态三：走势连续自相关（Autocorrelation）
若残差散点前后相连，正残差后面倾向于跟随正残差，负残差后面跟随负残差，说明样本误差之间不满足独立同分布假设，存在自相关性。这常见于时间序列或具有空间邻近性的数据，通常提示特征工程中遗漏了重要的滞后项（Lag terms）或时序周期特征。

---

## 动手验证：基于 scikit-learn 的评估与诊断代码

在代码实践中，回归评估通常包括“指标数值计算”与“残差散点图可视化”两部分。下面使用 Python 与 `scikit-learn` 构造一个包含非线性与异方差特征的对比实验。需要明确的是，本实验主要用于直观观察拟合形态缺陷与数据异方差在残差分布上的几何表现；在真实业务中评估模型的泛化性能时，指标计算与残差诊断均应在独立的测试集（未参与训练的数据）上执行，避免在训练集上直接打分导致对模型泛化质量的高估。

![端到端回归评估工程闭环](/images/content-rebuild/5cc549ae93b829ff/fig-6.png)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. 生成包含二次项关系与异方差噪声的模拟数据
np.random.seed(42)
X = np.linspace(10, 100, 200)
# 噪声标准差随 X 线性放大，构造异方差现象
noise = np.random.normal(loc=0.0, scale=0.15 * X, size=len(X))
y_true = 0.05 * (X ** 2) + 2.0 * X + 15.0 + noise

# 2. 拟合两个对比模型：简单线性模型 vs 二次多项式模型
# 模型 A：欠拟合的一阶线性回归
coeffs_linear = np.polyfit(X, y_true, deg=1)
y_pred_linear = np.polyval(coeffs_linear, X)

# 模型 B：匹配真实数据趋势的二阶多项式拟合
coeffs_poly = np.polyfit(X, y_true, deg=2)
y_pred_poly = np.polyval(coeffs_poly, X)

# 3. 封装综合评估函数
def evaluate_regression(y_real, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    r2 = r2_score(y_real, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}\n")
    return y_real - y_pred

res_linear = evaluate_regression(y_true, y_pred_linear, "简单线性回归 (欠拟合)")
res_poly = evaluate_regression(y_true, y_pred_poly, "二次曲线回归 (形态匹配)")
```

运行后可以看到，二次曲线回归的 $R^2$ 显著高于简单线性回归，且 MAE 与 RMSE 均出现明显下降。

接下来绘制残差图，直观查看两个模型的误差分布形态：

```python
# 4. 残差诊断可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 线性模型残差图：呈现明显的系统性 U 型弧线
axes[0].scatter(y_pred_linear, res_linear, alpha=0.6, color="crimson")
axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
axes[0].set_title("Linear Model: Systematic U-shape Pattern")
axes[0].set_xlabel("Predicted (y_hat)")
axes[0].set_ylabel("Residual (y - y_hat)")

# 多项式模型残差图：消除 U 型，暴露出原始数据的异方差喇叭口
axes[1].scatter(y_pred_poly, res_poly, alpha=0.6, color="steelblue")
axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
axes[1].set_title("Polynomial Model: Random Mean with Heteroscedasticity")
axes[1].set_xlabel("Predicted (y_hat)")
axes[1].set_ylabel("Residual (y - y_hat)")

plt.tight_layout()
plt.show()
```

残差图清晰地揭示了标量指标反映不出的信息：
1. 简单线性模型的残差分布呈现强烈的凹形弯曲，直接证明特征与目标之间存在未被模型捕捉的高阶非线性规律；
2. 二次模型修正了函数形态，残差均值回归至 $0$ 平衡线附近，但残差扩散范围随着预测值增大呈喇叭口扩散，直接暴露出数据自带的**异方差性**。这为进一步采取对数变换或加权回归提供了直观依据。

在完整的机器学习评测体系中，无论是传统回归分析、树模型（如 [决策树、随机森林和GBDT](/fundamentals/ml-basics/tree-models.md)），还是更为复杂的预测任务（见 [AI 评测体系](/architecture/ai-evaluation.md) 与 [大模型如何评估？常见 Benchmark 有哪些？](/llm-100/evaluation/llm-evaluation-benchmarks.md)），在独立的测试集上结合标量指标度量与残差分布诊断，是检验模型真实泛化质量的严谨评估流程。
