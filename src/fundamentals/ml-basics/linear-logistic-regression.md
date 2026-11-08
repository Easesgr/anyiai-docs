---
title: 线性回归与逻辑回归：如何预测连续数值与离散概率？
icon: mdi:chart-line
tag:
- 机器学习
- 监督学习
- 经典算法
description: 用具体算例拆解线性回归与逻辑回归：分析连续拟合与概率映射，对比正规方程与梯度下降，推导二分类交叉熵损失与Softmax多分类计算。
---

在机器学习中，我们经常遇到两类基本任务：
1. **连续数值预测**：根据房屋面积、房龄预测二手房成交价，输出是一个连续的实数金额；
2. **离散类别判断**：根据患者的体检指标判断是否患有某种疾病，输出是离散标签（如“患病/未患病”），或者输出属于该类别的后验概率。

线性回归（Linear Regression）与逻辑回归（Logistic Regression）分别对应这两种任务的典型解法。在上一讲[/fundamentals/ml-basics/loss-gradient-descent.md](/fundamentals/ml-basics/loss-gradient-descent.md)中，我们了解了“用损失函数度量偏差，用梯度下降调整参数”的基本思路。本讲我们将以一个具体的预测场景贯穿始终，搞清这两个模型的数学假设、参数求解方法以及适用边界。

![线性回归与逻辑回归在几何映射上的核心差异与演进流程](/images/content-rebuild/dfbf1caa728de471/cover.png)

---

## 线性回归：用超平面拟合连续标签

### 假设函数与几何意义
假设我们要预测一套房屋的价格 $y$。手头有 $n$ 个特征，例如面积 $x_1$、卧室数量 $x_2$、离地铁站距离 $x_3$ 等，组成特征向量 $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top$。

线性回归的核心假设是：目标变量 $y$ 可以由输入特征的线性加权组合外加一个偏置项来近似表示：

$$\hat{y} = f(\mathbf{x}) = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}^\top \mathbf{x} + b$$

其中 $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top$ 为权重向量，$b$ 为偏置标量。为了书写简便，我们常把偏置 $b$ 吸收到权重向量中，令增广特征向量 $\tilde{\mathbf{x}} = [1, x_1, x_2, \dots, x_n]^\top$，增广权重向量 $\tilde{\mathbf{w}} = [b, w_1, w_2, \dots, w_n]^\top$，预测公式可简化为：

$$\hat{y} = \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}$$

从几何上看：
* 当只有 1 个特征时，假设函数是平面直角坐标系中的一条直线；
* 当有 2 个特征时，假设函数是三维空间中的一个平面；
* 当有 $n$ 个特征时，假设函数是 $n+1$ 维空间中的一个 **$n$ 维超平面（Hyperplane）**。

![超平面拟合数据点与残差垂线投影视意图](/images/content-rebuild/dfbf1caa728de471/fig-1.png)

对于数据集中的第 $i$ 个样本点 $(\mathbf{x}_i, y_i)$，真实值与预测值之间的差距称为**残差（Residual）**：

$$e_i = y_i - \hat{y}_i = y_i - \tilde{\mathbf{w}}^\top \tilde{\mathbf{x}}_i$$

在标准普通最小二乘法（Ordinary Least Squares, OLS）中，统计视角通常假设标签中的噪声满足独立同分布的高斯分布，即 $y_i = \mathbf{w}^\top \mathbf{x}_i + b + \epsilon_i$，其中 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$。这意味着真实值围绕拟合平面上下对称分布，且方差 $\sigma^2$ 不随特征变化而改变（同方差假设）。需要注意：这里的残差度量的是**沿目标轴 $y$ 方向的竖直差距**，而非样本点到超平面的空间正交最短距离。

---

## 求解线性回归：正规方程与梯度下降

线性回归通常采用均方误差（Mean Squared Error, MSE）作为损失函数。设数据集包含 $m$ 个样本，损失函数定义为残差平方和的均值（分母乘 2 是为了求导时抵消平方项掉下来的指数）：

$$J(\tilde{\mathbf{w}}) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 = \frac{1}{2m} \|\mathbf{X}\tilde{\mathbf{w}} - \mathbf{y}\|_2^2$$

其中设计矩阵 $\mathbf{X} \in \mathbb{R}^{m \times (n+1)}$ 的每一行是一个样本的增广特征向量，$\mathbf{y} \in \mathbb{R}^m$ 是所有样本的真实标签向量。

![正规方程正交投影闭式解与梯度下降迭代下坡路径对比](/images/content-rebuild/dfbf1caa728de471/fig-2.png)

因为 $J(\tilde{\mathbf{w}})$ 是关于 $\tilde{\mathbf{w}}$ 的二次凸函数，求解其全局最小值有两种常用途径：

### 1. 正规方程闭式解（Normal Equation）
对二次损失函数直接求导并令梯度为零：

$$\nabla_{\tilde{\mathbf{w}}} J(\tilde{\mathbf{w}}) = \frac{1}{m} \mathbf{X}^\top (\mathbf{X}\tilde{\mathbf{w}} - \mathbf{y}) = 0$$

移项整理得到正规方程：

$$\mathbf{X}^\top \mathbf{X} \tilde{\mathbf{w}} = \mathbf{X}^\top \mathbf{y}$$

如果矩阵 $\mathbf{X}^\top \mathbf{X}$ 可逆，可以直接写出解析解：

$$\tilde{\mathbf{w}}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**适用条件与限制**：
* **可逆前提**：矩阵 $\mathbf{X}^\top \mathbf{X}$ 可逆的前提是样本数不少于特征数（$m \ge n+1$），且特征之间不存在完全多重共线性（如某个特征是另一特征的倍数）。当特征高度相关或样本量不足时，矩阵会不可逆或接近奇异，需要引入 L2 正则化（岭回归）。
* **计算开销**：求逆矩阵的时间复杂度约为 $\mathcal{O}(n^3)$。当特征数量 $n$ 达到数万时，直接求逆会消耗大量内存与计算资源。

### 2. 梯度下降迭代解（Gradient Descent）
当数据规模或特征维度较大时，通常采用迭代法沿梯度负方向逐步更新权重：

$$\tilde{\mathbf{w}}^{(t+1)} = \tilde{\mathbf{w}}^{(t)} - \alpha \cdot \frac{1}{m} \mathbf{X}^\top (\mathbf{X}\tilde{\mathbf{w}}^{(t)} - \mathbf{y})$$

单步更新的时间复杂度为 $\mathcal{O}(mn)$，不仅对高维特征更加稳健，还能通过小批量（Mini-batch）切片支持流式数据训练。

### 正规方程手算算例
为了直观理解正规方程的运算流程，我们使用 $m=3$ 个一维特征样本进行手算：

* 样本 1：$x_1 = 1, y_1 = 2$
* 样本 2：$x_2 = 2, y_2 = 3$
* 样本 3：$x_3 = 3, y_3 = 5$

构造增广矩阵 $\mathbf{X}$ 与目标向量 $\mathbf{y}$：

$$\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}$$

**第一步：计算 $\mathbf{X}^\top \mathbf{X}$**
$$\mathbf{X}^\top \mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 1+1+1 & 1+2+3 \\ 1+2+3 & 1+4+9 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$

**第二步：求逆矩阵 $(\mathbf{X}^\top \mathbf{X})^{-1}$**
计算二阶行列式：$\det(\mathbf{X}^\top \mathbf{X}) = (3 \times 14) - (6 \times 6) = 42 - 36 = 6 \neq 0$，行列式非零，矩阵可逆。
应用二阶伴随矩阵求逆公式：
$$(\mathbf{X}^\top \mathbf{X})^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}$$

**第三步：计算 $\mathbf{X}^\top \mathbf{y}$**
$$\mathbf{X}^\top \mathbf{y} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = \begin{bmatrix} 2+3+5 \\ 2+6+15 \end{bmatrix} = \begin{bmatrix} 10 \\ 23 \end{bmatrix}$$

**第四步：求出最优参数 $\tilde{\mathbf{w}}^*$**
$$\tilde{\mathbf{w}}^* = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} \begin{bmatrix} 10 \\ 23 \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 140 - 138 \\ -60 + 69 \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 2 \\ 9 \end{bmatrix} = \begin{bmatrix} \frac{1}{3} \\ \frac{3}{2} \end{bmatrix} \approx \begin{bmatrix} 0.333 \\ 1.5 \end{bmatrix}$$

计算得到偏置 $b = 0.333$，权重 $w_1 = 1.5$。拟合出的直线方程为 $\hat{y} = 1.5x + 0.333$。

<ReadMoreLock />

---

## 为什么不能直接用线性回归做分类？

在二分类场景中，标签是离散的：$y \in \{0, 1\}$。一个朴素的想法是：既然线性回归能拟合数值，我们把标签直接当作 $0$ 和 $1$，用线性回归拟合，然后设定阈值（比如 $\hat{y} \ge 0.5$ 判定为类别 1），这样做可行吗？

在实际工程和数学上，直接使用线性回归处理分类存在三个主要问题：

![线性回归拟合分类标签时受离群点拉扯产生边界偏移的失效图示](/images/content-rebuild/dfbf1caa728de471/fig-3.png)

1. **输出超出概率边界**：线性函数的输出范围是 $(-\infty, +\infty)$。当遇到较小的特征值时，模型预测值可能变成 $-0.5$；遇到较大的特征值时，可能变成 $1.8$。负数或大于 1 的值无法直接作为概率使用。
2. **极值样本拉偏分类分界面**：最小二乘法根据平方误差做调整。如果出现了一个特征值极大且类别明确为 1 的样本，模型预测值为 $2.0$。在分类逻辑中，该样本分类正确且置信度高；但均方误差会认为存在 $(2.0 - 1)^2 = 1.0$ 的较大残差，为了降低该残差，拟合直线会被强行拉向该离群样本，导致原本在 $0.5$ 附近划分准确的临界样本被误判。
3. **破坏误差的同方差假设**：在二分类问题中，输出变量服从伯努利分布，其方差为 $\text{Var}(y|\mathbf{x}) = p(\mathbf{x})(1 - p(\mathbf{x}))$。方差与输入特征直接相关，不再是固定常数，违背了普通最小二乘法对误差具有相同方差的假设。

---

## 逻辑回归：从几率到 Sigmoid 映射

为了解决上述问题，逻辑回归保留了特征的线性加权形式 $\mathbf{w}^\top \mathbf{x} + b$，但引入了一个非线性变换，将实数输出压缩至 $(0, 1)$ 区间内，用来表示正类的后验概率 $p = P(y=1|\mathbf{x})$。

### 几率与对数几率（Logit）
在概率论中，一个事件发生的**几率（Odds）**定义为“该事件发生概率”与“不发生概率”的比值：

$$\text{Odds} = \frac{p}{1 - p}$$

* 当事件发生概率 $p$ 从 0 趋向 1 时，几率 $\text{Odds}$ 从 0 趋向 $+\infty$。
* 对几率取自然对数，得到**对数几率（Log-odds，即 Logit）**：

$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$$

当 $p \in (0, 1)$ 时，对数几率的取值范围被扩展至整个实数空间 $(-\infty, +\infty)$。

逻辑回归的建模思路是：**假设样本属于正类的对数几率与输入特征呈线性关系**：

$$\ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^\top \mathbf{x} + b$$

![Logit对数几率与Sigmoid函数相互映射的关系结构图](/images/content-rebuild/dfbf1caa728de471/fig-4.png)

### 导出 Sigmoid 函数
为了根据输入特征反求出后验概率 $p$，我们对上式两端取指数：

$$\frac{p}{1 - p} = e^{\mathbf{w}^\top \mathbf{x} + b}$$

令未压缩的线性得分标量 $z = \mathbf{w}^\top \mathbf{x} + b$：

$$p = e^z(1 - p) \implies p(1 + e^z) = e^z \implies p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}$$

这一变换函数即为 **Sigmoid 函数**（记作 $\sigma(z)$）：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

逻辑回归预测正类的条件概率公式为：

$$\hat{y} = P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

### 决策边界仍是线性的
若以 $0.5$ 作为判定阈值：当 $\hat{y} \ge 0.5$ 时判定为正类，否则为负类。
因为 $\sigma(z) \ge 0.5$ 等价于 $z \ge 0$，所以判定分界面实际上是：

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

这表明：**逻辑回归虽然通过 Sigmoid 将输出压缩为概率，但在特征空间中划定类别的决策边界依然是一个线性超平面**。若不经过特征交叉或引入非线性核，逻辑回归无法直接解决异或（XOR）等非线性分类问题。

---

## 损失函数：从最大似然估计推导交叉熵

在线性回归中，我们直接使用残差均方误差。如果把 Sigmoid 函数代入均方误差：

$$J_{\text{MSE}}(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^m \left( \sigma(\mathbf{w}^\top \mathbf{x}_i + b) - y_i \right)^2$$

由于 Sigmoid 函数在两端梯度接近 0，复合后的损失函数曲面存在多个局部平坦区和非全局极小值，使用梯度下降难以保证收敛到理想解。因此，逻辑回归采用**最大似然估计（Maximum Likelihood Estimation, MLE）**构造损失函数。

![从独立样本概率连乘到对数相加再到交叉熵凸曲面的优化演进流程](/images/content-rebuild/dfbf1caa728de471/fig-5.png)

### 构建似然函数与交叉熵损失
在二分类中，样本标签 $y_i \in \{0, 1\}$。预测概率为 $\hat{y}_i = P(y_i=1|\mathbf{x}_i)$。单个样本的概率质量函数可以统一写作：

$$P(y_i | \mathbf{x}_i) = \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$

* 当 $y_i = 1$ 时，上式结果为 $\hat{y}_i^1 (1 - \hat{y}_i)^0 = \hat{y}_i$；
* 当 $y_i = 0$ 时，上式结果为 $\hat{y}_i^0 (1 - \hat{y}_i)^1 = 1 - \hat{y}_i$。

假设所有样本独立同分布，整个数据集的联合似然函数为：

$$L(\mathbf{w}, b) = \prod_{i=1}^m \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$

为了避免连乘导致的数值下溢并简化求导，两边取自然对数，得到对数似然：

$$\ell(\mathbf{w}, b) = \ln L(\mathbf{w}, b) = \sum_{i=1}^m \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

最大似然估计要求最大化 $\ell(\mathbf{w}, b)$。为了转化成机器学习中习惯的“最小化损失”，我们加上负号并除以样本量 $m$，得到**二分类交叉熵损失（Binary Cross-Entropy, BCE）**：

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

该损失函数关于权重向量 $\mathbf{w}$ 和偏置 $b$ 是凸函数，保证了梯度下降法能够收敛到全局最优点。

### 梯度推导与参数更新
求损失函数对第 $j$ 个权重参数 $w_j$ 的偏导。我们先针对单个样本 $i$ 的二分类交叉熵损失 $J_i = - [ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) ]$ 应用链式法则：

$$\frac{\partial J_i}{\partial w_j} = \frac{\partial J_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_j}$$

已知 Sigmoid 函数的导数具有性质：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$。逐步代入求导项：
1. $\frac{\partial J_i}{\partial \hat{y}_i} = -\left( \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right) = \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)}$
2. $\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1 - \hat{y}_i)$
3. $\frac{\partial z_i}{\partial w_j} = x_{ij}$

三项相乘后，分母与中间导数项相互抵消：

$$\frac{\partial J_i}{\partial w_j} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)} \cdot \hat{y}_i(1 - \hat{y}_i) \cdot x_{ij} = (\hat{y}_i - y_i) x_{ij}$$

最终求得的平均梯度为：

$$\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) x_{ij}$$

偏置项的梯度同理为：

$$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$

**梯度形式的联系**：逻辑回归使用交叉熵得到的梯度公式，在形式上与线性回归使用均方误差求出的梯度完全一致。两者的核心区别在于：线性回归中的 $\hat{y}_i$ 是未经约束的线性结果 $\mathbf{w}^\top \mathbf{x}_i + b$，而逻辑回归中的 $\hat{y}_i$ 是经过 Sigmoid 函数压缩后的概率值 $\sigma(\mathbf{w}^\top \mathbf{x}_i + b)$。

---

## 多分类推广：Softmax 函数

当目标从二分类扩展到 $K$ 个互斥类别（$y \in \{1, 2, \dots, K\}$）时，单个 Sigmoid 不再适用，需要推广为 Softmax 回归（多元逻辑回归）。

![Logits经Softmax指数竞争与归一化输出概率柱状图](/images/content-rebuild/dfbf1caa728de471/fig-6.png)

### 假设函数
模型为每个类别维护一组参数 $\mathbf{w}_k$ 和 $b_k$。输入特征 $\mathbf{x}$ 首先分别计算出每个类别的未归一化线性得分（Logits）：

$$z_k = \mathbf{w}_k^\top \mathbf{x} + b_k, \quad k \in \{1, 2, \dots, K\}$$

Softmax 函数通过指数放大和全局归一化，将得分向量转换为概率分布：

$$P(y = k | \mathbf{x}) = \text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

每个类别的输出值均位于 $(0, 1)$ 区间内，且所有类别的概率和严格为 1。当类别数 $K=2$ 时，Softmax 函数在代数上直接退化为标准 Sigmoid 函数。

### 多分类交叉熵损失
结合独热编码（One-Hot Encoding）标签向量 $\mathbf{y}_i$（仅真实类别位置为 1，其余为 0），损失函数定义为：

$$J(\mathbf{W}) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{ik} \ln \hat{y}_{ik}$$

### 数值计算细节：防止指数上溢
在计算机中直接计算 $e^{z_k}$ 时，如果得分 $z_k$ 较大（如 $z_k = 1000$），浮点数会触发上溢（Overflow）变成 `inf`；如果分母所有项都很小，则可能产生除以 0 的错误。

工程中通常利用指数性质进行平移缩放：

$$\frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{-C} \cdot e^{z_k}}{e^{-C} \cdot \sum_j e^{z_j}} = \frac{e^{z_k - C}}{\sum_j e^{z_j - C}}$$

实际实现时令 $C = \max(\mathbf{z})$。在计算指数前，先让得分向量中的每个元素减去当前向量的最大值：

$$z'_k = z_k - \max(z_1, \dots, z_K)$$

此时指数项的最大输入为 $e^0 = 1$，有效防止了数值溢出，同时数学结果保持等价。

---

## 最小可执行代码：NumPy 实现逻辑回归二分类

以下脚本使用 NumPy 实现逻辑回归的核心训练循环，包含 Sigmoid 映射、BCE 损失计算以及梯度下降更新：

```python
import numpy as np

def sigmoid(z):
    # 避免数值溢出：截断在合理范围内
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_bce_loss(y_true, y_pred, eps=1e-15):
    # eps 防止 log(0) 产生 NaN
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    m = y_true.shape[0]
    return - (1.0 / m) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

def train_logistic_regression(X, y, lr=0.5, epochs=300):
    m, n = X.shape
    # 初始化权重和偏置
    w = np.zeros(n)
    b = 0.0
    
    for epoch in range(epochs):
        # 1. 线性得分计算
        z = np.dot(X, w) + b
        # 2. Sigmoid 概率映射
        y_pred = sigmoid(z)
        
        # 3. 计算梯度 (y_pred - y)
        error = y_pred - y
        grad_w = (1.0 / m) * np.dot(X.T, error)
        grad_b = (1.0 / m) * np.sum(error)
        
        # 4. 参数更新
        w -= lr * grad_w
        b -= lr * grad_b
        
        if epoch % 100 == 0:
            loss = compute_bce_loss(y, y_pred)
            print(f"Epoch {epoch:3d} | BCE Loss: {loss:.4f}")
            
    return w, b

if __name__ == "__main__":
    # 准备 4 个二分类数据点（2个特征）
    X_samples = np.array([
        [1.0, 1.0],
        [2.0, 1.5],
        [4.0, 4.5],
        [5.0, 4.0]
    ])
    y_samples = np.array([0.0, 0.0, 1.0, 1.0])
    
    weights, bias = train_logistic_regression(X_samples, y_samples, lr=0.5, epochs=301)
    print("\n训练结果：")
    print(f"权重 w: {weights.round(4)}")
    print(f"偏置 b: {bias:.4f}")
    
    # 预测一个靠近类别 0 的未知测试点
    test_pt = np.array([1.5, 1.2])
    prob = sigmoid(np.dot(test_pt, weights) + bias)
    print(f"测试点 {test_pt} 预测为类别 1 的概率: {prob:.4f}")
```

运行结果中，损失值随迭代逐步下降，最终得到的线性决策参数能将正负两类点有效分开。

---

## 总结与承接

* **线性回归**：用超平面拟合连续标签。参数可以通过正规方程一次性求出解析解，也可以使用梯度下降在大数据量下迭代逼近。
* **逻辑回归**：对线性得分建立对数几率，利用 Sigmoid 函数将实数映射为 $(0, 1)$ 的概率。其决策分界面依然是线性的。
* **损失函数选择**：线性回归在高斯同方差假设下使用均方误差；逻辑回归在伯努利分布假设下使用最大似然估计推导出交叉熵，保证了损失曲面的凸性。
* **多分类扩展**：Softmax 通过指数放大和全部分母归一化输出互斥类别的概率，并在计算中通过减去最大值保证数值稳定。

在训练模型时，无论是回归还是分类，我们不能只看训练集上的损失。如果模型记住了训练样本中的特定噪声，就会在新的测试数据上表现恶化。在下一讲[/fundamentals/ml-basics/train-val-test.md](/fundamentals/ml-basics/train-val-test.md)中，我们将讨论如何科学划分训练集、验证集与测试集，建立可靠的模型评估流程。
