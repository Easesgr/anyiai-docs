---
title: 神经网络是怎么计算的？从单神经元到多层感知机
icon: mdi:brain
tag:
- 深度学习
- 神经网络
- 多层感知机
description: 以贯穿的二维分类算例，拆解单神经元的加权求和与偏置平移，推导无激活多层网络的折叠退化，并用张量维度与PyTorch代码验证前向计算。
---

在上一篇[深度学习到底是什么？它和机器学习有什么区别？](/fundamentals/dl-basics/what-is-dl.md)中，我们明确了深度学习的核心特征是通过多层网络自动从数据中学习特征表示。本文进一步拆解这一表示学习的基础计算构件：**人工神经元（Artificial Neuron）**以及由其构成的**多层感知机（Multilayer Perceptron, MLP）**。

我们将从单个神经元的数学形式出发，观察其几何意义，推导为什么仅堆叠线性层无法扩展模型的表达能力，并通过具体的张量维度与代码实验，验证多层感知机在前向传播中的张量变换逻辑。

![人工神经元到多层感知机的计算流程总览](/images/content-rebuild/19c1bbb7c07e697c/cover.png)

---

## 人工神经元的计算步骤：加权求和、偏置平移与激活输出

人工神经元是神经网络最基础的信息处理单元。它的计算过程接收一个由多个特征构成的数值向量，经过加权组合与平移后，输出一个标量数值。

设输入特征向量为 $\mathbf{x} = [x_1, x_2, \dots, x_D]^T \in \mathbb{R}^D$，对应的可学习参数包含：
1. **权重向量（Weight Vector）**：$\mathbf{w} = [w_1, w_2, \dots, w_D]^T \in \mathbb{R}^D$，用来量化各输入特征对该神经元激活状态的重要程度；
2. **偏置标量（Bias）**：$b \in \mathbb{R}$，用来平移神经元的基准阈值。

神经元的计算分为两步：**仿射变换（Affine Transformation）**与**非线性激活（Nonlinear Activation）**。

![单个神经元的两阶段计算流程图](/images/content-rebuild/19c1bbb7c07e697c/fig-1.png)

第一阶段计算加权和与偏置，得到净激活值（Net Input）$z$：
$$
z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^D w_i x_i + b
$$

第二阶段将标量 $z$ 输入非线性激活函数 $\sigma(\cdot)$，得到神经元的最终输出 $a$：
$$
a = \sigma(z) = \sigma\left(\sum_{i=1}^D w_i x_i + b\right)
$$

如果省去偏置 $b$（即令 $b = 0$），当输入全为零向量 $\mathbf{x} = \mathbf{0}$ 时，无论权重 $\mathbf{w}$ 取何值，净激活值恒为 $z = 0$。偏置的作用在于打破输入空间必须过原点的限制，使神经元能够在坐标系中自由平移响应边界。

---

## 单神经元的几何解释：超平面划分与线性决策边界

为了直观理解公式 $z = \mathbf{w}^T \mathbf{x} + b$，我们考察二维输入空间 $\mathbf{x} = [x_1, x_2]^T$ 下的分类任务。

当神经元的激活状态以阈值 $z = 0$ 为分界时，方程：
$$
w_1 x_1 + w_2 x_2 + b = 0
$$
在二维平面上定义了一条决策直线。在这条直线的某一侧，$z > 0$；在另一侧，$z < 0$。

![二维空间中单神经元超平面的几何划分机制](/images/content-rebuild/19c1bbb7c07e697c/fig-2.png)

在几何上，这条直线的法向量正是权重向量 $\mathbf{w} = [w_1, w_2]^T$。法向量垂直于直线并指向 $z > 0$ 的正半平面。偏置 $b$ 则控制直线的位置：坐标原点到该直线的几何垂直距离为：
$$
d = \frac{|b|}{\|\mathbf{w}\|_2}
$$
当 $b=0$ 时，直线必过原点；改变 $b$ 即可沿法向量方向平行移动该分界线。

推广到 $D$ 维特征空间，$z = \mathbf{w}^T \mathbf{x} + b = 0$ 对应的是一个 $(D-1)$ 维的**超平面（Hyperplane）**。该超平面将 $D$ 维输入空间划分为两个互不相交的半空间。

这种几何特性表明：**单个没有级联的神经元是一个线性二分类器**。它仅能正确划分线性可分（Linearly Separable）的数据分布。如果样本分布呈现同心圆环形，或经典的异或逻辑（XOR）分布，没有任何单一超平面能够将其区分开。需要说明的是，传统机器学习同样可以通过支持向量机（配合非线性核函数）或决策树等非线性模型处理此类数据，而神经网络选择的路径是通过多层网络与激活函数的组合来实现非线性拟合。

<ReadMoreLock />

---

## 为什么单纯线性层堆叠无法扩展表达能力：复合线性变换的折叠推导

既然单神经元的划分能力局限于单一超平面，能否在不引入激活函数的前提下，通过堆叠多个全连接层来提升网络的拟合能力？

答案是否定的。在代数上，多个线性变换（仿射变换）的复合仍然严格是一个仿射变换。以下给出具体推导。

![双层无激活全连接网络折叠为等价单层的推导图](/images/content-rebuild/19c1bbb7c07e697c/fig-3.png)

假设我们构建一个两层全连接网络，输入向量为 $\mathbf{x} \in \mathbb{R}^{D_{in}}$，第一层隐藏维度为 $D_1$，第二层输出维度为 $D_{out}$。如果每一层均不使用激活函数，其计算流程为：
1. 第一层前向计算：
$$
\mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1
$$
其中 $\mathbf{W}_1 \in \mathbb{R}^{D_1 \times D_{in}}$，$\mathbf{b}_1 \in \mathbb{R}^{D_1}$。

2. 第二层前向计算：
$$
\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
$$
其中 $\mathbf{W}_2 \in \mathbb{R}^{D_{out} \times D_1}$，$\mathbf{b}_2 \in \mathbb{R}^{D_{out}}$。

我们将第一层的输出 $\mathbf{h}$ 直接代入第二层方程：
$$
\mathbf{y} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)
$$

令：
$$
\mathbf{W}_{\text{eff}} = \mathbf{W}_2 \mathbf{W}_1 \in \mathbb{R}^{D_{out} \times D_{in}}
$$
$$
\mathbf{b}_{\text{eff}} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2 \in \mathbb{R}^{D_{out}}
$$

则两层网络的端到端映射直接折叠为：
$$
\mathbf{y} = \mathbf{W}_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}
$$

这一结论对任意深度的线性堆叠均成立：对于一个包含 $L$ 层的无激活函数网络，整体映射可等价缩减为单层网络 $\mathbf{y} = \mathbf{W}' \mathbf{x} + \mathbf{b}'$。此时无论增加多少层，其可表示的函数族均受限于线性映射，无法拟合弯曲的决策边界。

这就是神经网络必须在层间插入逐元素非线性激活函数（如 ReLU、GELU 等）的根本原因。关于各类激活函数的收敛特性与梯度流动细节，我们将在[激活函数有什么用？Sigmoid、Tanh、ReLU 到底怎么选？](/fundamentals/dl-basics/activation-functions.md)中系统讨论。

---

## 多层感知机（MLP）的矩阵表达与批量张量维度推导

在实际工程实现中，硬件（如 GPU）擅长批量张量运算。框架不会对样本逐一进行循环，而是将多个样本打包为一个批次（Batch）进行矩阵乘法。

我们以包含单个隐藏层的多层感知机为例，推导批处理模式下的张量形状演变。

![批量多层感知机全连接运算的张量维度演变与广播对齐](/images/content-rebuild/19c1bbb7c07e697c/fig-4.png)

### 1. 维度符号定义
* $B$：批大小（Batch Size），即单次前向传播同时处理的样本数；
* $D_{in}$：输入特征维度；
* $D_{hid}$：隐藏层神经元数量；
* $D_{out}$：输出层神经元数量。

### 2. 前向矩阵运算流程
**输入张量**：批数据组织为矩阵 $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$，其中每一行代表一个样本的特征向量。

**隐藏层仿射变换**：
$$
\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1
$$
* 权重矩阵 $\mathbf{W}_1 \in \mathbb{R}^{D_{in} \times D_{hid}}$；
* 偏置向量 $\mathbf{b}_1 \in \mathbb{R}^{D_{hid}}$。

在矩阵加法 $\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1$ 中，$\mathbf{X} \mathbf{W}_1$ 的形状为 $(B, D_{hid})$，而 $\mathbf{b}_1$ 的形状为 $(D_{hid},)$。运算中触发了**广播机制（Broadcasting）**：偏置向量 $\mathbf{b}_1$ 自动沿着批次维度对齐为 $(B, D_{hid})$，使批次内的每个样本均加上相同的偏置量。

> **框架实现注记**：在 PyTorch 的 `nn.Linear(in_features, out_features)` 内部，参数矩阵 `weight` 的存储形状为 `(out_features, in_features)`，底层前向计算采用的是转置乘法 $\mathbf{X} \mathbf{W}^T + \mathbf{b}$。两者在代数本质上一致，但在直接操作张量权重时须注意这一转置关系。

**隐藏层激活**：
$$
\mathbf{H} = \sigma(\mathbf{Z}_1) \in \mathbb{R}^{B \times D_{hid}}
$$
其中 $\sigma(\cdot)$ 为逐元素（element-wise）激活函数，不改变张量形状。

**输出层仿射变换**：
$$
\mathbf{Z}_2 = \mathbf{H} \mathbf{W}_2 + \mathbf{b}_2 \in \mathbb{R}^{B \times D_{out}}
$$
其中 $\mathbf{W}_2 \in \mathbb{R}^{D_{hid} \times D_{out}}$，$\mathbf{b}_2 \in \mathbb{R}^{D_{out}}$。

### 3. 具体算例演算
设 $B = 2, D_{in} = 3, D_{hid} = 2$。给定输入矩阵：
$$
\mathbf{X} = \begin{bmatrix} 1.0 & 0.0 & 2.0 \\ -1.0 & 1.0 & 0.0 \end{bmatrix}_{2 \times 3}
$$
权重矩阵与偏置向量：
$$
\mathbf{W}_1 = \begin{bmatrix} 0.5 & -0.2 \\ 1.0 & 0.0 \\ -0.5 & 0.8 \end{bmatrix}_{3 \times 2}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.1 & -0.3 \end{bmatrix}_{1 \times 2}
$$

计算矩阵乘积 $\mathbf{X} \mathbf{W}_1$：
* 第 1 个样本：
  * 节点 1：$1.0 \times 0.5 + 0.0 \times 1.0 + 2.0 \times (-0.5) = -0.5$
  * 节点 2：$1.0 \times (-0.2) + 0.0 \times 0.0 + 2.0 \times 0.8 = 1.4$
* 第 2 个样本：
  * 节点 1：$(-1.0) \times 0.5 + 1.0 \times 1.0 + 0.0 \times (-0.5) = 0.5$
  * 节点 2：$(-1.0) \times (-0.2) + 1.0 \times 0.0 + 0.0 \times 0.8 = 0.2$

加上偏置向量 $\mathbf{b}_1$：
$$
\mathbf{Z}_1 = \begin{bmatrix} -0.5 & 1.4 \\ 0.5 & 0.2 \end{bmatrix} + \begin{bmatrix} 0.1 & -0.3 \\ 0.1 & -0.3 \end{bmatrix} = \begin{bmatrix} -0.4 & 1.1 \\ 0.6 & -0.1 \end{bmatrix}_{2 \times 2}
$$

若采用 $\text{ReLU}(z) = \max(0, z)$ 激活，则：
$$
\mathbf{H} = \text{ReLU}(\mathbf{Z}_1) = \begin{bmatrix} 0.0 & 1.1 \\ 0.6 & 0.0 \end{bmatrix}_{2 \times 2}
$$

---

## 隐藏层如何解决非线性问题：异或（XOR）坐标变换

多层感知机能够解决非线性问题的关键，在于**隐藏层将原始输入特征空间进行了非线性折叠与映射**，使得原本线性不可分的点集在新坐标系下变为线性可分。

以经典的异或（XOR）问题为例。输入空间包含 4 个二维点：
* 类别 0：$(0, 0)$ 和 $(1, 1)$
* 类别 1：$(0, 1)$ 和 $(1, 0)$

在二维平面 $(x_1, x_2)$ 中，没有任何一条单一直线能够同时将这两组点分割开来。

![XOR问题在隐层空间的非线性折叠与线性可分映射](/images/content-rebuild/19c1bbb7c07e697c/fig-5.png)

现在引入一个包含 2 个隐藏神经元的网络，令两个隐藏单元使用带有阈值的线性整流激活（ReLU）：
* 隐藏单元 1：$h_1 = \text{ReLU}(x_1 + x_2 - 0.5)$
* 隐藏单元 2：$h_2 = \text{ReLU}(x_1 + x_2 - 1.5)$

我们计算四个点在隐藏特征空间 $(h_1, h_2)$ 中的新坐标映射：
1. 点 $(0, 0) \implies z_1 = -0.5, z_2 = -1.5 \implies (h_1, h_2) = (0.0, 0.0)$；
2. 点 $(0, 1) \implies z_1 = 0.5, z_2 = -0.5 \implies (h_1, h_2) = (0.5, 0.0)$；
3. 点 $(1, 0) \implies z_1 = 0.5, z_2 = -0.5 \implies (h_1, h_2) = (0.5, 0.0)$；
4. 点 $(1, 1) \implies z_1 = 1.5, z_2 = 0.5 \implies (h_1, h_2) = (1.5, 0.5)$。

分析这一坐标变化：
* 原本在对角两侧的 $(0, 1)$ 和 $(1, 0)$，由于激活函数的下限截断（负值置零），被映射到了隐藏空间的同一个坐标点 $(0.5, 0.0)$；
* 此时类别 1 集中在 $(0.5, 0.0)$，而类别 0 的两个样本分别位于 $(0.0, 0.0)$ 与 $(1.5, 0.5)$。

在新的 $(h_1, h_2)$ 坐标系下，类别 0 的两点 $(0, 0)$ 与 $(1.5, 0.5)$ 所在的直线方程为 $h_1 - 3 h_2 = 0$。而类别 1 的点 $(0.5, 0)$ 代入该式得到 $0.5 - 0 = 0.5 > 0$。因此，选取一条平行直线（例如 $h_1 - 3 h_2 - 0.2 = 0$），即可实现完全线性切分：
* 类别 0 的点 $(0, 0) \implies 0 - 0 - 0.2 = -0.2 < 0$；
* 类别 0 的点 $(1.5, 0.5) \implies 1.5 - 3(0.5) - 0.2 = -0.2 < 0$；
* 类别 1 的点 $(0.5, 0.0) \implies 0.5 - 0 - 0.2 = +0.3 > 0$。

通过隐藏层的加权和非线性激活，不可分的数据分布被映射为线性可分分布，后续输出层仅需执行一次超平面划分即可完成分类。

---

## 使用 PyTorch 验证最小全连接网络的前向张量计算

为验证上述数学计算，我们使用 [PyTorch 自动微分与张量基础教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) 中的核心算子，构建一个两层多层感知机的前向计算脚本，核对张量形状与数值等价性。

```python
import torch
import torch.nn as nn

# 固定随机种子以便复现输出
torch.manual_seed(42)

# 1. 定义批大小与各层特征维度
B = 4        # Batch Size (样本数)
D_in = 8     # 输入特征维度
D_hid = 16   # 隐藏层维度
D_out = 2    # 输出特征维度（例如二分类 logits）

# 2. 生成模拟输入数据张量
X = torch.randn(B, D_in)
print(f"输入张量 X 形状: {X.shape}")  # [4, 8]

# -------------------------------------------------------------
# 方法 A: 手动维护参数并使用基础矩阵算子完成前向计算
# -------------------------------------------------------------
# 定义第一层参数 (隐藏层)
W1 = torch.randn(D_in, D_hid)
b1 = torch.randn(D_hid)

# 定义第二层参数 (输出层)
W2 = torch.randn(D_hid, D_out)
b2 = torch.randn(D_out)

# 第一层前向计算：矩阵乘法 + 偏置广播 + ReLU 激活
Z1_manual = torch.matmul(X, W1) + b1
A1_manual = torch.relu(Z1_manual)
print(f"手动计算 - 隐层线性输出 Z1 形状: {Z1_manual.shape}")  # [4, 16]
print(f"手动计算 - 隐层激活输出 A1 形状: {A1_manual.shape}")  # [4, 16]

# 第二层前向计算：输出仿射变换
Z2_manual = torch.matmul(A1_manual, W2) + b2
print(f"手动计算 - 输出层 Logits Z2 形状: {Z2_manual.shape}") # [4, 2]

# -------------------------------------------------------------
# 方法 B: 使用 torch.nn.Linear 模块并对齐参数验证等价性
# -------------------------------------------------------------
class MinimalMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out

model = MinimalMLP(D_in, D_hid, D_out)

# 将方法 A 的权重精确同步到模型中（注意 nn.Linear 的权重转置关系）
with torch.no_grad():
    model.fc1.weight.copy_(W1.t())
    model.fc1.bias.copy_(b1)
    model.fc2.weight.copy_(W2.t())
    model.fc2.bias.copy_(b2)

# 运行模块前向传播
Z2_module = model(X)

# 3. 验证两种实现前向张量数值是否一致
max_diff = torch.max(torch.abs(Z2_manual - Z2_module)).item()
print(f"手动算子与 nn.Module 输出最大绝对误差: {max_diff:.6e}")
assert max_diff < 1e-6, "前向传播数值计算存在偏差！"
print("前向传播张量维度与数值对齐验证通过。")
```

![PyTorch前向网络执行与张量形状断言验证机制](/images/content-rebuild/19c1bbb7c07e697c/fig-6.png)

在上述代码中，各个步骤体现了前文的推导：
1. 输入张量 $\mathbf{X}$ 形状为 `[B, D_in]`；
2. 经权重矩阵变换后，第一层输出映射为 `[B, D_hid]`；
3. 偏置向量形状为 `[D_hid]`，通过底层广播机制自动扩充并累加到批次各样本；
4. `nn.Linear` 模块内部存储的权重形状为 `(out_features, in_features)`，手动赋值时使用 `.t()` 转置以确保代数对齐。

前向计算给出了给定输入下的预测输出。在模型训练过程中，还需通过损失函数衡量预测与真实标签的偏差，并利用反向传播算法（借助自动微分引擎构建的动态计算图）逐层求解参数梯度以完成更新。相关机制将在后续的[前向传播和反向传播是什么？神经网络是怎么训练的？](/fundamentals/dl-basics/forward-backward-propagation.md)中展开讨论。
