---
title: 前向传播和反向传播是什么？神经网络是怎么训练的？
icon: mdi:arrow-up-down
tag:
- 深度学习
- 反向传播
- 模型训练
description: 通过一个具体算例，讲清神经网络前向计算、动态计算图构建、链式法则反向求导与梯度累加的完整机制。
---

训练神经网络的目标，是找到一组参数让模型预测尽可能准确。上一讲在[激活函数有什么用？Sigmoid、Tanh、ReLU 到底怎么选？](/fundamentals/dl-basics/activation-functions.md)中，我们了解了非线性变换对特征空间的扭曲机制。但在实际代码中，包含成千上万个参数的复合函数，框架究竟是如何准确、快速地算出每个参数应该怎么调整的？

神经网络单次迭代训练的核心，由两个计算阶段构成：
1. **前向传播（Forward Propagation）**：数据从输入层逐层计算，得出预测结果并算出标量损失。在此过程中，框架会在内存中构建计算图，并暂存求导所需的中间结果；
2. **反向传播（Backward Propagation）**：从损失值出发，依据微积分链式法则逆向遍历计算图，计算损失关于每个可学习参数的梯度。

需要特别强调一个工程分界：**反向传播只负责计算梯度，并不直接修改参数**。参数具体怎么根据梯度进行更新，是由后续的优化器（如 SGD、Adam）根据更新规则独立执行的。

![前向计算生成损失与反向微分回传梯度的整体流程概览](/images/content-rebuild/a9f81661a312331f/cover.png)

---

## 贯穿算例：一个极简神经元的正向与反向计算

为了避免抽象概念，我们先通过一个具体的数值算例，完整走一遍前向和反向的过程。

假设有一个单层神经元：
- **输入特征**：$x = [1.0, 2.0]$
- **可学习参数**：权重 $w = [2.0, -1.0]^T$，偏置 $b = 0.5$
- **目标标签**：$y = 1.0$
- **激活函数**：Sigmoid，公式为 $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **损失函数**：均方误差（MSE），公式为 $\mathcal{L} = (a - y)^2$

### 1. 前向传播步骤（算数值）

第一步，计算线性加权和 $z$：
$$z = w_1 x_1 + w_2 x_2 + b = (2.0 \times 1.0) + ((-1.0) \times 2.0) + 0.5 = 0.5$$

第二步，通过 Sigmoid 计算激活输出 $a$：
$$a = \sigma(0.5) = \frac{1}{1 + e^{-0.5}} \approx 0.622459$$

第三步，计算与真实标签的误差损失 $\mathcal{L}$：
$$\mathcal{L} = (0.622459 - 1.0)^2 = (-0.377541)^2 \approx 0.142537$$

到这里，前向传播完成，得到一个标量损失值 $0.142537$。

### 2. 反向传播步骤（算导数）

反向传播要回答的问题是：如果微调 $w$ 和 $b$，损失 $\mathcal{L}$ 会怎么变？我们需要求出 $\frac{\partial \mathcal{L}}{\partial w_1}$、$\frac{\partial \mathcal{L}}{\partial w_2}$ 和 $\frac{\partial \mathcal{L}}{\partial b}$。

根据多元微积分的链式法则，我们从损失节点逆向一步步分解：

1. **损失对激活输出的导数**：
   $$\frac{\partial \mathcal{L}}{\partial a} = 2(a - y) = 2 \times (0.622459 - 1.0) = -0.755081$$

2. **激活输出对线性加权和的导数**（Sigmoid 的导数性质 $\sigma'(z) = \sigma(z)(1-\sigma(z))$）：
   $$\frac{\partial a}{\partial z} = a(1 - a) = 0.622459 \times (1 - 0.622459) \approx 0.235004$$

3. **损失对线性加权和的导数**（链式相乘）：
   $$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} = -0.755081 \times 0.235004 \approx -0.177447$$

4. **损失对权重和偏置的最终梯度**：
   - 对于 $w_1$：$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z} \cdot x_1 = -0.177447 \times 1.0 = -0.177447$
   - 对于 $w_2$：$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial z} \cdot x_2 = -0.177447 \times 2.0 = -0.354894$
   - 对于 $b$：$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial b} = \frac{\partial \mathcal{L}}{\partial z} \cdot 1.0 = -0.177447$

反向传播结束，每个参数都拿到了对应的偏导数数值。整个网络后续的参数优化，正是基于这些数值展开的。

---

## 前向传播与显存开销：为什么要暂存中间激活值？

在前向计算过程中，除了产出最终的标量损失，还有一个关键动作：**框架必须在显存中缓存部分前向中间变量**。

回顾前面的推导：
- 求 $\frac{\partial a}{\partial z}$ 时，需要用到前向算出的 $a$（即 $0.622459$）；
- 求 $\frac{\partial \mathcal{L}}{\partial w}$ 时，需要用到输入 $x$。

![前向传播计算管线与中间激活值在内存中的缓存机制](/images/content-rebuild/a9f81661a312331f/fig-1.png)

如果在前向计算完成后就把中间值直接丢弃，反向传播时就不得不重新执行一遍前向计算来求导。因此，深度学习框架会默认在前向过程中把求导所需的激活值保存在显存（VRAM）中。这也是为什么训练大模型时显存占用远高于纯推理的原因——训练显存中很大一部分是这些为反向传播预留的“激活缓存”（Activation Memory）。

---

## 动态计算图（DAG）：Tensor 与 Function 对象的运行时链接

在 PyTorch 中，计算图不是提前静态编译好的，而是在前向传播代码运行的同时动态构建的（Define-by-Run）。

计算图在数学上表现为**有向无环图（Directed Acyclic Graph, DAG）**，由两种核心元素组成：
- **Tensor（张量）**：数据的载体，可以是输入数据、模型参数或中间结果；
- **Function（算子节点）**：计算操作的载体（如加法 `Add`、乘法 `Mul`、矩阵乘法 `MatMul`）。

我们来看代码中的实际表现：

```python
import torch

# 用户输入数据，通常不需要求导
x = torch.tensor([2.0, 3.0], requires_grad=False)
# 模型参数，需要求导
w = torch.tensor([[1.0], [-1.0]], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

# 前向运算，动态构建计算图
z = torch.matmul(x, w) + b
loss = (z - 5.0) ** 2

print(f"z 的反向函数指针: {z.grad_fn}")
print(f"loss 的反向函数指针: {loss.grad_fn}")
```

输出：
```text
z 的反向函数指针: <AddBackward0 object at 0x7fa2c418>
loss 的反向函数指针: <PowBackward0 object at 0x7fa2c420>
```

当张量设置了 `requires_grad=True` 并参与运算时，运算结果张量会自动持有一个 `grad_fn` 属性。该属性指向生成它的 `Function` 对象。

每个 `Function` 内部都实现了两个关键逻辑：
1. `forward()`：执行具体的前向数学运算；
2. `backward()`：接收下游传回的梯度，结合前向缓存的张量，计算当前算子对各个输入的偏导数并向上回传。

![PyTorch 动态计算图 DAG 中 Tensor 与 Function 对象的连接拓扑](/images/content-rebuild/a9f81661a312331f/fig-2.png)

在计算图中：
- 由用户直接创建、不是通过算子计算得出的张量称为**叶子节点（Leaf Node）**，例如模型权重 `w`、偏置 `b`、输入 `x`。叶子节点的 `grad_fn` 为 `None`；
- 中间运算产生的张量（如 `z`、`loss`）为**非叶子节点**，其 `grad_fn` 记录了回溯路径。

<ReadMoreLock />

---

## 向量-雅可比积（VJP）：为什么工程上不求完整雅可比矩阵？

在很多机器学习理论介绍中，多维映射的导数是用雅可比矩阵（Jacobian Matrix）表示的。如果一个向量映射函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 将输入 $x$ 映射为输出 $y$，其雅可比矩阵 $J$ 的维度是 $m \times n$：

$$J = \begin{bmatrix} 
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ 
\vdots & \ddots & \vdots \\ 
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} 
\end{bmatrix}$$

如果上游回传的标量损失关于 $y$ 的梯度向量为 $v = \nabla_y \mathcal{L} \in \mathbb{R}^{1 \times m}$，那么损失关于 $x$ 的梯度在数学上就是向量与矩阵的乘积：
$$\nabla_x \mathcal{L} = v \cdot J$$

但在实际工程中，**深度学习框架绝不会先完整算出 $m \times n$ 的雅可比矩阵，再去乘向量 $v$**。

原因很简单：在大模型中，中间隐藏层的维度 $m$ 和 $n$ 动辄几万甚至几十万。如果显式物化一个几万乘几万的浮点矩阵，瞬间就会吃掉数十 GB 显存并拖垮计算。

反向传播的实际实现方式是**向量-雅可比积（Vector-Jacobian Product, VJP）**：直接根据算子的数学求导解析式，把上游传过来的向量 $v$ 直接变换为维度与 $x$ 一致的梯度，全过程无需存储任何高维矩阵。

以最常见的全连接矩阵乘法 $Y = XW$ 为例（假设 $X \in \mathbb{R}^{B \times d_{in}}$，$W \in \mathbb{R}^{d_{in} \times d_{out}}$）：
- 当上游回传梯度 $\frac{\partial \mathcal{L}}{\partial Y}$（尺寸与 $Y$ 相同）时，框架算子直接执行两个低开销矩阵乘法：
  $$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^T, \quad \frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y}$$

输入与输出维度完全匹配，内存使用极度紧凑。

![显式实例化全雅可比矩阵与反向传播向量-雅可比积（VJP）的计算路径对比](/images/content-rebuild/a9f81661a312331f/fig-3.png)

---

## backward() 梯度回传与叶子节点累加规则

当在代码中调用 `loss.backward()` 时，Autograd 引擎正式启动反向传播：
1. **注入初始梯度**：对于标量损失，引擎隐式赋予种子导数 $1.0$（即 $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$）；
2. **逆拓扑遍历**：从根节点的 `grad_fn` 开始，逆着 DAG 的连线访问每个 `Function` 节点；
3. **计算并回传**：每个节点调用自身的反向方法，将求得的梯度向上游传递，直到抵达叶子节点。

### 为什么叶子节点的梯度是“+=”累加而不是覆盖？

当梯度最终传递到参数张量（叶子节点）时，PyTorch 会把算出的梯度写入张量的 `.grad` 属性中。底层的核心行为是：**累加（Accumulate），即 `param.grad += new_grad`，而不是直接赋值覆盖**。

![autograd 执行 backward() 时的逆拓扑回传与叶子节点 += 累加机制](/images/content-rebuild/a9f81661a312331f/fig-4.png)

这种设计主要服务于两个常见场景：
1. **计算图中的分支汇合（多路径求导）**：如果一个参数在网络前向中被多个分支共用（例如残差连接、多任务输出或权重共享），根据多变量微积分求导法则，该参数的总梯度是各个分支偏导数的代数和。通过 `+=` 累加，框架能够自然处理多分支汇合的情况；
2. **梯度累积（Gradient Accumulation）**：当单卡显存不足以容纳大 Batch 时，开发者可以将大 Batch 拆为多个小 Mini-batch。每次前向计算后调用 `loss.backward()`，参数的梯度会在 `.grad` 中自动累加。连续累加若干步后，再调用优化器更新一次参数，即可在受限显存下等效实现大 Batch 的训练效果。

这也解释了为什么 PyTorch 的标准训练循环中，每次迭代必须显式写出 `optimizer.zero_grad()`：**如果不手动清零，上一轮迭代计算出的梯度就会直接叠加到当前这一轮，导致梯度数值翻倍失真**。

---

## 叶子节点的生命周期与显存释放

为了尽可能节约训练显存，PyTorch 制定了严格的生命周期与回收策略：

### 1. 叶子节点与非叶子节点的梯度存储
在默认情况下，执行 `loss.backward()` 后，**只有 `requires_grad=True` 的叶子节点（即模型权重与偏置）会保留 `.grad` 属性**。所有中间过程中的非叶子张量（如前向中的激活值和求和结果），其反向梯度在传递完毕后会被底层 C++ 引擎立刻释放，访问其中间张量的 `.grad` 会直接返回 `None`。

如果为了调试或特定算法需要查看某个中间张量的梯度，必须在前向之后、反向之前显式声明：
```python
z.retain_grad()  # 告知 autograd 引擎保留非叶子节点 z 的梯度
loss.backward()
print(z.grad)    # 此时可以正常打印出梯度张量
```

### 2. 动态计算图的单次销毁机制
在调用一次 `loss.backward()` 完成求导后，PyTorch 默认会立即销毁当前计算图以释放显存。如果在没有重新执行前向传播的情况下连续调用两次 `loss.backward()`，会触发运行时错误：
`RuntimeError: Trying to backward through the graph a second time`。

如果业务逻辑确实需要对同一个前向图进行多次反向传播（例如某些二阶求导算法），必须显式传入参数保留计算图：`loss.backward(retain_graph=True)`。

![叶子节点与非叶子节点在 backward() 前后的显存驻留与释放生命周期](/images/content-rebuild/a9f81661a312331f/fig-5.png)

---

## 推理阶段优化：torch.no_grad() 与 detach()

在模型验证（Evaluation）或线上推理（Inference）时，我们只需要前向计算得到预测结果，根本不需要计算梯度。如果不加控制，框架仍会默认构建计算图并缓存中间激活值，带来巨大的无谓开销。

### 1. torch.no_grad() 上下文管理器
使用 `with torch.no_grad():` 可以全局暂时关闭梯度跟踪。在该代码块内部：
- 新生成的张量其 `requires_grad` 会被强制设为 `False`；
- 框架不会构建任何 `Function` 节点或计算图连线；
- 前向计算算子不再缓存任何求导所需的激活值。

这能直接砍掉中间激活缓存所占用的显存，同时避免了图结构的维护开销，提升前向吞吐速度。

```python
model.eval()
with torch.no_grad():
    predictions = model(test_input)
```

### 2. tensor.detach() 算子
`detach()` 是针对单个张量的截断操作。它会返回一个与原张量**共享同一块底层内存**的新张量，但新张量的 `grad_fn` 为 `None`，且 `requires_grad=False`。

```python
z = x @ w
z_detached = z.detach()  # z_detached 退出计算图，反向传播无法穿透此节点
```

两者的分工很明确：
- **`torch.no_grad()`**：作用于一段完整代码，适合推理、验证或不用反向传播的评测循环；
- **`detach()`**：作用于单个张量，适合在复杂网络中局部阻断梯度传递，比如生成对抗网络（GAN）交替训练时固定其中一个子网络的输入。

![正常前向传播与包裹 torch.no_grad() 推理模式的显存缓冲区与图构建差异对比](/images/content-rebuild/a9f81661a312331f/fig-6.png)

---

## 完整代码验证：手动推导与 PyTorch 运行结果精确比对

我们用一段可运行的代码，把文章开头手动推导的数值与 PyTorch autograd 的真实计算结果进行严谨比对：

```python
import torch

def run_autograd_verification():
    # 1. 准备常量输入与目标标签（叶子节点，无需梯度）
    x = torch.tensor([1.0, 2.0], requires_grad=False)
    y = torch.tensor(1.0, requires_grad=False)
    
    # 2. 准备模型参数（叶子节点，需要梯度）
    w = torch.tensor([2.0, -1.0], requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)
    
    # 3. 前向传播：计算 z -> a -> loss
    z = torch.dot(x, w) + b
    a = torch.sigmoid(z)
    loss = (a - y) ** 2
    
    # 4. 手动推导的理论值
    expected_w1 = -0.177447
    expected_w2 = -0.354894
    expected_b  = -0.177447
    
    # 5. 反向传播（保留计算图以方便测试二次 backward）
    loss.backward(retain_graph=True)
    
    print("=== 第一次 backward 结果比对 ===")
    print(f"w 梯度理论值: [{expected_w1:.6f}, {expected_w2:.6f}] | autograd 实际值: [{w.grad[0].item():.6f}, {w.grad[1].item():.6f}]")
    print(f"b 梯度理论值: {expected_b:.6f} | autograd 实际值: {b.grad.item():.6f}")
    
    # 6. 测试未清零时的累加机制
    loss.backward()
    print("\n=== 未清零时二次 backward（梯度应翻倍）===")
    print(f"w 累加理论值: [{expected_w1*2:.6f}, {expected_w2*2:.6f}] | autograd 实际值: [{w.grad[0].item():.6f}, {w.grad[1].item():.6f}]")
    
    # 7. 手动清零验证
    w.grad.zero_()
    b.grad.zero_()
    print("\n=== 手动 zero_() 清零后 ===")
    print(f"w.grad: {w.grad.tolist()}, b.grad: {b.grad.item()}")

if __name__ == "__main__":
    run_autograd_verification()
```

运行该脚本，控制台输出如下：
```text
=== 第一次 backward 结果比对 ===
w 梯度理论值: [-0.177447, -0.354894] | autograd 实际值: [-0.177447, -0.354894]
b 梯度理论值: -0.177447 | autograd 实际值: -0.177447

=== 未清零时二次 backward（梯度应翻倍）===
w 累加理论值: [-0.354894, -0.709788] | autograd 实际值: [-0.354894, -0.709788]

=== 手动 zero_() 清零后 ===
w.grad: [0.0, 0.0], b.grad: 0.0
```

测试结果完全印证了三个核心结论：
1. Autograd 反向遍历与微积分链式法则的手动推导结果在数值上完全一致；
2. 连续调用 `backward()` 时，底层确实是在已有 `.grad` 上执行 `+=` 累加；
3. 在进入下一次迭代前，必须执行清零操作，才能确保梯度数值的准确性。

---

## 总结与延伸思考

前向传播计算损失并保存必要的中间结果，反向传播以向量-雅可比积形式自根向叶逆向分发梯度。二者配合，完成了神经网络训练中最繁重的数学求导任务。

然而，当网络从单层拓展到几十甚至上百层时，链式求导连乘过程中，权重大小与激活函数导数的微小偏差都会被指数级放大或缩小——这便是深层神经网络著名的失效问题。关于这一机制，我们在下一讲[梯度消失和梯度爆炸是什么？为什么深层网络不好训练？](/fundamentals/dl-basics/gradient-vanishing-exploding.md)中深入展开；而反向传播算出的梯度究竟如何引导参数更新，我们将在后续的[优化器是什么？SGD、Momentum、Adam 有什么区别？](/fundamentals/dl-basics/optimizers.md)中详细拆解。
