![image-20230605165529237](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306051655445.png)

# 1. NN

## 1.1 人工神经元

人工神经元：人类神经元中抽象出来的数学模型

![image-20230609090640994](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306090906421.png)

- Threshold：阈值函数，当超过T值，这个函数就会激活，这就对应了神经元的两种状态（激活、抑制）



**人工神经网络：**大量神经元以某种连接方式构成的机器学习模型

![image-20230609091306356](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306090913400.png)



**第一个神经网络**：1958年，计算科学家Rosentblatt提出Perceptron（感知机）

![image-20230609091459113](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306090914144.png)

- $o=\sigma(<w,x>+b)$ 
- $\sigma = \begin{cases}1& \text{ if } x> 0 \\0& \text{ if } x= otherwise\end{cases}$ 
- 感知机致命缺点：Minsky在1969年证明Perceptron无法解决异或问题【?】

感知机是线性的模型，其不能表达复杂的函数，不能出来线性不可分的问题，其连异或问题(XOR）都无法解决，因为异或问题是线性不可分的，怎样解决这个问题呢，通常可以：

1. 用更多的感知机去进行学习，这也就是人工神经网络的由来。
2. 用非线性模型，核技巧，如SVM进行处理。



## 1.2 多层感知机

**多层感知机（Multi Layer Perceptron, MLP）**：单层神经网络基础上引入一个或多个隐藏层，使神经网络有多个网络层，因而的命多层感知机。

- 只会计算有权重参数的层

![image-20230609093215588](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306090932642.png)![image-20230609093225352](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306090932392.png)

## 1.3 激活函数

如果没有**激活函数**，网络会退化为**单层网络**

- $H = XW_h + b_h$ 
- $O = HW_o + b_o = (XW_h + b_h)W_o + b_o = XW_hW_o +b_hW_o + b_o = XW + b$  

隐藏层加入**激活函数**，**可以避免网络退化**。

- $h = \sigma(W_1X + b_1)$ 
- $o = W_2^{\top} h + b_2$ 

> **激活函数**：
>
> 1. 让多层感知机成为真正的多层，否则等价于一层
> 2. 引入非线性，使网络可以逼近任意非线性函数（万能逼近定理，universal approximator）
>
> **激活函数需要具备以下性质：**
>
> 1. 连续并可导（允许少数点上不可导），便于利用数值优化的方法来学习网络参数
> 2. 激活函数及其导函数要尽可能的简单，有利于提高网络计算效率
> 3. 激活函数的导函数的值域要在合适区间内，不能太大也不能太小，否则会影响训练的效率和稳定性。



**常见的激活函数：Sigmoid（S型），Tanh（双曲正切），ReLu（修正线性单元）**

![image-20230609103959348](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306091040771.png)



## 1.4 反向传播

**前向传播**：输入层数据开始从前向后，数据逐步传递至输出层

**反向传播**：**损失函数**开始从后向前，**梯度**逐步传递至第一层

> **反向传播的作用：**用于权重的更新，使网络模型输出更接近标签
>
> **损失函数：**衡量模型输出于真是标签的差异，$Loss = f(\hat{y},y)$ 

**反向传播原理**：微积分中的链式求导法则

- $y=f(u), u=g(x)$
- $\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \frac{\partial u}{\partial x} $



![image-20230609105444099](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306091054148.png)

- 方框表示数据，圆圈表示运算

- 计算符定义：prod(x, y) 表示 x 与 y 根据形状做必要的变换，然后相乘
  $$
  \frac{\partial L}{\partial W^{(1)}} = prod(\frac{\partial L}{\partial o}, \frac{\partial o}{\partial W^{(1)}}  ) = \frac{\partial L}{\partial o}h^{\top}
  $$

  $$
  \frac{\partial L}{\partial h} = prod(\frac{\partial L}{\partial o}, \frac{\partial o}{\partial h}  ) =W^{(1)^{\top}} \frac{\partial L}{\partial o}
  $$

  $$
  \frac{\partial L}{\partial z} = prod(\frac{\partial L}{\partial o}, \frac{\partial o}{\partial h}, \frac{\partial h}{\partial z}  ) =\frac{\partial L}{\partial h} \odot \phi^{'}(z)
  $$

  $$
  \frac{\partial L}{\partial  W^{(1)}} = prod(\frac{\partial L}{\partial o}, \frac{\partial o}{\partial h}, \frac{\partial h}{\partial z}, \frac{\partial z}{\partial  W^{(1)}}  ) =\frac{\partial L}{\partial z} \cdot X^{\top}
  $$

  



# 3. RNN

## 3.1 序列数据

序列数据是常见的数据类型，前后数据通常具有 **`关联性`**

<img src="https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306051717784.png" alt="image-20230605171739722" style="zoom:80%;" />

## 3.2 语言模型

语言模型是自然语言处理重要技术

NLP中常把文本看为离散时间序列，一段长度为T的文本的词依次为$w_1,w_2,...,w_T$，其中$w_t(1 \le t \le T)$ 是 **`时间步（Time Step）`** t 的输出或标签

语言模型将会计算该序列概率 $P(w_1,w_2,...,w_T)$ 

> **例句：我在听课**
>
> - T = 4
> - $P(我在听课) = P(我)\cdot P(在|我)\cdot P(听|我在)\cdot P(课|我在听)$

统计**`语料库（Corpus）`**中的词频，得到以上概率，最终得到 $P(我在听课)$ 

**缺点：**时间步 t 的词需要考虑 t-1 步的词，其计算量随 t 呈指数增长。

## 3.3 RNN

RNN是针对序列数据而生的神经网络结构，核心在于循环使用网络层参数，避免时间步增大带来的参数激增，并引入 **`隐藏状态（Hidden State）`** 用于记录历史信息，有效的处理数据的前后关联性。

![image-20230605191030654](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306051910699.png)

- X：输入
- O：输出
- h：隐藏状态
- U、W、V：权值矩阵，循环使用
- $h_t = x_t \cdot U + h_{t-1}\cdot V$ 
- $O_t = h_t \cdot W$ 

---

**`隐藏状态（Hidden State）`** 用于记录历史信息，有效处理数据的前后关联性。

激活函数采用 Tanh，将输出值域限制在 (-1, 1)，防止数值呈指数级变化。

![image-20230605192012167](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306051920211.png)

- $H_t$：表示当前步的隐藏状态
- $H_t = \phi(X_tW_{xh}+H_{t-1}W_{hh}+b_h)$ ，$\phi$ 是激活函数
- $W_{xh},X_{hh},X_{hq}$：权重矩阵
- $b_h,b_q$：偏置项

---

RNN构建语言模型，实现文本生成。假设文本序列：想，要，有，直，升，机

![image-20230605192807501](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306051928539.png)

**RNN特性**：

1. 循环神经网络的 **`隐藏状态`** 可以捕捉截至当前时间步的序列的历史信息
2. 循环神经网络模型参数的数量不随时间步的增加而增加

---

**RNN的`通过（穿越）时间反向传播`（Back-propagation through time）** 

![image-20230605193721571](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306051937608.png)
$$
\frac{\partial L}{\partial W_{qh}} = \sum_{t=1}^T prod(\frac{\partial L}{\partial O_t},\frac{\partial O_t}{\partial W_{qh}} ) =\sum_{t=1}^T \frac{\partial L}{\partial O_t}h_t^{\top }
$$

$$
\frac{\partial L}{\partial h_T} = prod(\frac{\partial L}{\partial O_T},\frac{\partial O_T}{\partial h_T}  ) = W_{qh}^{\top}\frac{\partial L}{\partial O_T}
$$

$$
\frac{\partial L}{\partial h_t} = \sum_{i=1}^T(W_{hh}^{\top})^{T-i}W_{qh}^{\top}\frac{\partial L}{\partial O_{T+t-i}}  
$$

$$
\frac{\partial L}{\partial W_{hz}} =\sum_{t=1}^Tprod(\frac{\partial L}{\partial h_t},\frac{\partial h_t}{\partial W_{hz}}  ) = \sum_{t=1}^T\frac{\partial L}{\partial h_t}x_t^{\top} 
$$

$$
\frac{\partial L}{\partial W_{hh}} =\sum_{t=1}^Tprod(\frac{\partial L}{\partial h_t},\frac{\partial h_t}{\partial W_{hh}}  ) = \sum_{t=1}^T\frac{\partial L}{\partial h_t}h_{t-1}^{\top} 
$$

梯度随时间 t 呈指数变化，易引发 **`梯度消失`** 或 **`梯度爆炸`** 

![image-20230605200604444](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306052006481.png)

## 3.4 GRU

门控循环单元（Gated Recurrent Unit）

> **引入门的 循环网络**
>
> 缓解RNN梯度小时带来的问题，引入门的概念，来控制信息流动，使模型更好的记住长远时期的信息，并缓解梯度消失

**`重制门`**：哪些信息需要遗忘

**`更新门`**：哪些信息需要注意

**激活函数**为：Sigmoid，值域为（0, 1），0 表示遗忘，1 表示保留

![](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306052009723.png)

$$
R_t = \sigma(X_tW_{xr}+H_{t-1}W_{hr}+b_r)
$$

$$
Z_t = \sigma(X_tW_{xz}+H_{t-1}W_{hz}+b_z)
$$

---

![image-20230606095035096](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306060950153.png)

**`候选隐藏状态`**：输出 和 上一时间步隐藏状态 共同计算得到 候选隐藏状态，用于隐藏状态计算。通过重置门，对上一时间步隐藏状态进行 **选择性遗忘** ，对历史信息进行更好的选择。

- GRU：$\tilde{H_t}=\tanh (X_t W_{xh} + (R_t \odot H_{t-1})W_{hh} + b_h) $ 
- RNN：$H_t = \phi(X_t W_{xh} + H_{t-1}W_{hh} + b_h)$ 

上述是 GRU 和 RNN 的对比，GRU 多了个重制门的概念，对上一时间步隐藏状态 $H_{t-1}$ 进行选择性遗忘，其他步骤都与 RNN 类似。都是【输入 $X_t$ 与权重矩阵 $W_{xh}$ 相乘】加上【上一步隐藏状态 $H_{t-1}$ 与权重矩阵 $W_{hh}$ 相乘】，再加一个偏置项 $b_h$ 。

---

![image-20230606101012182](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306061010221.png)

**隐藏状态**：隐藏状态由 **候选隐藏状态** 及 **上一时间步隐藏状态** 组合而来。

- 隐藏状态： $H_t = Z_t \odot H_{t-1} + (1-Z_t) \odot \tilde{H_t}$ 

---

**GRU——引入门控机制的循环网络**

GRU特点：

1. 门机制采用Sigmoid激活函数，使门的值域为（０，１），０表示遗忘，１表示保留
2. 若更新门自第一个时间步到t-１时间过程中，一直保持为１，**信息可有效传递到当前时间步**。



## 3.5 LSTM

LSTM（Long Short-Term Memory）：长短期记忆网络，引入 **3个门** 和 **记忆细胞**，控制信息传递

- 遗忘门：哪些信息需要遗忘
- 输入门：哪些信息需要流入当前记忆细胞
- 输出门：哪些记忆信息流入隐藏状态
- 记忆细胞：特殊的隐藏状态，记忆历史信息

![image-20230606103122689](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306061031729.png)

$$
I_t = \sigma(X_t W_{xi} +H_{t-1}W_{hi} + b_i)
$$

$$
F_t = \sigma(X_t W_{xf} + H_{t-1}W_{hf} + b_f)
$$

$$
O_t = \sigma(X_tW_{xo} + H_{t-1}W_{ho} +b_o)
$$

---

![image-20230606103714915](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306061037952.png)

**`候选记忆细胞`**：可以理解为 **特殊隐藏状态**，存储历史时刻信息。

- $\tilde{C_t}=\tanh{(X_tW_{xc} +H_{t-1}W_{hc} +b_c)}$ 

![image-20230606104756898](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Typora/202306061047936.png)

记忆细胞由 **候选记忆细胞** 及 **上一时间步记忆细胞** 组合而来

- $C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C_t}$ 

由输出们控制记忆细胞信息，流入 **隐藏状态**

- $H_t = O_t \odot \tanh{(C_t)}$ 







# 0. right now

## 0.1 cuda如何使用共享GPU内存

共享GPU内存是在CUDA程序中提高性能的一个有效方法。共享内存是指在同一个块中的多个线程之间共享的小型高速缓存。CUDA共享内存可以通过以下步骤使用：
1. 在kernel函数中声明共享内存：使用__shared__关键字声明一个共享内存数组。
2. 在内核函数中初始化共享内存：在kernel函数的开始处，使用共享内存的线程（通常是线程号为0的线程）初始化共享内存中的数组元素。
3. 在内核函数中使用共享内存：在内核函数中，使用共享内存中的值进行计算。共享内存的速度很快，因此使用共享内存可以提高计算速度。
4. 使用同步函数：如果有多个线程访问共享内存，则需要使用同步函数保证所有线程在访问时不会发生竞争条件。
下面是一个简单的示例，演示如何在CUDA中使用共享内存：
```py
__global__ void add(int *a, int *b, int *c)
{
    __shared__ int shared_mem[512];
    int t_id = threadIdx.x;
    shared_mem[t_id] = a[t_id] + b[t_id];
    __syncthreads();
    c[t_id] = shared_mem[t_id];
}
```
在这个例子中，我们使用了一个大小为512的共享内存数组shared_mem。线程t_id读取a[t_id]和b[t_id]的值，并将它们存储到共享内存数组shared_mem中的相同位置。然后，我们使用__syncthreads()函数同步所有的线程，以确保线程都已经完成了写入操作。最后，线程再次使用共享内存来读取相应位置的值，将结果存储到c[t_id]中。
请注意，使用共享内存时需要注意内存的使用限制，因为共享内存的大小是有限制的，这个大小通常是每个线程块中的内存总量的一小部分。























































































































