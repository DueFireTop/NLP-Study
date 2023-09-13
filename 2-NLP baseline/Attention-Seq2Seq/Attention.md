# 1. Seq2Seq

## 1.1 Seq2Seq的目标

1. 一个文本做输入，一个文本做输出
   - 机器翻译
   - 文本摘要
2. 两个文本做输入，做分类，结构化预测的输出
   - 机器阅读理解问题



**End to End**：把输入和输出之间建立一个连接关系（模型），只管把数据扔进去，希望模型建立一个输入到输出的映射。

<img src="https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/image-20230901102254315.png" alt="image-20230901102254315" style="zoom:67%;" />

<center><p style="color:blue">Figure 1.1: End to End</p></center>

## 1.2 Seq2Seq的结构

![Seq2SeqAgriculture.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Seq2SeqAgriculture.drawio.png)

<center><p style="color:blue">Figure 1.2: Seq2Seq Agriculture</p></center>

> SANs：Self Attention Networks

- encoder：用于学习输入信息的神经元表达
- decoder：用于构建输出序列



## 1.3 Seq2Seq的能力

Seq2Seq 不仅仅用于机器翻译，其他 NLP 任务也可以借助于 Seq2Seq 的力量。

- 机器翻译：原文本转为目标文本
- 文本总结：长文本转为短文本
- 对话回复生成：上一个对话转为下一个对话
- 原本是判别结构的任务也可以转化为 Seq2Seq 的方式
  - 句法的逆袭可以转化为句法序列树
  - 情感分析可以转化为 类别转化为句子 的表达



# 2. Seq2Seq with LSTM

![Seq2Seq with LSTM.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Seq2Seq%20with%20LSTM.drawio.png)

<center><p style="color:blue">Figure 2.1: Seq2Seq with LSTM</p></center>

- input sentence: $X_{1:n} = x_1, x_2, ..., x_n$ 
- output sentence: $Y_{1:m} = y_1, y_2, ..., y_m$ 

> training 阶段，使用正确的结果 $y_1, y_2,...,y_m$ 来进行 decoder 的训练，更新LSTM的权重（生成模型）
>
> inference 阶段，使用预测的结果 $y_1, y_2,...,y_m$ 来作为输入（模型开始预测）
>
> $<s>, </s>$ 提示模型，表示序列开始生成和序列生成结束

Training 阶段采用 Teaching Forcing 强制学习，防止反向传播（BPTT）时的 loss 太大。

![Seq2Seq with BiLSTM.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Seq2Seq%20with%20BiLSTM.drawio.png)

<center><p style="color:blue">Figure 2.2: Seq2Seq with BiLSTM</p></center>

> - **encoder**：
>   - $\tilde{h}^{enc}_j = LSTM(\tilde{h}^{enc}_{j+1}, emb(x_j))$ 从右往左
>   - $\vec{h}^{enc}_j = LSTM(\vec{h}^{enc}_{j-1}, emb(x_j))$ 从左往右
>   - $h^{enc} = [\vec{h}^{enc}_n,\tilde{h}^{enc}_1]$ 
> - **decoder**：
>   - $h^{dec}_i = LSTM(h^{dec}_{i-1}, emb(y_{i-1}))$ 
>   - $o_i = W \cdot h^{dec}_i + b$ 
>   - $p_i = softmax(o_i), \space p_i \in R^{|V|}$ 

- Encoder的技巧：
  - 在输入文本的开头和结尾分别加上 Start sign 和 End sign
  - 分别使用两个初始化隐状态向量
    - 全0向量
    - 随机向量
    - 基于输入文本的 word embedding 加权平均向量
- Decoder的技巧：
  - 输出的过程中，可以使用 Greedy Search，也可以使用 BeamSearch 的 Top-K 和 Top-P。
  - 输出结束表示 End sign `</s>` 
  - 为了确保不会无限制输出，可以设定一个 Max Length
  - 模型遇到字典中没有出现过的词，在 Encoder 的时候可以使用 OOV 进行表述，也可以使用 UNK 代表所有未登录的 OOV 在输出字典中能到表达。训练时将低频词汇主动替换成 UNK，训练得到一个 UNK 的 word embedding，测试时就可以使用 UNK 代替这个词。实际使用中，可以强制模型不输出 UNK，也可以允许输出，但是增加 post processing 过程，将其替换成相对合理的词汇

$$
L = -\sum_{k=1}^N \sum_{j=1}^{m_k} \log{(P(y_i^k|X^k, Y_{1:i-1}^k))}
$$



# 3. Seq2Seq with Attention

在长序列文本输入/输出的场景下，Seq2Seq 天生的记忆力缺陷无法忽视，会丢失很多先前信息，即 Seq2Seq 在长序列下表现很差。

Attention：在当前时刻下，隐状态的结果一定与当前时刻的输入信息有最大相关性。

![Seq2Seq with Attention.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Seq2Seq%20with%20Attention.drawio.png)

<center><p style="color:blue">Figure 3.1: Seq2Seq with Attention</p></center>

## 3.1 Alignment Weight

![Alignment Weight in Attention.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Alignment%20Weight%20in%20Attention.drawio.png)

<center><p style="color:blue">Figure 3.2: Alignment Weight in Attention</p></center>

> 对于decoder某一时刻的隐状态，与encoder中所有隐状态相比，哪个有最大的相关性
>
> - $\alpha_i$：权重参数
> - $h_i$：i 时刻 encoder 的隐状态
> - $s_t$：t 时刻 decoder 的隐状态
> - $[\alpha_1,\alpha_2,...,\alpha_n]$：decoder中 t 时刻的隐状态与 encoder 中所有隐状态的相关性
> - $\alpha_i$ 也可以表示为 $\alpha_i = align(h_i, s_t)$ 



## 3.2 Context Vector

![Context Vector in Attention.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/Context%20Vector%20in%20Attention.drawio.png)

<center><p style="color:blue">Figure 3.3: Context Vector in Attention</p></center>

> - $c_t = \alpha_1 \cdot h_1 + \alpha_2 \cdot h_2 + ... + \alpha_n \cdot h_n$：decoder 中在 t 时刻下的 context vector
> - $s_t = tanh(A' \cdot \begin{bmatrix}
>   y_i \\
>   s_{i-1} \\
>   c_{i-1}
>   \end{bmatrix} + b)$：decoder 中在 t 时刻下的隐状态
> - $s_0$：encoder 的最终输出状态（包含 $h_1,h_2,...,h_n$ 的信息）



# 4. LSTM with Attention

![LSTM with Attention.drawio](https://bucket-zly.oss-cn-beijing.aliyuncs.com/img/LSTM%20with%20Attention.drawio.png)

<center><p style="color:blue">Figure 4.1: LSTM with Attention</p></center>

除了依赖于 encoder 的输出状态外，从 encoder 的每一时刻的隐藏状态去寻找特征

- $c_i = \sum_{j=1}^n \alpha_{ij}\cdot h_j^{enc}$ 
- $\alpha_{ij} = \frac{\exp{e_{ij}}}{\sum_{k=1}^n \exp{e_{ik}}}$ 
- $e_{ij} = V_{\alpha}^T \tanh{(W_{\alpha}h_{i-1}^{dec} + U_{\alpha}h_j^{enc})}$ 

Attention 大范围增强了模型的准确率，model 不会对 input 输入文本信息遗忘，decoder 部分将会知道需要关注的原文本，增强了翻译的准确度。但是同样也增加了很多参数需要训练，耗时增加极多。



# 5. Copy From Source

> I love ChatGPT ---> 我爱 ChatGPT
>
> Hello, my name is Tom. ---> Nice to meet you, Tom

翻译场景下无需翻译特有名词，对话场景下有些内容也无需生成，于是有了 Copy From Source。

除了考虑文本的生成概率，还要考虑文本的复制概率。
$$
Target Vocab = OriginalTargetVocab + SourceVocab
$$

- Pointer Network

  将注意力机制中的权重作为概率分布的参考。

  使用注意力的权重得分 $\alpha_{ij}$ 作为指向原词的概率分布：$score_c(y_i = x_k) = \alpha_{ik},\space k\in[1, ..., n]$ 。

  对于相同单词，计算其加和结果：$score_c(y_i = u_i) = \sum_{k:x_k = u_k}a_{ik},\space u_i\in U,j\in[1, ..., |U|]$ 

  最终的计算得分使用线性插值，分别处理生成得分和复制得分：$Score(y_i) = \lambda score_g(y_i) + (1-\lambda)score_c(y_i)$ 

- Copying Network

  使用独立的神经网络计算单词复制得分



