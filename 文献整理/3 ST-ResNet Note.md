# [1701.02543] Predicting Citywide Crowd Flows Using Deep Spatio-Temporal Residue

![](https://cdn.nlark.com/yuque/0/2020/png/1428756/1603610175805-b612a945-0d6f-43e4-81a3-22f1abbb9faa.png)## Introduction 引言

- 预测交通流量的重要性
- 预测目标
   - 流入流（inflow）：给定时间区间内进入一个区域的交通总流量（包含行人、汽车和公共汽车等）
   - 流出流（outflow）：给定时间区间内流出一个区域的交通总流量
- 预测交通流问题→时空预测问题
- 两种主流深度神经网络架构考虑了部分时间或空间属性：
   - 卷积神经网络（CNN）：空间结构
   - 递归神经网络（RNN）：时间依赖关系
- 深度学习在时空预测问题上遇到的挑战（需要考虑的因素）
   - 空间依赖性
      - 近处（nearby）：一个区域的流出流影响邻近区域的流入流；一个区域的流入流也会影响该区域的流出流
      - 远处（distant）：住在其他地方的上班族需要进入该区域工作，因此一个区域的流入流还受到距离较远区域流出流的影响
   - 时间依赖性
      - closeness：同一天邻近时间点的影响
      - period：每天同一时间段时空流间高度相关
      - trend：rush hour的出现与季节温度有关（冬天温度下降，白天变短，人们起的越来越迟）
   - 外部因素影响：天气因素、突发事件、特殊事件等
- 文章的主要贡献（主要工作）：
   - ST-ResNet：基于卷积残差网络（Convolutional-based Residual Network），使得网络能够反映近处/远处两种空间依赖性，同时模型的预测精度不受神经网络的深层结构的影响。
   - 总结了交通流量的时间特性：邻近（closeness）、时段（period）、趋势（trend），并在ST-ResNet中分别使用不同的残差网络描述了这三种特性
   - ST-ResNet动态聚合前述三个残差网络的输出，并为不同区域赋予了不同的权重，并且纳入了外部因素的影响
   - 将构建的网络在北京出租车轨迹及气象数据集，以及纽约自行车轨迹数据集上进行了验证。结果证明文章的方法优于其他的九种方法。
   - 基于该网络搭建了实时监控和预测区域交通流的系统。并且，该系统基于云和GPU，提供了高效灵活的计算环境支持。
- 文章结构：
   - S2：交通流量预测问题的基本概念
   - S3：系统的基本架构
   - S4：搭建的基于DNN的预测模型
   - S5：模型性能的评估
   - S6：相关工作
- 与前人工作区别（创新点）：
   - 实际应用：部署了一个基于云平台的系统（3），能够利用实时数据来预测贵阳市任一区域交通流量
   - 预测跨度更大：能够进行多步预测，能对较长期流量进行预测（4.4）
   - 更全面的试验：证实系统鲁棒性和有效性
      - 与其他方法的对比（5.2）
      - 不同网络结构的对比（5.3）
      - 添加多步预测实验（5.4）
      - 在不同云平台进行测试（5.5）
   - 探索了相关领域研究进展，阐述与别人工作的区别与联系（6）
## Preliminary 预备知识
### 交通流量预测
#### 区域（location）
将城市划分为$$I\times J$$的网格，每个网格代表一个区域（坐标范围）
#### 流入流/流出流
假设$$\mathbb{P}$$为发生在$$t^{th}$$时间区间的轨迹（trajectories）集合，对区域$$(i,j)$$而言，流入流出流被定义为：
$$x^{in,i,j}_t=\sum_{Tr\in\mathbb{P}}|\{k>1|g_{k-1}\notin(i,j)\wedge g_k\in(i,j)\}|
\\
x^{out,i,j}_t=\sum_{Tr\in\mathbb{P}}|\{k>1|g_k\in(i,j)\wedge g_{k+1}\notin(i,j)\}|$$
其中，$$Tr:g_1\rightarrow g_2 \rightarrow \cdots \rightarrow g_{|Tr|}$$是$$\mathbb{P}$$中的一条轨迹，$$g_$$是空间坐标，$$g_k\in(i,j)$$代表该坐标在区域$$(i,j)$$中。$$|·|$$代表集合的势（集合包含元素数）。
因此，在$$t^{th}$$时间区间，$$I\times J$$区域的流入流和流出流可以用张量$$X_t\in\mathbb{R}^{2\times I\times J}$$表示，其中$$(X_t)_{0,i,j}=x_t^{in,i,j}$$，$$(X_t)_{1,i,j}=x_t^{out,i,j}$$。
#### 问题定义
因此，交通流量预测问题就可以定义为：
给定历史观测数据$$\{X_t|t=0,\cdots,n-1\}$$，预测$$X_n$$
### 深度残差学习
深度残差学习下，CNN的深度可以达到100甚至多于1000层。这种学习方法在多种复杂识别任务上都展现了很高水准的性能。
一个残差块可以表示为：
$$X^{(l+1)}=X^l+\mathcal{F}(X^{(l)})$$
$$X^{(l)}$$和$$X^{(l+1)}$$分别为$$l^{th}$$残差块的输入和输出，$$\mathcal{F}$$是需要学习的残差函数（比如：通过两层3×3卷积层来实现残差映射）。残差学习，核心在于通过学习残差函数间接得到输出来避免网络退化导致的效果下降（？同时使用了浅层和深层的输出）。

> 这里要补充DRN的知识储备，然后DRN的介绍里又有BN......
> Deep Residual Network（DRN）参考资料：

> Batch Normalization layer（BN）参考资料：

## System Architecture 系统架构
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1600867351085-bb7eea8b-3b7a-496e-857c-7bb428f42f47.png)
系统框架
由三部分组成：本地GPU服务器、云服务平台和客户端（Web App）

### 云端
爬取天气和实时轨迹数据存入数据库，虚拟机计算流入/流出流并抽取天气特征，存入虚拟机存储器。云端只保存两天的数据，每隔一段时间把历史数据备份到本地

### 本地GPU

- 使用计算模型对大量历史轨迹数据进行计算，得到流入、流出流数据存入本地
- 从气象数据集种提取天气特征存入本地
- 训练模型（本地+下载的云上实时数据），模型训练完毕后上传至云端进行预测，输出预测数据存入Redis
### 用户界面
建立Web App来展现预测的交通流（热力图形式），可以以动画的形式播放交通流变化。
## Deep Spatio-Temporal Residual Networks 深度时空残差网络
### LSTM的不足
LSTM这类RNN网络能够学习较长时间内的时间依赖性，然而为了描述时间的period和trend特征，需要输入很长的时空数据序列，使得训练过程变得复杂。
基于时空领域知识，可以通过**提取关键帧**的方式来进行预测。因此，文章使用了closeness、period和trend来选取关键帧。

### 数据转化

- 轨迹数据：每一时段轨迹数据→ inflow&outflow→ 2-channel tensor → 按时间堆积得到类似视频流数据
- 气象/外部因素数据：提取特征 → 按时间堆积得到时序数据

![](https://cdn.nlark.com/yuque/0/2020/png/1428756/1600946048425-b715d09a-5778-4107-bc54-f818ba761b1e.png#align=left&display=inline&height=288&margin=%5Bobject%20Object%5D&originHeight=610&originWidth=793&size=0&status=done&style=none&width=375)
### 整体建模架构
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1600941067927-8f769307-58cb-497e-a60f-4dc188f8878b.png)
由4个模块组成：

- closeness
- period
- trend
- external

前三者各使用一个DRN来训练，将结果乘以参数矩阵得到混合后的$$X_{Res}$$。再与输入external提取的特征进行训练的全连接网络输出的$$X_{Ext}$$相加，通过tanh（收敛快于S型神经元）得到最终的模型输出。通过缩小该输出和实际流量的差异（代价）来训练整个网络。
> 理解：关键帧如何选取，图中的每个格子代表关键帧输入的 2*n*n tensor（即2channel的输入matrix）？

### closeness, period and trend
结构一致，由**卷积单元**和**残差单元**组成。
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1600947588235-7ea7df34-5891-40d7-a3ae-a4f994367673.png)

#### 卷积单元
CNN擅长处理层次化的空间结构信息，但传统深层CNN会出现分辨率损失的问题（降采样，池化层的使用使得像素产生损失，subsampling=pooling），导致空间依赖关系的丢失。因此，文章的卷积单元中没有使用池化层，而只用到了卷积层，使得网络能够同时保留nearby dependencies和distant dependencies。
> 理解：为什么卷积+残差能够学习邻近区域和较远区域之间的空间依赖性？
> 单层卷积的卷积核只能覆盖小区域（如3*3），卷积核反映出小区域内部的空间依赖性（也就是所说的nearby dependencies）。随着卷积层加深，新的卷积核在之前的特征图上进行特征映射，即此时的卷积核覆盖了多个小区域，反映了各个小区域之间的空间依赖性，这样不断叠加，就能渐渐描绘出较远区域之间的空间依赖性（distant dependencies）。但，由于要描述distant dependencies，层数就会很多，训练会出现退化的情况，因此才采用了residue。
> 理解：分辨率损失会对空间依赖关系的描述产生什么影响？
> 个人理解，这里的分辨率损失就好像是nearby dependencies的损失，传统的CNN使用pooling不断地进行降采样（如2*2pooling就只保留1个像素），当层数很多的时候，相邻区域的空间依赖关系就会丢失。

这里以closeness为例，选取recent time部分的一些关键帧（关键时间区间）的交通流三维张量组成序列：
$$[X_{t-l_c},X_{t-(l_c-1)},\dots,X_{t-1}]$$
在时间维度上将这些2channel的张量连接起来，得到一个新的张量$$X^{(0)}_c\in\mathbb{R}^{2l_c\times I\times J}$$，作为Conv1的输入，进行如下卷积：
$$
\mathrm{X}_c^{(1)}=f(W_c^{(1)}*\mathrm{X}_c^{(0)}+b_c^{(1)})
$$

后续Conv2同上，均使用ReLU神经元。

> 对于具有多个channel的输入图象，使用的依然是二维卷积核（二维是指卷积核只能在高和宽两个方向上移动，可以用$$(c, k_h, k_w)$$表示，$$c$$表示channel数），但对每一channel都有一个权重矩阵，各channel内部加权再相加，加上偏置，激活函数输出后，就得到卷积层最终的输出值。

卷积类型有三种（full, same, narrow/valid），为了使得输入输出大小相同，这里采用的是same convolution，即加上一圈大小为1的padding。
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1600951880667-bfd7c587-33ac-4e57-acdd-4f20247c3049.png)

> 关于几种convolution的介绍：

#### 残差单元
主要是为了解决网络层数增多导致的训练退化问题。
在Conv1和Conv2之间，共使用了$$L$$个残差单元，表达式如下：
$$
\mathrm{X}_c^{(l+1)}=\mathrm{X}_c^{(l)}+\mathcal{F}(\mathrm{X}_c^{(l)};\theta_c^{(l)}),l=1,\cdots,L
$$

$$\mathcal{F}$$是由残差单元中的两层卷积+两个ReLU函数表示的映射，ReLU前都使用了BN技术（这个好像是在残差网络提出的论文随后一篇优化论文里提出的，对比了多种结构，实验证明了BN加在ReLU前更好）。

> 推文看忘了，要看一下NG里的BN视频加深理解

另外两个特征period, trend类似：

- period：$$[X_{t-l_p\cdot p},X_{t-(l_p-1)\cdot p},\dots,X_{t-p}]$$
- trend：$$[X_{t-l_q\cdot q},X_{t-(l_q-1)\cdot q},\dots,X_{t-q}]$$

$$l_c,l_p,l_q$$代表序列长度，$$p,q$$代表跨度（实际取了1天和1周，closeness取1，即一个区间间隔），最后三个网络的Conv2分别输出$$X^{(L+2)}_c,X^{(L+2)}_p,X^{(L+2)}_q$$。
### external component
主要提取的特征：

- holiday：是否为节假日
- weather：是否为雨天、温度、风速等
- weekday/weekend：工作日/休息日
- DayOfWeek：星期几

用$$E_t$$表示这些外部因素在时间间隔$$t$$的特征向量。
相对于其他外部因素，将要预测时刻$$t$$的天气数据是无法预测的，文章使用了前一刻$$t-1$$的天气数据来近似替代。
网络包括两层全连接层，前一层用作embedding，后一层用作低维到高维的映射（映射为与其他三个时间特征输出的$$X_t$$相同维度的数据），最终的输出用$$X_{Ext}$$表示。

> Embedding的目的是进行数据的降维和向量化，比如图像用CNN卷了之后也可以叫做Embedding，Auto-Encoder里面前面的那一部分也可以叫做Embedding，LSTM可以视作将状态Embedding等等。所以**Embedding描述的是一种功能：数据降维和稠密表示(≈向量化)，且通常所指的Embedding是中间的产物，为了方便后面的处理。**

### Fusion
- closeness：
  横轴代表取的time gap大小，纵轴代表该gap下任意两个时段inflow之间的平均比率。→ time gap越小相关性越高，且在工作和居住区下可能拥有不同closeness特性。
- period：
  都具有明显日周期性。不同类型区域特征不同。
- trend：
  随三月到五月同一时刻办公区流入减少，居住区流入增多。
  以上观察可以得出：
  **不同区域都具有这三个特性，但影响程度不同。**
  ![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601035316067-68a2b4eb-f26a-4b63-a2c1-899e0c52997f.png)
  基于以上观察提出：

#### 基于参数矩阵的融合
将时间上的三个特性以参数矩阵方式融合：
$$
\mathrm{X}_{Res}=\mathrm{W}_c\odot\mathrm{X}_c^{(L+2)}+\mathrm{W}_p\odot\mathrm{X}_p^{(L+2)}+\mathrm{W}_q\odot\mathrm{X}_q^{(L+2)}
$$
$\odot$代表Hadamand乘积（对应相乘），$$W_c,W_p,W_q$$是可学习的、反映了三个特性影响程度的参数。

#### 加入外部因素影响

求和并进行tanh映射（将其映射至-1到1的区间内）：
$$
\hat{\mathrm{X}}_t=\tanh(\mathrm{X}_{Res}+\mathrm{X}_{Ext})
$$
以代价函数最小化为目标，对该网络进行训练：
$$
\mathcal{L}(\theta)=||\mathrm{X}_t-\hat{\mathrm{X}}_t||^2_2
$$

其中$$\theta$$代表网络中的所有参数。

### Algorithms and Optimization 算法和优化
#### 训练

使用BP+Adam进行训练：
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601038210880-0a2b4cf4-8917-4110-9093-f42c3afcded8.png)

#### 预测
使用训练得到的模型进行预测，区别在于天气数据换用了预测的天气数据（训练时用的真实天气数据）
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601039280515-fa04ed16-f81c-466e-b0c2-a8878e53c27a.png)

## Experiments 实验
### Settings 实验设定
#### 数据集
使用了两个包含轨迹数据和气象数据的数据集：

- TaxiBJ
- BikeNYC

![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601117614547-3b08093e-fcc8-4656-80ae-5ebba1a1415c.png)
#### Baseline 基准

- HA：同一时间段历史数据平均
- ARIMA：closeness
- SARIMA：增加季节因素，可以反映closeness+period
- VAR：能捕捉所有流量中的成对关系，但计算成本很高
> 向量自回归（VAR,Vector Auto regression），对于相互联系的时间序列变量系统是有效的预测模型，同时也被频繁地用于分析不同类型的随机误差项对系统变量的动态影响。如果变量之间不仅存在滞后影响，还存在同期影响关系，则适合建立VAR模型，因为VAR模型实际上是把当期关系隐含到了随机扰动项之中。

- ST-ANN：提取时间（前8个时间段）和空间（邻近8个区域）特征作为ANN输入
- DeepST：当前最先进的预测流量DNN之一
- RNN：RNN-3, RNN-6, RNN-12, RNN-24, RNN-48, RNN-336
- LSTM：RNN的一种，能学习较长期的时间依赖性，同样有LSTM-3, LSTM-6, LSTM-12, LSTM-24, LSTM-48, LSTM-336
- GRU：RNN的一种，，能学习较长期的时间依赖性，同样有GRU-3, GRU-6, GRU-12, GRU-24, GRU-48, GRU-336
#### 预处理
##### inflow/outflow
使用Min-Max归一化将数据映射到[-1,1]上，以进行cost计算
由于输出层激活函数：tanh → [-1,1]，评估时将预测值rescale回原来量纲值
##### 外部因素

- 类别变量（非数值变量）：如是否为节假日、星期几等等，采用独热编码（One-Hot），即各类别变量用寄存器表示，几个取值取几位，最后连在一起，得到二进制向量
- 数值变量：如气温与风速，使用Min-Max归一化将数据映射到[-1,1]上
#### 超参数

- 使用Keras内置的均匀分布来初始化超参数。
- Conv1和所有的残差单元使用64个大小为3×3的卷积核，Conv2使用2个大小为3×3的卷积核
- 使用Adam优化算法，batch_size=32
- TaxiBJ使用12个残差单元，BikeNYC使用4个残差单元
- 时间区间间隔$$p,q$$固定，$$p$$取1天，$$q$$取1周
- $$l_c\in\{1,2,3,4,5\},l_p\in\{1,2,3,4\},l_q\in\{1,2,3,4\}$$，得到80个不同的模型
- 在训练集中，90%用作训练，10%用作验证
- 采用early-stop，在best validation score时停止训练，然后再使用所有训练集数据训练一段时间

![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601119717543-257a8867-64b9-4024-b4c5-19de36226255.png)
#### 评估方式
RMSE作为代价函数。
$$
RMSE=\sqrt{\frac1z\sum_i(x_i-\hat{x}_i)^2}
$$

#### 实验环境

![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601120229831-63ad4d02-f902-438b-9f91-81545fc0183e.png)
### 单步预测结果
单步预测（One-Step Ahead），即使用历史观测数据预测$$t$$时刻交通流
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601120518171-e44de418-693c-43c7-97ff-d56e1494c08c.png)
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601120657570-d40b0edd-31e7-47e8-8b5d-ecaf55b66b29.png)

### 不同网络结构下的结果对比

- network configuration
- network depth
- filter size and number
#### Network configuration

- BN or not
- fusion：加权/直接相加（parametric matrix / add up straightforward）
- residual unit：1/2层卷积

![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601120849586-7617d121-f8f2-4a1d-ad2e-073e0b50d498.png)
（新知识：w/o代表without，w代表with）

#### Network Depth
卷积单元数量（网络深度）
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601121120172-b6efc8c6-10ab-4ef6-aeec-f5c51f0c0734.png)

#### Filter size and number
卷积核大小和数量
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601121200680-ac19cf39-d098-477f-8456-521d7b069134.png)

#### closeness/period/trend各序列长度
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601121338278-8c1108c7-004b-42c4-a335-0adf18d61a27.png)
![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601121471963-a4ea55aa-f8ff-4fdf-8ad4-8e113b935101.png)
不同的地区各个因素（factor）影响程度不同，通过使用热力图可视化学习到的混合参数权重矩阵，可以看到在近邻性特征中，外环区域的影响不是很大，因为长期车辆较少，而在周期性方面，朝阳公园具有很强的影响，因人们外出锻炼、逛公园具有很强的规律性和时间周期性，故这方面影响（权重）较大。

### 多步预测结果
利用历史观测数据和最近预测数据来预测后续时间区间的人群流量，即多步预测（Multi-step Ahead Prediction）。

- [BN]-使用BN
- [CP]-不包含trend
- [C]-只使用closeness和external

![image.png](https://cdn.nlark.com/yuque/0/2020/png/1428756/1601121834852-a1d23086-dfe0-42fb-8b6b-372c98c9666f.png)
LSTM-12在step更大时表现更好：可能是由于LSTM同时输入了过去12个时段数据，而ST-ResNet只是用了前三个时段的数据。

### 计算资源的选择
对比了两种虚拟机配置下的计算效率和效果
## Related Work 相关研究
### 交通流量预测
其他文献：

- 根据轨迹历史位置来预测个体行动轨迹，而不是预测流量
- 聚焦于全市或某条道路的流量预测，而不像本文预测某个区域（街区）的流入流出
### 传统时间序列预测
线性模型

- AN：非动态的，历史平均
- ARIMA：不适用于含缺失值数据
- SARIMA：耗时的
- VAR：忽略了预测值和残差关系

非线性模型

- ANN：非线性建模突出，但对于线性关系描述仍然不足
### 深度神经网络

- CNN/RNN：只能捕捉时间/空间关系中的一种
- convolutional LSTM：可以同时捕捉，但不能对较长时间间隔的时间依赖关系建模，且随着层数加深训练难度变得更大
### Urban Computing 城市计算
新的研究领域，旨在利用城市中的一切传感器所得的数据来进行分析和计算，以解决现代城市中影响居民生活的问题（如交通拥挤、能源消耗、大气污染等），为城市和居民服务。论文作者称在文章发表前该领域从未有过像本文进行端到端交通流量预测的研究。
## Conclusion and Future Work 结论和展望
未来将利用其它形式的交通流数据（公共交通数据）来进行更多类型的交通流预测，使用更合适的融合方法来预测区域的总体交通流量。
## 可学习笔记

- [精读论文：Predicting Citywide Crowd Flows Using Deep Spatio-Temporal Residual Networks](https://blog.csdn.net/u013982164/article/details/84112679)
- [基于深度ST-残差网络的城市人流量预测 读书笔记](https://blog.csdn.net/zzyy0929/article/details/83997923?utm_medium=distribute.pc_relevant.none-task-blog-title-1&spm=1001.2101.3001.4242)
