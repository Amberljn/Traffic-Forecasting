# A Comprehensive Survey on Traffic Prediction

## 摘要

- 总结了现有的流量预测方法，并给出它们的分类。
- 列出了交通预测的常见任务和这些任务的最新技术。
- 收集和组织现有文献中广泛使用的公共数据集
- 讨论了未来可能的发展方向。

## 导论

智能交通系统(ITS)是智慧城市不可缺少的一部分，而交通预测是其发展的**基石**。

例如：

- 交通量预测可以帮助城市缓解拥堵
- 网约车需求预测可以促使汽车共享公司将车辆预先分配到高需求地区。

### 挑战

- 复杂空间依赖性。
  - 不同位置对预测位置的影响不同，相同位置对预测位置的影响也随着时间的变化而变化。不同位置之间的空间相关性是高度动态的。
- 动态时间依赖性。
  - 同一位置不同时间的观测值呈现非线性变化，远时步长的交通状态与预测时间步长的相关性比近时要大。
- 外部因素
  - 交通时空序列数据还受到天气、事件或道路属性等外部因素的影响。

### 主要任务

- 交通流
  - 在一定时间内通过道路上某一点的车辆数量。
- 速度
  - 车辆单位时间内行驶的距离
- 需求
  - 使用历史请求数据来预测未来时间戳中某个区域的请求数量
- 占用率
  - 车辆占用道路空间的程度。
- 出行时间
  - 从一点到另一点的耗费时间

## 方法

#### 传统方法

##### 经典统计模型

以ARIMA一类经典统计模型为代表，问题在于不适合处理复杂、动态的时间序列数据。此外，由于通常只考虑时间信息，交通数据的空间依赖性被忽略或很少考虑

##### 机器学习模型

机器学习方法可以建模更复杂数据,大致可分为三类:

- 基于特征的模型
  - 训练基于人工工程交通特征的回归模型来解决交通预测问题这些方法易于实现，并能在一些实际情况下提供预测。尽管存在这种可行性，但基于特性的模型有一个重要的局限性:模型的性能严重依赖于人工设计的特征。
- 高斯过程模型
  - 高斯过程通过不同的核函数来模拟交通数据的内在特征，这些核函数需要同时包含空间和时间的相关性。虽然这类方法在流量预测([25]-[27])中被证明是有效可行的，但是它们具有较高的计算负荷和存储压力，在有大量训练样本的情况下不合适。
- 状态空间模型
  - 该模型的优点是能够自然地对系统的不确定性进行建模，并能更好地捕捉到时空数据的潜在结构。然而，这些模型([28]-[38])的整体非线性是有限的，大多数情况下，它们对复杂动态交通数据的建模不是最优的。

### 深度学习模型

- 建模空间依赖性
  - CNN
    - 将不同时刻的交通网络结构转换成图像，并将这些图像划分为标准网格，每个网格代表一个区域。这样CNNs就可以用来学习不同区域之间的空间特征。
  - GCN
    - 统的CNN仅限于欧氏数据的建模，因此使用GCN来建模非欧氏空间结构数据，更符合交通道路网络的结构。
- 建模时间依赖性
  - CNN
    - 一维卷积
  - RNN
    - 遗忘
    - 无法提取长期相关性

### 数据集

- 常用的公共时空序列数据
- 提高预测精度的外部数据

**PeMS:** 

It is an abbreviation from the California Trans-portation Agency Performance Measurement System(PeMS), which is displayed on the map and collected n real-time by more than 39000 independent detectors.
These sensors span the freeway system across all major
metropolitan areas of the State of California. The source
is available at: http://pems.dot.ca.gov/. Based on this sys-
tem, several sub-dataset versions (PeMSD3/4/7(M)/7/8/-
SF/-BA Y) have appeared and are widely used. The main
difference is the range of time and space, as well as the
number of sensors included in the data collection.
PeMSD3: This dataset is a piece of data processed by
Song et al. It includes 358 sensors and flow information
from 9/1/2018 to 11/30/2018. A processed version is
available at: https://github.com/Davidham3/STSGCN.
**PeMSD4:**  

It  describes  the  San  Francisco  Bay
Area, and contains 3848 sensors on 29 roads
dated  from  1/1/2018  until  2/28/2018,  59  days
in  total.  A  processed  version  is  available  at:
https://github.com/Davidham3/ASTGCN/tree/master/data/
PEMS04.

**PeMSD7(M): **

It describes the District 7 of California
containing  228  stations,  and  The  time  range
of it is in the weekdays of May and June
of 2012. A processed version is available at:
https://github.com/Davidham3/STGCN/tree/master/
datasets.
**PeMSD7:** 

This version was publicly released by Song
et al. It contains traffic flow information from 883
sensor stations, covering the period from 7/1/2016
to 8/31/2016. A processed version is available at:
https://github.com/Davidham3/STSGCN.
**PeMSD8:**  

It  depicts  the  San  Bernardino  Area,
and  contains  1979  sensors  on  8  roads  dated
from  7/1/2016  until  8/31/2016,  62  days  in
total.  A  processed  version  is  available  at:
https://github.com/Davidham3/ASTGCN/tree/master/
data/PEMS08.
**PeMSD-SF:** 

This dataset describes the occupancy rate,
between 0 and 1, of different car lanes of San Francisco
bay area freeways. The time span of these measure-
ments is from 1/1/2008 to 3/30/2009 and the data is
sampled every 10 minutes. The source is available at:
http://archive.ics.uci.edu/ml/datasets/PEMS-SF.
**PeMSD-BAY:** 

It contains 6 months of statistics on traffic
speed, ranging from 1/1/2017 to 6/30/2017, including
325 sensors in the Bay area. The source is available at:
https://github.com/liyaguang/DCRNN.

 **LOOP: I**

it is collected from loop detectors deployed
on four connected freeways (I-5, I-405, I-90 and SR-
520) in the Greater Seattle Area. It contains traffic
state data from 323 sensor stations over the entirely of
2015 at 5-minute intervals. The source is available at:
https://github.com/zhiyongc/Seattle-Loop-Data.

**Los-loop:** 

This dataset is collected in the highway of
Los Angeles County in real time by loop detectors. It
includes 207 sensors and its traffic speed is collected
from 3/1/2012 to 3/7/2012. These traffic speed data is
aggregated every 5 minutes. The source is available at:
https://github.com/lehaifeng/T-GCN/tree/master/data.
**TaxiBJ**: 

Trajectory data is the taxicab GPS data and
meteorology data in Beijing from four time intervals:
1st Jul. 2013 - 30th Otc. 2013, 1st Mar. 2014 -
30th Jun. 2014, 1st Mar. 2015 - 30th Jun. 2015, 1st
Nov. 2015 - 10th Apr. 2016. The source is avail-
able at: https://github.com/lucktroy/DeepST/tree/master/
data/TaxiBJ.
**SZ-taxi:** 

This is the taxi trajectory of Shenzhen from
Jan.1 to Jan.31, 2015. It contains 156 major roads
of Luohu District as the study area. The speed of
traffic on each road is calculated every 15 minutes.
The source is available at: https://github.com/lehaifeng/T-
GCN/tree/master/data.
**NYC Bike:** 

The bike trajectories are collected from
NYC CitiBike system. There are about 13000
bikes and 800 stations in total. The source is
available  at:  https://www.citibikenyc.com/system-
data.  A  processed  version  is  available  at:
https://github.com/lucktroy/DeepST/tree/master/data/
BikeNYC.
**NYC Taxi**: 

The trajectory data is taxi GPS data for
New Y ork City from 2009 to 2018. The source is
available at: https://www1.nyc.gov/site/tlc/about/tlc-trip-
record-data.page.
• **Q-Traffic dataset:** 

It consists of three sub-datasets:
query sub-dataset, traffic speed sub-dataset and road
network sub-dataset. These data are collected in Bei-
jing, China between April 1, 2017 and May 31, 2017,
from the Baidu Map. The source is available at:
https://github.com/JingqingZ/BaiduTraffic#Dataset.
**• Chicago:** 

This is the trajectory of shared bikes in
Chicago from 2013 to 2018. The source is available at:
https://www.divvybikes.com/system-data.
**• BikeDC:** 

It is taken from the Washington D.C.Bike Sys-
tem. The dataset includes data from 472 stations and four
time intervals of 2011, 2012, 2014 and 2016. The source
is available at: https://www.capitalbikeshare.com/system-
data.
• **ENG-HW:**  

It  contains  traffic flow information
from inter-city highways between three cities,
recorded by British Government, with a time
range of 2006 to 2014. The source is available at:
http://tris.highwaysengland.co.uk/detail/trafficflowdata.
• **T-Drive:** 

It consists of tremendous amounts of trajectories
of Beijing taxicabs from Feb.1st, 2015 to Jun. 2nd 2015.
These trajectories can be used to calculate the traffic
flow in each region. The source is available at:
https://www.microsoft.com/en-us/research/publication/t-
drive-driving-directions-based-on-taxi-trajectories/.
• **I-80:** 

It is collected detailed vehicle trajectory data
on eastbound I-80 in the San Francisco Bay area in
Emeryville, CA, on April 13, 2005. The dataset is 45
minutes long, and the vehicle trajectory data provides
the precise location of each vehicle in the study area
every tenth of a second. The source is available at:
http://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm.
• **DiDi chuxing:**

 DiDi gaia data open program provides real
and free desensitization data resources to the academic
community. It mainly includes travel time index, travel
and trajectory datasets of multiple cities. The source is
available at: https://gaia.didichuxing.com.
**• Travel Time Index data:**
The dataset includes the travel time index of Shenzhen,
Suzhou, Jinan, and Haikou, including travel time index
and average driving speed of city-level, district-level,
and road-level, and time range is from 1/1/2018 to
12/31/2018. It also includes the trajectory data of the Didi
taxi platform from 10/1/2018 to 12/1/2018 in the second
ring road area of Chengdu and Xi’an, as well as travel
time index and average driving speed of road-level in
the region, and Chengdu and Xi’an city-level. Moreover,
the city-level, district-level, road-level travel time index
and average driving speed of Chengdu and Xi’an from
1/1/2018 to 12/31/2018 is contained.
**Travel data:**
This dataset contains daily order data from 5/1/2017 to
10/31/2017 in Haikou City, including the latitude and
longitude of the start and end of the order, as well as
the order attribute of the order type, travel category, and
number of passengers.
**Trajectory data:**
This dataset comes from the order driver trajectory data
of the Didi taxi platform in October and November 2016
in the Second Ring Area of Xi’an and Chengdu. The
trajectory point collection interval is 2-4s. The trajectory
points have been processed for road binding, ensuring that
the data corresponds to the actual road information. The
driver and order information were encrypted, desensitized
and anonymized.

#### 外部数据

- 天气
- 驾驶员ID
  - 由于驾驶员个人情况的不同，预测会产生一定的影响，因此需要对驾驶员进行标签，该信息主要用于个人预测
- 活动
  - 包括各种节日、交通管制、交通事故、体育赛事、音乐会等活动。
- 时间信息
  - 工作日和周末
  - 
    每天的不同时段

## 实验

总结

- 图神经是未来
- 注意力机制很有用

## 未来研究方向

- 少样本问题
- 知识图融合问题:
- 长期预测
- 多源数据问题