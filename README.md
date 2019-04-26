# QSAR_package使用说明
$\to$
## 1. 使用前准备：
下载所有脚本，把把所有文件解压后存放至一个目录，如```$/myPackage/```

<img src="https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-25_12-47-38.png" alt="Sample"  width="800">
<img src="https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-37-48.png" alt="Sample"  width="800">

新建一个文本文件，把上述目录的路径粘贴至这个文件内，然后把后缀改为```.pth```，如```myPackage.pth```

<img src="https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-47-04.png" alt="Sample"  width="800">

打开cmd，输入```python```进入Python交互界面  
```python
import sys
```  
```python
sys.path
```  
找到一个类似```..\\lib\\site-packages```的路径  

<img src="https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-51-27.png" alt="Sample"  width="800">

然后进入这个文件夹，把刚才创建的```myPackage.pth```文件放入这个路径，

<img src="https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_11-08-25.png" alt="Sample"  width="800">

以上操作的目的是把自己的脚本库路径加入到Python的环境变量中

## 2. 提取数据/随机划分训练集测试集——必选步骤，三种方式三选一

**在存放描述符数据的文件中，一定要把标签列放于第一个特征（描述符）列的前一列**

### 2.1 输入一个总描述符文件，采用随机划分的方式产生训练集和测试集
```python
from QSAR_package.data_split import randomSpliter
```

```python
file_name = "C:/OneDrive/Jupyter_notebook/regression_new/data/spla2_296_rdkit2d.csv"  # 描述符数据文件路径
spliter = randomSpliter(test_size=0.25,random_state=42)
spliter.ExtractTotalData(file_name,label_name='Activity') #注意指定标签（活性）列的列名
spliter.SplitData()
tr_x = spliter.tr_x
tr_y = spliter.tr_y
te_x = spliter.te_x
te_y = spliter.te_y
```
如果想保存训练集测试集标签，则加入以下代码：
```python
spliter.saveTrainTestLabel('C:/OneDrive/Jupyter_notebook/regression_new/data/sPLA2_296_trOte42.csv')
```
### 2.2 根据训练集和测试集标签文件提取训练集和测试集
```python
from QSAR_package.data_split import extractData
```

```python
data_path = 'C:/OneDrive/Jupyter_notebook/regression_new/data/spla2_296_rdkit2d.csv'  # 描述符数据文件路径
trOte_path = 'C:/OneDrive/Jupyter_notebook/regression_new/data/sPLA2_296_trOte0.csv'  # 训练集和测试集标签文件路径
spliter = extractData()
spliter.ExtractTrainTestFromLabel(data_path, trOte_path, label_name='Activity') #注意指定标签（活性）列的列名
tr_x = spliter.tr_x
tr_y = spliter.tr_y
te_x = spliter.te_x
te_y = spliter.te_y
```
### 2.3 如果已经提前分好训练集测试集，且训练集和测试集文件存放于两个文件中，则使用以下代码

```python
from QSAR_package.data_split import extractData
```

```python
train_path = 'C:/OneDrive/Jupyter_notebook/Deep_learning/spla2_som_train_312_maccs.csv'  # 训练集数据文件路径
test_path = 'C:/OneDrive/Jupyter_notebook/Deep_learning/spla2_som_test_140_maccs.csv'  # 测试集数据文件路径
spliter = extractData()
spliter.ExtractTrainTestData(train_path, test_path, label_name='Activity') #注意指定标签（活性）列的列名
tr_x = spliter.tr_x 
tr_y = spliter.tr_y
te_x = spliter.te_x
te_y = spliter.te_y
```

## 3. Pearson相关性筛选/RFE排序/数据压缩
### 3.1 Pearson相关性筛选（按训练集数据筛选）——可选步骤（一般都会用上）
```python
from QSAR_package.feature_preprocess import correlationSelection
```

```python
corr = correlationSelection()
corr.PearsonXX(tr_x, tr_y,threshold_low=0.1, threshold_up=0.9)
```

筛选结果的描述符顺序已经按照其跟活性的Pearson相关性从高到低排好序，筛选之后的数据可通过```corr.selected_tr_x```获取，该属性是筛选之后的DataFrame对象，然后将此结果输入数据压缩环节。

### 3.2 数据压缩——必要步骤

数据压缩模块`dataScale`可以将所有描述符数据压缩至指定的区间范围（如0.1到0.9），此处直接使用上一步骤Pearson相关性筛选产生的训练集数据`corr.selected_tr_x`拟合压缩器，然后对测试集数据进行压缩，此模块能自动识别连续型的描述符数据和指纹描述符数据，如果输入的是指纹描述符数据，则压缩之后数据不会有变化，所以，为了减少代码的改动，保证变量的统一，可以让指纹描述符也经过数据压缩过程，其数值不会发生变化。

```python
from QSAR_package.data_scale import dataScale
```

```python
tr_scaled_x = scaler.FitTransform(corr.selected_tr_x)
te_scaled_x = scaler.Transform(te_x,DataSet='test')  
```

### 3.3 RFE（递归消除法）排序——可选步骤
经过压缩之后，数据就可以直接输入参数寻优环节了，如果还需要将描述符的顺序换为RFE（递归消除法）排序的顺序，则运行以下代码：

```python
from QSAR_package.feature_preprocess import RFE_ranking
```

```python
rfe = RFE_ranking(estimator='SVR',features_num=1)   # "SVR" 为用于实现RFE排序的学习器（分类或回归算法）
rfe.Fit(tr_scaled_x, tr_y)
tr_ranked_x = rfe.tr_ranked_x
te_ranked_x = te_scaled_x.loc[:,tr_ranked_x.columns]
```
目前支持字符串指定的学习器有"SVC"(分类)、"RFC"（分类）、"SVR"（回归）、"RFR"（回归），如果想尝试其他学习器，可以直接让```estimator```参数等于一个自定义的学习器对象，前提是该学习器对象有```coef_```或```feature_importance_```属性，详见[sklearn文档中RFE算法的介绍](https://scikit-learn.org/stable/modules/feature_selection.html#rfe"")

## 4. 参数寻优
### 不带描述符数量的重复网格寻优
- 使用`gridSearchBase`模块可以自定义传入学习器、参数字典、打分器对象，进行重复网格寻优，此处以`SVC`算法的寻优为例
```python
from QSAR_package.grid_search import gridSearchBase
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,make_scorer
```

```python
grid_estimator = SVC()
grid_dict = {'C':[1,0.1,0.01],'gamma':[1,0.1,0.01]}
grid_scorer = make_scorer(accuracy_score,greater_is_better=True)
grid = gridSearchBase(fold=5, grid_estimator=grid_estimator, grid_dict=grid_dict, grid_scorer=grid_scorer, 
                      repeat=10, early_stop=None, scoreThreshold=None, stratified=False)

```
