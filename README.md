# QSAR_package使用说明
## 1. 使用前准备：
下载所有脚本，把把所有文件解压后存放至一个目录，如```$/myPackage/```
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-25_12-47-38.png)
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-37-48.png)
新建一个文本文件，把上述目录的路径粘贴至这个文件内，然后把后缀改为```.pth```，如```myPackage.pth```
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-47-04.png)
打开cmd，输入```python```进入Python交互界面  
```python
import sys
```  
```python
sys.path
```  
找到一个类似```..\\lib\\site-packages```的路径  
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-51-27.png)
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_11-03-33.png)
然后进入这个文件夹，把刚才创建的```myPackage.pth```文件放入这个路径，
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_11-08-25.png)
以上操作的目的是把自己的脚本库路径加入到Python的环境变量中
## 2. 提取数据/随机划分训练集测试集（三种方式），```在存放描述符数据的文件中，一定要把标签列放于第一个特征（描述符）列的前一列```
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
