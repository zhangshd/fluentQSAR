# QSAR_package使用说明
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
### 4.1 不带描述符数量的重复网格寻优
- 使用`gridSearchBase`模块可以自定义传入学习器、参数字典、打分器对象，进行重复网格寻优，此处以`SVC`算法的寻优为例

    ```python
    from QSAR_package.grid_search import gridSearchBase
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score,make_scorer
    ```
    
    ```python
    grid_estimator = SVC() # 学习器对象
    grid_dict = {'C':[1,0.1,0.01],'gamma':[1,0.1,0.01]}  # 对应学习器的参数字典
    grid_scorer = make_scorer(accuracy_score,greater_is_better=True)  # 打分器对象
    grid = gridSearchBase(fold=5, grid_estimator=grid_estimator, grid_dict=grid_dict, grid_scorer=grid_scorer, repeat=10)
    grid.Fit(tr_scaled_x,tr_y)
    ```
    其中`fold`为网格寻优中交叉验证的重数，`repeat`为网格寻优的重复次数；
    然后可以通过`grid.best_params`获取最优参数，通过`grid.best_estimator`获取拟合好的学习器。
- 使用`gridSearchPlus`模块可以通过字符串直接指定预定义好的学习器和对应的参数字典及打分器，现支持的算法有"SVC"、"DTC"、"RFC"、"SVR"、"RFR"，调用代码（不带描述符数量的寻优）如下：
    ```python
    from QSAR_package.grid_search import gridSearchPlus
    ```
    ```python
    grid = gridSearchPlus(grid_estimatorName='SVC', fold=5, repeat=5)
    grid.Fit(tr_scaled_x,tr_y)
    ```
    然后可以通过`grid.best_params`获取最优参数，通过`grid.best_estimator`获取拟合好的学习器。
### 4.2 带描述符数量的重复网格寻优
因为前面已经介绍了可以通过Pearson相关性或者RFE方法对描述符数据排序，得到一个在列方向上有序的二维数据（DataFrame或numpy数组），如此以来，便可以将描述符的数量`n`也作为一个超参数，参与寻优过程，在网格寻优的外层套一个循环，每次循环取前`n`个描述符的数据，再用此数据进行重复网格寻优，最后找出在交叉验证的得分最高的描述符数量与参数组合。
- 使用`gridSearchBase`模块进行带描述符数量的重复网格寻优
    ```python
    grid_estimator = SVC() # 学习器对象
    grid_dict = {'C':[1,0.1,0.01],'gamma':[1,0.1,0.01]}  # 对应学习器的参数字典
    grid_scorer = make_scorer(accuracy_score,greater_is_better=True)  # 打分器对象
    grid = gridSearchBase(fold=5, grid_estimator=grid_estimator, grid_dict=grid_dict, 
                          grid_scorer=grid_scorer, repeat=10)
    grid.FitWithFeaturesNum(tr_scaled_x, tr_y,features_range=(5,20))  # features_range为描述符数量的迭代范围，参数为元组或列表形式
    ```
    然后可以通过`grid.best_params`获取最优参数，通过`grid.best_estimator`获取拟合好的学习器，还可以通过`grid.best_features`获取最终选择的若干个描述符名称。
-  使用`gridSearchPlus`模块进行带描述符数量的重复网格寻优
    ```python
    grid = gridSearchPlus(grid_estimatorName='SVC', fold=5, repeat=5)
    grid.FitWithFeaturesNum(tr_scaled_x, tr_y,features_range=(5,20))
    ```
    然后可以通过`grid.best_params`获取最优参数，通过`grid.best_estimator`获取拟合好的学习器，还可以通过`grid.best_features`获取最终选择的若干个描述符名称。
### 4.3 Early_stop策略——降低过拟合程度
正常情况gridsearch所选的最优参数组合是交叉验证平均得分（mean_test_score）最高的参数组合，如果采用Early_stop策略，则会从（mean_test_score）最高分开始向下寻找（分值按降序排列）得分与最高分有显著差异的次优参数组合， 显著差异的标准就是该分值与最高分的差值占该分值的比率（取绝对值）大于指定的early_stop数值，最终选择的参数组合是降序排名在上述次优参数组合前一名的参数组合，在`gridSearchBase`和`gridSearchPlus`中都可以设置`early_stop`参数，默认为`None`，有效的`early_stop`参数值为`0`到`1`之间的浮点数，具体例子如下：
```python
grid_estimator = SVC() # 学习器对象
grid_dict = {'C':[1,0.1,0.01],'gamma':[1,0.1,0.01]}  # 对应学习器的参数字典
grid_scorer = make_scorer(accuracy_score,greater_is_better=True)  # 打分器对象
grid = gridSearchBase(fold=5, grid_estimator=grid_estimator, grid_dict=grid_dict, 
                      grid_scorer=grid_scorer, repeat=10, early_stop=0.01)
...
```
或者
```python
grid = gridSearchPlus(grid_estimatorName='SVC', fold=5, repeat=5, early_stop=0.01)
...
```

## 拟合模型/评价模型/保存结果
- 用`modeling`模块可以传入一个学习器对象及对应的一组超参数，然后使用训练集进行拟合（`modeling.Fit`），同时也可以用来对测试集样本进行预测（`modeling.Predict`），还可以用训练集做交叉验证（通过sklearn中`metrics`模块下的`cross_val_predict`实现，通过`modeling.CrossVal`调用）。分类任务的预测结果评价值包括`Accuracy`、`MCC`、`SE`、`SP`、`tp`、`tn`、`fp`、`fn`，回归任务的预测结果评价值包括`R2`、`RMSE`。评价结果可以通过`modeling.ShowResults`打印出来，如果想看训练集和测试集预测结果的散点图（回归任务），可以设定参数`make_fig=True`，该参数默认为`False`。评价结果及模型的超参数可以通过`modeling.SaveResults`方法保存，保存的机制是以追加的方式写入一个csv文件，如果在使用`modeling.ShowResults`设置了`make_fig=True`，则散点图也会保存出来（tif格式），同时，这组结果对应的模型文件也会保存（.model后缀），如果不需要，则可以在`modeling.SaveResults`中设置`save_model=False`。
    -   `modeling`模块可以直接接收上一环节网格寻优的结果（`grid.best_estimator`、`grid.best_params`、`grid.best_features`），使用示例如下：

        ```python
        from QSAR_package.model_evaluation import modeling
        ```
        ```python
        model = modeling(estimator=grid.best_estimator,params=grid.best_params)
        model.Fit(tr_scaled_x.loc[:,grid.best_features], tr_y)
        model.Predict(te_scaled_x.loc[:,grid.best_features],te_y)
        model.CrossVal(cv="LOO")
        model.ShowResults(show_cv=True, make_fig=False)
        model.SaveResults('./results.csv',notes='自己定义的一些备注信息')
        ```
