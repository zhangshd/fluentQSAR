# coding: utf-8

from QSAR_package.data_split import extractData,randomSpliter
from QSAR_package.feature_preprocess import correlationSelection,RFE_ranking
from QSAR_package.data_scale import dataScale
from QSAR_package.grid_search import gridSearchPlus,gridSearchBase
from QSAR_package.model_evaluation import modeling
import pandas as pd
import numpy as np
import os
os.system("")

if __name__ == '__main__':

    """=============================参数设定（常变）================================"""
    # ##输入文件路径，包含所有样本的描述符数据文件
    file_name = "./spla2_296_rdkit2d.csv"
    
    # ##标签列的列名，label列请务必放于第一个feature列的前一列
    label_name = 'pIC50'
    
    # ##随机划分训练集和测试集的随机种子
    randx = 0
    
    # ##描述符权重排序方法，可选["Pearson","SVM-RFE","RF-RFE"]
    ranker = 'Pearson'
    
    # ##建模算法，可选["SVC","DTC","RFC","SVR","RFR"]，也可直接传入一个学习器对象
    estimatorName = 'SVR'
    
    # ##网格寻优中交叉验证的重数
    gridCV = 3
    
    # ##网格寻优的重复次数
    repeat = 3
    
    # ##触发early_stop的最小mean_test_score变化率，如果设为0，则不执行early_stop
    early_stop = 0.01
    
    # ##采用带描述符数量的重复网格寻优，指定描述符数量的迭代范围，如果不迭代描述符数量，则删除或注释掉此参数
    # features_range = (19,20)
    
    # ##是否做散点图，回归任务可以设为True，分类任务请设为False
    make_fig = False
    
    """========================参数设定（不常变）=========================="""
    
    # ##测试集的划分比例
    test_size = 0.25                        
    
    # ##Pearson相关性筛选中描述符与活性相关性的下界，默认值为0.1
    corr_low = 0.1
    
    # ##Pearson相关性筛选中描述符之间两两相关性的上界，默认值为0.9
    corr_up = 0.9
    
    # ##连续型特征数据的压缩范围
    scale_range = (0.1,0.9)
    
    # ##最后拟合的模型做交叉验证的重数，可以是整数，或字符串"LOO"，不做cv可设为0
    make_cv = 5
    
    # ##是否保存最终模型文件
    save_model = True
    
    
    """========输入总描述符CSV文件→随机分集→相关性筛选(排序)→数据压缩 [ →RFE排序 ]=========="""
    
    spliter = randomSpliter(test_size=test_size,random_state=randx)
    spliter.ExtractTotalData(file_name,label_name=label_name)
    spliter.SplitData()
    
    tr_x = spliter.tr_x
    tr_y = spliter.tr_y
    te_y = spliter.te_y

    corr = correlationSelection()
    corr.PearsonXX(tr_x, tr_y,threshold_low=corr_low, threshold_up=corr_up)

    scaler = dataScale(scale_range=scale_range)

    tr_scaled_x = scaler.FitTransform(corr.selected_tr_x.iloc[:,:])
    te_scaled_x = scaler.Transform(spliter.te_x,DataSet='test')

    if ranker == 'Pearson':
        tr_ranked_x = tr_scaled_x
        te_ranked_x = te_scaled_x
    elif ranker == 'SVM-RFE':
        if np.unique(tr_y) <= 5:
            rfe = RFE_ranking('SVC',features_num=1)
        else:
            rfe = RFE_ranking('SVR',features_num=1)
        rfe.Fit(tr_scaled_x, tr_y)
        tr_ranked_x = rfe.tr_ranked_x
        te_ranked_x = te_scaled_x.loc[:,tr_ranked_x.columns]
        
    elif  ranker == 'RF-RFE':
        if np.unique(tr_y) <= 5:
            rfe = RFE_ranking('RFC',features_num=1)
        else:
            rfe = RFE_ranking('RFR',features_num=1)
        rfe.Fit(tr_scaled_x, tr_y)
        tr_ranked_x = rfe.tr_ranked_x
        te_ranked_x = te_scaled_x.loc[:,tr_ranked_x.columns]

    """=========repeat寻优，找出最佳参数 [ 及描述符数量 ]============="""

    grid = gridSearchPlus(grid_estimatorName=estimatorName, fold=gridCV, repeat=repeat, early_stop=early_stop, scoreThreshold=None)
    if('features_range' in vars()):
        grid.FitWithFeaturesNum(tr_ranked_x, tr_y,features_range=features_range)   # 用带描述符数量的寻优策略
    else:
        grid.Fit(tr_scaled_x,tr_y)  # 不带描述符数量的寻优


    """==============拟合模型，评价模型，保存结果================"""

    model = modeling(grid.best_estimator,params=grid.best_params)
    model.Fit(tr_scaled_x.loc[:,grid.best_features], tr_y)
    model.Predict(te_scaled_x.loc[:,grid.best_features],te_y)
    if make_cv:
        model.CrossVal(cv=make_cv)
        model.ShowResults(show_cv=True, make_fig=make_fig)
    else:
        model.ShowResults(show_cv=False, make_fig=make_fig)
    model.SaveResults(file_name[:-4]+'_results.csv',notes='{},split_seed={},gridCV={}'.format(ranker,randx,gridCV),
                      save_model=save_model)