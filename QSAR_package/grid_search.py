# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>
#         Qin Zijian <zijianqin@foxmail.com>
#         Tu GuiPing <>

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold
from sklearn.metrics import make_scorer,accuracy_score,mean_squared_error,matthews_corrcoef
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import pandas as pd
import numpy as np
from time import time
import copy

__all__ = ["gridSearchBase","gridSearchPlus"]

class gridSearchBase(object):
    """gridsearch通用版，自定义传入寻优所用学习器、寻优字典、打分器"""
    def __init__(self,fold=5,grid_estimator=None,grid_dict=None,grid_scorer=None,repeat=10,
                 early_stop=None,scoreThreshold=None,stratified=False):
        """参数：
           -----
           fold：int型，交叉验证重数
           grid_estimator：object型，需参数寻优的学习器
           grid_dict：dict型，寻优参数字典
           grid_scorer：string或object型，寻优所用打分器，可通过string选择的仅有'accuracy'和'mse'
           repeat：int型，gridsearch重复次数，参考文献Krstajic D, Journal of Cheminformatics, 2014, 6(1):1-15
           scoreThreshold：float型，第一轮gridsearch后用于筛选参数组合的阈值，将grid_scorer打分低于阈值的参数组合舍去，不进行后面repeat-1轮的gridsearch，从而提高运算效率，对于分越高越好的scorer(如accuracy_score)，阈值设定区间宜为[0,1)，对于分越低越好的scorer(如mean_squared_error)，阈值设定区间宜为[-1,0)，阈值越接近上界筛选力度越大
           stratified：bool型，决定是否采用分层抽取来产生交叉验证数据集
           early_stop：float型或其他数据类型，正常情况gridsearch所选的最优参数组合是交叉验证平均得分（mean_test_score）最高的参数组合，\
           如果设置这个参数为float型数字（大于0小于1），则会从最高分开始向下寻找（分值按降序排列）得分与最高分有显著差异的次优参数组合，\
           显著差异的标准就是该分值与最高分的差值占该分值的比率（取绝对值）大于指定的early_stop数值，最终选择的参数组合是降序排名在上述\
           次优参数组合前一名的参数组合"""
        self.fold = fold
        self.grid_estimator = grid_estimator
        self.best_estimator = grid_estimator
        self.grid_dict = grid_dict
        self.repeat = repeat
        self.early_stop = early_stop
        self.scoreThreshold = scoreThreshold
        self.__Stratify(stratified)
        self.__Scorer(grid_scorer)
    
    def __Scorer(self,grid_scorer):
        if grid_scorer == 'accuracy':
            self.grid_scorer = make_scorer(matthews_corrcoef,greater_is_better=True)
            
        elif grid_scorer == 'mse':
            self.grid_scorer = make_scorer(mean_squared_error,greater_is_better=False)
            
        else:
            self.grid_scorer = grid_scorer
    
    def __Stratify(self,stratified): 
        if stratified == True:
            self.grid_cv = [StratifiedKFold(n_splits=self.fold, shuffle=True,random_state=10*i) for i in range(self.repeat)]
        else:
            self.grid_cv = [KFold(n_splits=self.fold, shuffle=True,random_state=10*i) for i in range(self.repeat)]
    
    def __Sec2Time(self,seconds):  # convert seconds to time
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))
    
    
    def Fit(self,tr_scaled_x, tr_y,verbose=True):
        """执行gridsearch
           参数：
           -----
           tr_scaled_x：array型，训练样本的feature数据
           tr_y：array型(一维)，训练样本的label数据"""
        t0 = time()
        if len(np.unique(tr_y)) <= 5:
            self.__Stratify(stratified=True)
        for i in range(len(self.grid_cv)):
            t1 = time()
            # print(self.grid_dict)
            self.grid = GridSearchCV(self.grid_estimator,self.grid_dict,scoring=self.grid_scorer,iid=False,
                                     cv=self.grid_cv[i],n_jobs=-1,return_train_score=False)
            print('训练集数据shape:{}'.format(tr_scaled_x.shape))
            self.grid.fit(tr_scaled_x, tr_y)
            if verbose:
                print("第{}/{}次gridsearch，此轮耗时{}".format(i+1,self.repeat,self.__Sec2Time(time()-t1)))
            if i == 0:
                self.cv_results = pd.DataFrame(self.grid.cv_results_).loc[:,['params','mean_test_score']]
                if self.scoreThreshold == None:
                    self.scoreThreshold = self.cv_results.mean_test_score.median()
                    # print(self.scoreThreshold)
                elif self.scoreThreshold > self.cv_results.mean_test_score.quantile(0.8):
                # 如果设置的打分阈值太高，可能没有cv_results符合条件，则重新设置阈值为所有打分的0.8分位数
                    self.scoreThreshold = self.cv_results.mean_test_score.quantile(0.8)
                    print("\033[41;1mscoreThreshold值太高，重新设定为所有打分的0.8分位数{}\033[0m".format(self.scoreThreshold))
                self.cv_results = self.cv_results.loc[self.cv_results.mean_test_score >= self.scoreThreshold,:].reset_index(drop=True)
                
                
                # if len(self.cv_results) == 0:
                    # break
                self.grid_dict = copy.deepcopy(self.cv_results.params.tolist())
                # 把grid_dict中的值变成单元素列表
                for d in self.grid_dict:
                    for key in d.keys(): 
                        d[key] = [d[key]]
                
                if self.repeat==1: 
                    #如果寻优重复次数为1（即不重复），则将该次寻优结果的'mean_test_score'复制一份，便于求其均值
                    self.cv_results = pd.concat([self.cv_results,self.cv_results.loc[:,'mean_test_score']],axis=1)
                
            else:
                self.cv_results = pd.concat([self.cv_results,pd.DataFrame(self.grid.cv_results_).loc[:,'mean_test_score']],axis=1)
                
            
        # ### 提取每次gridsearch中所有参数组合的测试集打分结果，
        # ### 对于每个参数组合，求取所有轮gridsearch的测试集打分平均值，根据平均值选择最好的参数组合
        # if len(self.cv_results) == 0:
            # print("\033[41;1mscoreThreshold值太高，没有符合要求的参数组合，请重新设定！！！\033[0m")
            
        # else:
        self.cv_results["repeat_mean"] = self.cv_results.mean_test_score.mean(axis=1)
        self.cv_results.sort_values(by="repeat_mean",ascending=False,inplace=True)
        
        if type(self.early_stop) == float:
            print('\033[45m执行early_stop\033[0m')
            top_score = self.cv_results.repeat_mean[0]
            for k in range(1,len(self.cv_results)):
                score = self.cv_results.repeat_mean[k]
                if abs((top_score-score)/score) < self.early_stop:
                    best_k = k
                    continue
                else:
                    best_k = k-1
                    break
        else:
            best_k = 0 #如果不设定early_stop，则取打分最高的参数组合
        self.best_params = self.cv_results.iloc[best_k,0]
        self.best_features = tr_scaled_x.columns
        self.best_estimator.set_params(**self.best_params)
        self.best_estimator.fit(tr_scaled_x, tr_y)
        print('{}次gridsearch执行完毕，总耗时{}，可通过best_params属性查看最优参数，通过cv_results属性查看所有结果'\
             .format(self.repeat,self.__Sec2Time(time()-t0)))
    
    
    
    def FitWithFeaturesNum(self,tr_scaled_x,tr_y,features_range=None,verbose=True):
        """执行带描述符数量迭代的gridsearch
           参数：
           -----
           tr_scaled_x：array型，训练样本的feature数据
           tr_y：array型(一维)，训练样本的label数据
           features_range：(start, stop[, step])，描述符数量迭代范围"""
        t0 = time()
        if len(np.unique(tr_y)) <= 5:
            self.__Stratify(stratified=True)
        if features_range == None:
            self.features_range = (1,len(tr_scaled_x.columns),1)
        else:
            self.features_range = features_range
    
        def loop():
            self.featureNum_notNone = self.features_range[0]
            for i in range(*self.features_range):
                grid_dict = self.grid_dict
                
                for j in range(self.repeat):
                    t1 = time()
                    self.grid = GridSearchCV(self.grid_estimator,grid_dict,scoring=self.grid_scorer,iid=False,
                                             cv=self.grid_cv[j],n_jobs=-1,return_train_score=False)
                    self.grid.fit(tr_scaled_x.iloc[:,:i], tr_y)
                    if verbose:
                        print("{}个特征，第{}/{}次gridsearch，此轮耗时{}".format(i,j+1,self.repeat,self.__Sec2Time(time()-t1)))
                    if j == 0:
                        temp0 = pd.DataFrame(self.grid.cv_results_).loc[:,['params','mean_test_score']]
                        # print(temp0)
                        if self.scoreThreshold == None:
                            self.scoreThreshold = temp0.mean_test_score.median()
                            # print(self.scoreThreshold)
                        temp0 = temp0.loc[temp0.mean_test_score >= self.scoreThreshold,:].reset_index(drop=True)
                        # print(len(temp0))
                        if len(temp0) == 0:
                            if not hasattr(self,'cv_results'):  # 判断self.cv_results属性是否存在
                                self.featureNum_notNone += 1
                            if verbose:
                                print("\033[31m{}个特征的gridsearch中，没有符合要求的参数组合\033[0m".format(i))
                            break
                        grid_dict = copy.deepcopy(temp0.params.tolist())
                        # 把grid_dict中的值变成单元素列表
                        for d in grid_dict:
                            for key in d.keys(): 
                                d[key] = [d[key]]
                        
                    else:
                        temp0 = pd.concat([temp0,pd.DataFrame(self.grid.cv_results_).loc[:,'mean_test_score']],axis=1)
                        # print(len(temp0))
                
                if len(temp0) != 0:
                    if self.repeat == 1:
                        #如果寻优重复次数为1（即不重复），则将该次寻优结果的'mean_test_score'复制一份
                        temp0 = pd.concat([temp0,temp0.loc[:,'mean_test_score']],axis=1)
                        
                    temp0["repeat_mean"] = temp0.mean_test_score.mean(axis=1)
                    temp0.sort_values(by="repeat_mean",ascending=False,inplace=True)
                    # print(len(temp0))
                    if type(self.early_stop) == float:
                        print('\033[45m执行early_stop\033[0m')
                        top_score = temp0.repeat_mean[0]
                        for k in range(len(temp0)):
                            score = temp0.repeat_mean[k]
                            if abs((top_score-score)/score) < self.early_stop:
                                best_k = k
                                continue
                            else:
                                best_k = k-1
                                break
                        temp0 = temp0.iloc[[best_k],:]
                        
                    if i == self.featureNum_notNone:
                        temp0['n_features'] = len(tr_scaled_x.iloc[:,:i].columns)
                        # print(len(tr_scaled_x.iloc[:,:i].columns))
                        self.cv_results = temp0
                        
                    elif i > self.featureNum_notNone:
                        temp0['n_features'] = len(tr_scaled_x.iloc[:,:i].columns)
                        # print(len(tr_scaled_x.iloc[:,:i].columns))
                        self.cv_results = pd.concat([self.cv_results,temp0],axis=0,ignore_index=True)
                        
                
        # ### 提取每次gridsearch中所有参数组合的测试集打分结果，
        # ### 对于每个参数组合，求取所有轮gridsearch的测试集打分平均值，根据平均值选择最好的参数组合
        loop()
        if not hasattr(self,'cv_results'):
            self.scoreThreshold = None
            print("\033[41;1mscoreThreshold值太高，没有符合要求的参数组合，已重置为默认设定并重新执行寻优过程！\033[0m")
            loop()  # 重新执行寻优过程
        
        # self.cv_results["repeat_mean"] = self.cv_results.mean_test_score.mean(axis=1)
        self.cv_results.sort_values(by="repeat_mean",ascending=False,inplace=True)
        self.best_params = self.cv_results.iloc[0,0]
        self.best_features = tr_scaled_x.iloc[:,:self.cv_results.iloc[0,-1]].columns
        self.best_estimator.set_params(**self.best_params)
        self.best_estimator.fit(tr_scaled_x.loc[:,self.best_features], tr_y)
        print('{}×{}次gridsearch执行完毕，总耗时{}，可通过best_params属性查看最优参数，通过cv_results属性查看所有结果'\
        .format(len(range(*self.features_range)),self.repeat,self.__Sec2Time(time()-t0)))
    def SaveGridCVresults(self,path):
        self.cv_results.to_csv(path,index=False)
            
        
class gridSearchPlus(gridSearchBase):
    """gridsearch特别版，只适用于SVC、SVR、DTC、RFC、RFR等算法的寻优，无需传入寻优所用学习器、寻优字典、打分器，只需给定支持的算法名"""
    def __init__(self,grid_estimatorName='SVR',fold=5,repeat=10,early_stop=None,
                 scoreThreshold=None,stratified=False,random_state=0):
        """参数：
           -----
           grid_estimatorName：string型，指定需参数寻优的学习器名字，可选项有'SVC'、'SVR'、'DTC'、'RFC'、'RFR'
           fold：int型，交叉验证重数
           repeat：int型，gridsearch重复次数，参考文献Krstajic D, Journal of Cheminformatics, 2014, 6(1):1-15
           scoreThreshold：float型，第一轮gridsearch后用于筛选参数组合的阈值，将grid_scorer打分低于阈值的参数组合舍去，不进行后面repeat-1轮的gridsearch，从而提高运算效率，对于分越高越好的scorer(如accuracy_score)，阈值设定区间宜为[0,1)，对于分越低越好的scorer(如mean_squared_error)，阈值设定区间宜为[-1,0)，阈值越接近上界筛选力度越大
           stratified：bool型，决定是否采用分层抽取来产生交叉验证数据集
           random_state：int型，控制学习器（如随机森林）的随机状态
           early_stop：float型或其他数据类型，正常情况gridsearch所选的最优参数组合是交叉验证平均得分（mean_test_score）最高的参数组合，\
           如果设置这个参数为float型数字（大于0小于1），则会从最高分开始向下寻找（分值按降序排列）得分与最高分有显著差异的次优参数组合，\
           显著差异的标准就是该分值与最高分的差值占该分值的比率（取绝对值）大于指定的early_stop数值，最终选择的参数组合是降序排名在上述\
           次优参数组合前一名的参数组合"""
        self.__grid_estimatorName = grid_estimatorName
        self.fold = fold
        self.repeat = repeat
        self.random_state = random_state
        self.early_stop = early_stop
        self.scoreThreshold = scoreThreshold
        super()._gridSearchBase__Stratify(stratified)
        self.__SelectEstimator()
        self.best_estimator = self.grid_estimator
        
    def __SelectEstimator(self):
        """根据输入grid_estimatorName选择使用的算法及其对应的寻优参数字典"""
        if self.__grid_estimatorName[-1] == 'C':
            self.grid_scorer = make_scorer(matthews_corrcoef,greater_is_better=True)   # 分类任务选择准确率打分器
            
            if self.__grid_estimatorName == 'SVC':
                self.grid_dict = {'C':[2**i for i in range(-10,11)],
                                  'gamma':[2**i for i in range(-10,11)]}
                self.grid_estimator = SVC()
            if self.__grid_estimatorName == 'DTC':
                self.grid_dict = {'max_leaf_nodes':[i for i in range(10,51,2)]}
                self.grid_estimator = DecisionTreeClassifier(random_state=self.random_state)
            if self.__grid_estimatorName == 'RFC':
                self.grid_dict = {'max_leaf_nodes':[i for i in range(10,51,2)],
                                  'n_estimators':[i for i in range(10,101,5)]}
                self.grid_estimator = RandomForestClassifier(random_state=self.random_state)
        if self.__grid_estimatorName[-1] == 'R':
            self.grid_scorer = make_scorer(mean_squared_error,greater_is_better=False)      # 回归任务选择均方误差打分器
            
            if self.__grid_estimatorName == 'SVR':
                self.grid_dict = {'C':[2**i for i in range(-10,11)],
                                  'gamma':[2**i for i in range(-10,11)],
                                  'epsilon':[2**i for i in range(-15,0)]}
                self.grid_estimator = SVR()
            if self.__grid_estimatorName == 'RFR':
                self.grid_dict = {'max_leaf_nodes':[i for i in range(20,61,2)],
                                  'n_estimators':[i for i in range(50,101,10)]}
                self.grid_estimator = RandomForestRegressor(random_state=self.random_state)

if __name__ == '__main__':
    pass