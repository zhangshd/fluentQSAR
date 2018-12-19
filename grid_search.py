# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold
from sklearn.metrics import make_scorer,accuracy_score,mean_squared_error
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import pandas as pd
from time import time
import copy

class gridSearchBase(object):
    """gridsearch通用版，自定义传入寻优所用学习器、寻优字典、打分器"""
    def __init__(self,fold=5,grid_estimator=None,grid_dict=None,grid_scorer=None,repeat=10,scoreThreshold=None,stratified=False):
        """参数：
           -----
           fold：int型，交叉验证重数
           grid_estimator：object型，需参数寻优的学习器
           grid_dict：dict型，寻优参数字典
           grid_scorer：string或object型，寻优所用打分器，可通过string选择的仅有'accuracy'和'mse'
           repeat：int型，gridsearch重复次数，参考文献Krstajic D, Journal of Cheminformatics, 2014, 6(1):1-15
           scoreThreshold：float型，第一轮gridsearch后用于筛选参数组合的阈值，将grid_scorer打分低于阈值的参数组合舍去，不进行后面repeat-1轮的gridsearch，从而提高运算效率，对于分越高越好的scorer(如accuracy_score)，阈值设定区间宜为[0,1)，对于分越低越好的scorer(如mean_squared_error)，阈值设定区间宜为[-1,0)，阈值越接近上界筛选力度越大
           stratified：bool型，决定是否采用分层抽取来产生交叉验证数据集"""
        self.fold = fold
        self.grid_estimator = grid_estimator
        self.grid_dict = grid_dict
        self.repeat = repeat
        self.scoreThreshold = scoreThreshold
        self.__stratify(stratified)
        self.__scorer(grid_scorer)
    
    def __scorer(self,grid_scorer):
        if grid_scorer == 'accuracy':
            self.grid_scorer = make_scorer(accuracy_score,greater_is_better=True)
            
        elif grid_scorer == 'mse':
            self.grid_scorer = make_scorer(mean_squared_error,greater_is_better=False)
            
        else:
            self.grid_scorer = grid_scorer
    
    def __stratify(self,stratified): 
        if stratified == True:
            self.grid_cv = [StratifiedKFold(n_splits=self.fold, shuffle=True,random_state=10*i) for i in range(self.repeat)]
        else:
            self.grid_cv = [KFold(n_splits=self.fold, shuffle=True,random_state=10*i) for i in range(self.repeat)]
    def __sec2time(self,seconds):  # convert seconds to time
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))
    
    
    def fit(self,tr_scaled_x, tr_y,verbose=True):
        """执行gridsearch
           参数：
           -----
           tr_scaled_x：array型，训练样本的feature数据
           tr_y：array型(一维)，训练样本的label数据"""
        t0 = time()
        for i in range(len(self.grid_cv)):
            t1 = time()
            self.grid = GridSearchCV(self.grid_estimator,self.grid_dict,scoring=self.grid_scorer,
                                     cv=self.grid_cv[i],n_jobs=-1,return_train_score=False)
            self.grid.fit(tr_scaled_x, tr_y)
            if verbose:
                print("第{}/{}次gridsearch，此轮耗时{}".format(i+1,self.repeat,self.__sec2time(time()-t1)))
            if i == 0:
                self.cv_results = pd.DataFrame(self.grid.cv_results_).loc[:,['params','mean_test_score']]
                if self.scoreThreshold == None:
                    self.scoreThreshold = self.cv_results.mean_test_score.median()
                self.cv_results = self.cv_results.loc[self.cv_results.mean_test_score > self.scoreThreshold,:].reset_index(drop=True)
                if len(self.cv_results) == 0:
                    break
                self.grid_dict = copy.deepcopy(self.cv_results.params.tolist())
                # 把grid_dict中的值变成单元素列表
                for d in self.grid_dict:
                    for key in d.keys(): 
                        d[key] = [d[key]]
            else:
                self.cv_results = pd.concat([self.cv_results,pd.DataFrame(self.grid.cv_results_).loc[:,'mean_test_score']],axis=1)
                
            
        # ### 提取每次gridsearch中所有参数组合的测试集打分结果，
        # ### 对于每个参数组合，求取所有轮gridsearch的测试集打分平均值，根据平均值选择最好的参数组合
        if len(self.cv_results) == 0:
            print("\033[41;1mscoreThreshold值太高，没有符合要求的参数组合，请重新设定！！！\033[0m")
        else:
            self.cv_results["mean"] = self.cv_results.mean_test_score.mean(axis=1)
            self.cv_results.sort_values(by="mean",ascending=False,inplace=True)
            self.best_params = self.cv_results.iloc[0,0]
            print('{}次gridsearch执行完毕，总耗时{}，可通过best_params属性查看最优参数，通过cv_results属性查看所有结果'\
                 .format(self.repeat,self.__sec2time(time()-t0)))
    
    
    
    def fit_with_nFeatures(self,tr_scaled_x,tr_y,features_range=None,verbose=True):
        """执行带描述符数量迭代的gridsearch
           参数：
           -----
           tr_scaled_x：array型，训练样本的feature数据
           tr_y：array型(一维)，训练样本的label数据
           features_range：(start, stop[, step])，描述符数量迭代范围"""
        t0 = time()
        if features_range == None:
            self.features_range = (1,len(tr_scaled_x.columns),1)
        else:
            self.features_range = features_range
        
        
        self.featureNum_notNone = self.features_range[0]
        for i in range(*self.features_range):
            grid_dict = self.grid_dict
            
            for j in range(self.repeat):
                t1 = time()
                self.grid = GridSearchCV(self.grid_estimator,grid_dict,scoring=self.grid_scorer,
                                         cv=self.grid_cv[j],n_jobs=-1,return_train_score=False)
                self.grid.fit(tr_scaled_x.iloc[:,:i], tr_y)
                if verbose:
                    print("{}个特征，第{}/{}次gridsearch，此轮耗时{}".format(i,j+1,self.repeat,self.__sec2time(time()-t1)))
                if j == 0:
                    temp0 = pd.DataFrame(self.grid.cv_results_).loc[:,['params','mean_test_score']]
                    if self.scoreThreshold == None:
                        self.scoreThreshold = temp0.mean_test_score.median()
                    temp0 = temp0.loc[temp0.mean_test_score > self.scoreThreshold,:].reset_index(drop=True)
                    if len(temp0) == 0 :
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
                
                    
            
            if i == self.featureNum_notNone:
                temp0['n_features'] = len(tr_scaled_x.iloc[:,:i].columns)
                self.cv_results = temp0
                
            elif i > self.featureNum_notNone and len(temp0) != 0:
                temp0['n_features'] = len(tr_scaled_x.iloc[:,:i].columns)
                self.cv_results = self.cv_results.append(temp0)
                
        # ### 提取每次gridsearch中所有参数组合的测试集打分结果，
        # ### 对于每个参数组合，求取所有轮gridsearch的测试集打分平均值，根据平均值选择最好的参数组合
        if not hasattr(self,'cv_results'):
            print("\033[41;1mscoreThreshold值太高，没有符合要求的参数组合，请重新设定！！！\033[0m")
        else:
            self.cv_results["mean"] = self.cv_results.mean_test_score.mean(axis=1)
            self.cv_results.sort_values(by="mean",ascending=False,inplace=True)
            self.best_params = self.cv_results.iloc[0,0]
            print('{}×{}次gridsearch执行完毕，总耗时{}，可通过best_params属性查看最优参数，通过cv_results属性查看所有结果'\
            .format(len(range(*self.features_range)),self.repeat,self.__sec2time(time()-t0)))
            
        
class gridSearchPlus(gridSearchBase):
    """gridsearch特别版，只适用于SVC、SVR、DTC、RFC、RFR等算法的寻优，无需传入寻优所用学习器、寻优字典、打分器，只需给定支持的算法名"""
    def __init__(self,grid_estimatorName='SVR',fold=5,repeat=10,scoreThreshold=None,stratified=False,random_state=0):
        """参数：
           -----
           grid_estimatorName：string型，指定需参数寻优的学习器名字，可选项有'SVC'、'SVR'、'DTC'、'RFC'、'RFR'
           fold：int型，交叉验证重数
           repeat：int型，gridsearch重复次数，参考文献Krstajic D, Journal of Cheminformatics, 2014, 6(1):1-15
           scoreThreshold：float型，第一轮gridsearch后用于筛选参数组合的阈值，将grid_scorer打分低于阈值的参数组合舍去，不进行后面repeat-1轮的gridsearch，从而提高运算效率，对于分越高越好的scorer(如accuracy_score)，阈值设定区间宜为[0,1)，对于分越低越好的scorer(如mean_squared_error)，阈值设定区间宜为[-1,0)，阈值越接近上界筛选力度越大
           stratified：bool型，决定是否采用分层抽取来产生交叉验证数据集
           random_state：int型，控制随机状态"""
        self.__grid_estimatorName = grid_estimatorName
        self.fold = fold
        self.repeat = repeat
        self.random_state = random_state
        self.scoreThreshold = scoreThreshold
        super()._gridSearchBase__stratify(stratified)
        self.__selectEstimator()
        
    def __selectEstimator(self):
        """根据输入grid_estimatorName选择使用的算法及其对应的寻优参数字典"""
        if self.__grid_estimatorName[-1] == 'C':
            self.grid_scorer = make_scorer(accuracy_score,greater_is_better=True)   # 分类任务选择准确率打分器
            
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
                self.grid_dict = {'max_leaf_nodes':[i for i in range(10,51,2)],
                                  'n_estimators':[i for i in range(10,101,5)]}
                self.grid_estimator = RandomForestRegressor(random_state=self.random_state)

if __name__ == '__main__':
    pass