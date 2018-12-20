# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVR,SVC
import pandas as pd

class classificationRFE(object):
    """用于分类任务的'Recursive Feature Elimination'方法"""
    def __init__(self,features_num=1):
        """
        参数：
        -----
        features_num: 递归完成后留下的特征数量(设置为1则能够让每个特征都有唯一的排名次序)，
                      对应sklearn的RFE中'n_features_to_select'参数"""
        self.feature_num = features_num
        
    def RF_RFE(self,tr_scaled_x,tr_y,n_estimators=50,random_state=0):
        """使用随机森林学习器的RFE
        参数：
        -----
        tr_scaled_x: 数组样类型(连续型数据要经过压缩或标准化处理)，样本feature数据
        tr_y: 样本label数据，长度(行数)需与tr_scaled_x一致
        n_estimators: 正整数，随机森林中树的数量
        random_state: 任意整数，不同的数字代表不同随机状态"""
        self.estimator = RandomForestClassifier(n_estimators=n_estimators,random_state=random_state,n_jobs=-1)
        self.filter = RFE(estimator=self.estimator, n_features_to_select=self.feature_num).fit(tr_scaled_x, tr_y)
        self.descriptors_list = pd.Series(index=self.filter.ranking_,data=tr_scaled_x.columns).sort_index().tolist()
        self.tr_x_ranked = tr_scaled_x.loc[:,self.descriptors_list]
        
    def SVM_RFE(self,tr_scaled_x,tr_y,C=1.0):
        """使用线性SVM(只有线性核的SVM能用于RFE)学习器的RFE
        参数：
        -----
        tr_scaled_x: 数组样类型(连续型数据要经过压缩或标准化处理)，样本feature数据
        tr_y: 样本label数据，长度(行数)需与tr_scaled_x一致
        C: 浮点数，SVM的惩罚参数C"""
        self.estimator = SVC(kernel='linear',C=C)
        self.filter = RFE(estimator=self.estimator, n_features_to_select=self.feature_num).fit(tr_scaled_x, tr_y)
        self.descriptors_list = pd.Series(index=self.filter.ranking_,data=tr_scaled_x.columns).sort_index().tolist()
        self.tr_x_ranked = tr_scaled_x.loc[:,self.descriptors_list]
        
    def otherRFE(self,tr_scaled_x,tr_y,estimator=None):
        """使用其他学习器的RFE，需传入自定义的学习器"""
        self.estimator = estimator
        self.filter = RFE(estimator=self.estimator, n_features_to_select=self.feature_num).fit(tr_scaled_x, tr_y)
        self.descriptors_list = pd.Series(index=self.filter.ranking_,data=tr_scaled_x.columns).sort_index().tolist()
        self.tr_x_ranked = tr_scaled_x.loc[:,self.descriptors_list]
        
class regressionRFE(classificationRFE):
    """用于回归任务的'Recursive Feature Elimination'方法"""
    
    def RF_RFE(self,tr_scaled_x,tr_y,n_estimators=50,random_state=0):
        """使用随机森林学习器的RFE
        参数：
        -----
        tr_scaled_x: 数组样类型(连续型数据要经过压缩或标准化处理)，样本feature数据
        tr_y: 样本label数据，长度(行数)需与tr_scaled_x一致
        n_estimators: 正整数，随机森林中树的数量
        random_state: 任意整数，不同的数字代表不同随机状态"""
        self.estimator = RandomForestRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=-1)
        self.filter = RFE(estimator=self.estimator, n_features_to_select=self.feature_num).fit(tr_scaled_x, tr_y)
        self.descriptors_list = pd.Series(index=self.filter.ranking_,data=tr_scaled_x.columns).sort_index().tolist()
        self.tr_x_ranked = tr_scaled_x.loc[:,self.descriptors_list]
        
    def SVM_RFE(self,tr_scaled_x,tr_y,C=1.0,epsilon=0.1):
        """使用线性SVM(只有线性核的SVM能用于RFE)学习器的RFE
        参数：
        -----
        tr_scaled_x: 数组样类型(连续型数据要经过压缩或标准化处理)，样本feature数据
        tr_y: 样本label数据，长度(行数)需与tr_scaled_x一致
        C: 浮点数，SVM的惩罚参数C
        epsilon: 浮点数，SVM损失函数中的ε参数"""
        self.estimator = SVR(kernel='linear',C=C,epsilon=epsilon)
        self.filter = RFE(estimator=self.estimator, n_features_to_select=self.feature_num).fit(tr_scaled_x, tr_y)
        self.descriptors_list = pd.Series(index=self.filter.ranking_,data=tr_scaled_x.columns).sort_index().tolist()
        self.tr_x_ranked = tr_scaled_x.loc[:,self.descriptors_list]

if __name__ == '__main__':
    pass