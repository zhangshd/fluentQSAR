
# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVR,SVC
import pandas as pd

__all__ = ["correlationSelection","RFE_ranking"]

class correlationSelection(object):
    """Pearson相关系数筛选：
        PearsonXY()：删除与label相关性小于设定值的features
        PearsonXX()：删除与label相关性小于设定值的features，并且删除与label高相关特征的冗余特征(两两相关性大于设定值)"""
    def __init__(self):
        self.del_list_XX = []
        
    def PearsonXY(self,tr_x,tr_y,threshold_low=0.1):
        """找出与label的Pearson相关性大于设定值的feature
           
           参数:
           -----
           tr_x: Dataframe类型，为所有样本待筛选的feature值
           tr_y: Series类型，为所有样本的label值
           threshold_low: float数值，范围(0,1)，label相关性的下界"""
        self.adjustSTD = tr_x.std()/(tr_x.max()-tr_x.min())
        self.corrXY = tr_x.loc[:,self.adjustSTD>self.adjustSTD.quantile(q=0.1)].corrwith(tr_y).abs()
        # ### 求出每个feature与label的Pearson相关性(取绝对值)，筛除相关性低于设定值的feature，并且将剩下的feature按相关性降序排序
        self.high_corrXY_list = self.corrXY[self.corrXY>threshold_low].sort_values(ascending=False,inplace=False).index
        self.del_list_XY = tr_x.columns.drop(self.high_corrXY_list)
        
    def PearsonXX(self,tr_x,tr_y,threshold_low=0.1,threshold_up=0.9):
        """找出与label的Pearson相关性大于设定值的feature，对于两两Pearson相关性大于设定值的feature，再删除其中与label相关性较低者
           
           参数:
           -----
           tr_x: Dataframe类型，为所有样本待筛选的feature值
           tr_y: Series类型，为所有样本的label值
           threshold_low: float数值，范围(0,1)，label相关性的下界
           threshold_up: float数值，范围(0,1)，两两相关性的上界"""
        # ### 求出每个feature与label的Pearson相关性，筛除相关性低于设定值的feature，并且将剩下的feature按相关性降序排序
        self.PearsonXY(tr_x,tr_y,threshold_low=threshold_low)
        # ### 按相关性降序排列的feature从tr_x切片，再计算Pearson相关性矩阵(取绝对值)
        self.corrXX = tr_x.loc[:,self.high_corrXY_list].corr().abs()
        # ### 迭代矩阵对角线右上角的每行每列，如果发现有大于设定值的数值，则将该数值所对应的列名(feature)加到del_list_XX中
        for i in range(len(self.corrXX)):
            if self.corrXX.index[i] not in self.del_list_XX:
                for j in range (i+1,len(self.corrXX)):
                    if self.corrXX.columns[j] not in self.del_list_XX:
                        if self.corrXX.iloc[i,j]>threshold_up:
                            self.del_list_XX.append(self.corrXX.columns[j])
        # ### 从high_corrXY_list中删掉del_list_XX中包含的所有feature
        self.selected_feature = self.high_corrXY_list.drop(self.del_list_XX)
        self.selected_tr_x = tr_x.loc[:,self.selected_feature]
        
class RFE_ranking(object):
    """用于分类任务的'Recursive Feature Elimination'方法"""
    def __init__(self,estimator,features_num=1,**params):
        """
        参数：
        -----
        estimator: string或object型，指定学习器对象，string型仅限于'SVC'、'SVR'、'RFC'、'RFR'
        features_num: int型，递归完成后留下的特征数量(设置为1则能够让每个特征都有唯一的排名次序)，
                      对应sklearn的RFE中'n_features_to_select'参数
        **params: 学习器的超参数，如SVM有C、gamma、epsilon等，RF有n_estimators、max_leaf_nodes、random_state等"""
        self.feature_num = features_num
        if type(estimator) == str:
            if estimator == 'SVC':
                self.estimator = SVC(kernel='linear')
            elif estimator == 'SVR':
                self.estimator = SVR(kernel='linear')
            elif estimator == 'RFC':
                self.estimator = RandomForestClassifier(n_estimators=50,random_state=0,n_jobs=-1)
            elif estimator == 'RFR':
                self.estimator = RandomForestRegressor(n_estimators=50,random_state=0,n_jobs=-1)
            else:
                print("通过字符串指定estimator仅限于'SVC'、'SVR'、'RFC'、'RFR'!")
        else:
            self.estimator = estimator
        if len(params) != 0:
            self.estimator.set_params(**params)
            
    def Fit(self,tr_scaled_x,tr_y):
        """执行RFE
        参数：
        -----
        tr_scaled_x: DataFrame类型(连续型数据要经过压缩或标准化处理)，样本feature数据
        tr_y: array样类型(一维)，样本label数据，长度(行数)需与tr_scaled_x一致"""
        self.filter = RFE(estimator=self.estimator, n_features_to_select=self.feature_num).fit(tr_scaled_x, tr_y)
        self.descriptors_list = pd.Series(index=self.filter.ranking_,data=tr_scaled_x.columns).sort_index().tolist()
        self.tr_ranked_x = tr_scaled_x.loc[:,self.descriptors_list]
        

if __name__ == '__main__':
    pass