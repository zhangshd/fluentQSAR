
# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

__all__ = ["extractData","randomSpliter"]

class extractData(object):
    """方法介绍(使用此类的前提是\033[35;1mlabel列放于第一个feature列的前一列\033[0m)：
        ExtractTotalData()：从总数据集文件提取总数据集
        ExtractTrainTestData()：从训练集、测试集文件提取训练集测试集
        ExtractTrainTestFromLabel()：从总样本数据文件及训练集测试集标签文件提取训练集测试集"""
    def __init__(self):
        pass
    def ExtractTotalData(self,data_path,label_name='Activity'):
        """从总样本数据文件提取总数据，\033[35;1m！！注意：\033[30mlabel列必须放于第一个feature列的前一列\033[0m
        参数：
        -----
        data_path：string型，总数据集文件路径
        label_name：string型，label列的名字"""
        self.init_df = pd.read_csv(data_path,dtype=np.str)
        self.norm_df = self.init_df.loc[:,label_name:].astype(np.float32)
        
    def ExtractTrainTestData(self,train_path,test_path,label_name='Activity'):
        """从训练集、测试集文件提取训练集测试集，\033[35;1m！！注意：\033[30mlabel列必须放于第一个feature列的前一列\033[0m
        参数：
        -----
        train_path：string型，训练集文件路径
        test_path：string型，测试集文件路径
        label_name：string型，label列的名字
        """
        self.ExtractTotalData(train_path,label_name)
        self.tr_x = self.norm_df.iloc[:,1:]
        self.tr_y = self.norm_df.loc[:,label_name]
        self.ExtractTotalData(test_path,label_name)
        self.te_x = self.norm_df.iloc[:,1:]
        self.te_y = self.norm_df.loc[:,label_name]
        del self.init_df,self.norm_df
        if len(self.tr_y.unique())<=5: #判断是否为分类数据的标签，如果是的话，把标签列转换为整型
            self.tr_y = self.tr_y.astype(np.int32)
            self.te_y = self.te_y.astype(np.int32)
    def ExtractTrainTestFromLabel(self,data_path,trOte_path,label_name='Activity'):
        """从总样本数据文件及训练集测试集标签文件(训练集、测试集分别以"tr"、"te"表示)提取训练集测试集，\
        \033[35;1m！！注意：\033[30mlabel列必须放于第一个feature列的前一列\033[0m
        参数：
        -----
        data_path：string型，总数据集文件路径
        trOte_path：string型，训练集测试集标签文件路径，文件中训练集、测试集分别以"tr"、"te"表示
        label_name：string型，label列的名字
        """
        self.trOte = pd.read_csv(trOte_path,squeeze=True)
        self.ExtractTotalData(data_path,label_name)
        self.feature_all = self.norm_df.iloc[:,1:]
        self.label_all = self.norm_df.loc[:,label_name]
        self.tr_x = self.feature_all.loc[self.trOte=='tr',:]
        self.te_x = self.feature_all.loc[self.trOte=='te',:]
        self.tr_y = self.label_all[self.trOte=='tr']
        self.te_y = self.label_all[self.trOte=='te'] 
        if len(self.tr_y.unique())<=5:
            self.tr_y.astype(np.int32)
            self.te_y.astype(np.int32)
    
    def ExtractFeatureFromTXT(self,feature_txt_path=None):
        """在已经提取了训练集和测试集数据的情况下（通过文件提取或随机划分得到），通过一个包含若干特征名的txt文件提取的部分特征数据，\
        不在txt文件中的特征会被排除
        参数：
        -----
        feature_txt_path：string型，存放所需特征名的txt文件路径，每个特征名之间用换行符分隔
        """
        if type(feature_txt_path) == str:
            self.des_list = np.loadtxt(feature_txt_path,dtype=np.str).tolist()
            self.tr_x = self.tr_x.loc[:,self.des_list]
            self.te_x = self.te_x.loc[:,self.des_list]
        
class randomSpliter(extractData):
    """将数据集随机分成训练集和测试集，对于分类任务的数据，采取分层抽样；对于回归任务的数据，label值最大和最小的样本会被分到训练集中
       example：
       -------------以下为必要步骤-----------------
       spliter = randomSpliter(test_size=0.25,random_state=0)
       spliter.ExtractTotalData(data_path,label_name='Activity')
       spliter.SplitRegressionData()
       -------------以下为可选步骤-----------------
       spliter.SaveTrainTestLabel(trOte_path)"""
    def __init__(self,test_size=0.25,validation_size=None,random_state=0):
        """参数：
           -----
           test_size：float型(范围(0,1))，测试集比例
           random_state：int型，控制随机状态
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
    def SplitData(self):
        if len(self.norm_df.iloc[:,0].unique()) <= 5:
            self.SplitClassificationData()
        else:
            self.SplitRegressionData()
    
    def SplitClassificationData(self):
        """把用于分类建模的数据集分为训练集和测试集，\033[35;1m！！注意：\033[30mlabel列必须放于第一个feature列的前一列\033[0m
           
           参数：
           -----
           data：string或DataFrame型，string视为数据CSV文件路径，DataFrame视为数据对象
           label_name：string型，label列的列名"""
        
        self.feature_all = self.norm_df.iloc[:,1:]
        self.label_all = self.norm_df.iloc[:,0].astype(np.int32)
        
        self.tr_x,self.te_x = train_test_split(self.feature_all,test_size=self.test_size,
                                               random_state=self.random_state,stratify=self.label_all)
        
        if type(self.validation_size) == float:
            self.tr_x,self.val_x = train_test_split(self.tr_x,test_size=self.validation_size,
                                                    random_state=self.random_state,stratify=self.label_all[self.tr_x.index])
            self.val_ids = self.val_x.index
            self.val_y = self.label_all[self.val_ids]
            
        self.tr_ids = self.tr_x.index
        self.te_ids = self.te_x.index
        
        self.tr_y = self.label_all[self.tr_ids]
        self.te_y = self.label_all[self.te_ids]

    def SplitRegressionData(self):
        """把用于回归建模的数据集分为训练集和测试集，\033[35;1m！！注意：\033[30mlabel列必须放于第一个feature列的前一列\033[0m
           
           参数：
           -----
           data：string或DataFrame型，string视为数据CSV文件路径，DataFrame视为数据对象
           label_name：string型，label列的列名"""
        
        self.feature_all = self.norm_df.iloc[:,1:]
        self.label_all = self.norm_df.iloc[:,0]
        self.max_min_x = self.feature_all.loc[[self.label_all.idxmax(),self.label_all.idxmin()],:]
        self.x_withoutMaxMin = self.feature_all.drop(index=[self.label_all.idxmax(),self.label_all.idxmin()])
        
        self.tr_x,self.te_x = train_test_split(self.x_withoutMaxMin,test_size=self.test_size,random_state=self.random_state)
        if type(self.validation_size) == float:
            self.tr_x,self.val_x = train_test_split(self.tr_x,test_size=self.validation_size,random_state=self.random_state)
            self.val_ids = self.val_x.index
            self.val_y = self.label_all[self.val_ids]
            
        self.tr_x = self.tr_x.append(self.max_min_x)
        
        self.tr_ids = self.tr_x.index
        self.te_ids = self.te_x.index
        
        self.tr_y = self.label_all[self.tr_ids]
        self.te_y = self.label_all[self.te_ids]
        
    def SaveTrainTestLabel(self,trOte_path):
        """将训练集测试集的划分结果以"tr"、"te"标签的形式保存到文件
        参数：
        -----
        trOte_path：string型，训练集测试集标签文件保存路径"""
        self.trOte = pd.Series(data=['tr']*len(self.tr_ids),index=self.tr_ids,name='trOte')\
                     .append(pd.Series(data=['te']*len(self.te_ids),index=self.te_ids,name='trOte'))
        self.trOte.sort_index(inplace=True)
        self.trOte.to_csv(trOte_path,header=True,index=False)
        

if __name__ == '__main__':
    pass