
# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

__all__ = ["randomSpliter"]

class randomSpliter(object):

    def __init__(self,test_size=0.25,random_state=0):
        """将数据集随机分成训练集和测试集，对于分类任务的数据，采取分层抽样；对于回归任务的数据，label值最大和最小的样本会被分到训练集中
           参数：
           -----
           test_size：float型(范围(0,1))，测试集比例
           random_state：int型，控制随机状态
        """
        self.test_size = test_size
        self.random_state = random_state
        

    def __ExtractFile(self,file_name,label_name):
        """从CSV文件(CSV文件中label列在第一列feature前)提取数据"""
        self.init_df = pd.read_csv(file_name,dtype=np.str)
        self.__norm_df = self.init_df.loc[:,label_name:].astype(np.float64)
    
        
    def splitClassificationData(self,data,label_name='activity'):
        """把用于分类建模的数据集分为训练集和测试集，\033[35;1m！！注意：\033[30m第一个feature列必须紧接在label列之后\033[0m
           
           参数：
           -----
           data：string或DataFrame型，string视为数据CSV文件路径，DataFrame视为数据对象
           label_name：string型，label列的列名"""
        
        if type(data) is str:
            self.__ExtractFile(data,label_name)
        else:
            self.init_df = data
            self.__norm_df = self.init_df.loc[:,label_name:].astype(np.float64)
        
        self.feature_all = self.__norm_df.iloc[:,1:]
        self.label_all = self.__norm_df.loc[:,label_name].astype(np.int8)
        
        self.tr_x,self.te_x = train_test_split(self.feature_all,test_size=self.test_size,
                                               random_state=self.random_state,stratify=self.label_all)
        
        self.tr_ids = self.tr_x.index
        self.te_ids = self.te_x.index
        
        self.tr_y = self.label_all[self.tr_ids]
        self.te_y = self.label_all[self.te_ids]

    def splitRegressionData(self,data,label_name='pIC50'):
        """把用于回归建模的数据集分为训练集和测试集，\033[35;1m！！注意：\033[30m第一个feature列必须紧接在label列之后\033[0m
           
           参数：
           -----
           data：string或DataFrame型，string视为数据CSV文件路径，DataFrame视为数据对象
           label_name：string型，label列的列名"""
        if type(data) is str:
            self.__ExtractFile(data,label_name)
        else:
            self.init_df = data
            self.__norm_df = self.init_df.loc[:,label_name:].astype(np.float64)
        
        self.feature_all = self.__norm_df.iloc[:,1:]
        self.label_all = self.__norm_df.loc[:,label_name]
        self.max_min_x = self.feature_all.loc[[self.label_all.idxmax(),self.label_all.idxmin()],:]
        self.x_withoutMaxMin = self.feature_all.drop(index=[self.label_all.idxmax(),self.label_all.idxmin()])
        
        self.tr_x,self.te_x = train_test_split(self.x_withoutMaxMin,test_size=self.test_size,random_state=self.random_state)
        self.tr_x = self.tr_x.append(self.max_min_x)
        
        self.tr_ids = self.tr_x.index
        self.te_ids = self.te_x.index
        
        self.tr_y = self.label_all[self.tr_ids]
        self.te_y = self.label_all[self.te_ids]
        
        

if __name__ == '__main__':
    pass