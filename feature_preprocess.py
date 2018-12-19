
# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

import pandas as pd

class correlationSelection(object):
    def __init__(self):
        self.del_list_XX = []
        
    def pearsonXY(self,tr_x,tr_y,threshold_low=0.1):
        """找出与label的Pearson相关性大于设定值的feature
           
           参数:
           -----
           tr_x: Dataframe类型，为所有样本待筛选的feature值
           tr_y: Series类型，为所有样本的label值
           threshold_low: float数值，范围(0,1)，label相关性的下界"""
        self.corrXY = tr_x.corrwith(tr_y).abs()
        # ### 求出每个feature与label的Pearson相关性(取绝对值)，筛除相关性低于设定值的feature，并且将剩下的feature按相关性降序排序
        self.high_corrXY_list = self.corrXY[self.corrXY>threshold_low].sort_values(ascending=False,inplace=False).index
        
    def pearsonXX(self,tr_x,tr_y,threshold_low=0.1,threshold_up=0.9):
        """找出与label的Pearson相关性大于设定值的feature，对于两两Pearson相关性大于设定值的feature，再删除其中与label相关性较低者
           
           参数:
           -----
           tr_x: Dataframe类型，为所有样本待筛选的feature值
           tr_y: Series类型，为所有样本的label值
           threshold_low: float数值，范围(0,1)，label相关性的下界
           threshold_up: float数值，范围(0,1)，两两相关性的上界"""
        # ### 求出每个feature与label的Pearson相关性，筛除相关性低于设定值的feature，并且将剩下的feature按相关性降序排序
        self.pearsonXY(tr_x,tr_y,threshold_low=threshold_low)
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
        

if __name__ == '__main__':
    pass