# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

import pandas as pd

__all__ = ["dataScale"]

class dataScale(object):
    """数据区间压缩，只压缩连续型数据，二进制数据不压缩，被压缩的数据必须包含拟合压缩器所用的所有feature"""
    def __init__(self,scale_range=(0.1, 0.9)):
        """
           参数：
           -----
           scale_range: 数据区间压缩的范围，包含两个数值的序列(升序)"""
        
        self.scale_range = scale_range
        
    def Fit(self,tr_x):
        """拟合压缩器，压缩范围为类参数'scale_range'
        
           参数：
           -----
           tr_x: 数组样类型，为训练集所有样本的feature值"""
        self.tr_max = tr_x.max(axis=0)
        self.tr_min = tr_x.min(axis=0)
        self.feature = tr_x.columns
    def Transform(self,x,DataSet='test'):
        """用压缩器变换数据，如果为连续型数据则执行区间压缩变换，否则不进行变换
        
           参数：
           -----
           x: 数组样类型，为数据集样本的feature值
           DataSet: 字符串类型，如果为'train'则将压缩后的数据存入属性self.tr_scaled_x，
                    如果为'test'，则将压缩后的数据存入属性self.te_scaled_x
           """
        if self.tr_min.min() == 0 and self.tr_max.max() == 1:
            self.scaled_x = x.loc[:,self.feature]
        else:
            self.scaled_x = (x.loc[:,self.feature]-self.tr_min)/(self.tr_max-self.tr_min)*\
                            (self.scale_range[1]-self.scale_range[0])+self.scale_range[0]
        if DataSet == 'train':
            self.tr_scaled_x = self.scaled_x
        elif DataSet == 'validation':
            self.val_scaled_x = self.scaled_x
        elif DataSet == 'test':
            self.te_scaled_x = self.scaled_x
        return self.scaled_x
    
    def FitTransform(self,tr_x):
        """用于训练集数据Fit + Transform"""
        self.Fit(tr_x)
        self.Transform(tr_x,DataSet='train')
        return self.scaled_x


if __name__ == '__main__':
    pass